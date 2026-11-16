"""Round and game drivers.  Deterministic given a seed.

RNG layout (common random numbers): ``SeedSequence(seed).spawn(1 + N)`` gives
one stream for dealing and one per seat for that seat's policy.  Deals depend
only on the seed, never on which policies are seated or how much randomness
they consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .cards import BIT, SUIT_MASK, SUIT_OF, mask_of
from .rules import Rules
from .state import BID, DONE, PLAY, IllegalAction, InfoState, RoundState


def make_rngs(seed, n_players: int) -> tuple[np.random.Generator, list[np.random.Generator]]:
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    children = ss.spawn(1 + n_players)
    return np.random.default_rng(children[0]), [np.random.default_rng(c) for c in children[1:]]


def deal(rules: Rules, n_cards: int, rng: np.random.Generator) -> list[int]:
    perm = rng.permutation(rules.deck_size).tolist()
    N = rules.n_players
    return [mask_of(perm[p * n_cards:(p + 1) * n_cards]) for p in range(N)]


@dataclass
class RoundResult:
    n_cards: int
    trump: int
    dealer: int
    bids: list[int]
    tricks: list[int]
    scores: list[float]
    trace: dict | None = None


@dataclass
class GameResult:
    totals: list[float]
    wins: list[float]          # 1/k to each of k tied leaders
    rounds: list[RoundResult] = field(default_factory=list)
    trace: dict | None = None


def run_round(state: RoundState, policies: Sequence, rngs: Sequence[np.random.Generator],
              check: bool = True) -> RoundState:
    """Drive ``state`` to the end with the given per-seat policies.

    Works from any point (bidding or mid-play).  This is the hot loop: legal
    card masks are computed inline and the trick is resolved incrementally.
    With ``check`` the policy's returned action is validated.
    """
    N = state.N
    infos = [InfoState(state, p) for p in range(N)]
    while state.phase == BID:
        p = state.to_act
        legal = state.legal_bids()
        b = policies[p].bid(infos[p], legal, rngs[p])
        if check and b not in legal:
            raise IllegalAction(f"policy {policies[p]!r} bid {b}, legal {legal}")
        state.apply_bid(b)
    hands = state.hands
    apply = state._apply_card
    while state.phase == PLAY:
        p = state.to_act
        hand = hands[p]
        trick = state.trick
        if trick:
            legal = hand & SUIT_MASK[SUIT_OF[trick[0][1]]] or hand
        else:
            legal = hand
        c = policies[p].play(infos[p], legal, rngs[p])
        if check and not (0 <= c < 52 and legal & BIT[c]):
            raise IllegalAction(f"policy {policies[p]!r} played illegal card {c}")
        apply(c)
    return state


def play_round(rules: Rules, policies: Sequence, seed=None, *, round_index: int = 0,
               n_cards: int | None = None, trump: int | None = None, dealer: int | None = None,
               hands: list[int] | None = None, deal_rng: np.random.Generator | None = None,
               policy_rngs: Sequence[np.random.Generator] | None = None,
               record_trace: bool = False) -> RoundResult:
    """Play one round.  Schedule fields default from ``round_index`` under ``rules``."""
    N = rules.n_players
    if len(policies) != N:
        raise ValueError("need one policy per seat")
    if deal_rng is None or policy_rngs is None:
        d, prs = make_rngs(seed, N)
        if deal_rng is None:
            deal_rng = d
        if policy_rngs is None:
            policy_rngs = prs
    n = rules.ladder()[round_index] if n_cards is None else n_cards
    t = rules.trump_for_round(round_index) if trump is None else trump
    dl = rules.dealer_for_round(round_index) if dealer is None else dealer
    if hands is None:
        hands = deal(rules, n, deal_rng)
    st = RoundState(rules, n, t, dl, hands)
    initial = list(st.hands)
    run_round(st, policies, policy_rngs)
    res = RoundResult(n, t, dl, list(st.bids), list(st.tricks_won), st.scores())
    if record_trace:
        from .trace import round_trace
        res.trace = round_trace(st, initial, round_index)
    return res


def fractional_wins(totals: Sequence[float]) -> list[float]:
    top = max(totals)
    k = sum(1 for x in totals if x == top)
    return [1.0 / k if x == top else 0.0 for x in totals]


def play_game(rules: Rules, policies: Sequence, seed, record_trace: bool = True) -> GameResult:
    N = rules.n_players
    if len(policies) != N:
        raise ValueError("need one policy per seat")
    deal_rng, prngs = make_rngs(seed, N)
    totals = [0.0] * N
    rounds = []
    for r, n in enumerate(rules.ladder()):
        rr = play_round(rules, policies, round_index=r, deal_rng=deal_rng, policy_rngs=prngs,
                        record_trace=record_trace)
        for p in range(N):
            totals[p] += rr.scores[p]
        rounds.append(rr)
    res = GameResult(totals, fractional_wins(totals), rounds)
    if record_trace:
        from .trace import game_trace
        res.trace = game_trace(rules, seed, rounds, totals, res.wins,
                               [getattr(p, "name", type(p).__name__) for p in policies])
    return res

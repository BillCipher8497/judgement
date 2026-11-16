"""Simulation helpers for Phase 1: data generation for the bidder and a
bid-conditional Monte-Carlo oracle (needed because target-aware play makes the
trick distribution depend on the hero's own bid)."""
from __future__ import annotations

import numpy as np

from ..cards import NT
from ..engine import deal, make_rngs, run_round
from ..sampling import HandSampler
from ..state import BID, RoundState, state_from_info
from .model import Agent, info_features, score_matrix


def random_trump(rng, p_nt=0.4) -> int:
    return NT if rng.random() < p_nt else int(rng.integers(4))


def model_bid(model, rules, x, n, trump, legal, S_cache) -> tuple[int, np.ndarray]:
    S = S_cache.get(n)
    if S is None:
        S = S_cache[n] = score_matrix(rules, n)
    P = model.get(n, trump).dist_all_bids(x)
    ev = (P * S).sum(1)
    return max(legal, key=lambda b: (ev[b], -b)), ev


def simulate_rounds(rules, play_policy, n: int, n_rounds: int, seed: int, model=None,
                    fallback_bidder=None, explore_prob: float = 0.0, p_nt: float = 0.4,
                    trump: int | None = None, score_rules=None):
    """Rounds with dealer 0; every seat bids with ``model`` (or the fallback) and
    plays with ``play_policy``.  With prob ``explore_prob`` one random seat bids a
    random offset in {-2,-1,1,2} from the model bid (nearest legal).
    Returns a dict of numpy arrays, one row per (round, seat)."""
    rng = np.random.default_rng(seed)
    N = rules.n_players
    pols = [play_policy] * N
    prng = [rng] * N
    S_cache: dict = {}
    srules = score_rules or rules
    out = {k: [] for k in ("round", "n", "trump", "pos", "seat", "x", "bid", "tricks", "score",
                           "explored", "sum_bids")}
    for r in range(n_rounds):
        tr = trump if trump is not None else random_trump(rng, p_nt)
        hands = deal(rules, n, rng)
        st = RoundState(rules, n, tr, 0, hands)
        explorer = int(rng.integers(N)) if rng.random() < explore_prob else -1
        xs, expl = {}, {}
        while st.phase == BID:
            p = st.to_act
            info = st.info_state(p)
            x = info_features(info)
            legal = st.legal_bids()
            if model is not None:
                b, _ = model_bid(model, srules, x, n, tr, legal, S_cache)
            else:
                b = fallback_bidder.bid(info, legal, rng)
            e = False
            if p == explorer:
                off = int(rng.choice([-2, -1, 1, 2]))
                tgt = min(max(b + off, 0), n)
                b2 = min(legal, key=lambda v: (abs(v - tgt), v))
                e = b2 != b
                b = b2
            xs[p], expl[p] = x, e
            st.apply_bid(b)
        run_round(st, pols, prng, check=False)
        sb = sum(st.bids)
        sc = st.scores()
        for p in range(N):
            out["round"].append(r)
            out["n"].append(n)
            out["trump"].append(tr)
            out["pos"].append((p - 1) % N)
            out["seat"].append(p)
            out["x"].append(xs[p])
            out["bid"].append(st.bids[p])
            out["tricks"].append(st.tricks_won[p])
            out["score"].append(sc[p])
            out["explored"].append(expl[p])
            out["sum_bids"].append(sb)
    return {k: np.array(v) for k, v in out.items()}


class FixedBid:
    """Bidder that always bids a given value (used for the hero in the oracle)."""

    def __init__(self, b):
        self.b = b

    def bid(self, info, legal, rng):
        return self.b


def bid_conditional_dists(st: RoundState, hero: int, agents, candidates, n_samples: int,
                          rng: np.random.Generator) -> dict[int, np.ndarray]:
    """For a state where ``hero`` is about to bid: P_b(tricks) for each candidate b.
    Hidden hands are resampled ``n_samples`` times; the *same* worlds are used for
    every candidate (common random numbers).  Later bidders and all play use
    ``agents`` (hero plays with agents[hero].play)."""
    assert st.phase == BID and st.to_act == hero
    info = st.info_state(hero)
    sampler = HandSampler(info)
    N = st.N
    n = st.n_cards
    counts = {b: np.zeros(n + 1) for b in candidates}
    prngs = [rng] * N
    for _ in range(n_samples):
        hidden = sampler.sample(rng)
        for b in candidates:
            s2 = state_from_info(info, hidden)
            s2.apply_bid(b)
            run_round(s2, agents, prngs, check=False)
            counts[b][s2.tricks_won[hero]] += 1
    return {b: c / n_samples for b, c in counts.items()}


def bid_free_dist(st: RoundState, hero: int, agents, n_samples: int, rng) -> np.ndarray:
    """P(tricks) for bid-agnostic play (bids made by ``agents``)."""
    info = st.info_state(hero)
    sampler = HandSampler(info)
    n = st.n_cards
    c = np.zeros(n + 1)
    prngs = [rng] * st.N
    for _ in range(n_samples):
        s2 = state_from_info(info, sampler.sample(rng))
        run_round(s2, agents, prngs, check=False)
        c[s2.tricks_won[hero]] += 1
    return c / n_samples


__all__ = ["simulate_rounds", "bid_conditional_dists", "bid_free_dist", "FixedBid", "random_trump",
           "model_bid", "Agent"]

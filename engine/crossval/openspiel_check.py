"""Differential test of judgement against OpenSpiel's ``oh_hell``.

For each seeded deal we force *identical* inputs into both engines:
  * OpenSpiel chance nodes: num_tricks = n, dealer = d, then the deal (OpenSpiel
    gives the k-th dealt card to player k % N), then the face-up trump card.
  * judgement: RoundState(n, trump = suit of that trump card, dealer = d, hands).
    The trump card is simply absent from all hands.
Bids and cards are chosen by a seeded random policy on the judgement side and
applied to both.  At every decision we assert the acting seat and the legal
bid / card sets match; after every trick we assert the same winner; at the end
we assert tricks won and scores (OpenSpiel's scoring re-expressed through a
judgement ``Rules`` callable).

What is *not* compared (documented rule differences, see REPORT.md):
  trump selection method (turned card vs fixed cycle), no-trump rounds (absent
  in OpenSpiel), max hand size (OpenSpiel reserves the trump card), scoring.

Run:  python -m crossval.openspiel_check [--deals 10000] [--players 4]
      python -m crossval.openspiel_check --naive   # also diff vs naive_ref
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np

from judgement.cards import SUITS, RANKS, card_str, cards_of, make_card, mask_of
from judgement.rules import Rules
from judgement.state import RoundState

from . import naive_ref

try:
    import pyspiel
except ImportError:  # pragma: no cover
    pyspiel = None


# OpenSpiel's scoring rules, expressed as judgement Rules (hit -> 10 + bid in both).
def os_miss_default(bid: int, tricks: int) -> int:
    return tricks              # points_per_trick * tricks, no bonus


def os_miss_off_bid(bid: int, tricks: int) -> int:
    return -abs(bid - tricks)  # off_bid_penalty=True


class Mismatch(AssertionError):
    pass


def build_card_map(game) -> tuple[dict[int, int], dict[int, int]]:
    """Map OpenSpiel card actions <-> judgement cards via their string names."""
    st = game.new_initial_state()
    st.apply_action(st.chance_outcomes()[0][0])  # num tricks
    st.apply_action(st.chance_outcomes()[0][0])  # dealer
    os2j, j2os = {}, {}
    for a, _ in st.chance_outcomes():
        name = st.action_to_string(a)          # e.g. "CA", "H10"? -> suit then rank
        suit_ch, rank_s = name[0], name[1:]
        rank_s = "T" if rank_s == "10" else rank_s
        c = make_card(SUITS.index(suit_ch), RANKS.index(rank_s))
        os2j[a] = c
        j2os[c] = a
    assert len(os2j) == 52
    return os2j, j2os


def _tricks_won_from_string(s: str, N: int) -> list[int]:
    m = re.search(r"Tricks Won:\s*([-\d ]+)", s)
    return [int(x) for x in m.group(1).split()][:N]


def check_deal(game, rules_j: Rules, os2j, j2os, rng: np.random.Generator, n: int, dealer: int,
               stats: dict, naive: bool = False) -> None:
    N = rules_j.n_players
    perm = rng.permutation(52).tolist()
    deal_seq = perm[:n * N]                      # k-th card -> player k % N
    trump_card = perm[n * N]
    hands = [mask_of(deal_seq[k] for k in range(p, n * N, N)) for p in range(N)]
    trump = trump_card // 13

    os_st = game.new_initial_state()
    os_st.apply_action(n)        # chance action value == number of tricks
    os_st.apply_action(dealer)
    for c in deal_seq:
        os_st.apply_action(j2os[c])
    os_st.apply_action(j2os[trump_card])

    st = RoundState(rules_j, n, trump, dealer, hands)
    nv = None
    if naive:
        nv = naive_ref.NaiveRound([[card_str(c) for c in cards_of(h)] for h in hands],
                                  SUITS[trump], dealer)

    # ---- bidding ----
    while not st.is_terminal and st.phase == 0:
        p = st.to_act
        if os_st.current_player() != p:
            raise Mismatch(f"bid turn: os={os_st.current_player()} j={p}")
        lj = st.legal_bids()
        los = sorted(a - 52 for a in os_st.legal_actions())
        if lj != los:
            raise Mismatch(f"legal bids differ: j={lj} os={los}")
        if nv is not None and nv.legal_bids() != lj:
            raise Mismatch(f"naive legal bids differ: {nv.legal_bids()} vs {lj}")
        if len(lj) == n:
            stats["hook_constrained"] += 1
        b = lj[int(rng.random() * len(lj))]
        st.apply_bid(b)
        os_st.apply_action(52 + b)
        if nv is not None:
            nv.bid(b)
        stats["bid_checks"] += 1

    # ---- play ----
    while not st.is_terminal:
        p = st.to_act
        if os_st.current_player() != p:
            raise Mismatch(f"play turn: os={os_st.current_player()} j={p}")
        lj = set(cards_of(st.legal_cards()))
        los = {os2j[a] for a in os_st.legal_actions()}
        if lj != los:
            raise Mismatch(f"legal cards differ: j={sorted(map(card_str, lj))} os={sorted(map(card_str, los))}")
        if nv is not None and nv.legal_cards() != {card_str(c) for c in lj}:
            raise Mismatch("naive legal cards differ")
        stats["legal_checks"] += 1
        lst = sorted(lj)
        c = lst[int(rng.random() * len(lst))]
        n_done = len(st.trick_winners)
        st.apply_card(c)
        os_st.apply_action(j2os[c])
        nw = nv.play(card_str(c)) if nv is not None else None
        if len(st.trick_winners) > n_done:
            w = st.trick_winners[-1]
            stats["trick_checks"] += 1
            if nv is not None and nw != w:
                raise Mismatch(f"naive trick winner {nw} vs {w}")
            if not os_st.is_terminal() and os_st.current_player() != w:
                raise Mismatch(f"trick winner: os next={os_st.current_player()} j={w}")
    if not os_st.is_terminal():
        raise Mismatch("OpenSpiel not terminal when judgement is")
    tw = _tricks_won_from_string(str(os_st), N)
    if tw != st.tricks_won:
        raise Mismatch(f"tricks won differ os={tw} j={st.tricks_won}")
    if [float(x) for x in st.scores()] != list(os_st.returns()):
        raise Mismatch(f"scores differ os={os_st.returns()} j={st.scores()}")
    stats["deals"] += 1


def run(n_deals: int, n_players: int, seed: int = 0, off_bid_penalty: bool = False,
        naive: bool = False) -> dict:
    if pyspiel is None:
        raise RuntimeError("pyspiel not installed")
    game = pyspiel.load_game("oh_hell", {"players": n_players, "off_bid_penalty": off_bid_penalty})
    rules_j = Rules(n_players=n_players,
                    miss_penalty=os_miss_off_bid if off_bid_penalty else os_miss_default)
    os2j, j2os = build_card_map(game)
    max_n = (52 - 1) // n_players
    stats = dict(deals=0, bid_checks=0, legal_checks=0, trick_checks=0, hook_constrained=0,
                 players=n_players, off_bid_penalty=off_bid_penalty, max_n=max_n)
    root = np.random.SeedSequence(seed)
    t0 = time.perf_counter()
    for i, child in enumerate(root.spawn(n_deals)):
        rng = np.random.default_rng(child)
        n = 1 + int(rng.integers(max_n))
        dealer = int(rng.integers(n_players))
        try:
            check_deal(game, rules_j, os2j, j2os, rng, n, dealer, stats, naive)
        except Mismatch as e:
            stats.setdefault("failures", []).append((i, n, dealer, str(e)))
    stats["seconds"] = round(time.perf_counter() - t0, 2)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=10000)
    ap.add_argument("--players", type=int, nargs="*", default=[4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--off-bid-penalty", action="store_true")
    ap.add_argument("--naive", action="store_true", help="also diff against naive_ref")
    a = ap.parse_args(argv)
    ok = True
    for N in a.players:
        s = run(a.deals, N, a.seed, a.off_bid_penalty, a.naive)
        fails = s.pop("failures", [])
        print(s, "failures:", len(fails))
        for f in fails[:10]:
            print("  ", f)
        ok &= not fails
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

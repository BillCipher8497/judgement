"""Greedy-baseline reproduction (SPEC correctness item 5).

4 players, 13-card round, trump = spades, dealer = seat 0, hero = seat 1 (first
bidder, so the hook never binds).  All four seats play GreedyPolicy.  For each of
5 fixed hands we estimate P(hero takes k tricks) with the MC oracle and compare:
  * oracle bid  = argmax_b (10 + b) * P(b)      (expected score, miss = 0)
  * mean bid    = round(E[tricks])
Hands 0-2: the hero's hand from deals with seeds 0..2.
Hand 3: the first deal (seeds 1000, 1001, ...) where the hero holds >= 6 trumps
        and a void.  (Under all-greedy play this one turns out *unimodal*.)
Hand 4: the first such deal whose trick distribution is actually bimodal: an
        interior k with P(k) at least 0.03 below the max on each side, and both
        side maxima >= 0.08 (screened at 2,000 samples, reported at --samples).
Hand 5 (extra): the first bimodal long-trump+void deal where round(E[t]) lands
        in the trough, i.e. where the mean-tricks bid is at its worst.

Run:  python bench/baseline.py [--samples 10000]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from judgement.cards import SUIT_MASK, hand_str  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.oracle import mc_trick_distribution, mean_tricks_bid, optimal_bid  # noqa: E402
from judgement.policies import GreedyPolicy  # noqa: E402
from judgement.rules import Rules  # noqa: E402
from judgement.state import RoundState  # noqa: E402

RULES = Rules()
TRUMP, DEALER, HERO, N_CARDS = 0, 0, 1, 13


def hand_for_seed(seed):
    return deal(RULES, N_CARDS, np.random.default_rng(seed))


def is_long_trump_void(h):
    return (h & SUIT_MASK[TRUMP]).bit_count() >= 6 and any((h & SUIT_MASK[s]) == 0 for s in range(4))


def greedy_dist(hands, samples, seed):
    st = RoundState(RULES, N_CARDS, TRUMP, DEALER, hands)
    info = st.info_state(HERO)
    return info, mc_trick_distribution(info, [GreedyPolicy()] * 4, samples, np.random.default_rng(seed))


def dip_points(d, thr=0.03, side=0.08):
    return [k for k in range(1, len(d) - 1)
            if d[k] < max(d[:k]) - thr and d[k] < max(d[k + 1:]) - thr
            and max(d[:k]) >= side and max(d[k + 1:]) >= side]


def find_deal(bimodal: bool, mean_in_trough: bool = False):
    seed = 1000
    while True:
        hands = hand_for_seed(seed)
        if is_long_trump_void(hands[HERO]):
            if not bimodal:
                return seed, hands
            d = greedy_dist(hands, 2000, seed)[1]
            dips = dip_points(d)
            if dips and (not mean_in_trough or mean_tricks_bid(d)[0] in dips):
                return seed, hands
        seed += 1


def cases():
    out = [(f"seed {s}", s, hand_for_seed(s)) for s in range(3)]
    s, hands = find_deal(bimodal=False)
    out.append((f"seed {s}, first long-trump+void", s, hands))
    s, hands = find_deal(bimodal=True)
    out.append((f"seed {s}, first bimodal long-trump+void", s, hands))
    s, hands = find_deal(bimodal=True, mean_in_trough=True)
    out.append((f"seed {s}, first bimodal with mean in trough (extra)", s, hands))
    return out


def run(samples: int):
    rows = []
    for label, seed, hands in cases():
        info, dist = greedy_dist(hands, samples, seed + 1)  # fresh stream vs the screen
        ob, ev = optimal_bid(info, dist)
        mb, mean = mean_tricks_bid(dist)
        rows.append(dict(label=label, hand=hand_str(hands[HERO]), dist=dist, oracle_bid=ob,
                         oracle_ev=ev[ob], mean=mean, mean_bid=mb, mean_bid_ev=ev[mb],
                         sd=float(np.sqrt(sum(p * (k - mean) ** 2 for k, p in enumerate(dist))))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10000)
    a = ap.parse_args(argv)
    rows = run(a.samples)
    print(f"4p, 13 cards, trump S, hero seat {HERO}, all-greedy play, {a.samples} samples/hand\n")
    print("| # | hero hand | source | E[t] (sd) | mean-tricks bid -> E[score] | oracle bid -> E[score] |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        flag = "" if r["mean_bid"] == r["oracle_bid"] else " **(diverges)**"
        print(f"| {i} | `{r['hand']}` | {r['label']} | {r['mean']:.2f} ({r['sd']:.2f}) | "
              f"{r['mean_bid']} -> {r['mean_bid_ev']:.2f} | {r['oracle_bid']} -> {r['oracle_ev']:.2f}{flag} |")
    print("\nTrick distributions P(k):\n")
    print("| # | " + " | ".join(str(k) for k in range(14)) + " |")
    print("|---|" + "---|" * 14)
    for i, r in enumerate(rows):
        cells = [f"{p:.3f}" if p >= 0.0005 else "." for p in r["dist"]]
        print(f"| {i} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

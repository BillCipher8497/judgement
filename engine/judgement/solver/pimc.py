"""Perfect-Information Monte Carlo (PIMC) play policy.

At each decision: draw ``n_samples`` hidden-hand assignments uniformly from
those consistent with the seat's InfoState (``sampling.HandSampler``), rebuild
each full state, solve it exactly with max^n (neutral tie-break), and play the
legal card with the highest average own score (ties -> lowest card index).
Known weakness: strategy fusion / non-locality (it assumes it will know the
cards after this move), and it models opponents as perfect-information max^n
players with the same bids.
"""
from __future__ import annotations

import numpy as np

from ..cards import cards_of
from ..sampling import HandSampler
from ..state import state_from_info
from .core import NEUTRAL, MaxN
from .heuristic import OracleGreedyBidder


def pimc_move_scores(info, legal_mask, n_samples, rng, mode=NEUTRAL):
    legal = cards_of(legal_mask)
    tot = {c: 0.0 for c in legal}
    if len(legal) == 1:
        return tot
    sampler = HandSampler(info)
    seat = info.seat
    for _ in range(n_samples):
        st = state_from_info(info, sampler.sample(rng))
        mv = MaxN(st.rules, st.trump, st.bids, st.N, mode).move_values(st)
        for c in legal:
            tot[c] += mv[c][seat]
    return {c: v / n_samples for c, v in tot.items()}


class PIMCPolicy:
    def __init__(self, n_samples: int = 20, bidder=None, mode: str = NEUTRAL):
        self.n_samples = n_samples
        self.mode = mode
        self.bidder = bidder or OracleGreedyBidder()
        self.name = f"pimc{n_samples}"
        self.last_scores = None

    def bid(self, info, legal, rng):
        return self.bidder.bid(info, legal, rng)

    def play(self, info, legal_mask, rng):
        sc = pimc_move_scores(info, legal_mask, self.n_samples, rng, self.mode)
        self.last_scores = sc
        best = max(sorted(sc), key=lambda c: sc[c])   # max returns first max -> lowest index on ties
        return best

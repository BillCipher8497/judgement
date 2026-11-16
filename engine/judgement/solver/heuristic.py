"""Minimal target-aware heuristic play policy (Phase 2's own; uses only InfoState).

need = own bid - own tricks won.
  need > 0  following: cheapest card that beats the current winner (lowest
            same-suit overcard, else lowest trump when void), else discard the
            lowest card (non-trump preferred).
            leading:   lowest "boss" card (highest outstanding in its suit,
            trumps first); else the highest-ranked card (trump first).
  need == 0 following: highest card that does not beat the current winner
            (a sure loser of this trick; highest rank, non-trump preferred on
            ties); if every legal card beats the winner: the highest if last to
            play (the trick is ours anyway), else the lowest.
            leading:   lowest-ranked card (non-trump preferred).
  need < 0  lowest-ranked card (non-trump preferred).
Bidding: oracle bid from a Monte-Carlo trick distribution under greedy play
(``OracleGreedyBidder``) — shared by every seat in Phase-2 imperfect-info runs.
"""
from __future__ import annotations

import numpy as np

from ..cards import NT, SUIT_MASK, SUIT_OF, cards_of
from ..oracle import mc_trick_distribution, optimal_bid
from ..policies.greedy import GreedyPolicy

RANK = [c % 13 for c in range(52)]


def _key_low(c, trump):           # lowest rank, non-trump first
    return (RANK[c], SUIT_OF[c] == trump, c)


def _key_high(c, trump):          # highest rank, non-trump first on ties
    return (RANK[c], SUIT_OF[c] != trump, -c)


def beats(c, bc, trump):
    s, sb = SUIT_OF[c], SUIT_OF[bc]
    return (s == sb and c > bc) or (s != sb and s == trump)


def boss_cards(hand, played, trump=None):
    """Own cards that are the highest outstanding card of their suit."""
    out = []
    outside = ~(hand | played) & ((1 << 52) - 1)
    for c in cards_of(hand):
        s = SUIT_OF[c]
        higher = SUIT_MASK[s] & ~((1 << (c + 1)) - 1)
        if not (higher & outside):
            out.append(c)
    return out


def info_need(info):
    bid = dict(info.bids)[info.seat]
    return bid - info.tricks_won[info.seat]


def heuristic_card(info, legal_mask):
    trump = info.trump
    need = info_need(info)
    legal = cards_of(legal_mask)
    wc = info.winning_card
    low = lambda cs: min(cs, key=lambda c: _key_low(c, trump))
    high = lambda cs: max(cs, key=lambda c: _key_high(c, trump))
    if need < 0:
        return low(legal)
    if wc < 0:  # leading
        if need == 0:
            return low(legal)
        boss = boss_cards(info.hand, info.played_mask)
        if trump != NT:
            bt = [c for c in boss if SUIT_OF[c] == trump]
            if bt:
                return low(bt)
        if boss:
            return low(boss)
        if trump != NT:
            t = [c for c in legal if SUIT_OF[c] == trump]
            if t:
                return max(t)
        return high(legal)
    win = [c for c in legal if beats(c, wc, trump)]
    lose = [c for c in legal if not beats(c, wc, trump)]
    if need > 0:
        if win:
            return min(win, key=lambda c: (SUIT_OF[c] == trump, RANK[c]))
        nt = [c for c in lose if SUIT_OF[c] != trump] or lose
        return low(nt)
    # need == 0
    if lose:
        return high(lose)
    last = info.trick_len == info.n_players - 1
    return high(win) if last else low(win)


class TargetHeuristic:
    name = "target_heuristic"

    def bid(self, info, legal, rng):
        return GreedyPolicy().bid(info, legal, rng)

    def play(self, info, legal_mask, rng):
        return heuristic_card(info, legal_mask)


class OracleGreedyBidder:
    """argmax_b E[score] from an MC trick distribution under all-greedy play."""

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def bid(self, info, legal, rng):
        dist = mc_trick_distribution(info, [GreedyPolicy()] * info.n_players, self.n_samples, rng)
        b, _ = optimal_bid(info, dist, legal)
        return b

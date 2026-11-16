"""Naive baseline: always try to win the current trick, ignoring the bid.

Play rule (deterministic, rng unused):
  * Leading: play the strongest card — highest trump if holding trump, otherwise
    the highest-ranked card (ties across suits -> lowest suit index S<H<D<C).
  * Following: if any legal card would beat the card currently winning the
    trick, play the highest such card.  Otherwise discard the lowest-ranked
    legal card, preferring non-trumps.

Bid rule (simple, used only when a GreedyPolicy bids in play_game; the
research baseline instead bids with ``oracle.optimal_bid``): count aces, plus
trump K/Q, plus trumps beyond the third; snap to the nearest legal bid.
"""
from __future__ import annotations

from ..cards import NT, RANK_OF, SUIT_MASK, SUIT_OF
from .base import nearest_legal

_RANKMASK = 0x1FFF


def _highest_rank_card(mask: int) -> int:
    """Highest-ranked card in mask; ties -> lowest suit index."""
    best = -1
    best_r = -1
    for s in range(4):
        m = mask & SUIT_MASK[s]
        if m:
            c = m.bit_length() - 1
            r = c - 13 * s
            if r > best_r:
                best_r, best = r, c
    return best


def _lowest_rank_card(mask: int) -> int:
    best = -1
    best_r = 99
    for s in range(4):
        m = mask & SUIT_MASK[s]
        if m:
            c = (m & -m).bit_length() - 1
            r = c - 13 * s
            if r < best_r:
                best_r, best = r, c
    return best


class GreedyPolicy:
    name = "greedy"

    def bid(self, info, legal, rng):
        hand, trump = info.hand, info.trump
        est = 0
        for s in range(4):
            m = (hand >> (13 * s)) & _RANKMASK
            if m >> 12 & 1:
                est += 1  # ace
            if s == trump:
                est += (m >> 11 & 1) + (m >> 10 & 1)  # K, Q of trump
                est += max(0, m.bit_count() - 3)
        return nearest_legal(min(est, info.n_cards), legal)

    def play(self, info, legal_mask, rng):
        trump = info.trump
        wc = info.winning_card
        if wc < 0:  # leading
            if trump != NT:
                t = legal_mask & SUIT_MASK[trump]
                if t:
                    return t.bit_length() - 1
            return _highest_rank_card(legal_mask)
        ws = SUIT_OF[wc]
        # cards that would beat the current winner
        same = legal_mask & SUIT_MASK[ws] & ~((1 << (wc + 1)) - 1)
        if same:
            return same.bit_length() - 1
        if trump != NT and ws != trump:
            t = legal_mask & SUIT_MASK[trump]
            if t:
                return t.bit_length() - 1
        # cannot win: discard lowest, preferring non-trump
        if trump != NT:
            nt = legal_mask & ~SUIT_MASK[trump]
            if nt:
                return _lowest_rank_card(nt)
        return _lowest_rank_card(legal_mask)

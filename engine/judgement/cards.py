"""Card encoding.

A card is an int 0..51: ``card = suit * 13 + rank`` with suits S=0, H=1, D=2, C=3
and ranks 0..12 = 2..A.  A hand (or any set of cards) is a 52-bit int bitmask
with bit ``card`` set.  Because ranks are contiguous within a suit, comparing two
cards of the same suit is plain integer comparison, and the highest card of a suit
inside a mask is ``(mask & SUIT_MASK[s]).bit_length() - 1``.
"""
from __future__ import annotations

from typing import Iterable

SUITS = "SHDC"
RANKS = "23456789TJQKA"
NT = -1  # "no trump" sentinel; never equals a real suit index

N_SUITS = 4
N_RANKS = 13
DECK_SIZE = 52
FULL_DECK = (1 << DECK_SIZE) - 1
SUIT_MASK: tuple[int, ...] = tuple(((1 << N_RANKS) - 1) << (N_RANKS * s) for s in range(N_SUITS))
SUIT_OF: tuple[int, ...] = tuple(c // N_RANKS for c in range(DECK_SIZE))
RANK_OF: tuple[int, ...] = tuple(c % N_RANKS for c in range(DECK_SIZE))
BIT: tuple[int, ...] = tuple(1 << c for c in range(DECK_SIZE))


def make_card(suit: int, rank: int) -> int:
    return suit * N_RANKS + rank


def card_str(c: int) -> str:
    """0 -> '2S', 51 -> 'AC'."""
    return RANKS[RANK_OF[c]] + SUITS[SUIT_OF[c]]


def parse_card(s: str) -> int:
    """Parse 'AS', 'TD', '10H', 'as' (rank then suit)."""
    s = s.strip().upper()
    if s.startswith("10"):
        s = "T" + s[2:]
    if len(s) != 2 or s[0] not in RANKS or s[1] not in SUITS:
        raise ValueError(f"bad card string {s!r}")
    return make_card(SUITS.index(s[1]), RANKS.index(s[0]))


def mask_of(cards: Iterable[int]) -> int:
    m = 0
    for c in cards:
        m |= 1 << c
    return m


def cards_of(mask: int) -> list[int]:
    """Cards in a mask, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def parse_hand(s: str) -> int:
    return mask_of(parse_card(t) for t in s.replace(",", " ").split())


def hand_str(mask: int) -> str:
    """Suit-grouped, high to low: 'S:AK2 H:- D:T C:QJ'."""
    parts = []
    for s in range(N_SUITS):
        ranks = [RANKS[RANK_OF[c]] for c in reversed(cards_of(mask & SUIT_MASK[s]))]
        parts.append(f"{SUITS[s]}:{''.join(ranks) or '-'}")
    return " ".join(parts)


def suit_str(trump: int) -> str:
    return "NT" if trump == NT else SUITS[trump]


def parse_suit(s: str) -> int:
    s = s.strip().upper()
    if s == "NT":
        return NT
    return SUITS.index(s)


def popcount(mask: int) -> int:
    return mask.bit_count()


def nth_card(mask: int, i: int) -> int:
    """The i-th (0-based, ascending) card in mask."""
    while i:
        mask &= mask - 1
        i -= 1
    return (mask & -mask).bit_length() - 1

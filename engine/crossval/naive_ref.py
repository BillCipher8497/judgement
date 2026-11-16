"""Deliberately naive reference implementation of the v1 rules.

Written independently of ``judgement`` (no imports from it): cards are strings
like "AS", hands are Python lists, everything is recomputed from scratch.
Slow and obvious on purpose — used for differential testing, especially for
what OpenSpiel cannot cover (no-trump rounds, 13-card rounds at 4 players,
the game schedule and scoring).
"""
from __future__ import annotations

RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_ORDER = ["S", "H", "D", "C"]


def rank_value(card: str) -> int:
    return RANK_ORDER.index(card[0])


def suit(card: str) -> str:
    return card[1]


def legal_plays(hand: list[str], trick_so_far: list[str]) -> set[str]:
    if len(trick_so_far) == 0:
        return set(hand)
    led = suit(trick_so_far[0])
    same = [c for c in hand if suit(c) == led]
    if len(same) > 0:
        return set(same)
    return set(hand)


def trick_winner(plays: list[tuple[int, str]], trump: str | None) -> int:
    """plays = [(seat, card), ...] in play order; trump=None for no-trump."""
    led = suit(plays[0][1])
    trumps = [(s, c) for s, c in plays if trump is not None and suit(c) == trump]
    if len(trumps) > 0:
        candidates = trumps
    else:
        candidates = [(s, c) for s, c in plays if suit(c) == led]
    best_seat, best_card = candidates[0]
    for s, c in candidates:
        if rank_value(c) > rank_value(best_card):
            best_seat, best_card = s, c
    return best_seat


def legal_bids(n_cards: int, bids_so_far: list[int], n_players: int, hook: bool = True) -> list[int]:
    options = list(range(0, n_cards + 1))
    if hook and len(bids_so_far) == n_players - 1:
        options = [b for b in options if sum(bids_so_far) + b != n_cards]
    return options


def score(bid: int, tricks: int, mode: str = "zero", hit_base: int = 10) -> int:
    if bid == tricks:
        return hit_base + bid
    if mode == "zero":
        return 0
    if mode == "neg_bid":
        return -bid
    if mode == "neg_10":
        return -10
    if mode == "neg_diff":
        return -10 * abs(bid - tricks)
    raise ValueError(mode)


def schedule(n_players: int) -> list[tuple[int, str | None, int]]:
    """[(n_cards, trump or None, dealer)] for each round."""
    max_cards = 52 // n_players
    sizes = []
    k = max_cards
    while k >= 1:
        sizes.append(k)
        k -= 1
    k = 2
    while k <= max_cards:
        sizes.append(k)
        k += 1
    cycle = ["S", "H", "D", "C", None]
    out = []
    for i, n in enumerate(sizes):
        out.append((n, cycle[i % 5], i % n_players))
    return out


class NaiveRound:
    """Holds state as plain lists; caller picks bids and cards."""

    def __init__(self, hands: list[list[str]], trump: str | None, dealer: int):
        self.hands = [list(h) for h in hands]
        self.n_players = len(hands)
        self.n_cards = len(hands[0])
        self.trump = trump
        self.dealer = dealer
        self.bids: list[int] = []
        self.bidders: list[int] = []
        self.tricks = [0] * self.n_players
        self.current: list[tuple[int, str]] = []
        self.leader = (dealer + 1) % self.n_players
        self.winners: list[int] = []

    def next_bidder(self) -> int:
        return (self.dealer + 1 + len(self.bids)) % self.n_players

    def legal_bids(self, hook: bool = True) -> list[int]:
        return legal_bids(self.n_cards, self.bids, self.n_players, hook)

    def bid(self, b: int) -> None:
        assert b in self.legal_bids()
        self.bidders.append(self.next_bidder())
        self.bids.append(b)

    def next_player(self) -> int:
        return (self.leader + len(self.current)) % self.n_players

    def legal_cards(self) -> set[str]:
        return legal_plays(self.hands[self.next_player()], [c for _, c in self.current])

    def play(self, card: str) -> int | None:
        p = self.next_player()
        assert card in self.legal_cards()
        self.hands[p].remove(card)
        self.current.append((p, card))
        if len(self.current) == self.n_players:
            w = trick_winner(self.current, self.trump)
            self.tricks[w] += 1
            self.winners.append(w)
            self.current = []
            self.leader = w
            return w
        return None

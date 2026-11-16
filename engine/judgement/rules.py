"""All game rules live in the frozen ``Rules`` dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

from .cards import DECK_SIZE, NT, parse_suit, suit_str

PenaltyFn = Callable[[int, int], float]
MISS_MODES = ("zero", "neg_bid", "neg_10", "neg_diff")


@dataclass(frozen=True)
class Rules:
    n_players: int = 4
    hit_base: int = 10
    # "zero" | "neg_bid" | "neg_10" | "neg_diff" | callable(bid, tricks) -> score on a miss
    miss_penalty: Union[str, PenaltyFn] = "zero"
    hook: bool = True                      # last bidder may not make sum(bids) == n
    trump_cycle: tuple[int, ...] = (0, 1, 2, 3, NT)  # S, H, D, C, NT by round index
    first_dealer: int = 0
    max_cards: int | None = None           # None -> floor(52 / n_players)
    deck_size: int = field(default=DECK_SIZE, repr=False)

    def __post_init__(self) -> None:
        if not 3 <= self.n_players <= 7:
            raise ValueError("n_players must be in 3..7")
        if isinstance(self.miss_penalty, str) and self.miss_penalty not in MISS_MODES:
            raise ValueError(f"unknown miss_penalty {self.miss_penalty!r}")
        if not (isinstance(self.miss_penalty, str) or callable(self.miss_penalty)):
            raise TypeError("miss_penalty must be a mode string or callable")
        if self.M < 1 or self.M * self.n_players > self.deck_size:
            raise ValueError("max_cards out of range")
        if not self.trump_cycle:
            raise ValueError("trump_cycle must be non-empty")

    # ---- schedule -------------------------------------------------------
    @property
    def M(self) -> int:
        return self.max_cards if self.max_cards is not None else self.deck_size // self.n_players

    def ladder(self) -> tuple[int, ...]:
        """M, M-1, ..., 2, 1, 2, ..., M  (2M-1 rounds)."""
        M = self.M
        return tuple(range(M, 0, -1)) + tuple(range(2, M + 1))

    @property
    def n_rounds(self) -> int:
        return 2 * self.M - 1

    def trump_for_round(self, r: int) -> int:
        return self.trump_cycle[r % len(self.trump_cycle)]

    def dealer_for_round(self, r: int) -> int:
        # "rotates left": seat d+1 is to the left of seat d
        return (self.first_dealer + r) % self.n_players

    # ---- scoring --------------------------------------------------------
    def score(self, bid: int, tricks: int) -> float:
        if bid == tricks:
            return self.hit_base + bid
        mp = self.miss_penalty
        if mp == "zero":
            return 0
        if mp == "neg_bid":
            return -bid
        if mp == "neg_10":
            return -10
        if mp == "neg_diff":
            return -10 * abs(bid - tricks)
        return mp(bid, tricks)

    # ---- serialisation --------------------------------------------------
    def to_dict(self) -> dict:
        mp = self.miss_penalty
        return {
            "n_players": self.n_players,
            "hit_base": self.hit_base,
            "miss_penalty": mp if isinstance(mp, str) else f"callable:{getattr(mp, '__qualname__', repr(mp))}",
            "hook": self.hook,
            "trump_cycle": [suit_str(t) for t in self.trump_cycle],
            "first_dealer": self.first_dealer,
            "max_cards": self.max_cards,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rules":
        if str(d["miss_penalty"]).startswith("callable:"):
            raise ValueError("trace used a callable miss_penalty; pass rules= explicitly to replay()")
        return cls(
            n_players=d["n_players"],
            hit_base=d["hit_base"],
            miss_penalty=d["miss_penalty"],
            hook=d["hook"],
            trump_cycle=tuple(parse_suit(s) for s in d["trump_cycle"]),
            first_dealer=d["first_dealer"],
            max_cards=d["max_cards"],
        )


def ladder(rules: Rules) -> tuple[int, ...]:
    return rules.ladder()


def trump_for_round(rules: Rules, r: int) -> int:
    return rules.trump_for_round(r)


def score(rules: Rules, bid: int, tricks: int) -> float:
    return rules.score(bid, tricks)

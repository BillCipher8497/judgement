"""Policy protocol.

A policy is any object with::

    bid(info, legal: list[int], rng: numpy.random.Generator) -> int
    play(info, legal_mask: int, rng: numpy.random.Generator) -> int   # a card

``info`` is an ``InfoState`` (read-only view for the acting seat).  ``legal_mask``
is a card bitmask (see ``judgement.cards``).  Policies must only use ``rng`` for
randomness so runs are reproducible.  Policies may be stateful but must not
retain ``info`` across calls (it is a live view).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    name: str

    def bid(self, info, legal: list[int], rng: np.random.Generator) -> int: ...

    def play(self, info, legal_mask: int, rng: np.random.Generator) -> int: ...


def nearest_legal(target: float, legal: list[int]) -> int:
    """Legal bid closest to target; ties go to the lower bid (deterministic)."""
    return min(legal, key=lambda b: (abs(b - target), b))

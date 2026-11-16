"""Hybrid play: TA+ early, PIMC (judgement.solver.pimc, max^n on sampled worlds)
once the seat holds at most ``pimc_cards`` cards (any round size, so the last
tricks of big rounds are searched too).  ``pimc_in_nt=False`` keeps TA+ for the
whole of no-trump rounds (Phase 2: PIMC's edge is in trump rounds)."""
from __future__ import annotations

from ..cards import NT
from ..solver.pimc import pimc_move_scores
from .ta_plus import TAPlusPolicy


class HybridPolicy:
    def __init__(self, n_samples: int = 16, pimc_cards: int = 5, base=None, pimc_in_nt: bool = True,
                 name: str | None = None):
        self.n_samples = n_samples
        self.pimc_cards = pimc_cards
        self.pimc_in_nt = pimc_in_nt
        self.base = base or TAPlusPolicy()
        self.name = name or f"hybrid_k{n_samples}_c{pimc_cards}" + ("" if pimc_in_nt else "_ntbase")
        self.n_pimc = 0

    def bid(self, info, legal, rng):
        return self.base.bid(info, legal, rng)

    def play(self, info, legal_mask, rng):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        if info.hand.bit_count() <= self.pimc_cards and (self.pimc_in_nt or info.trump != NT):
            self.n_pimc += 1
            sc = pimc_move_scores(info, legal_mask, self.n_samples, rng)
            return max(sorted(sc), key=lambda c: sc[c])
        return self.base.play(info, legal_mask, rng)


__all__ = ["HybridPolicy"]

from __future__ import annotations

from ..cards import nth_card


class RandomPolicy:
    """Uniform over legal bids and legal cards."""

    name = "random"

    def bid(self, info, legal, rng):
        return legal[int(rng.random() * len(legal))]

    def play(self, info, legal_mask, rng):
        return nth_card(legal_mask, int(rng.random() * legal_mask.bit_count()))

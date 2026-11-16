"""TA+3 : ta_plus_v2 + "lead the highest sure loser at target" (math M9, Prop. L).

At target (need == 0) and on lead, TA leads its lowest card (min P(win), then
lowest value).  M9 showed the exchange argument: when >= 2 cards are SURE
losers (estimated P(win if led) == 0) lead the highest-value one instead, keeping
the low cards as later safe exits; when < 2 sure losers exist, TA's choice
stands.  ``lead_loser``: 1 = any suit (M9 "L", default), 2 = only within the
suit of TA's choice (M9 "L_same"), 0 = off (== ta_plus_v2).  Same logic as
math/m9_play.py ``with_variants(L=...)``.
"""
from __future__ import annotations

from ..cards import SUIT_OF
from .ta_plus import TAPlusPolicy


class TAPlus3Policy(TAPlusPolicy):
    name = "ta_plus3"

    def __init__(self, lead_loser: int = 1, **kw):
        kw.setdefault("win_high", 2)
        kw.setdefault("win_margin", -0.5)
        super().__init__(**kw)
        self.lead_loser = lead_loser

    def _lose(self, tm, cards, pw, val):
        d = super()._lose(tm, cards, pw, val)
        if self.lead_loser and tm.wc < 0:
            losers = [c for c in cards if pw[c] <= 0.0]
            if self.lead_loser == 2:
                losers = [c for c in losers if SUIT_OF[c] == SUIT_OF[d]]
            if len(losers) >= 2:
                return max(losers, key=lambda c: (val(c), -c))
        return d


def ta_plus3(**kw) -> TAPlus3Policy:
    return TAPlus3Policy(**kw)


__all__ = ["TAPlus3Policy", "ta_plus3"]

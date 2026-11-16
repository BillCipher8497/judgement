"""TA+ : target-aware play with overshoot control and table-scarcity awareness.

Builds on ``TargetAwarePolicy`` (same TrickModel win-probability estimates).
Motivation (Phase 1/2): 59-69% of TA misses are overshoots, and the solver
rules show "take it cheaply when last" is right only 72% of the time when the
table has surplus tricks (others will gift tricks later) vs 95% when scarce.

Extra quantities per decision (seat to act, r = cards in hand incl. this trick):

* ``lead_pw(c)`` -- P(card c would win if *led* now against all three opponents)
  (same estimator as TA, ``later`` = every opponent).  ``E_w = sum lead_pw`` over
  the hand is a crude count of the tricks the hand can still take by force.
* ``slack = r - sum_q need_q`` over the live players q (0 < need_q <= r),
  including me: > 0 means the table has more tricks than anyone wants (surplus:
  tricks will be dumped on someone), <= 0 means tricks are scarce.

Changes vs TA (each switchable by a flag; defaults = the variant selected by a
1-vs-3 screen on seeds 31,000,000+, evaluated afterwards on fresh seeds):

1. ``shed`` (overshoot control, ON) -- need > 0, following, and I will not win
   this trick (no card beats the winner, or TA chose to duck): instead of the
   cheapest card, throw the sure-loser with the highest ``lead_pw`` as long as
   ``E_w - lead_pw(c) >= need + shed_margin`` (default margin -0.5: E_w ignores
   ruffing and long-suit tricks, so it under-counts force).  Sheds surplus
   winners on tricks someone else is taking.
2. ``lead_high`` (R9-alt, ON, =3) -- need > 0, leading, no sure winner: lead the
   highest side-suit card (solver: lead-highest 84% vs lead-lowest 70% optimal).
   =3 trump rounds only (default: in NT it cost -0.08/round on the validation
   seeds), =2 all rounds, =1 only when TA's choice is not "urgent enough"; 0 = TA.
3. ``theorems`` (ON) -- need == r (must win every trick): follow with the
   cheapest sure winner if any (R4|need=rem, 100%), else the strongest card;
   lead the strongest card (R7 96.7%, R7f 93.7%).  need == 0 stays TA (R1
   highest sure loser 96.5%, R2 lead low), which already matches the solver.
4. ``surplus_duck`` (OFF) -- need > 0, following, table slack > 0, own force
   E_w - need >= duck_margin, holds a sure loser: duck/shed instead of winning
   (``duck_last_only`` restricts to last seat, the R4|slack>0 case).
5. ``scarce`` (OFF) -- need > 0, following, slack <= 0, no sure winner: play
   the max-P(win) card (R5-alt).
   4 and 5 are the scarcity/surplus-aware winning rules; neither helped in the
   screen (4 hurt, 5 neutral), so they are off by default.
6. ``win_high`` (experiments/play; 1 = following, 2 = also when leading) --
   need > 0 and >= 2 sure winners: win with the sure winner of highest
   ``lead_pw`` such that ``E_w - lead_pw(c) >= need - 1 + win_margin`` (at
   need == 1: the most dangerous card), instead of TA's cheapest sure winner.
"""
from __future__ import annotations

from ..cards import NT, SUIT_OF
from .target_aware import TargetAwarePolicy, TrickModel, _bits, card_value


class TAPlusPolicy(TargetAwarePolicy):
    name = "ta_plus"

    def __init__(self, shed: bool = True, surplus_duck: bool = False, theorems: bool = True,
                 shed_margin: float = -0.5, duck_margin: float = 1.0, duck_last_only: bool = False,
                 lead_high: int = 3, scarce: bool = False, win_high: int = 0, win_margin: float = 0.0,
                 name: str | None = None):
        super().__init__(joint=True)
        self.shed = shed
        self.surplus_duck = surplus_duck
        self.theorems = theorems
        self.shed_margin = shed_margin
        self.duck_margin = duck_margin
        self.duck_last_only = duck_last_only
        self.lead_high = lead_high
        self.scarce = scarce
        self.win_high = win_high
        self.win_margin = win_margin
        if name:
            self.name = name

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def lead_pw(info, joint=True) -> dict[int, float]:
        """P(each card in hand wins if led now against all opponents)."""
        tm = TrickModel(info, joint)
        tm.later = [q for q in range(tm.N) if q != tm.me]
        tm.wc = -1
        return {c: tm.p_survive(c, SUIT_OF[c]) for c in _bits(info.hand)}

    @staticmethod
    def table_slack(info) -> int:
        r = info.hand.bit_count()
        tot = 0
        won = info.tricks_won
        for q, b in info.bids:
            nq = b - won[q]
            # players who already played to this trick have one fewer trick to go
            if 0 < nq <= r:
                tot += nq
        return r - tot

    # ---- play ----------------------------------------------------------------
    def _win(self, info, tm, cards, pw, need, r, trump, val):
        if self.theorems and need == r:
            # must win every remaining trick: strongest card (ties -> highest value)
            sure = [c for c in cards if pw[c] >= 1.0]
            if tm.wc >= 0 and sure:
                return min(sure, key=val)            # R4|need=rem: cheapest sure winner
            if tm.wc < 0 and sure:
                return super()._win(info, tm, cards, pw, need, r, trump, val)
            return max(cards, key=lambda c: (round(pw[c], 9), val(c)))
        if self.win_high and (tm.wc >= 0 or self.win_high >= 2):
            c = self._win_high_card(info, cards, pw, need)
            if c is not None:
                return c
        if tm.wc < 0:
            if self.lead_high and (self.lead_high != 3 or trump != NT) and not any(pw[c] >= 1.0 for c in cards):
                choice = super()._win(info, tm, cards, pw, need, r, trump, val)
                if pw[choice] < 1.0 and (self.lead_high >= 2 or pw[choice] < 0.5 * (1.0 - need / r)):
                    # R9-alt: no sure winner, not urgent -> lead the highest side card
                    side = [c for c in cards if SUIT_OF[c] != trump] or cards
                    return max(side, key=lambda c: (c % 13, -c))
                return choice
            return super()._win(info, tm, cards, pw, need, r, trump, val)
        if not (self.shed or self.surplus_duck):
            return super()._win(info, tm, cards, pw, need, r, trump, val)
        # following
        losers = [c for c in cards if pw[c] <= 0.0]
        if not losers:
            return super()._win(info, tm, cards, pw, need, r, trump, val)
        lpw = self.lead_pw(info)
        E_w = sum(lpw.values())
        can_win = len(losers) < len(cards)
        if can_win:
            if self.scarce and self.table_slack(info) <= 0 and not any(pw[c] >= 1.0 for c in cards):
                # tricks are scarce: compete with the strongest card (R5-alt)
                return max(cards, key=lambda c: (round(pw[c], 9), -val(c)))
            if not self.surplus_duck:
                choice = super()._win(info, tm, cards, pw, need, r, trump, val)
                if pw[choice] > 0.0:
                    return choice
                return self._shed_card(losers, lpw, E_w, need, trump, val) if self.shed else choice
            last = not tm.later
            if (last or not self.duck_last_only) and self.table_slack(info) > 0 \
                    and E_w - need >= self.duck_margin:
                return self._shed_card(losers, lpw, E_w, need, trump, val)
            choice = super()._win(info, tm, cards, pw, need, r, trump, val)
            if pw[choice] > 0.0 or not self.shed:
                return choice
            return self._shed_card(losers, lpw, E_w, need, trump, val)
        # cannot win this trick
        if self.shed:
            return self._shed_card(losers, lpw, E_w, need, trump, val)
        return super()._win(info, tm, cards, pw, need, r, trump, val)

    def _win_high_card(self, info, cards, pw, need):
        """win_high: when a sure winner takes this trick, win it with the sure winner
        carrying the most future winning potential that I can spare (overshoot
        control: at need == 1 this dumps the most dangerous card before need hits 0)."""
        sure = [c for c in cards if pw[c] >= 1.0]
        if len(sure) < 2:
            return None
        lpw = self.lead_pw(info)
        E_w = sum(lpw.values())
        ok = [c for c in sure if E_w - lpw.get(c, 0.0) >= need - 1 + self.win_margin]
        if not ok:
            return None
        return max(ok, key=lambda c: (round(lpw.get(c, 0.0), 6), c % 13, -c))

    def _shed_card(self, losers, lpw, E_w, need, trump, val):
        """Throw the sure loser with the most winning potential that I can spare,
        else the cheapest loser (TA's discard: lowest side-suit card)."""
        ok = [c for c in losers if E_w - lpw.get(c, 0.0) >= need + self.shed_margin]
        if ok:
            best = max(ok, key=lambda c: (round(lpw.get(c, 0.0), 6), val(c)))
            if lpw.get(best, 0.0) > 0.0:
                return best
        return min(losers, key=lambda c: (SUIT_OF[c] == trump and trump != NT, val(c)))


def ta_plus_v2(**kw) -> TAPlusPolicy:
    """TA+ as selected in experiments/play: round-2 defaults plus win_high=2,
    win_margin=-0.5 (chosen on the screen seeds 91,000,000+g)."""
    kw.setdefault("win_high", 2)
    kw.setdefault("win_margin", -0.5)
    kw.setdefault("name", "ta_plus2")
    return TAPlusPolicy(**kw)


__all__ = ["TAPlusPolicy", "ta_plus_v2", "card_value"]

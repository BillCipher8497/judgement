"""Spoiler: an opponent that ignores its own score and plays to make one
designated ``target`` seat MISS its bid (round 3, H1: hidden-information
analogue of the max^n - paranoid cooperation surplus).

Hidden information only: uses the target's bid, tricks won, the cards played,
known voids and the same hypergeometric card-tracking estimates as TA
(``TrickModel``).  Several spoilers with the same target coordinate only
implicitly (same objective, same public information); there is no signalling.

Target state, with need = bid_t - won_t and R = tricks the target can still
take (its cards left, +1 if it has already played to the current trick):

* decided (need < 0 or need > R: the target has missed whatever happens) ->
  fall back to the ``base`` self-interested policy (TA+ by default), so the
  spoiler salvages what it can for itself;
* DENY  (need > 0): minimise P(target wins this trick);
* GIVE  (need == 0): maximise P(target wins this trick).

P(target wins | I play c):
  - target already played: 0 if c beats the current winner or the target is not
    winning; otherwise P(no later player beats it) (TrickModel.p_survive);
  - target still to play: proxy P(target can beat the trick's best card after c)
    (TrickModel.p_beat, per-player hypergeometric incl. ruffs).
Ties (equal estimate, within 1e-9): DENY keeps strength (plays the cheapest card),
GIVE sheds strength (plays the most valuable card) -- in DENY mode high cards and
trumps are what later overtake the target; in GIVE mode low cards are what let
the spoiler duck under the target later.

Bidding is delegated (wrap in ``Agent(ModelBidder(...), Spoiler(t))``): by default
spoilers bid like everyone else, so the hero's information environment (prior
bids, hook) is unchanged and only the play differs.
"""
from __future__ import annotations

import numpy as np

from .ta_plus import ta_plus_v2
from .target_aware import TrickModel, _bits, card_value

_EPS = 1e-9


class SpoilerPolicy:
    name = "spoiler"

    def __init__(self, target: int, base=None):
        self.target = target
        self.base = base if base is not None else ta_plus_v2()

    # bidding fallback only (normally delegated to a ModelBidder)
    def bid(self, info, legal, rng):
        return self.base.bid(info, legal, rng)

    def mode(self, info) -> int:
        """+1 GIVE, -1 DENY, 0 decided (target already missed)."""
        t = self.target
        bid = None
        for p, b in info.bids:
            if p == t:
                bid = b
                break
        if bid is None:
            return 0
        need = bid - info.tricks_won[t]
        played = any(p == t for p, _ in info.current_trick)
        R = info.cards_left(t) + (1 if played else 0)
        if need < 0 or need > R:
            return 0
        return 1 if need == 0 else -1

    def p_target_wins(self, info, tm: TrickModel, c: int) -> float:
        t = self.target
        trick = info.current_trick
        played = any(p == t for p, _ in trick)
        if trick:
            led = tm.led
            best = c if tm.beats(c, tm.wc) else tm.wc
        else:
            led = c // 13
            best = c
        if played:
            if tm.ws != t or best != tm.wc:
                return 0.0
            return tm.p_survive(tm.wc, led)
        return tm.p_beat(t, best, led)

    def play(self, info, legal_mask, rng):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        m = self.mode(info)
        if m == 0 or info.seat == self.target:
            return self.base.play(info, legal_mask, rng)
        tm = TrickModel(info, True)
        trump = info.trump
        best_c, best_key = -1, None
        for c in _bits(legal_mask):
            p = self.p_target_wins(info, tm, c)
            v = card_value(c, trump)
            # GIVE: maximise p, tie -> most valuable; DENY: minimise p, tie -> cheapest
            key = (round(-p, 9), -v, -c) if m > 0 else (round(p, 9), v, c)
            if best_key is None or key < best_key:
                best_c, best_key = c, key
        return best_c


class PIMCSpoilerPolicy(SpoilerPolicy):
    """Stronger spoiler: PIMC with a paranoid objective.  At each decision (target still
    live) draw ``n_samples`` hidden-hand assignments consistent with this seat's InfoState,
    and for every legal card c solve the resulting perfect-information state exactly with
    ``Paranoid(target)`` (all three others minimise the target's score, the target
    maximises); play the card with the lowest mean target value (ties -> the heuristic
    spoiler's preference order).  Same strategy-fusion caveat as PIMCPolicy."""

    name = "spoiler_pimc"

    def __init__(self, target: int, n_samples: int = 16, base=None):
        super().__init__(target, base)
        self.n_samples = n_samples

    def play(self, info, legal_mask, rng):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        m = self.mode(info)
        if m == 0 or info.seat == self.target:
            return self.base.play(info, legal_mask, rng)
        from ..sampling import HandSampler
        from ..solver.core import Paranoid
        from ..state import state_from_info
        cards = _bits(legal_mask)
        tot = dict.fromkeys(cards, 0.0)
        sampler = HandSampler(info)
        for _ in range(self.n_samples):
            st = state_from_info(info, sampler.sample(rng))
            par = Paranoid(st.rules, st.trump, st.bids, self.target, st.N)
            for c in cards:
                s2 = st.copy()
                s2.apply_card(c)
                tot[c] += par.value(s2)
        # tie-break with the heuristic order
        tm = TrickModel(info, True)
        trump = info.trump

        def key(c):
            p = self.p_target_wins(info, tm, c)
            v = card_value(c, trump)
            h = (round(-p, 9), -v, -c) if m > 0 else (round(p, 9), v, c)
            return (round(tot[c], 6),) + h
        return min(cards, key=key)


class OmniscientSpoilerPolicy(SpoilerPolicy):
    """CHEATING reference (experiment only): solves the TRUE full state (read through the
    InfoState's underlying RoundState) with Paranoid(target) for every legal card.  The
    coalition is then a perfect-information paranoid player, against whatever the hero
    actually plays -- it isolates what hidden information costs the spoilers."""

    name = "spoiler_omni"

    def play(self, info, legal_mask, rng):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        m = self.mode(info)
        if m == 0 or info.seat == self.target:
            return self.base.play(info, legal_mask, rng)
        from ..solver.core import Paranoid
        st = info._s
        par = Paranoid(st.rules, st.trump, st.bids, self.target, st.N)
        tm = TrickModel(info, True)
        trump = info.trump
        vals = {}
        for c in _bits(legal_mask):
            s2 = st.copy()
            s2.apply_card(c)
            vals[c] = par.value(s2)

        def key(c):
            p = self.p_target_wins(info, tm, c)
            v = card_value(c, trump)
            h = (round(-p, 9), -v, -c) if m > 0 else (round(p, 9), v, c)
            return (round(vals[c], 6),) + h
        return min(vals, key=key)


class MCSpoilerPolicy(SpoilerPolicy):
    """Hidden-information spoiler WITH a model of the target's policy: for every legal card c,
    play out the same ``n_samples`` sampled worlds (consistent with this seat's InfoState) to
    the end with ``target_policy`` (default TA+, i.e. the true hero policy in the H1 runs) for
    the target and heuristic SpoilerPolicy for the other seats; play the card with the lowest
    mean target score (ties -> heuristic order)."""

    name = "spoiler_mc"

    def __init__(self, target: int, n_samples: int = 8, target_policy=None, base=None):
        super().__init__(target, base)
        self.n_samples = n_samples
        self.target_policy = target_policy if target_policy is not None else ta_plus_v2()

    def play(self, info, legal_mask, rng):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        m = self.mode(info)
        if m == 0 or info.seat == self.target:
            return self.base.play(info, legal_mask, rng)
        from ..engine import run_round
        from ..sampling import HandSampler
        from ..state import state_from_info
        cards = _bits(legal_mask)
        tot = dict.fromkeys(cards, 0.0)
        sampler = HandSampler(info)
        pols = [SpoilerPolicy(self.target, self.base) for _ in range(info.n_players)]
        pols[self.target] = self.target_policy
        for _ in range(self.n_samples):
            st = state_from_info(info, sampler.sample(rng))
            seed = int(rng.integers(1 << 31))
            for c in cards:
                s2 = st.copy()
                s2.apply_card(c)
                rr = [np.random.default_rng([seed, q]) for q in range(info.n_players)]
                run_round(s2, pols, rr, check=False)
                tot[c] += s2.scores()[self.target]
        tm = TrickModel(info, True)
        trump = info.trump

        def key(c):
            p = self.p_target_wins(info, tm, c)
            v = card_value(c, trump)
            h = (round(-p, 9), -v, -c) if m > 0 else (round(p, 9), v, c)
            return (round(tot[c], 6),) + h
        return min(cards, key=key)


__all__ = ["SpoilerPolicy", "PIMCSpoilerPolicy", "OmniscientSpoilerPolicy", "MCSpoilerPolicy"]

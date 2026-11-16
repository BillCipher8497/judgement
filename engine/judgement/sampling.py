"""Sample the other players' current hands uniformly from all assignments
consistent with an InfoState.

Known: own hand, every played card, each opponent's remaining card count, and
voids (an opponent who failed to follow a suit holds none of it).  When fewer
than 52 cards are dealt the undealt stock is one more, never-void "holder".

Algorithm (exact uniform, no rejection in the common path):
  * Holders with no known voids are exchangeable w.r.t. suits, so they are
    merged into one *pool*; once the pool's card set is chosen, splitting it
    among them is a plain shuffle.
  * The remaining *constrained* holders are handled with a suit-by-suit counting
    DP over their remaining capacities (the pool's capacity is implied).  Each
    suit's composition is drawn with probability proportional to
    multinomial(k_s; c) * completions(next state).
  * The DP's cost grows like prod(cap_i + 1), so only the most constraining
    holders (up to DP_STATE_LIMIT) go into the DP; any other constrained holders
    are put in the pool and their voids are enforced by rejection.  A proposal
    that is uniform over deals satisfying a subset of the constraints, accepted
    only when all constraints hold, is exactly uniform over consistent deals.
"""
from __future__ import annotations

from math import comb

import numpy as np

from .cards import FULL_DECK, SUIT_MASK, cards_of

DP_STATE_LIMIT = 1_500
REJECTION_TRIES = 1_000_000


class InconsistentInfo(ValueError):
    pass


def _compositions(k: int, caps: tuple[int, ...]):
    """All tuples c with 0 <= c[i] <= caps[i] and sum(c) <= k (remainder goes to the pool)."""
    if not caps:
        yield ()
        return
    first, rest = caps[0], caps[1:]
    for c in range(min(first, k) + 1):
        for tail in _compositions(k - c, rest):
            yield (c,) + tail


class HandSampler:
    def __init__(self, info):
        self.seat = info.seat
        N = info.n_players
        self.opps = [p for p in range(N) if p != info.seat]
        unknown = FULL_DECK & ~info.hand & ~info.played_mask
        self.unknown = cards_of(unknown)
        need = [info.cards_left(p) for p in self.opps]
        stock = len(self.unknown) - sum(need)
        if stock < 0:
            raise InconsistentInfo("card counts exceed unseen cards")
        voids = info.voids
        # holders: opponents then stock (index len(opps))
        self.holder_need = need + [stock]
        self.holder_void = [voids[p] for p in self.opps] + [0]
        constrained = [i for i, v in enumerate(self.holder_void) if v and self.holder_need[i] > 0]
        # most constraining first: more void suits, then more cards to place
        constrained.sort(key=lambda i: (-self.holder_void[i].bit_count(), -self.holder_need[i], i))
        self.con: list[int] = []
        size = 1
        for i in constrained:
            if size * (self.holder_need[i] + 1) <= DP_STATE_LIMIT:
                self.con.append(i)
                size *= self.holder_need[i] + 1
        self.con.sort()
        self.rej = [i for i in constrained if i not in self.con]   # enforced by rejection
        self.free = [i for i in range(len(self.holder_need)) if i not in self.con]
        self.mode = "shuffle"
        if self.con:
            self.suit_cards = [[c for c in self.unknown if (1 << c) & SUIT_MASK[s]] for s in range(4)]
            self.suit_left = [sum(len(self.suit_cards[t]) for t in range(s, 4)) for s in range(5)]
            # per suit: which constrained holders (by position in self.con) may receive it
            self.allowed = [tuple(j for j, i in enumerate(self.con) if not self.holder_void[i] >> s & 1)
                            for s in range(4)]
            self.pool_need = sum(self.holder_need[i] for i in self.free)
            caps = tuple(self.holder_need[i] for i in self.con)
            self.mode = "dp" if not self.rej else "dp+reject"
            self._memo: dict = {}
            self._opts: dict = {}
            self.caps0 = caps
            if self._count(0, caps) == 0:
                raise InconsistentInfo("no deal satisfies the void constraints")
        elif self.rej:
            self.mode = "reject"

    # ---- counting DP -------------------------------------------------------
    def _transitions(self, s: int, caps: tuple[int, ...]):
        k = len(self.suit_cards[s])
        allowed = self.allowed[s]
        pool_cap = self.suit_left[s] - sum(caps)  # implied remaining pool capacity
        for comp in _compositions(k, tuple(caps[j] for j in allowed)):
            to_pool = k - sum(comp)
            if to_pool > pool_cap:
                continue
            new = list(caps)
            ways = 1
            rem = k
            for j, c in zip(allowed, comp):
                new[j] -= c
                ways *= comb(rem, c)
                rem -= c
            yield comp, to_pool, ways, tuple(new)

    def _count(self, s: int, caps: tuple[int, ...]) -> int:
        if s == 4:
            return 1 if not any(caps) else 0
        key = (s, caps)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        total = 0
        for _, _, ways, new in self._transitions(s, caps):
            sub = self._count(s + 1, new)
            if sub:
                total += ways * sub
        self._memo[key] = total
        return total

    def _options(self, s: int, caps: tuple[int, ...]):
        key = (s, caps)
        opt = self._opts.get(key)
        if opt is None:
            comps, weights, nexts = [], [], []
            for comp, _, ways, new in self._transitions(s, caps):
                sub = self._count(s + 1, new)
                if sub:
                    comps.append(comp)
                    weights.append(ways * sub)
                    nexts.append(new)
            tot = sum(weights)
            cum = np.cumsum([w / tot for w in weights])
            opt = (comps, cum, nexts)
            self._opts[key] = opt
        return opt

    # ---- sampling ------------------------------------------------------------
    def _split(self, cards: list[int], holders: list[int], masks: list[int], rng) -> None:
        perm = rng.permutation(len(cards))
        pos = 0
        for i in holders:
            k = self.holder_need[i]
            m = 0
            for t in perm[pos:pos + k]:
                m |= 1 << cards[t]
            masks[i] = m
            pos += k

    def _ok(self, masks) -> bool:
        for i in self.rej:
            v = self.holder_void[i]
            for s in range(4):
                if v >> s & 1 and masks[i] & SUIT_MASK[s]:
                    return False
        return True

    def sample(self, rng: np.random.Generator) -> dict[int, int]:
        """Return {seat: current_hand_mask} for every opponent."""
        H = len(self.holder_need)
        if self.mode == "shuffle":
            masks = [0] * H
            self._split(self.unknown, list(range(H)), masks, rng)
            return dict(zip(self.opps, masks))
        for _ in range(REJECTION_TRIES):
            masks = self._propose(rng)
            if not self.rej or self._ok(masks):
                return dict(zip(self.opps, masks))  # drops the stock
        raise InconsistentInfo("rejection sampling failed to find a consistent deal")

    def _propose(self, rng) -> list[int]:
        H = len(self.holder_need)
        masks = [0] * H
        if self.mode == "reject":
            self._split(self.unknown, list(range(H)), masks, rng)
        else:
            caps = self.caps0
            pool: list[int] = []
            con = self.con
            for s in range(4):
                comps, cum, nexts = self._options(s, caps)
                j = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
                j = min(j, len(comps) - 1)
                cards = self.suit_cards[s]
                perm = rng.permutation(len(cards))
                pos = 0
                for jj, c in zip(self.allowed[s], comps[j]):
                    i = con[jj]
                    for t in perm[pos:pos + c]:
                        masks[i] |= 1 << cards[t]
                    pos += c
                pool.extend(cards[t] for t in perm[pos:])
                caps = nexts[j]
            self._split(pool, self.free, masks, rng)
        return masks


def sample_hands(info, rng: np.random.Generator) -> dict[int, int]:
    return HandSampler(info).sample(rng)

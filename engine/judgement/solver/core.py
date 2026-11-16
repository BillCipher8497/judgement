"""Exact perfect-information (double-dummy) solvers for the play phase of one round.

Positions are given as a ``RoundState`` in the PLAY phase (any point, including
mid-trick).  Bids are fixed (taken from the state).  Three solution concepts:

* ``MaxN``     max^n / subgame-perfect: every player maximises their own round
               score.  Ties among own-indifferent moves are broken by ``mode``:
               ``"neutral"``    first move in ascending card-index order,
               ``"spiteful"``   minimise the sum of the other players' scores,
               ``"benevolent"`` maximise the sum of the other players' scores
               (remaining ties -> ascending card index).
* ``paranoid`` security value of a target player when all others jointly
               minimise its score (zero-sum reduction).
* ``social``   maximum of the sum of all scores (cooperative upper bound).

Implementation notes
  * Memo (transposition table) at trick boundaries keyed on
    (hands tuple, leader, needs), need = bid - tricks won.  A player whose need
    is < 0 or > tricks remaining can no longer hit; when the miss score does not
    depend on the exact trick count (miss modes "zero", "neg_bid", "neg_10") all
    such needs are canonicalised to -1 ("dead") so more positions share entries.
  * Rank-equivalent cards (same hand, same suit, no card between them still held
    by anyone else or lying in the current trick) give isomorphic subgames, so
    only the lowest card of each class is searched.  This is exact.
  * Neutral max^n with a constant miss score only depends on each player's
    hit/miss indicator, so values are stored as 0/1 indicator vectors and the
    table is independent of the bid *magnitudes*: one table can be shared by all
    bid vectors on the same deal (used by bid enumeration / backward induction).
  * Cut-offs: a mover who reaches its best possible value stops (exact under the
    first-maximum tie-break); paranoid and social use the trivial global bounds.
"""
from __future__ import annotations

from ..cards import BIT, NT, SUIT_MASK, SUIT_OF
from ..state import PLAY

NEUTRAL, SPITEFUL, BENEVOLENT = "neutral", "spiteful", "benevolent"
MODES = (NEUTRAL, SPITEFUL, BENEVOLENT)
CONST_MISS = ("zero", "neg_bid", "neg_10")


def move_classes(legal: int, others: int) -> list[list[int]]:
    """Partition the legal cards into rank-equivalence classes (ascending)."""
    out: list[list[int]] = []
    prev = -1
    while legal:
        low = legal & -legal
        c = low.bit_length() - 1
        legal ^= low
        if prev >= 0 and SUIT_OF[c] == SUIT_OF[prev] and not (others & (low - (2 << prev))):
            out[-1].append(c)
        else:
            out.append([c])
        prev = c
    return out


def _reps(legal: int, others: int) -> list[int]:
    out = []
    prev = -1
    while legal:
        low = legal & -legal
        c = low.bit_length() - 1
        legal ^= low
        if not (prev >= 0 and SUIT_OF[c] == SUIT_OF[prev] and not (others & (low - (2 << prev)))):
            out.append(c)
        prev = c
    return out


def _unpack(state):
    if state.phase != PLAY:
        raise ValueError("solver needs a state in the PLAY phase")
    N = state.N
    hands = tuple(state.hands)
    trick = state.trick
    inplay = 0
    for _, c in trick:
        inplay |= BIT[c]
    led = SUIT_OF[trick[0][1]] if trick else -1
    rem = state.n_cards - len(state.trick_winners)  # tricks left incl. the current one
    needs = tuple(state.bids[p] - state.tricks_won[p] for p in range(N))
    return N, hands, state.leader, len(trick), led, state.best_card, state.best_seat, inplay, rem, needs


def last_trick_winner(hands, leader, N, trump):
    """Winner of the final trick (every hand holds exactly one card)."""
    bc = hands[leader].bit_length() - 1
    bs = leader
    for k in range(1, N):
        q = (leader + k) % N
        c = hands[q].bit_length() - 1
        s, sb = SUIT_OF[c], SUIT_OF[bc]
        if (s == sb and c > bc) or (s != sb and s == trump):
            bc, bs = c, q
    return bs


class _Base:
    def __init__(self, rules, trump: int, bids, N: int):
        self.rules = rules
        self.trump = trump
        self.bids = tuple(bids)
        self.N = N
        mp = rules.miss_penalty
        self.clamp = isinstance(mp, str) and mp in CONST_MISS
        self.hit = tuple(rules.score(b, b) for b in self.bids)
        self.miss = tuple(rules.score(b, b + 1) for b in self.bids) if self.clamp else None
        self.nodes = 0

    def canon(self, needs, rem):
        if not self.clamp:
            return needs
        return tuple(-1 if (x < 0 or x > rem) else x for x in needs)

    def final_scores(self, needs):
        sc = self.rules.score
        return tuple(sc(b, b - x) if x >= 0 or not self.clamp else self.miss[p]
                     for p, (b, x) in enumerate(zip(self.bids, needs)))


class MaxN(_Base):
    """max^n solver.  ``value(state)`` -> score vector; ``move_values(state)`` ->
    {card: score vector} for the player to act."""

    def __init__(self, rules, trump: int, bids, N: int | None = None, mode: str = NEUTRAL,
                 memo: dict | None = None):
        N = N or rules.n_players
        super().__init__(rules, trump, bids, N)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        # indicator representation (bid-magnitude free) for neutral + constant miss
        self.indicator = (mode == NEUTRAL and self.clamp)
        self.memo = {} if memo is None else memo

    # value representation -> real scores
    def to_scores(self, v):
        if self.indicator:
            return tuple(self.miss[p] + v[p] * (self.hit[p] - self.miss[p]) for p in range(self.N))
        return v

    def _leaf(self, needs):
        if self.indicator:
            return tuple(1 if x == 0 else 0 for x in needs)
        return self.final_scores(needs)

    def _best_own(self, p, needs):
        """Best value p can still get (cut-off threshold)."""
        if self.indicator:
            return 1 if needs[p] >= 0 else 0
        if self.clamp:
            return self.hit[p] if needs[p] >= 0 else self.miss[p]
        return self.hit[p]

    def _node(self, hands, leader, needs, rem):
        key = (hands, leader, needs) if self.indicator else (hands, leader, needs, self.bids)
        v = self.memo.get(key)
        if v is None:
            if rem == 0:
                v = self._leaf(needs)
            elif rem == 1:
                w = last_trick_winner(hands, leader, self.N, self.trump)
                nn = list(needs)
                nn[w] -= 1
                v = self._leaf(self.canon(tuple(nn), 0))
            else:
                v = self._trick(hands, leader, needs, rem, 0, -1, -1, -1, 0)
            self.memo[key] = v
        return v

    def _trick(self, hands, leader, needs, rem, i, led, bc, bs, inplay):
        self.nodes += 1
        N = self.N
        p = (leader + i) % N
        hand = hands[p]
        legal = (hand & SUIT_MASK[led]) if i else hand
        if not legal:
            legal = hand
        others = inplay
        for q in range(N):
            if q != p:
                others |= hands[q]
        mode = self.mode
        best = None
        bestkey = None
        cut = self._best_own(p, needs)
        trump = self.trump
        for c in _reps(legal, others):
            nh = list(hands)
            nh[p] = hand ^ BIT[c]
            nh = tuple(nh)
            if i == 0:
                nbc, nbs, nled = c, p, SUIT_OF[c]
            else:
                nled = led
                s = SUIT_OF[c]
                sb = SUIT_OF[bc]
                if (s == sb and c > bc) or (s != sb and s == trump):
                    nbc, nbs = c, p
                else:
                    nbc, nbs = bc, bs
            if i == N - 1:
                nn = list(needs)
                nn[nbs] -= 1
                r2 = rem - 1
                v = self._node(nh, nbs, self.canon(tuple(nn), r2), r2)
            else:
                v = self._trick(nh, leader, needs, rem, i + 1, nled, nbc, nbs, inplay | BIT[c])
            if mode == NEUTRAL:
                k = v[p]
                if best is None or k > bestkey:
                    best, bestkey = v, k
                    if k == cut:
                        break
            else:
                o = sum(v) - v[p]
                k = (v[p], -o if mode == SPITEFUL else o)
                if best is None or k > bestkey:
                    best, bestkey = v, k
        return best

    # ---- public API -----------------------------------------------------------
    def _root(self, state):
        N, hands, leader, i, led, bc, bs, inplay, rem, needs = _unpack(state)
        return hands, leader, self.canon(needs, rem), rem, i, led, bc, bs, inplay

    def raw_value(self, state):
        hands, leader, needs, rem, i, led, bc, bs, inplay = self._root(state)
        if i == 0:
            return self._node(hands, leader, needs, rem)
        return self._trick(hands, leader, needs, rem, i, led, bc, bs, inplay)

    def value(self, state):
        return self.to_scores(self.raw_value(state))

    def move_values(self, state) -> dict[int, tuple]:
        """Score vector after each legal card of the player to act (exact)."""
        hands, leader, needs, rem, i, led, bc, bs, inplay = self._root(state)
        N = self.N
        p = (leader + i) % N
        hand = hands[p]
        legal = state.legal_cards()
        others = inplay
        for q in range(N):
            if q != p:
                others |= hands[q]
        out = {}
        trump = self.trump
        for cls in move_classes(legal, others):
            c = cls[0]
            nh = list(hands)
            nh[p] = hand ^ BIT[c]
            nh = tuple(nh)
            if i == 0:
                nbc, nbs, nled = c, p, SUIT_OF[c]
            else:
                nled = led
                s, sb = SUIT_OF[c], SUIT_OF[bc]
                if (s == sb and c > bc) or (s != sb and s == trump):
                    nbc, nbs = c, p
                else:
                    nbc, nbs = bc, bs
            if i == N - 1:
                nn = list(needs)
                nn[nbs] -= 1
                v = self._node(nh, nbs, self.canon(tuple(nn), rem - 1), rem - 1)
            else:
                v = self._trick(nh, leader, needs, rem, i + 1, nled, nbc, nbs, inplay | BIT[c])
            v = self.to_scores(v)
            for cc in cls:
                out[cc] = v
        return out

    def best_move(self, state) -> int:
        """The card max^n (with this tie-break) plays."""
        mv = self.move_values(state)
        p = state.to_act
        mode = self.mode
        best, bk = None, None
        for c in sorted(mv):
            v = mv[c]
            if mode == NEUTRAL:
                k = v[p]
            else:
                o = sum(v) - v[p]
                k = (v[p], -o if mode == SPITEFUL else o)
            if best is None or k > bk:
                best, bk = c, k
        return best


class Paranoid(_Base):
    """Security value of ``target``: others jointly minimise its score."""

    def __init__(self, rules, trump: int, bids, target: int, N: int | None = None, memo: dict | None = None):
        N = N or rules.n_players
        super().__init__(rules, trump, bids, N)
        self.t = target
        self.memo = {} if memo is None else memo
        self.hi = self.hit[target]

    def _dead_or_leaf(self, need, rem):
        b = self.bids[self.t]
        if rem == 0:
            return self.rules.score(b, b - need) if need >= 0 or not self.clamp else self.miss[self.t]
        if self.clamp and (need < 0 or need > rem):
            return self.miss[self.t]
        return None

    def _node(self, hands, leader, need, rem):
        v = self._dead_or_leaf(need, rem)
        if v is not None:
            return v
        key = (hands, leader, need, self.bids[self.t])   # bid in key: memo may be shared across bids
        v = self.memo.get(key)
        if v is None:
            if rem == 1:
                w = last_trick_winner(hands, leader, self.N, self.trump)
                v = self._dead_or_leaf(need - 1 if w == self.t else need, 0)
            else:
                v = self._trick(hands, leader, need, rem, 0, -1, -1, -1, 0)
            self.memo[key] = v
        return v

    def _trick(self, hands, leader, need, rem, i, led, bc, bs, inplay):
        self.nodes += 1
        N = self.N
        t = self.t
        p = (leader + i) % N
        hand = hands[p]
        legal = (hand & SUIT_MASK[led]) if i else hand
        if not legal:
            legal = hand
        others = inplay
        for q in range(N):
            if q != p:
                others |= hands[q]
        maxing = p == t
        best = None
        hi = self.hi
        lo = self.miss[t] if self.clamp else None
        trump = self.trump
        for c in _reps(legal, others):
            nh = list(hands)
            nh[p] = hand ^ BIT[c]
            nh = tuple(nh)
            if i == 0:
                nbc, nbs, nled = c, p, SUIT_OF[c]
            else:
                nled = led
                s, sb = SUIT_OF[c], SUIT_OF[bc]
                if (s == sb and c > bc) or (s != sb and s == trump):
                    nbc, nbs = c, p
                else:
                    nbc, nbs = bc, bs
            if i == N - 1:
                v = self._node(nh, nbs, need - 1 if nbs == t else need, rem - 1)
            else:
                v = self._trick(nh, leader, need, rem, i + 1, nled, nbc, nbs, inplay | BIT[c])
            if best is None or (v > best if maxing else v < best):
                best = v
                if maxing and v == hi:
                    break
                if not maxing and v == lo:
                    break
        return best

    def value(self, state):
        N, hands, leader, i, led, bc, bs, inplay, rem, needs = _unpack(state)
        need = needs[self.t]
        if i == 0:
            return self._node(hands, leader, need, rem)
        v = self._dead_or_leaf(need, rem)
        if v is not None and rem != 0:
            return v
        return self._trick(hands, leader, need, rem, i, led, bc, bs, inplay)


class Social(_Base):
    """Maximum achievable sum of all players' scores (players fully cooperate)."""

    def __init__(self, rules, trump: int, bids, N: int | None = None, memo: dict | None = None):
        N = N or rules.n_players
        super().__init__(rules, trump, bids, N)
        self.memo = {} if memo is None else memo

    def _ub(self, needs):
        if self.clamp:
            return sum(self.hit[p] if needs[p] >= 0 else self.miss[p] for p in range(self.N))
        return sum(self.hit)

    def _node(self, hands, leader, needs, rem):
        if rem == 0:
            return sum(self.final_scores(needs))
        if rem == 1:
            w = last_trick_winner(hands, leader, self.N, self.trump)
            nn = list(needs)
            nn[w] -= 1
            return sum(self.final_scores(self.canon(tuple(nn), 0)))
        key = (hands, leader, needs, self.bids)
        v = self.memo.get(key)
        if v is None:
            v = self._trick(hands, leader, needs, rem, 0, -1, -1, -1, 0)
            self.memo[key] = v
        return v

    def _trick(self, hands, leader, needs, rem, i, led, bc, bs, inplay):
        self.nodes += 1
        N = self.N
        p = (leader + i) % N
        hand = hands[p]
        legal = (hand & SUIT_MASK[led]) if i else hand
        if not legal:
            legal = hand
        others = inplay
        for q in range(N):
            if q != p:
                others |= hands[q]
        ub = self._ub(needs)
        best = None
        trump = self.trump
        for c in _reps(legal, others):
            nh = list(hands)
            nh[p] = hand ^ BIT[c]
            nh = tuple(nh)
            if i == 0:
                nbc, nbs, nled = c, p, SUIT_OF[c]
            else:
                nled = led
                s, sb = SUIT_OF[c], SUIT_OF[bc]
                if (s == sb and c > bc) or (s != sb and s == trump):
                    nbc, nbs = c, p
                else:
                    nbc, nbs = bc, bs
            if i == N - 1:
                nn = list(needs)
                nn[nbs] -= 1
                v = self._node(nh, nbs, self.canon(tuple(nn), rem - 1), rem - 1)
            else:
                v = self._trick(nh, leader, needs, rem, i + 1, nled, nbc, nbs, inplay | BIT[c])
            if best is None or v > best:
                best = v
                if v == ub:
                    break
        return best

    def value(self, state):
        N, hands, leader, i, led, bc, bs, inplay, rem, needs = _unpack(state)
        needs = self.canon(needs, rem)
        if i == 0:
            return self._node(hands, leader, needs, rem)
        return self._trick(hands, leader, needs, rem, i, led, bc, bs, inplay)


class TrickRange:
    """Min / max number of tricks seat ``p`` can end with if *all* players
    cooperate toward that goal (bounds for any outcome of any play)."""

    def __init__(self, N: int, trump: int):
        self.N, self.trump = N, trump
        self.memo: dict = {}

    def _node(self, hands, leader, p, sign, rem):
        if rem == 0:
            return 0
        if rem == 1:
            return 1 if last_trick_winner(hands, leader, self.N, self.trump) == p else 0
        key = (hands, leader, p, sign)
        v = self.memo.get(key)
        if v is None:
            v = self._trick(hands, leader, p, sign, rem, 0, -1, -1, -1, 0)
            self.memo[key] = v
        return v

    def _trick(self, hands, leader, p, sign, rem, i, led, bc, bs, inplay):
        N = self.N
        q = (leader + i) % N
        hand = hands[q]
        legal = (hand & SUIT_MASK[led]) if i else hand
        if not legal:
            legal = hand
        others = inplay
        for r in range(N):
            if r != q:
                others |= hands[r]
        best = None
        trump = self.trump
        target = rem if sign > 0 else 0
        for c in _reps(legal, others):
            nh = list(hands)
            nh[q] = hand ^ BIT[c]
            nh = tuple(nh)
            if i == 0:
                nbc, nbs, nled = c, q, SUIT_OF[c]
            else:
                nled = led
                s, sb = SUIT_OF[c], SUIT_OF[bc]
                if (s == sb and c > bc) or (s != sb and s == trump):
                    nbc, nbs = c, q
                else:
                    nbc, nbs = bc, bs
            if i == N - 1:
                v = (nbs == p) + self._node(nh, nbs, p, sign, rem - 1)
            else:
                v = self._trick(nh, leader, p, sign, rem, i + 1, nled, nbc, nbs, inplay | BIT[c])
            if best is None or (v > best if sign > 0 else v < best):
                best = v
                if v == target:
                    break
        return best

    def range(self, hands, leader, p, rem):
        hands = tuple(hands)
        return self._node(hands, leader, p, -1, rem), self._node(hands, leader, p, 1, rem)


# ---- convenience -------------------------------------------------------------------
def maxn_value(state, mode: str = NEUTRAL):
    return MaxN(state.rules, state.trump, state.bids, state.N, mode).value(state)


def paranoid_value(state, target: int):
    return Paranoid(state.rules, state.trump, state.bids, target, state.N).value(state)


def social_value(state):
    return Social(state.rules, state.trump, state.bids, state.N).value(state)


def play_out(state, choose):
    """Play ``state`` (a copy) to the end; ``choose(state) -> card``.  Returns final state."""
    st = state.copy()
    while st.phase == PLAY:
        st.apply_card(choose(st))
    return st

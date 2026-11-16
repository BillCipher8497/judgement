"""Bids for perfect-information experiments.

``rational_bids``  "perfect-info rational" bidding.  Scheme (stated precisely):
    bidders act in table order (left of dealer first, dealer last); everyone
    sees all hands.  The game after bidding is valued by max^n (given tie-break
    mode).  Bidding is solved by backward induction over the bid sequence: the
    k-th bidder picks the hook-legal bid maximising its own max^n value given
    the earlier bids, anticipating that later bidders do the same.  A bidder
    that can hit takes the highest hitting bid (strictly best); a bidder that
    cannot hit with any bid takes the best miss score, ties -> lowest bid.  This is the subgame-perfect equilibrium
    of the full (bid + play) perfect-information game under that selection.
    Leaves share one transposition table (bid-magnitude-free values in the
    neutral / constant-miss case), so the (n+1)^N leaves are cheap.

``legal_bid_vectors``  all hook-legal bid vectors (in seat order).
``greedy_trick_bids``      "naive" bids = tricks each seat wins when every seat
    plays GreedyPolicy on the full deal (dealer snapped to nearest hook-legal).
"""
from __future__ import annotations

from itertools import product

from ..engine import run_round
from ..policies.base import nearest_legal
from ..policies.greedy import GreedyPolicy
from ..state import RoundState
from .core import CONST_MISS, NEUTRAL, MaxN, TrickRange


def bid_order(N: int, dealer: int) -> list[int]:
    return [(dealer + 1 + k) % N for k in range(N)]


def make_play_state(rules, n, trump, dealer, hands, bids_by_seat) -> RoundState:
    st = RoundState(rules, n, trump, dealer, list(hands))
    for p in bid_order(rules.n_players, dealer):
        st.apply_bid(bids_by_seat[p])
    return st


def legal_bid_vectors(rules, n, dealer):
    """All hook-legal bid vectors, indexed by seat."""
    N = rules.n_players
    order = bid_order(N, dealer)
    out = []
    for seq in product(range(n + 1), repeat=N):
        if rules.hook and sum(seq) == n:
            continue
        v = [0] * N
        for p, b in zip(order, seq):
            v[p] = b
        out.append(tuple(v))
    return out


def rational_bids(rules, n, trump, dealer, hands, mode: str = NEUTRAL, memo: dict | None = None):
    """Returns (bids_by_seat, value_vector).  See module docstring."""
    N = rules.n_players
    order = bid_order(N, dealer)
    memo = {} if memo is None else memo
    leaf_cache: dict = {}

    sc = rules.score
    fast = None
    if mode == NEUTRAL:
        fast = MaxN(rules, trump, (0,) * N, N, mode, memo=memo)
    hands_t = tuple(hands)
    leader = (dealer + 1) % N
    mp = rules.miss_penalty
    const = isinstance(mp, str) and mp in CONST_MISS
    if const:
        tr = TrickRange(N, trump)
        ranges = [tr.range(hands_t, leader, p, n) for p in range(N)]

    def leaf(seq):
        v = [0] * N
        for p, b in zip(order, seq):
            v[p] = b
        v = tuple(v)
        r = leaf_cache.get(v)
        if r is None:
            if fast is not None and fast.indicator:
                # indicator values depend on the needs only -> one shared solver
                ind = fast._node(hands_t, leader, fast.canon(v, n), n)
                r = tuple(sc(b, b) if ind[p] else sc(b, b + 1) for p, b in enumerate(v))
            else:
                st = make_play_state(rules, n, trump, dealer, hands, v)
                r = MaxN(rules, trump, v, N, mode, memo=memo if mode == NEUTRAL else None).value(st)
            leaf_cache[v] = r
        return r

    def rec(seq):
        k = len(seq)
        if k == N:
            return leaf(seq), seq
        p = order[k]
        legal = range(n + 1)
        if k == N - 1 and rules.hook:
            forb = n - sum(seq)
            legal = [b for b in legal if b != forb]
        # Descending order: the first bid that hits is p's global optimum (a hit
        # with bid b scores hit_base + b > any miss), so we stop there.  If no bid
        # hits, p takes the best miss value, ties -> lowest bid.
        # With a constant miss score only bids inside p's feasible trick range
        # [lo, hi] (all players cooperating) can hit; if none of them hits, every
        # bid misses equally and p takes the lowest legal bid.
        if const:
            lo, hi = ranges[p]
            cand = [b for b in sorted(legal, reverse=True) if lo <= b <= hi]
            done = {}
            for b in cand:
                r = rec(seq + (b,))
                if r[0][p] == sc(b, b):
                    return r
                done[b] = r
            b0 = min(legal)
            return done[b0] if b0 in done else rec(seq + (b0,))
        best = None
        for b in sorted(legal, reverse=True):
            val, full = rec(seq + (b,))
            if val[p] == sc(b, b):
                return val, full
            if best is None or val[p] >= best[0][p]:
                best = (val, full)
        return best

    val, seq = rec(())
    bids = [0] * N
    for p, b in zip(order, seq):
        bids[p] = b
    return tuple(bids), tuple(val)


def greedy_trick_bids(rules, n, trump, dealer, hands):
    """Naive bids: tricks won under all-greedy perfect-info play; dealer hook-snapped."""
    N = rules.n_players
    st = RoundState(rules, n, trump, dealer, list(hands))
    # play with dummy bids (greedy ignores bids): bid 0 except hook
    for p in bid_order(N, dealer):
        L = st.legal_bids()
        st.apply_bid(L[0])
    run_round(st, [GreedyPolicy()] * N, [None] * N, check=False)
    tricks = list(st.tricks_won)
    order = bid_order(N, dealer)
    if rules.hook:
        d = order[-1]
        forb = n - sum(tricks[p] for p in order[:-1])
        legal = [b for b in range(n + 1) if b != forb]
        tricks[d] = nearest_legal(tricks[d], legal)
    return tuple(tricks)

"""Toy Judgement: same scoring (hit = 10 + bid, miss = 0) and hook as the full game,
reduced deck, one round, explicit game tree for exact tabular solving.

Game ``ToyGame(N, S, R, n, trump)``:
* deck of S suits x R ranks, card c = s*R + r (higher r = stronger); trump suit or NT (-1);
* every ordered deal of n-card hands to seats 0..N-1 equally likely (rest = undealt stock);
* dealer = seat N-1; bids 0..n in seat order 0..N-1, the dealer may not make sum(bids) == n;
* seat 0 leads trick 1, trick winner leads next; must follow suit, void -> any card;
* trick: highest trump, else highest of led suit; payoff = 10 + bid if exact else 0
  (``util='diff'``: 2-player zero-sum s_0 - s_1; ``util='rel'``: s_i - mean of others).

Information set of seat p = (p, own hand, public action sequence: bids then cards).
Moves with a single legal action are applied without creating a decision node (they stay
in the public sequence, so perfect recall is kept).  Action index: bid value for bids,
position of the card in the sorted original hand for plays; A = n + 1 slots per infoset.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

NT = -1


@dataclass(frozen=True)
class ToyGame:
    N: int = 3
    S: int = 2
    R: int = 4
    n: int = 2
    trump: int = 0
    hit_base: int = 10
    util: str = "raw"            # raw | diff (2p) | rel
    hook: bool = True

    @property
    def deck(self):
        return list(range(self.S * self.R))

    @property
    def A(self):
        return self.n + 1

    @property
    def dealer(self):
        return self.N - 1

    def suit(self, c):
        return c // self.R

    def legal_bids(self, bids):
        L = list(range(self.n + 1))
        if self.hook and len(bids) == self.N - 1:
            f = self.n - sum(bids)
            L = [b for b in L if b != f]
        return L

    def legal_cards(self, hand_left, trick):
        """hand_left: list of cards; trick: list of (seat, card) of the current trick."""
        if trick:
            led = self.suit(trick[0][1])
            fol = [c for c in hand_left if self.suit(c) == led]
            if fol:
                return fol
        return list(hand_left)

    def beats(self, c, w, led):
        """card c beats current winning card w in a trick led in suit led."""
        sc, sw = self.suit(c), self.suit(w)
        if sc == sw:
            return c > w
        return self.trump != NT and sc == self.trump

    def trick_winner(self, trick):
        led = self.suit(trick[0][1])
        ws, wc = trick[0]
        for s, c in trick[1:]:
            if self.beats(c, wc, led):
                ws, wc = s, c
        return ws

    def scores(self, bids, tricks):
        return [self.hit_base + b if b == t else 0 for b, t in zip(bids, tricks)]

    def utility(self, bids, tricks):
        s = self.scores(bids, tricks)
        if self.util == "raw":
            return s
        if self.util == "diff":
            assert self.N == 2
            return [s[0] - s[1], s[1] - s[0]]
        if self.util == "rel":
            tot = sum(s)
            return [x - (tot - x) / (self.N - 1) for x in s]
        raise ValueError(self.util)

    def deals(self):
        """All ordered deals as tuples of sorted hands (uniform chance)."""
        out = []

        def rec(rest, acc):
            if len(acc) == self.N:
                out.append(tuple(acc))
                return
            for h in combinations(rest, self.n):
                hs = set(h)
                rec([c for c in rest if c not in hs], acc + [h])
        rec(self.deck, [])
        return out

    def replay(self, hands, hist):
        """Reconstruct the play state from the public sequence: returns
        (hand_left per seat, bids, trick (list of (seat,card)), leader, tricks_won, to_move)."""
        N = self.N
        bids = list(hist[:N])
        left = [list(h) for h in hands] if hands is not None else None
        tricks = [0] * N
        trick, leader = [], 0
        for c in hist[N:]:
            s = (leader + len(trick)) % N
            if left is not None:
                left[s].remove(c)
            trick.append((s, c))
            if len(trick) == N:
                leader = self.trick_winner(trick)
                tricks[leader] += 1
                trick = []
        to_move = len(hist) if len(hist) < N else (leader + len(trick)) % N
        return left, bids, trick, leader, tricks, to_move


class Tree:
    """Explicit game tree; see module docstring.  Arrays:
    node_len, node_pl (-1 terminal), node_is, node_c (chance prob), U (terminal utilities);
    edges sorted by (parent length, parent): e_par, e_ch, e_is, e_a, e_pl;
    infosets: is_pl, is_valid (nIS x A bool), is_rep (one node of the infoset), is_key."""

    def __init__(self, g: ToyGame):
        self.g = g
        N, A = g.N, g.A
        node_len, node_pl, node_is, node_c = [], [], [], []
        U = {}
        e_par, e_ch, e_is, e_a = [], [], [], []
        isd: dict = {}
        is_pl, is_valid, is_rep, is_key = [], [], [], []
        roots = []
        deals = g.deals()
        pc = 1.0 / len(deals)

        def new_node(L, pl, s):
            node_len.append(L); node_pl.append(pl); node_is.append(s); node_c.append(pc)
            return len(node_len) - 1

        def infoset(p, hand, hist, legal_idx, nid):
            key = (p, hand, hist)
            s = isd.get(key)
            if s is None:
                s = len(is_pl)
                isd[key] = s
                is_pl.append(p)
                v = [False] * A
                for a in legal_idx:
                    v[a] = True
                is_valid.append(v)
                is_rep.append(nid)
                is_key.append(key)
            return s

        def rec(hands, left, bids, trick, leader, tricks, hist):
            # advance through forced moves; returns node id of next decision/terminal node
            while True:
                L = len(hist)
                if L < N:
                    p = L
                    legal = g.legal_bids(bids)
                    if len(legal) == 1:
                        bids = bids + [legal[0]]
                        hist = hist + (legal[0],)
                        continue
                    nid = new_node(L, p, -1)
                    s = infoset(p, hands[p], hist, legal, nid)
                    node_is[nid] = s
                    for b in legal:
                        ch = rec(hands, left, bids + [b], trick, leader, tricks, hist + (b,))
                        e_par.append(nid); e_ch.append(ch); e_is.append(s); e_a.append(b)
                    return nid
                if L == N + N * g.n:
                    nid = new_node(L, -1, -1)
                    U[nid] = g.utility(bids, tricks)
                    return nid
                p = (leader + len(trick)) % N
                legal = g.legal_cards(left[p], trick)
                if len(legal) == 1:
                    c = legal[0]
                    left, trick, leader, tricks = _play(g, left, trick, leader, tricks, p, c)
                    hist = hist + (c,)
                    continue
                nid = new_node(L, p, -1)
                h = hands[p]
                s = infoset(p, h, hist, [h.index(c) for c in legal], nid)
                node_is[nid] = s
                for c in legal:
                    l2, t2, ld2, tr2 = _play(g, left, trick, leader, tricks, p, c)
                    ch = rec(hands, l2, bids, t2, ld2, tr2, hist + (c,))
                    e_par.append(nid); e_ch.append(ch); e_is.append(s); e_a.append(h.index(c))
                return nid

        for hands in deals:
            roots.append(rec(hands, [list(h) for h in hands], [], [], 0, [0] * N, ()))

        self.n_deals = len(deals)
        self.roots = np.array(roots, np.int64)
        self.node_len = np.array(node_len, np.int16)
        self.node_pl = np.array(node_pl, np.int8)
        self.node_is = np.array(node_is, np.int64)
        self.node_c = np.array(node_c)
        self.n_nodes = len(node_len)
        self.U = np.zeros((self.n_nodes, N))
        for k, v in U.items():
            self.U[k] = v
        self.is_pl = np.array(is_pl, np.int8)
        self.is_valid = np.array(is_valid, bool).reshape(-1, A)
        self.is_rep = np.array(is_rep, np.int64)
        self.is_key = is_key
        self.n_is = len(is_pl)
        self.is_bid = np.array([len(k[2]) < N for k in is_key], bool)
        e_par = np.array(e_par, np.int64)
        order = np.lexsort((e_par, self.node_len[e_par]))
        self.e_par = e_par[order]
        self.e_ch = np.array(e_ch, np.int64)[order]
        self.e_is = np.array(e_is, np.int64)[order]
        self.e_a = np.array(e_a, np.int64)[order]
        self.e_pl = self.node_pl[self.e_par].astype(np.int64)
        self.e_ia = self.e_is * A + self.e_a
        self.n_edges = len(self.e_par)
        # chunks by parent length (ascending)
        pl_len = self.node_len[self.e_par]
        cuts = np.flatnonzero(np.diff(pl_len)) + 1
        self.chunks = []
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, self.n_edges]):
            par = self.e_par[a:b]
            seg = np.r_[0, np.flatnonzero(np.diff(par)) + 1]
            self.chunks.append((int(a), int(b), seg, par[seg]))

    def uniform(self):
        v = self.is_valid.astype(float)
        return v / v.sum(1, keepdims=True)


def _play(g, left, trick, leader, tricks, p, c):
    left = [list(x) for x in left]
    left[p].remove(c)
    trick = trick + [(p, c)]
    if len(trick) == g.N:
        leader = g.trick_winner(trick)
        tricks = list(tricks)
        tricks[leader] += 1
        trick = []
    return left, trick, leader, tricks

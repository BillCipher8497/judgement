"""Tabular versions of the project heuristics in the toy game (one-hot profiles).

* greedy play: lead the strongest card (highest trump, else highest rank); follow with the
  highest card that beats the current winner, else discard the lowest (non-trumps first).
* TA play (target-aware, simplified to the toy): exact P(card wins the trick) over uniform
  hidden deals consistent with voids, assuming any later player who *can* beat it does.
  need = bid - won, r = cards left.  0 < need <= r: cheapest sure winner; else (following)
  cheapest card within 0.1 of the best P(win) if that is > 0, else the cheapest card;
  (leading) the max-P(win) card.  need == 0: most valuable sure loser, else min P(win).
  Missed: cheapest card.  Value = rank + R if trump.
* myopic bidder: own trick distribution over uniform hidden deals (previous bids ignored),
  every seat playing greedy; bid argmax_b (10+b) P(t=b) over hook-legal b (ties: lower).
  For n = 1 this is exactly "bid 1 iff P(win) > 10/21".
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np

from .toy import NT, ToyGame


def _val(g, c):
    return c % g.R + (g.R if g.trump != NT and g.suit(c) == g.trump else 0)


def greedy_card(g: ToyGame, left, trick):
    legal = g.legal_cards(left, trick)
    if not trick:
        tr = [c for c in legal if g.trump != NT and g.suit(c) == g.trump]
        if tr:
            return max(tr)
        return max(legal, key=lambda c: (c % g.R, -c))
    led = g.suit(trick[0][1])
    w = trick[0][1]
    for _, c in trick[1:]:
        if g.beats(c, w, led):
            w = c
    win = [c for c in legal if g.beats(c, w, led)]
    if win:
        return max(win, key=lambda c: (_val(g, c), -c))
    return min(legal, key=lambda c: (_val(g, c), c))


def _voids(g, hist):
    N = g.N
    v = [set() for _ in range(N)]
    trick, leader = [], 0
    for c in hist[N:]:
        s = (leader + len(trick)) % N
        if trick and g.suit(c) != g.suit(trick[0][1]):
            v[s].add(g.suit(trick[0][1]))
        trick.append((s, c))
        if len(trick) == N:
            leader = g.trick_winner(trick)
            trick = []
    return v


def _worlds(g, p, hand, hist):
    """Uniform hidden deals consistent with p's view: list of per-seat remaining hands."""
    N, n = g.N, g.n
    played = set(hist[N:])
    _, _, trick, leader, _, _ = g.replay(None, hist)
    # cards played per seat
    cnt = [0] * N
    tr, ld = [], 0
    for c in hist[N:]:
        s = (ld + len(tr)) % N
        cnt[s] += 1
        tr.append((s, c))
        if len(tr) == N:
            ld = g.trick_winner(tr)
            tr = []
    voids = _voids(g, hist)
    unseen = [c for c in g.deck if c not in played and c not in hand]
    others = [q for q in range(N) if q != p]
    out = []

    def rec(i, rest, acc):
        if i == len(others):
            out.append(dict(acc))
            return
        q = others[i]
        k = n - cnt[q]
        for h in combinations(rest, k):
            if any(g.suit(c) in voids[q] for c in h):
                continue
            hs = set(h)
            acc[q] = list(h)
            rec(i + 1, [c for c in rest if c not in hs], acc)
        acc.pop(q, None)
    rec(0, unseen, {})
    return out, trick, leader


@lru_cache(maxsize=None)
def _pwin_cached(g, p, hand, play_hist):
    hist = (0,) * g.N + play_hist
    left = [c for c in hand if c not in play_hist]
    worlds, trick, _ = _worlds(g, p, hand, hist)
    legal = g.legal_cards(left, trick)
    led_of = lambda c: g.suit(trick[0][1]) if trick else g.suit(c)  # noqa: E731
    N = g.N
    later = [(p + i) % N for i in range(1, N - len(trick))]
    res = {}
    for c in legal:
        led = led_of(c)
        w = c
        if trick:
            w0 = trick[0][1]
            for _, x in trick[1:]:
                if g.beats(x, w0, led):
                    w0 = x
            if not g.beats(c, w0, led):
                res[c] = 0.0
                continue
        tr2 = list(trick) + [(p, c)]
        wins = 0
        for wd in worlds:
            ok = True
            for q in later:
                if any(g.beats(x, w, led) for x in g.legal_cards(wd[q], tr2)):
                    ok = False
                    break
            wins += ok
        res[c] = wins / len(worlds)
    return res


def ta_card(g: ToyGame, p, hand, hist):
    N = g.N
    _, bids, trick, _, tricks, _ = g.replay(None, hist)
    left = [c for c in hand if c not in hist[N:]]
    legal = g.legal_cards(left, trick)
    need, r = bids[p] - tricks[p], len(left)
    pw = _pwin_cached(g, p, hand, tuple(hist[N:]))
    cheap = lambda cs: min(cs, key=lambda c: (_val(g, c), c))  # noqa: E731
    dear = lambda cs: max(cs, key=lambda c: (_val(g, c), -c))  # noqa: E731
    if 0 < need <= r:
        sure = [c for c in legal if pw[c] >= 1.0]
        if sure:
            return cheap(sure)
        best = max(pw[c] for c in legal)
        if trick:
            if best > 0:
                return cheap([c for c in legal if pw[c] >= best - 0.1])
            return cheap(legal)
        return cheap([c for c in legal if pw[c] >= best])
    if need == 0:
        lose = [c for c in legal if pw[c] <= 0.0]
        if lose:
            return dear(lose)
        m = min(pw[c] for c in legal)
        return cheap([c for c in legal if pw[c] <= m])
    return cheap(legal)


@lru_cache(maxsize=None)
def trick_dist_greedy(g: ToyGame, p, hand):
    """P(own tricks = k) for seat p with ``hand``, uniform hidden deals, all seats greedy."""
    worlds, _, _ = _worlds(g, p, hand, (0,) * g.N)
    dist = np.zeros(g.n + 1)
    for wd in worlds:
        left = [list(wd[q]) if q != p else list(hand) for q in range(g.N)]
        trick, leader, tricks = [], 0, [0] * g.N
        for _ in range(g.N * g.n):
            s = (leader + len(trick)) % g.N
            c = greedy_card(g, left[s], trick)
            left[s].remove(c)
            trick.append((s, c))
            if len(trick) == g.N:
                leader = g.trick_winner(trick)
                tricks[leader] += 1
                trick = []
        dist[tricks[p]] += 1
    return dist / len(worlds)


def myopic_bid(g: ToyGame, p, hand, bids):
    d = trick_dist_greedy(g, p, hand)
    legal = g.legal_bids(list(bids))
    return max(legal, key=lambda b: ((g.hit_base + b) * d[b], -b))


def profile(tree, play="greedy", bid="myopic"):
    """One-hot (n_is x A) profile.  play in {greedy, ta}; bid in {myopic}."""
    g = tree.g
    sig = np.zeros(tree.is_valid.shape)
    for s, (p, hand, hist) in enumerate(tree.is_key):
        if len(hist) < g.N:
            a = myopic_bid(g, p, hand, hist)
        else:
            if play == "greedy":
                _, _, trick, _, _, _ = g.replay(None, hist)
                left = [c for c in hand if c not in hist[g.N:]]
                c = greedy_card(g, left, trick)
            else:
                c = ta_card(g, p, hand, hist)
            a = hand.index(c)
        assert tree.is_valid[s, a], (s, a)
        sig[s, a] = 1.0
    return sig

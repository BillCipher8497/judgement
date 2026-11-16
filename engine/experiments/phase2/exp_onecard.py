"""Stretch: exact analysis of the 1-card round (4 players, imperfect information).

Each player sees only its own card and the earlier bids; play is forced, so this
is a pure sequential bidding / signalling game.  Positions 0..3 bid in order
(0 = left of dealer = leader, 3 = dealer, hook applies to the dealer).  A pure
strategy of position k maps (own card, earlier-bid history) -> bid in {0,1}.

Iterated best response (Gauss-Seidel over positions 0..3), exact expectation over
all 52*51*50*49 ordered deals (vectorised): when position k best-responds it
anticipates how later positions' fixed strategies react to its bid (so bids act
as signals).  Ties -> bid 0.  Start: bid 1 iff P(win | own card, position) > 10/21
(the signal-free myopic threshold).  Stops when a full sweep changes nothing or
a strategy profile repeats (cycle).
Usage: python exp_onecard.py   -> results/onecard.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "results"
RANKS = "23456789TJQKA"
SUITS = "SHDC"


def all_deals():
    c = np.arange(52, dtype=np.int16)
    a = np.repeat(c, 51 * 50 * 49)
    # build lexicographic distinct 4-tuples via masks (memory: ~6.5M x 4)
    out = []
    for c0 in range(52):
        r1 = np.array([x for x in range(52) if x != c0], dtype=np.int16)
        for c1 in r1:
            r2 = r1[r1 != c1]
            g2, g3 = np.meshgrid(r2, r2, indexing="ij")
            m = g2 != g3
            n = int(m.sum())
            blk = np.empty((n, 4), dtype=np.int16)
            blk[:, 0] = c0
            blk[:, 1] = c1
            blk[:, 2] = g2[m]
            blk[:, 3] = g3[m]
            out.append(blk)
    del a
    return np.concatenate(out)


def winner(D, trump):
    suit = D // 13
    rank = D % 13
    led = suit[:, 0:1]
    if trump >= 0:
        key = np.where(suit == trump, 100 + rank, np.where(suit == led, rank, -1))
    else:
        key = np.where(suit == led, rank, -1)
    return key.argmax(axis=1)


def solve(trump, max_iter=30):
    t0 = time.time()
    D = all_deals()
    W = winner(D, trump)
    M = len(D)
    # strategies: S[k][card, hist] with hist = integer of earlier bits (k bits)
    pwin = [np.bincount(D[:, k], weights=(W == k), minlength=52) / np.bincount(D[:, k], minlength=52) for k in range(4)]
    S = [np.zeros((52, 1 << k), dtype=np.int8) for k in range(4)]
    for k in range(4):
        S[k][:] = (pwin[k] > 10 / 21)[:, None]
    history = []

    def play_bids(S, k_fixed=None, x=None):
        """Bids of all positions (vector over deals); if k_fixed given, force its bid to x."""
        bids = np.zeros((M, 4), dtype=np.int8)
        hist = np.zeros(M, dtype=np.int64)
        for k in range(4):
            if k == k_fixed:
                b = np.full(M, x, dtype=np.int8)
            else:
                b = S[k][D[:, k], hist]
            if k == 3:  # hook: sum must not equal 1
                s3 = bids[:, :3].sum(axis=1)
                b = np.where(s3 + b == 1, 1 - b, b).astype(np.int8)
            bids[:, k] = b
            hist = hist * 2 + b
        return bids

    def score(bids, k):
        t = (W == k).astype(np.int8)
        return np.where(bids[:, k] == t, 10 + bids[:, k], 0)

    sweeps = []
    converged = False
    for it in range(max_iter):
        changed = 0
        for k in range(4):
            # history index of position k under current strategies of earlier positions
            bids_cur = play_bids(S)
            hist_k = np.zeros(M, dtype=np.int64)
            for j in range(k):
                hist_k = hist_k * 2 + bids_cur[:, j]
            idx = D[:, k].astype(np.int64) * (1 << k) + hist_k
            ev = []
            for x in (0, 1):
                bx = play_bids(S, k, x)
                sc = score(bx, k).astype(np.float64)
                if k == 3:   # hook-illegal choice: mark as -inf
                    s3 = bids_cur[:, :3].sum(axis=1)
                    sc = np.where(s3 + x == 1, -1e9, sc)
                ev.append(np.bincount(idx, weights=sc, minlength=52 << k))
            cnt = np.bincount(idx, minlength=52 << k)
            new = (ev[1] > ev[0]).astype(np.int8).reshape(52, 1 << k)
            seen = (cnt > 0).reshape(52, 1 << k)
            old = S[k]
            diff = int(((new != old) & seen).sum())
            S[k] = np.where(seen, new, old)
            changed += diff
        key = tuple(s.tobytes() for s in S)
        sweeps.append(changed)
        if changed == 0:
            converged = True
            break
        if key in history:
            break
        history.append(key)
    bids = play_bids(S)
    vals = [float(score(bids, k).mean()) for k in range(4)]
    hit = [float((bids[:, k] == (W == k)).mean()) for k in range(4)]
    sumb = bids.sum(axis=1)
    # reachable histories per position
    rules = {}
    for k in range(4):
        hist_k = np.zeros(M, dtype=np.int64)
        for j in range(k):
            hist_k = hist_k * 2 + bids[:, j]
        rules[k] = {}
        for h in range(1 << k):
            reach = np.zeros(52, bool)
            reach[np.unique(D[hist_k == h, k])] = True
            if not reach.any():
                continue
            hs = format(h, f"0{k}b") if k else "-"
            ones = [c for c in range(52) if reach[c] and S[k][c, h] == 1]
            rules[k][hs] = describe(ones, trump)
    return {"trump": "NT" if trump < 0 else SUITS[trump], "converged": converged, "sweeps_changes": sweeps,
            "values_by_pos": vals, "hit_rate_by_pos": hit, "table_value": float(sum(vals)),
            "P(sum_bids=0)": float((sumb == 0).mean()), "P(sum_bids>=2)": float((sumb >= 2).mean()),
            "bid1_cards_by_pos_and_history": rules,
            "myopic_pwin_threshold_cards": {k: describe([c for c in range(52) if pwin[k][c] > 10 / 21], trump) for k in range(4)},
            "seconds": round(time.time() - t0, 1)}


def describe(cards, trump):
    """Summarise a card set as per-suit rank sets."""
    out = {}
    for s in range(4):
        rs = sorted(c % 13 for c in cards if c // 13 == s)
        if rs:
            out[SUITS[s]] = "".join(RANKS[r] for r in rs)
    return out or "none"


if __name__ == "__main__":
    res = [solve(0), solve(-1)]
    (OUT / "onecard.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

"""M8 follow-up: does ONE history-free hand-strength index (with seat/history-dependent thresholds)
describe the equilibrium bids?  Re-solves the hook equilibrium on half A (deterministic, as m8_eq.py),
then for candidate static indices (seat-0 E[t | class, b0 = b] for b = 0,1,2; lexicographic combos)
reports the mass fraction of each seat's bids explained by a monotone rule in that index
(weighted longest non-decreasing subsequence per history, summed).  Also lists seat-0 / dealer
equilibrium bids by hand type.  Output: results/m8_index.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m8_eq as m8  # noqa: E402


def consistency(g, S, k, idx_sort):
    prof = g.bids(S)
    hist = np.zeros(g.N, np.int64)
    for j in range(k):
        hist = hist * 3 + prof[:, j]
    nh = 3 ** k
    key = g.cls[:, k] * nh + hist
    cnt = np.bincount(key, minlength=g.K * nh)
    Sk = S[k].reshape(-1)
    num = den = 0.0
    for h in range(nh):
        rows = np.arange(g.K) * nh + h
        w = cnt[rows].astype(float)
        m = w > 0
        if w.sum() == 0:
            continue
        num += m8.lnds_mass(idx_sort[m], Sk[rows][m], w[m])
        den += w[m].sum()
    return num / den


def best_threshold(order, ev, w):
    """Best monotone rule (bid 0 | 1 | 2 by two cut points along `order`) maximising sum w*ev; ev (k,3), -inf = illegal."""
    e = np.where(np.isfinite(ev), ev, -1e9) * w[:, None]
    e = e[order]
    C = np.vstack([np.zeros(3), np.cumsum(e, 0)])  # (K+1, 3)
    K = len(order)
    t1 = np.arange(K + 1)[:, None]
    t2 = np.arange(K + 1)[None, :]
    val = C[t1, 0] + (C[t2, 1] - C[t1, 1]) + (C[K, 2] - C[t2, 2])
    val = np.where(t2 >= t1, val, -np.inf)
    i, j = np.unravel_index(np.argmax(val), val.shape)
    bids = np.zeros(K, np.int64)
    bids[i:j] = 1
    bids[j:] = 2
    out = np.zeros(K, np.int64)
    out[order] = bids
    return out


def rule_values(gA, gB, S, k, index):
    """Seat k: thresholds fit on A (ev tables of A), BR of A; both valued on B's ev tables (per deal)."""
    _, evA, cntA = gA.br(S, k)
    _, evB, cntB = gB.br(S, k)
    nh = 3 ** k
    Sk = S[k].reshape(-1)
    vbr = vth = 0.0
    for h in range(nh):
        rows = np.arange(gA.K) * nh + h
        wA, wB = cntA[rows].astype(float), cntB[rows].astype(float)
        if wA.sum() == 0:
            continue
        th = best_threshold(np.argsort(index, kind="stable"), evA[rows], wA)
        eB = np.where(np.isfinite(evB[rows]), evB[rows], 0.0)
        ok = wB > 0
        vbr += (wB * eB[np.arange(gA.K), Sk[rows]])[ok].sum()
        vth += (wB * eB[np.arange(gA.K), th])[ok].sum()
    return vbr / gB.N, vth / gB.N


def main():
    Hs, tr, cls, inv = m8.load()
    K, N = len(inv), len(cls)
    A = np.arange(N) < N // 2
    g = m8.Game(tr[A], cls[A], K, True)
    S, _ = m8.solve(g, m8.init_profile(K, True))
    cnt0 = np.bincount(g.cls[:, 0], minlength=K)
    E = {}
    for b in range(3):
        bb = g.bids(S, (0, np.full(g.N, b), True))
        _, t = g.scores(bb)
        E[b] = np.bincount(g.cls[:, 0], weights=t[:, 0], minlength=K) / np.maximum(cnt0, 1)
    # rank-based lexicographic combos (ties broken by second index)
    def lex(a, b):
        return np.lexsort((b, a)).argsort().astype(float)
    idx = {"E[t|b=0]": E[0], "E[t|b=1]": E[1], "E[t|b=2]": E[2], "E[t|b=1]+E[t|b=2]": E[1] + E[2],
           "lex(E2,E1)": lex(E[2], E[1]), "lex(E1,E2)": lex(E[1], E[2])}
    out = {"consistency": {nm: {k: consistency(g, S, k, v) for k in range(4)} for nm, v in idx.items()}}
    gB = m8.Game(tr[~A], cls[~A], K, True)
    rv = {}
    for nm in ("E[t|b=0]", "E[t|b=1]", "E[t|b=2]"):
        rv[nm] = {}
        for k in range(4):
            vbr, vth = rule_values(g, gB, S, k, idx[nm])
            rv[nm][k] = {"BR_value_B": vbr, "threshold_value_B": vth, "loss": vbr - vth}
        print(nm, {k: round(v["loss"], 3) for k, v in rv[nm].items()}, flush=True)
    out["threshold_rule_value_on_B"] = rv
    # seat-0 strategy by hand type (eq) and the 2-dim picture
    s0 = {}
    for c in range(K):
        typ = inv[c][0]
        s0.setdefault(typ, {0: [], 1: [], 2: []})[int(S[0][c])].append(inv[c][1:])
    out["seat0_bids_by_type"] = {t: {b: sorted(v, reverse=True) for b, v in d.items()} for t, d in s0.items()}
    json.dump(out, open(m8.RES / "m8_index.json", "w"), indent=1, default=str)
    for nm, d in out["consistency"].items():
        print(nm, {k: round(v, 3) for k, v in d.items()})
    for t, d in out["seat0_bids_by_type"].items():
        print(t, {b: len(v) for b, v in d.items()}, "bid2:", d[2][:12], "bid1 lowest:", d[1][-6:])


if __name__ == "__main__":
    main()

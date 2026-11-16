"""M4: conservation (constrained-error) model of per-player hit rate vs Σbids − n.

Data: all-TA self-play, experiments/phase1/results/games_ta.npz (10,000 games x 25 rounds).
Model: latent per-player errors e_i = t_i - b_i, independent with pmf pi_c (c = bid class),
support e in [max(-b,-K), min(n-b,K)], conditioned on the conservation law
Σ e_i = D := n - Σ b_i.     P(e | D) = Π pi_c(e_i) / Z(D).
Hit rate prediction: P(e_i = 0 | D) = pi(0) * conv_{j!=i} pi_j (D) / Z(D).
Variants: (A) one symmetric pi; (B) one free pi; (C) pi by bid class {0,1,2,3+}.
Fit by maximum likelihood (L-BFGS on logits) over rounds grouped by (classes, bounds, D).
Output: math/results/m4_conservation.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
K = 4  # error support truncation |e| <= K (rows outside are dropped; counted)
NCLS = 4


def load(fname="games_ta.npz"):
    G = np.load(ROOT / "experiments/phase1/results" / fname)
    b = G["bids"].astype(int).reshape(-1, 4)
    t = G["tricks"].astype(int).reshape(-1, 4)
    lad = list(range(13, 0, -1)) + list(range(2, 14))
    n = np.tile(np.array(lad), G["bids"].shape[0])
    return b, t, n


def pmfs(theta, variant):
    if variant == "sym":
        h = theta[:K + 1]
        th = np.concatenate([h[1:][::-1], h])[None, :]  # e=-K..K symmetric
    else:
        th = theta.reshape(-1, 2 * K + 1)
    z = np.exp(th - th.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def conv(a, b):
    return np.convolve(a, b)


def masked(pi, lo, hi):
    m = np.zeros(2 * K + 1)
    m[lo + K:hi + K + 1] = pi[lo + K:hi + K + 1]
    return m  # index j <-> e = j-K


def group_rows(b, t, n):
    e = t - b
    keep = (np.abs(e) <= K).all(1)
    groups = Counter()
    obs = Counter()  # (cls, lo, hi, e) counts for numerator
    for bi, ei, ni in zip(b[keep], e[keep], n[keep]):
        pl = tuple(sorted((min(x, NCLS - 1), max(-x, -K), min(ni - x, K)) for x in bi))
        D = int(ni - bi.sum())
        groups[(pl, D)] += 1
        for x, y in zip(bi, ei):
            obs[(min(x, NCLS - 1), y)] += 1
    return groups, obs, int((~keep).sum())


def nll(theta, variant, groups, obs):
    P = pmfs(theta, variant)
    cls_pi = (lambda c: P[0]) if P.shape[0] == 1 else (lambda c: P[c])
    ll = 0.0
    for (c, e), k in obs.items():
        ll += k * np.log(cls_pi(c)[e + K] + 1e-300)
    for (pl, D), k in groups.items():
        f = np.array([1.0])
        for c, lo, hi in pl:
            f = conv(f, masked(cls_pi(c), lo, hi))
        ll -= k * np.log(f[D + 4 * K] + 1e-300)
    return -ll


def predict_hits(theta, variant, groups):
    P = pmfs(theta, variant)
    cls_pi = (lambda c: P[0]) if P.shape[0] == 1 else (lambda c: P[c])
    num = Counter()
    den = Counter()
    for (pl, D), k in groups.items():
        ms = [masked(cls_pi(c), lo, hi) for c, lo, hi in pl]
        f = np.array([1.0])
        for m in ms:
            f = conv(f, m)
        Z = f[D + 4 * K]
        hs = 0.0
        for i in range(4):
            g = np.array([1.0])
            for j in range(4):
                if j != i:
                    g = conv(g, ms[j])
            # e_i = 0 -> others sum to D ; g index for sum s is s + 3K
            idx = D + 3 * K
            hs += ms[i][K] * (g[idx] if 0 <= idx < len(g) else 0.0) / Z
        key = int(max(min(-D, 4), -5))  # noqa  # Σb - n = -D, pooled at edges like E4
        num[key] += k * hs
        den[key] += 4 * k
    return {k: num[k] / den[k] for k in sorted(den)}


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "games_ta.npz"
    tag = "" if fname == "games_ta.npz" else "_" + fname.split(".")[0].replace("games_", "")
    b, t, n = load(fname)
    groups, obs, dropped = group_rows(b, t, n)
    # empirical hit rate by Σb-n (same pooling)
    sbn = b.sum(1) - n
    hits = (b == t)
    emp = {}
    for k in range(-5, 5):
        m = (sbn == k) if k > -5 else (sbn <= -5)
        if k == 4:
            m = sbn >= 4
        if m.sum():
            p = hits[m].mean()
            emp[k] = [float(p), float(1.96 * np.sqrt(p * (1 - p) / (4 * m.sum()))), int(m.sum())]
    dl = (np.arange(len(b)) % 25) % 4
    isd = np.zeros_like(hits)
    isd[np.arange(len(b)), dl] = True
    split = {}
    for k in range(-4, 5):
        m = sbn == k
        if m.sum() < 10:
            continue
        nd = hits[m][~isd[m]]
        dd = hits[m][isd[m]]
        f = lambda x: [float(x.mean()), float(1.96 * np.sqrt(x.mean() * (1 - x.mean()) / len(x))), int(len(x))]  # noqa: E731
        split[k] = {"non_dealer": f(nd), "dealer": f(dd), "P(dealer bid is hook-forced off model argmax)": None}
    out_split = split
    out = {"split_by_seat": out_split, "rows_total": int(len(b)), "rows_dropped_|e|>K": dropped, "groups": len(groups), "empirical": emp,
           "variants": {}}
    for variant, dim in (("sym", K + 1), ("one", 2 * K + 1), ("class", NCLS * (2 * K + 1))):
        x0 = np.zeros(dim)
        if variant == "class":
            x0 = np.tile(-np.abs(np.arange(-K, K + 1)) * 1.0, NCLS)
        else:
            x0 = -np.abs(np.arange(-K, K + 1)) * 1.0 if variant == "one" else -np.arange(K + 1) * 1.0
        r = minimize(nll, x0, args=(variant, groups, obs), method="L-BFGS-B", options={"maxiter": 300})
        pred = predict_hits(r.x, variant, groups)
        P = pmfs(r.x, variant)
        out["variants"][variant] = {"nll": float(r.fun), "k_params": dim, "pi": P.round(4).tolist(),
                                    "pred_hit_by_sum_minus_n": pred, "converged": bool(r.success)}
        print(variant, round(r.fun, 1), {k: round(v, 3) for k, v in pred.items()}, flush=True)
    print("emp", {k: round(v[0], 3) for k, v in emp.items()})
    json.dump(out, open(ROOT / f"math/results/m4_conservation{tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

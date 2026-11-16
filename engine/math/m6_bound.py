"""M6: upper side of L3'.  For T ~ PB(p), b* = argmax (H+b)P(b) satisfies b* < mu + 1,
with the sharp constant  sup (b* - mu) = max_{m<=N-1} (m+1)/(1 + c_m (m+1)),  c_m = (H+m)/(H+m+1).

Numerical checks (results/m6_bound.json):
 (a) ratio lemma R: mu <= m  =>  P(m+1) <= m/(m+1) P(m)   (random + hill-climb maximising the ratio)
 (b) b* - mu never exceeds the sharp constant s(N) (random + hill-climb maximising b* - mu)
 (c) sharpness: Bin(m+1, a) with a just above a*_m = c(m+1)/(1+c(m+1)) attains s within 1e-6.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb import H, best_bid, pb_pmf  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def c_m(m, H=H):
    return (H + m) / (H + m + 1)


def sharp_gap(N, H=H):
    """sup over PB on N trials of b* - mu (not attained)."""
    return max((m + 1) / (1 + c_m(m, H) * (m + 1)) for m in range(N))


def crit_a(m, H=H):
    c = c_m(m, H)
    return c * (m + 1) / (1 + c * (m + 1))


def gap(p, H=H):
    P = pb_pmf(p)
    return best_bid(P, H=H) - float(np.sum(p))


def ratio_violation(p):
    """max over integers m >= mu of P(m+1) - m/(m+1) P(m)  (<= 0 under lemma R)."""
    P = pb_pmf(p)
    mu = float(np.sum(p))
    worst = -np.inf
    for m in range(int(np.ceil(mu - 1e-12)), len(P) - 1):
        worst = max(worst, P[m + 1] - m / (m + 1) * P[m])
    return worst


def rand_p(rng, N):
    k = rng.integers(0, 4)
    if k == 0:
        return rng.random(N)
    if k == 1:
        return rng.beta(0.3, 0.3, N)
    if k == 2:
        return np.clip(rng.choice([0.0, 1.0, rng.random()], N) + rng.normal(0, .05, N), 0, 1)
    return rng.random(N) ** rng.uniform(.1, 3)


def hill(f, p0, rng, steps=1500, s0=0.1):
    p, best = p0.copy(), f(p0)
    for t in range(steps):
        q = np.clip(p + rng.normal(0, s0 * (1 - t / steps) + 1e-4, len(p)), 0, 1)
        v = f(q)
        if v > best:
            p, best = q, v
    return p, best


def main():
    rng = np.random.default_rng(606)
    out = {"H": H}
    rows = []
    for N in range(1, 14):
        s = sharp_gap(N)
        g_rand, r_rand = -np.inf, -np.inf
        for _ in range(3000):
            p = rand_p(rng, N)
            g_rand = max(g_rand, gap(p))
            r_rand = max(r_rand, ratio_violation(p))
        g_hc, r_hc = -np.inf, -np.inf
        for _ in range(20):
            _, v = hill(gap, rand_p(rng, N), rng, 800)
            g_hc = max(g_hc, v)
            _, v = hill(ratio_violation, rand_p(rng, N), rng, 400)
            r_hc = max(r_hc, v)
        # sharpness: Bin(m+1, a*+eps), m = N-1
        m = N - 1
        a = crit_a(m) + 1e-9
        g_bin = gap(np.full(m + 1, a))
        rows.append({"N": N, "sharp_sup": s, "argmax_m": int(np.argmax([(k + 1) / (1 + c_m(k) * (k + 1)) for k in range(N)])),
                     "a_star": crit_a(m), "gap_bin_at_a*+": g_bin,
                     "max_gap_random": g_rand, "max_gap_hillclimb": g_hc,
                     "max_ratio_viol_random": r_rand, "max_ratio_viol_hillclimb": r_hc,
                     "n_vectors": 3000 + 40})
        print(N, round(s, 4), round(g_bin, 6), round(g_rand, 4), round(g_hc, 4), f"{max(r_rand, r_hc):.2e}", flush=True)
    out["rows"] = rows
    json.dump(out, open(OUT / "m6_bound.json", "w"), indent=1)


if __name__ == "__main__":
    main()

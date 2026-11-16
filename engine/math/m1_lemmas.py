"""Numerical verification of L1-L3 on random and adversarially searched
Poisson-binomial vectors.  Output: math/results/m1_lemmas.json

Checks, for PB(p) with p in [0,1]^k:
  L1  pmf log-concave  (and (10+b)P(b) log-concave)
  L2  b* in {m_hi, m_hi+1}, m_hi = largest mode; b* = m_hi+1 iff P(m+1)/P(m) > (10+m)/(11+m)
  L3  floor(mu) <= b* <= ceil(mu)+1  (Darroch + L2)
  L3' (conjecture) b* <= floor(mu)+1 ;  and b* >= ceil(mu) - ?  (record min of b*-mu, max of b*-mu)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb import best_bid, ev_vec, is_log_concave, modes, pb_pmf  # noqa: E402


def check(p, H=10):
    P = pb_pmf(p)
    mu = float(np.sum(p))
    b = best_bid(P, H=H)
    ms = modes(P, 1e-13)
    m = ms[-1]
    rule = m + 1 if (m + 1 < len(P) and P[m + 1] * (H + m + 1) > P[m] * (H + m) + 1e-13) else m
    return {
        "lc": is_log_concave(P, 1e-13) and is_log_concave(ev_vec(P, H), 1e-12),
        "L2": b in (m, m + 1) and b >= ms[0], "L2rule": b == rule,
        "L3": np.floor(mu) - 1e-9 <= b <= np.ceil(mu) + 1 + 1e-9,
        "L3c": b <= np.floor(mu + 1e-12) + 1, "excess": b - mu, "b": b, "mu": mu,
    }


def main():
    rng = np.random.default_rng(2026)
    agg = {"lc": 0, "L2": 0, "L2rule": 0, "L3": 0, "L3c": 0}
    N = 0
    exc = []
    worst = None
    gens = []
    for k in range(1, 27):
        for _ in range(1500):
            mode = rng.integers(4)
            if mode == 0:
                p = rng.random(k)
            elif mode == 1:
                p = rng.beta(0.3, 0.3, k)
            elif mode == 2:
                p = np.clip(rng.normal(rng.random(), 0.05, k), 0, 1)
            else:
                p = rng.beta(0.5, 3, k)
            gens.append(p)
    for p in gens:
        r = check(p)
        N += 1
        for key in agg:
            agg[key] += bool(r[key])
        exc.append(r["excess"])
    # adversarial: maximise b* - mu (try to break L3c) by random-restart hill climbing
    best_ex = -9
    for restart in range(150):
        k = int(rng.integers(2, 16))
        p = rng.random(k)
        cur = check(p)["excess"]
        for it in range(300):
            q = np.clip(p + rng.normal(0, 0.05, k), 0, 1)
            e = check(q)["excess"]
            if e >= cur:
                p, cur = q, e
        r = check(p)
        N += 1
        for key in agg:
            agg[key] += bool(r[key])
        if cur > best_ex:
            best_ex, worst = cur, p.tolist()
    exc = np.array(exc)
    out = {"vectors": N, "pass_counts": agg, "excess_b_minus_mu_random": [float(exc.min()), float(exc.max())],
           "adversarial_max_excess": float(best_ex), "adversarial_p": worst,
           "note": "excess = b* - mu; L3c conjecture b* <= floor(mu)+1"}
    json.dump(out, open(Path(__file__).resolve().parent / "results" / "m1_lemmas.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "adversarial_p"}))


if __name__ == "__main__":
    main()

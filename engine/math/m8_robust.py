"""M8 robustness: iterate best responses from different starting profiles (all-0, all-1, all-2, random)
and on the other half of the deals; compare the resulting equilibria (bid disagreement mass on B and
held-out seat values).  Output: results/m8_robust.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m8_eq as m8  # noqa: E402


def legalize(S):
    for b0 in range(3):
        for b1 in range(3):
            for b2 in range(3):
                f = 2 - b0 - b1 - b2
                bad = S[3][:, b0, b1, b2] == f
                S[3][bad, b0, b1, b2] = 1 if f != 1 else 0
    return S


def main():
    Hs, tr, cls, inv = m8.load()
    K, N = len(inv), len(cls)
    A = np.arange(N) < N // 2
    gA, gB = m8.Game(tr[A], cls[A], K, True), m8.Game(tr[~A], cls[~A], K, True)
    rng = np.random.default_rng(88)
    starts = {}
    for v in (0, 1, 2):
        starts[f"all{v}"] = legalize([np.full(s.shape, v, np.int64) for s in m8.init_profile(K, True)])
    starts["random"] = legalize([rng.integers(0, 3, s.shape) for s in m8.init_profile(K, True)])
    res, prof = {}, {}
    for nm, S0 in starts.items():
        S, hist = m8.solve(gA, [s.copy() for s in S0])
        prof[nm] = S
        sc, b, _ = gB.values(S)
        res[nm] = {"sweeps": hist[-3:], "values_B": [m8.mci(sc[:, k])[:2] for k in range(4)]}
    # swap halves: BR on B, values on A
    S, hist = m8.solve(gB, m8.init_profile(K, True))
    sc, _, _ = gA.values(S)
    res["BR_on_B_values_on_A"] = {"sweeps": hist[-3:], "values_A": [m8.mci(sc[:, k])[:2] for k in range(4)]}
    bref = gB.bids(prof["all0"])
    for nm in prof:
        res[nm]["bid_disagree_vs_all0_B"] = [float((gB.bids(prof[nm])[:, k] != bref[:, k]).mean()) for k in range(4)]
    json.dump(res, open(m8.RES / "m8_robust.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()

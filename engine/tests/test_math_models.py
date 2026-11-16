"""Tests for M3/M4 helper models (math/m3_decay.py, math/m4_conservation.py)."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "math"))
import m4_conservation as m4  # noqa: E402
from m3_decay import gauss_gap  # noqa: E402


def test_conservation_hit_formula_matches_bruteforce():
    K = m4.K
    pi = np.array([0, 0, 0.05, 0.2, 0.5, 0.15, 0.07, 0.03, 0])
    theta = np.log(np.maximum(pi, 1e-300))
    pl = tuple([(1, -K, K)] * 4)
    for D in (-2, -1, 1, 2):
        groups = Counter({(pl, D): 1})
        pred = m4.predict_hits(theta, "one", groups)[-D]
        # brute force over all error vectors
        num = den = 0.0
        rng = range(-K, K + 1)
        for a in rng:
            for b in rng:
                for c in rng:
                    d = D - a - b - c
                    if -K <= d <= K:
                        w = pi[a + K] * pi[b + K] * pi[c + K] * pi[d + K]
                        den += w
                        num += w * ((a == 0) + (b == 0) + (c == 0) + (d == 0)) / 4
        assert abs(pred - num / den) < 1e-9


def test_symmetric_pi_gives_symmetric_hits():
    K = m4.K
    th = -np.arange(K + 1) * 1.3
    pl = tuple([(1, -K, K)] * 4)
    h = m4.predict_hits(th, "sym", Counter({(pl, 1): 1, (pl, -1): 1}))
    assert abs(h[1] - h[-1]) < 1e-12


def test_gauss_gap_decreases_with_spread():
    g = [gauss_gap(s, 2) for s in (0.3, 0.6, 1.0, 1.5)]
    assert all(x > y for x, y in zip(g, g[1:]))

"""M6: upper side of L3' (b* < mu + 1 for Poisson-binomial) and its sharp constant."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "math"))

from m6_bound import c_m, crit_a, gap, ratio_violation, sharp_gap  # noqa: E402
from pb import best_bid, pb_pmf  # noqa: E402


def test_ratio_lemma_random():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        N = rng.integers(1, 14)
        p = rng.random(N) ** rng.uniform(0.2, 3)
        assert ratio_violation(p) <= 1e-12


def test_bstar_below_mu_plus_sharp_constant():
    rng = np.random.default_rng(2)
    for _ in range(2000):
        N = int(rng.integers(1, 14))
        p = rng.beta(0.4, 0.4, N)
        assert gap(p) < sharp_gap(N) + 1e-12 < 1


def test_sharpness_binomial():
    for m in range(13):
        a = crit_a(m)
        lo, hi = np.full(m + 1, a - 1e-7), np.full(m + 1, a + 1e-7)
        assert best_bid(pb_pmf(lo)) == m and best_bid(pb_pmf(hi)) == m + 1
        s = (m + 1) / (1 + c_m(m) * (m + 1))
        assert abs(gap(hi) - s) < 1e-5


def test_integer_mean_bid_equals_mean():
    rng = np.random.default_rng(3)
    for _ in range(500):
        N = int(rng.integers(2, 14))
        p = rng.random(N)
        k = int(rng.integers(1, N))
        p = np.clip(p * k / p.sum(), 0, 1)
        if abs(p.sum() - k) > 1e-9:
            continue
        assert best_bid(pb_pmf(p)) == k

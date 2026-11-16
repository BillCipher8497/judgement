"""Tests for the math layer's Poisson-binomial lemmas (math/pb.py, math/NOTES)."""
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "math"))
from pb import best_bid, ev_vec, is_log_concave, kl, modes, pb_pmf, tv  # noqa: E402


def _rand_ps(rng, k):
    m = rng.integers(3)
    if m == 0:
        return rng.random(k)
    if m == 1:
        return rng.beta(0.3, 0.3, k)
    return rng.beta(0.5, 3, k)


def test_pb_pmf_matches_binomial():
    P = pb_pmf([0.3] * 7)
    ref = [comb(7, j) * 0.3 ** j * 0.7 ** (7 - j) for j in range(8)]
    assert np.allclose(P, ref)
    assert abs(P.sum() - 1) < 1e-12


def test_L1_log_concave_and_unique_bid():
    rng = np.random.default_rng(1)
    for _ in range(400):
        p = _rand_ps(rng, int(rng.integers(1, 14)))
        P = pb_pmf(p)
        assert is_log_concave(P, 1e-13)
        ev = ev_vec(P)
        assert is_log_concave(ev, 1e-12)
        top = np.flatnonzero(ev >= ev.max() - 1e-12)
        assert top[-1] - top[0] <= 1  # at most two adjacent maximisers (ties)


def test_L2_mode_rule_and_ratio():
    rng = np.random.default_rng(2)
    for _ in range(400):
        p = _rand_ps(rng, int(rng.integers(1, 14)))
        P = pb_pmf(p)
        b = best_bid(P)
        ms = modes(P, 1e-13)
        m = ms[-1]
        assert ms[0] <= b <= m + 1
        up = m + 1 < len(P) and P[m + 1] * (11 + m) > P[m] * (10 + m) + 1e-13
        assert b == (m + 1 if up else m)


def test_L2_lower_bound_any_distribution():
    rng = np.random.default_rng(3)
    for _ in range(300):
        P = rng.dirichlet(np.ones(int(rng.integers(2, 10))) * 0.5)
        assert best_bid(P) >= modes(P)[0]


def test_L3_window_and_conjecture():
    rng = np.random.default_rng(4)
    for _ in range(400):
        p = _rand_ps(rng, int(rng.integers(1, 14)))
        P = pb_pmf(p)
        mu = p.sum()
        b = best_bid(P)
        assert np.floor(mu) - 1e-9 <= b <= np.ceil(mu) + 1 + 1e-9  # proved
        assert abs(b - mu) < 1 + 1e-9  # L3' (verified numerically, proved for binomial)


def test_dispersion_identity_plugin():
    rng = np.random.default_rng(5)
    X = (rng.random((500, 6)) < rng.random(6)).astype(float)
    X[:, 1] = X[:, 0] * (rng.random(500) < 0.7)  # induce dependence
    T = X.sum(1)
    p = X.mean(0)
    cov = np.cov(X.T, bias=True)
    lhs = T.var() - (p * (1 - p)).sum()
    rhs = 2 * np.triu(cov, 1).sum()
    assert abs(lhs - rhs) < 1e-10


def test_bimodal_is_not_pb():
    # any strict interior dip violates log-concavity => cannot be PB
    assert not is_log_concave([0.4, 0.1, 0.5])


def test_tv_kl_basic():
    P = pb_pmf([0.2, 0.9])
    assert tv(P, P) == 0 and abs(kl(P, P)) < 1e-12

"""Round 3 (H2): lead-chain models in math/m10_analyze.py."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "math"))

import m10_analyze as m10  # noqa: E402
from m5_analyze import hist, pb_pmf  # noqa: E402


def test_chain_independent_equals_pb():
    p = [0.2, 0.7, 0.5, 0.9]
    P = m10.chain_pmf(lambda j, s, t: p[j], 4)
    assert np.allclose(P, pb_pmf(p))


def test_lct_kappa0_reproduces_empirical():
    rng = np.random.default_rng(1)
    n, R = 5, 400
    W = np.zeros((R, n), int)
    for r in range(R):           # lead-dependent process
        lead = 1
        for j in range(n):
            W[r, j] = rng.random() < (0.7 if lead else 0.3)
            lead = W[r, j]
    X = W.astype(float)          # card j won trick j (card order = trick order)
    suit = np.array([0, 0, 1, 2, 3])
    out = m10.fit_models(X, W, suit, n, kappa=0.0)
    emp = hist(W.sum(1), n)
    assert np.allclose(out["LCT"], emp, atol=1e-9)
    # LC1 with kappa 0 is the exact law of this first-order chain; close to empirical
    assert 0.5 * np.abs(out["LC1"] - emp).sum() < 0.06
    for k in m10.MODELS:
        assert abs(out[k].sum() - 1) < 1e-9

"""Poisson-binomial utilities and bid rules for the math layer (M1/M3)."""
from __future__ import annotations

import numpy as np

H = 10  # hit base: score(b, t) = (H + b) * 1{t == b}


def pb_pmf(p) -> np.ndarray:
    """PMF of a sum of independent Bernoulli(p_i), length len(p)+1 (exact DP)."""
    f = np.array([1.0])
    for q in np.asarray(p, float):
        g = np.zeros(len(f) + 1)
        g[:-1] += f * (1 - q)
        g[1:] += f * q
        f = g
    return f


def ev_vec(P, H=H) -> np.ndarray:
    """EV(b) = (H + b) P(b) for every b."""
    P = np.asarray(P, float)
    return (H + np.arange(len(P))) * P


def best_bid(P, legal=None, H=H) -> int:
    """argmax_b (H+b)P(b) over legal bids, ties -> lower bid (engine convention)."""
    ev = ev_vec(P, H)
    bs = range(len(P)) if legal is None else legal
    return max(bs, key=lambda b: (ev[b] - 1e-12 * b, -b))


def modes(P, tol=1e-12):
    P = np.asarray(P, float)
    return [int(k) for k in np.flatnonzero(P >= P.max() - tol)]


def is_log_concave(P, tol=1e-12) -> bool:
    P = np.asarray(P, float)
    nz = np.flatnonzero(P > tol)
    if len(nz) == 0:
        return True
    if nz[-1] - nz[0] + 1 != len(nz):
        return False  # internal zero
    Q = P[nz[0]:nz[-1] + 1]
    return bool(np.all(Q[1:-1] ** 2 >= Q[:-2] * Q[2:] - tol))


def tv(P, Q) -> float:
    return 0.5 * float(np.abs(np.asarray(P) - np.asarray(Q)).sum())


def kl(P, Q, eps=1e-9) -> float:
    """KL(P || Q) with Q floored at eps (P = empirical)."""
    P = np.asarray(P, float)
    Q = np.maximum(np.asarray(Q, float), eps)
    m = P > 0
    return float((P[m] * np.log(P[m] / Q[m])).sum())


def mean_var(P):
    P = np.asarray(P, float)
    t = np.arange(len(P))
    m = float((P * t).sum())
    return m, float((P * (t - m) ** 2).sum())

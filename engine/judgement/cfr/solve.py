"""Vectorised tabular CFR+ and exact best response / NashConv on a toy Tree.

A profile ``sigma`` is an (n_is x A) array of action probabilities (rows of each
infoset sum to 1 over valid actions; one array holds every player's strategy).
"""
from __future__ import annotations

import numpy as np


def reaches(t, sigma):
    """reach[p, node] = product of player p's action probs on the path (chance excluded)."""
    N = t.g.N
    reach = np.ones((N, t.n_nodes))
    sflat = sigma.ravel()
    for a, b, _, _ in t.chunks:
        par, ch, pl = t.e_par[a:b], t.e_ch[a:b], t.e_pl[a:b]
        reach[:, ch] = reach[:, par]
        reach[pl, ch] *= sflat[t.e_ia[a:b]]
    return reach


def cf_reach(t, reach):
    """cf[p, node] = chance * product of the other players' reaches."""
    N = t.g.N
    cf = np.empty_like(reach)
    for p in range(N):
        x = t.node_c.copy()
        for q in range(N):
            if q != p:
                x *= reach[q]
        cf[p] = x
    return cf


def values(t, sigma):
    """V[node, p] = expected utility of p from node under sigma."""
    V = t.U.copy()
    sflat = sigma.ravel()
    for a, b, seg, upar in reversed(t.chunks):
        contrib = V[t.e_ch[a:b]] * sflat[t.e_ia[a:b]][:, None]
        V[upar] = np.add.reduceat(contrib, seg, axis=0)
    return V


def profile_value(t, sigma, V=None):
    if V is None:
        V = values(t, sigma)
    return t.node_c[t.roots] @ V[t.roots]


def best_response(t, sigma, p, free=None, cf=None, return_policy=False):
    """Exact best-response value of player p against sigma_{-p}.  ``free``: bool mask over
    infosets where p may deviate (default: all of p's); elsewhere p plays sigma."""
    if cf is None:
        cf = cf_reach(t, reaches(t, sigma))[p]
    A = t.g.A
    is_free = (t.is_pl == p) if free is None else (free & (t.is_pl == p))
    v = t.U[:, p].copy()
    sflat = sigma.ravel()
    agg = np.zeros(t.n_is * A)
    best = np.zeros(t.n_is, np.int64)
    for a, b, seg, upar in reversed(t.chunks):
        ch, ia, isx = t.e_ch[a:b], t.e_ia[a:b], t.e_is[a:b]
        q = v[ch]
        fe = is_free[isx]
        if fe.any():
            np.add.at(agg, ia[fe], cf[t.e_par[a:b][fe]] * q[fe])
            fis = np.unique(isx[fe])
            m = np.where(t.is_valid[fis], agg.reshape(-1, A)[fis], -np.inf)
            best[fis] = m.argmax(1)
            w = np.where(fe, (t.e_a[a:b] == best[isx]).astype(float), sflat[ia])
        else:
            w = sflat[ia]
        v[upar] = np.add.reduceat(q * w, seg)
    val = float(t.node_c[t.roots] @ v[t.roots])
    if not return_policy:
        return val
    pol = sigma.copy()
    fi = np.flatnonzero(is_free)
    pol[fi] = 0.0
    pol[fi, best[fi]] = 1.0
    return val, pol


def nash_conv(t, sigma, free=None):
    """Returns (NashConv, per-player gains, per-player values)."""
    vals = profile_value(t, sigma)
    cf = cf_reach(t, reaches(t, sigma))
    br = np.array([best_response(t, sigma, p, free=free, cf=cf[p]) for p in range(t.g.N)])
    gains = br - vals
    return float(gains.sum()), gains, vals


class CFRPlus:
    """Tabular CFR+ (regret floor at 0, linear average weights), simultaneous updates."""

    def __init__(self, t):
        self.t = t
        self.R = np.zeros(t.is_valid.shape)
        self.S = np.zeros(t.is_valid.shape)
        self.T = 0
        self.sigma = t.uniform()

    def current(self):
        R = self.R * self.t.is_valid
        z = R.sum(1, keepdims=True)
        u = self.t.uniform()
        return np.where(z > 0, R / np.where(z > 0, z, 1), u)

    def average(self):
        z = self.S.sum(1, keepdims=True)
        return np.where(z > 0, self.S / np.where(z > 0, z, 1), self.t.uniform())

    def iterate(self, k=1):
        t = self.t
        A = t.g.A
        for _ in range(k):
            self.T += 1
            sig = self.current()
            reach = reaches(t, sig)
            cf = cf_reach(t, reach)
            V = values(t, sig)
            p = t.e_pl
            w = cf[p, t.e_par] * (V[t.e_ch, p] - V[t.e_par, p])
            inst = np.bincount(t.e_ia, weights=w, minlength=t.n_is * A).reshape(-1, A)
            self.R = np.maximum(self.R + inst, 0.0) * t.is_valid
            own = reach[t.is_pl, t.is_rep]
            self.S += self.T * own[:, None] * sig
        self.sigma = self.current()
        return self

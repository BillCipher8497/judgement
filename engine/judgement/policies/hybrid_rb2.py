"""Hybrid-RB2: Hybrid-RB with a *continuous* per-opponent bid model added to RB's discrete bidder types.

Continuous family (per opponent, a grid over (slope a, offset d, temperature T)):
    centre  c(H) = a * mu(H) + d,      mu(H) = sum_t t P_model(t | H, b = m(H))   (model's expected tricks at its
                                       own argmax bid m(H) over 0..n, i.e. ignoring the hook)  [centre='mu']
                                       or mu(H) = m(H)                                         [centre='m']
    P(b | H) = (1 - eps) softmax_{b' in legal}( -|b' - c(H)| / T )[b] + eps / |legal|
The full per-opponent mixture is {grid components} u {RB types: model, noisy.1/.25/.5, honour} u {unknown};
the prior puts ``u`` on unknown, ``cont_mass`` spread uniformly over the grid and the rest evenly on the RB types.
After every round the opponents' bid-time hands are public, so each component's log-weight is updated by exact
Bayes (log P_c(b_j | H_j)); within a round the PIMC worlds are weighted by the posterior-predictive mixture and
the RB tempering guard (ESS >= ess_min) is kept.  Everything else (play path, sampler) is Hybrid-RB's.
"""
from __future__ import annotations

import numpy as np

from ..bidding.model import score_matrix
from ..bidding2.features_vec import hand_features_batch
from .hybrid_bi import BidInformedWorlds, _bool
from .hybrid_rb import NOISE, BidderTypes, HybridRBPolicy, RobustWorlds

SLOPES = (0.8, 0.9, 1.0, 1.1)
OFFSETS = tuple(np.round(np.arange(-2.0, 2.01, 0.25), 2))
TEMPS = (0.15, 0.3, 0.5, 0.8, 1.2, 2.0)
RB_TYPES = ("model",) + tuple(f"noisy{p}" for p in NOISE) + ("honour",)


def model_ev_mu(lik, H, n, trump, pos, prior):
    """-> (ev (M, n+1), mu (M,) expected tricks at the unconstrained model argmax bid, m (M,) that bid)."""
    S = lik._S.get(n)
    if S is None:
        S = lik._S[n] = score_matrix(lik.rules, n)
    cl = lik.model.get(n, trump)
    X = hand_features_batch(H, trump, n, pos, prior, lik.rules.n_players)
    base = ((X - cl.mu) / cl.sd) @ cl.W.T
    z = base[:, None, :] + cl._gmat()[None, :, :]
    z -= z.max(2, keepdims=True)
    e = np.exp(z)
    P = e / e.sum(2, keepdims=True)                      # (M, b, t)
    ev = (P * S[None]).sum(2)
    m = ev.argmax(1)
    mu = (P[np.arange(len(m)), m] * np.arange(n + 1)[None]).sum(1)
    return ev, mu, m


class _EvShim:
    """Hands BidderTypes a precomputed EV matrix (so the model features are evaluated once)."""

    def __init__(self, ev):
        self.ev = ev

    def evs(self, *a):
        return self.ev


class ContTypes:
    """P_c(b | H) for every component c = (RB types..., grid..., unknown), vectorised over hands."""

    def __init__(self, lik, eps=0.02, noise=NOISE, slopes=SLOPES, offsets=OFFSETS, temps=TEMPS, centre="mu"):
        self.lik, self.eps, self.centre = lik, float(eps), centre
        self.rb = BidderTypes(lik, eps, noise)
        g = np.array([(a, d, t) for a in slopes for d in offsets for t in temps], float)
        self.grid = g                                     # (C, 3)
        self.names = RB_TYPES + tuple(f"c{a:g}/{d:+g}/{t:g}" for a, d, t in g) + ("unknown",)
        self.n_rb, self.n_grid = len(RB_TYPES), len(g)

    def probs(self, H, n, trump, pos, prior, b, legal) -> np.ndarray:
        ev, mu, m = model_ev_mu(self.lik, H, n, trump, pos, prior)
        P_rb = self._rb(ev, H, n, trump, pos, prior, b, legal)
        x = mu if self.centre == "mu" else m.astype(float)
        a, d, t = self.grid.T
        c = x[:, None] * a[None] + d[None]               # (M, C)
        L = np.asarray(sorted(legal), float)
        z = -np.abs(L[None, None, :] - c[:, :, None]) / t[None, :, None]   # (M, C, nl)
        z -= z.max(2, keepdims=True)
        ez = np.exp(z)
        ib = int(np.searchsorted(L, b))
        Pc = ez[:, :, ib] / ez.sum(2)
        nl = len(L)
        Pc = Pc * (1 - self.eps) + self.eps / nl
        return np.concatenate([P_rb[:, :-1], Pc, P_rb[:, -1:]], 1)

    def _rb(self, ev, H, n, trump, pos, prior, b, legal):
        self.rb.lik = _EvShim(ev)
        try:
            return self.rb.probs(H, n, trump, pos, prior, b, legal)
        finally:
            self.rb.lik = self.lik


def rb2_prior(types: ContTypes, u=0.3, cont_mass=0.3):
    rest = (1.0 - u - cont_mass) / types.n_rb
    return np.array([rest] * types.n_rb + [cont_mass / types.n_grid] * types.n_grid + [u])


class OpponentModel2:
    """Per-opponent log-posterior over all components; exact Bayes from revealed rounds."""

    def __init__(self, types: ContTypes, prior, n_players=4):
        self.types = types
        p = np.asarray(prior, float)
        self.log_prior = np.log(p / p.sum())
        self.lp = {s: self.log_prior.copy() for s in range(n_players)}

    def post(self, seat):
        x = np.exp(self.lp[seat] - self.lp[seat].max())
        return x / x.sum()

    def update(self, seat, H1, n, trump, pos, prior, b, legal):
        P = self.types.probs(H1[None, :], n, trump, pos, prior, b, legal)[0]
        self.lp[seat] = self.lp[seat] + np.log(P)
        self.lp[seat] -= self.lp[seat].max()

    def log_lik(self, seat, H, n, trump, pos, prior, b, legal):
        return np.log(self.types.probs(H, n, trump, pos, prior, b, legal) @ self.post(seat))

    def summary(self, seat) -> dict:
        """Family masses and grid-parameter posterior moments (conditional on the continuous family)."""
        T = self.types
        p = self.post(seat)
        out = {t: round(float(p[i]), 5) for i, t in enumerate(RB_TYPES)}
        pg = p[T.n_rb:T.n_rb + T.n_grid]
        out["cont"] = round(float(pg.sum()), 5)
        out["unknown"] = round(float(p[-1]), 5)
        q = pg / max(pg.sum(), 1e-300)
        for k, nm in enumerate(("a", "d", "t")):
            v = T.grid[:, k]
            mean = float(q @ v)
            out[nm] = round(mean, 4)
            out[nm + "_sd"] = round(float(np.sqrt(max(0.0, q @ (v - mean) ** 2))), 4)
        j = int(pg.argmax())
        out["map"] = [float(x) for x in T.grid[j]]
        return out


class HybridRB2Policy(HybridRBPolicy):
    """Hybrid-RB with the continuous component grid (see module doc).  post_log entries are summary dicts."""

    def __init__(self, lik, n_samples=16, pimc_cards=5, base=None, eps=0.02, noise=NOISE, u=0.3, cont_mass=0.3,
                 slopes=SLOPES, offsets=OFFSETS, temps=TEMPS, centre="mu", ess_min=4.0, m_max=512, learn=True,
                 name=None, validate=False):
        super().__init__(lik, n_samples=n_samples, pimc_cards=pimc_cards, base=base, eps=eps, noise=noise,
                         ess_min=ess_min, m_max=m_max, learn=learn, validate=validate,
                         name=name or f"hybrid_rb2_k{n_samples}_c{pimc_cards}")
        types = ContTypes(lik, eps, noise, slopes, offsets, temps, centre)
        self.opp = OpponentModel2(types, rb2_prior(types, u, cont_mass))
        self.worlds = RobustWorlds(self.opp, m_max=m_max, ess_min=ess_min)

    def observe(self, info):
        if info is self._prev:
            return
        prev, self._prev = self._prev, info
        if prev is None:
            return
        r = self.rounds_seen
        self.rounds_seen += 1
        if not self.learn:
            return
        played = {}
        for p, c in prev.history:
            played[p] = played.get(p, 0) | (1 << c)
        for p, pos, prior, b, legal in BidInformedWorlds.observations(prev):
            H1 = _bool([played.get(p, 0)])[0]
            if H1.sum() != prev.n_cards:
                continue
            self.opp.update(p, H1, prev.n_cards, prev.trump, pos, prior, b, legal)
            self.post_log.append((r, p, self.opp.summary(p)))


__all__ = ["HybridRB2Policy", "ContTypes", "OpponentModel2", "model_ev_mu", "rb2_prior", "SLOPES", "OFFSETS", "TEMPS"]

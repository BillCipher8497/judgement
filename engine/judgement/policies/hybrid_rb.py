"""Hybrid-RB: Hybrid-BI with *robust, online-learned* bid inference, plus an optional persistent particle
set per round (B2).

Bidder-type family (per opponent j, posterior pi_j over types t):
    model      ModelBidder argmax (M_TA)                         P = (1-eps)[b = m(H)] + eps/|legal|
    noisy.p    ModelBidder bid moved +-1 w.p. p (round4 NoisyBidder; illegal -> other sign -> keep), p in NOISE
    honour     GreedyPolicy honour count (aces, trump K/Q, trump length-3), nearest legal
    unknown    uniform over legal bids
After every round all opponents' bid-time hands are public (every card is played face up), so
    pi_j(t) <- pi_j(t) * P_t(b_j | H_j, situation_j)          (exact Bayes update, no hidden state).
Within a round the PIMC world weight is the posterior-predictive mixture
    w(h) = prod_j sum_t pi_j(t) P_t(b_j | h_j u played_j, situation_j).
The prior gives 'unknown' most of the mass, so early rounds behave close to plain Hybrid.
ESS guard: if SIR ends below ``ess_min`` (after ``m_max`` proposals) the log-weights are tempered by the
largest beta in [0, 1] that restores ESS >= ess_min (beta = 0 -> uniform worlds = plain Hybrid).

ParticleWorlds (B2): per round, a weighted pool of bid-time hands H_j.  At each decision particles that
contradict the plays since (a card played by j not in H_j, or a remaining card in a suit j has shown void)
are dropped - exact rejection, so survivors stay iid from the uniform-consistent proposal and keep their
(fixed within the round) bid weights; the pool is topped up with fresh proposals until ESS >= K or
``m_max`` new proposals; then the same tempering guard and systematic resampling to K worlds.
"""
from __future__ import annotations

import numpy as np

from ..cards import NT, SUIT_MASK
from ..sampling import HandSampler
from .hybrid_bi import BidInformedWorlds, HybridBIPolicy, _bool

NOISE = (0.1, 0.25, 0.5)
TYPES = ("model",) + tuple(f"noisy{p}" for p in NOISE) + ("honour", "unknown")
DEFAULT_PRIOR = (0.1, 0.1, 0.1, 0.1, 0.1, 0.5)


def honour_bids(H: np.ndarray, n: int, trump: int, legal) -> np.ndarray:
    """Vectorised GreedyPolicy.bid on (M, 52) bool hands."""
    A = H[:, 12::13].sum(1)                          # aces (rank bit 12 of each suit)
    est = A.astype(int)
    if 0 <= trump < 4:
        t = H[:, 13 * trump:13 * trump + 13]
        est = est + t[:, 11] + t[:, 10] + np.maximum(0, t.sum(1) - 3)
    est = np.minimum(est, n)
    L = np.array(sorted(legal))
    d = np.abs(est[:, None] - L[None, :])             # argmin takes the first (= lower) on ties
    return L[d.argmin(1)]


class BidderTypes:
    """P_t(b | H) for every type t, vectorised over hypothesised hands."""

    def __init__(self, lik, eps: float = 0.02, noise=NOISE):
        self.lik, self.eps, self.noise = lik, float(eps), tuple(noise)
        self.names = ("model",) + tuple(f"noisy{p}" for p in self.noise) + ("honour", "unknown")

    def probs(self, H, n, trump, pos, prior, b, legal) -> np.ndarray:
        """(M, T) probability of the observed bid b under each type."""
        M, nl = H.shape[0], len(legal)
        ev = self.lik.evs(H, n, trump, pos, prior)
        mask = np.full(n + 1, -np.inf)
        mask[list(legal)] = 0.0
        m = (ev + mask).argmax(1)                         # ModelBidder: ties -> lower bid
        lg = np.zeros(n + 3, bool)
        lg[np.array(legal) + 1] = True                    # offset 1 so m-1 = -1 indexes safely
        up_ok, dn_ok = lg[m + 2], lg[m]
        up = np.where(up_ok, m + 1, np.where(dn_ok, m - 1, m))
        dn = np.where(dn_ok, m - 1, np.where(up_ok, m + 1, m))
        hit = (m == b).astype(float)
        cols = [hit]
        for p in self.noise:
            cols.append((1 - p) * hit + 0.5 * p * (up == b) + 0.5 * p * (dn == b))
        cols.append((honour_bids(H, n, trump, legal) == b).astype(float))
        P = np.stack(cols, 1) * (1 - self.eps) + self.eps / nl
        return np.concatenate([P, np.full((M, 1), 1.0 / nl)], 1)


class OpponentModel:
    """Per-opponent log-posterior over bidder types; updated from revealed rounds."""

    def __init__(self, types: BidderTypes, prior=DEFAULT_PRIOR, n_players: int = 4):
        self.types = types
        p = np.asarray(prior, float)
        self.log_prior = np.log(p / p.sum())
        self.lp = {s: self.log_prior.copy() for s in range(n_players)}

    def post(self, seat) -> np.ndarray:
        x = np.exp(self.lp[seat] - self.lp[seat].max())
        return x / x.sum()

    def update(self, seat, H1, n, trump, pos, prior, b, legal):
        P = self.types.probs(H1[None, :], n, trump, pos, prior, b, legal)[0]
        self.lp[seat] = self.lp[seat] + np.log(P)
        self.lp[seat] -= self.lp[seat].max()

    def log_lik(self, seat, H, n, trump, pos, prior, b, legal) -> np.ndarray:
        return np.log(self.types.probs(H, n, trump, pos, prior, b, legal) @ self.post(seat))


def temper(lw: np.ndarray, ess_min: float):
    """-> (weights normalised, ess, beta).  beta = 1 unless ESS(lw) < ess_min."""
    def ess_of(beta):
        w = np.exp(beta * (lw - lw.max()))
        return float(w.sum() ** 2 / (w * w).sum()), w
    e, w = ess_of(1.0)
    beta = 1.0
    if e < ess_min and len(lw) > ess_min:
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if ess_of(mid)[0] >= ess_min:
                lo = mid
            else:
                hi = mid
        beta = lo
        e, w = ess_of(beta)
    return w / w.sum(), e, beta


def _systematic(w, K, rng):
    pos = (rng.random() + np.arange(K)) / K
    return np.minimum(np.searchsorted(np.cumsum(w), pos), len(w) - 1)


class RobustWorlds(BidInformedWorlds):
    """SIR worlds weighted by the learned per-opponent mixture likelihood (+ tempering guard)."""

    def __init__(self, opp: OpponentModel, batch=64, m_max=512, ess_target=1.0, ess_min=4.0):
        super().__init__(None, batch, m_max, ess_target)
        self.opp, self.ess_min = opp, float(ess_min)

    def _lw_H(self, info, Hs: dict, obs) -> np.ndarray:
        """log-weight of particles given bid-time hands Hs[seat] = (M,) int masks."""
        M = len(next(iter(Hs.values()))) if Hs else 0
        lw = np.zeros(M)
        for p, pos, prior, b, legal in obs:
            lw += self.opp.log_lik(p, _bool(Hs[p]), info.n_cards, info.trump, pos, prior, b, legal)
        return lw

    def log_weights(self, info, worlds, obs=None):
        obs = self.observations(info) if obs is None else obs
        played = {}
        for p, c in info.history:
            played[p] = played.get(p, 0) | (1 << c)
        Hs = {p: [w[p] | played.get(p, 0) for w in worlds] for p, *_ in obs}
        return self._lw_H(info, Hs, obs)

    def sample(self, info, K, rng):
        hs = HandSampler(info)
        obs = self.observations(info)
        worlds, lws = [], []
        while True:
            new = [hs.sample(rng) for _ in range(self.batch)]
            worlds += new
            lws.append(self.log_weights(info, new, obs))
            lw = np.concatenate(lws)
            w = np.exp(lw - lw.max())
            ess = float(w.sum() ** 2 / (w * w).sum())
            if ess >= self.ess_target * K or len(worlds) >= self.m_max:
                break
        w, ess2, beta = temper(lw, self.ess_min)
        idx = _systematic(w, K, rng)
        st = {"ess": ess2, "ess_raw": ess, "beta": beta, "M": len(worlds)}
        return [worlds[i] for i in idx], st


class ParticleWorlds(RobustWorlds):
    """B2: persistent per-round particle pool of bid-time hands (exact rejection filter + top-up)."""

    def __init__(self, opp, batch=64, m_max=512, ess_target=1.0, ess_min=4.0, pool_max=4096):
        super().__init__(opp, batch, m_max, ess_target, ess_min)
        self.pool_max = pool_max
        self._key = None

    def sample(self, info, K, rng):
        obs = self.observations(info)
        opps = [p for p in range(info.n_players) if p != info.seat]
        played = {p: 0 for p in opps}
        for p, c in info.history:
            if p in played:
                played[p] |= 1 << c
        kept = 0
        if self._key is not info:                 # new round (new InfoState object): empty pool
            self._key, self.H, self.lw = info, {p: np.zeros(0, np.uint64) for p in opps}, np.zeros(0)
        else:                                     # exact filter
            ok = np.ones(len(self.lw), bool)
            vo = info.voids
            for p in opps:
                Hp = self.H[p]
                pm = np.uint64(played[p])
                ok &= (Hp & pm) == pm
                vm = 0
                for s in range(4):
                    if vo[p] >> s & 1:
                        vm |= SUIT_MASK[s]
                ok &= (Hp & ~pm & np.uint64(vm)) == 0
            self.H = {p: self.H[p][ok] for p in opps}
            self.lw = self.lw[ok]
            kept = int(ok.sum())
        hs = HandSampler(info)
        new_n = 0
        while True:
            w = np.exp(self.lw - self.lw.max()) if len(self.lw) else np.zeros(0)
            ess = float(w.sum() ** 2 / (w * w).sum()) if len(w) else 0.0
            if (ess >= self.ess_target * K and len(self.lw) >= K) or new_n >= self.m_max:
                break
            new = [hs.sample(rng) for _ in range(self.batch)]
            new_n += len(new)
            Hn = {p: np.array([x[p] | played[p] for x in new], np.uint64) for p in opps}
            lwn = self._lw_H(info, Hn, obs)
            self.H = {p: np.concatenate([self.H[p], Hn[p]]) for p in opps}
            self.lw = np.concatenate([self.lw, lwn])
        if len(self.lw) > self.pool_max:          # keep the heaviest particles
            keep = np.argsort(-self.lw, kind="stable")[: self.pool_max]
            self.H = {p: self.H[p][keep] for p in opps}
            self.lw = self.lw[keep]
        wn, ess2, beta = temper(self.lw, self.ess_min)
        idx = _systematic(wn, K, rng)
        worlds = [{p: int(self.H[p][i]) & ~played[p] for p in opps} for i in idx]
        st = {"ess": ess2, "ess_raw": ess, "beta": beta, "M": new_n, "kept": kept, "pool": len(self.lw)}
        return worlds, st


class HybridRBPolicy(HybridBIPolicy):
    """Hybrid play; PIMC worlds weighted by the online-learned bidder-type mixture.

    The posterior is updated at the first decision of each new round from the finished previous round
    (its InfoState is kept; after the round every played card is public, so this is public information).
    ``fixed`` = a type name -> posterior pinned to that type (ablation; 'model' ~ Hybrid-BI)."""

    def __init__(self, lik, n_samples=16, pimc_cards=5, base=None, prior=DEFAULT_PRIOR, eps=0.02,
                 noise=NOISE, ess_min=4.0, m_max=512, particles=False, learn=True, name=None, validate=False):
        super().__init__(lik, n_samples=n_samples, pimc_cards=pimc_cards, base=base, informed=True,
                         m_max=m_max, validate=validate, name=name or f"hybrid_rb_k{n_samples}_c{pimc_cards}")
        self.opp = OpponentModel(BidderTypes(lik, eps, noise), prior)
        W = ParticleWorlds if particles else RobustWorlds
        self.worlds = W(self.opp, m_max=m_max, ess_min=ess_min)
        self.learn = learn
        self._prev = None
        self.rounds_seen = 0
        self.post_log: list = []         # (round index, seat, posterior) after each update

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
            self.post_log.append((r, p, self.opp.post(p).tolist()))

    def bid(self, info, legal, rng):
        self.observe(info)
        return self.base.bid(info, legal, rng)

    def play(self, info, legal_mask, rng):
        self.observe(info)
        return super().play(info, legal_mask, rng)


__all__ = ["HybridRBPolicy", "BidderTypes", "OpponentModel", "RobustWorlds", "ParticleWorlds", "honour_bids",
           "temper", "TYPES", "DEFAULT_PRIOR"]

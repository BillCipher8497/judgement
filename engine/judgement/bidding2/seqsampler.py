"""Sequential importance sampler for opponents' hands given their observed bids.

Target (bid time, hero = info.seat):
    p(h_opps | bids) ∝ U(h_opps | hero hand) * prod_{j earlier bidder} L_j(b_j | h_j, prior bids)
L_j depends only on h_j and the (observed) bids before j, so we sample the earlier
bidders one at a time.  For each particle and each earlier bidder j:
    draw B uniform candidate hands from the cards still unassigned in that particle,
    Zhat_j = mean_B L_j(candidate),   pick one candidate with prob ∝ L_j,
    log w += log Zhat_j.
Zhat_j is an unbiased estimate of the exact incremental weight
Z_j(prev) = E_{h ~ U(remaining)}[L_j(h)], so (random-weight IS) the weighted
particles target the exact posterior.  Later bidders' hands are uniform from the rest.
K worlds are drawn from M weighted particles by systematic resampling.
This replaces the previous attempt's rejection scheme, whose acceptance fell to
~1e-5 at the dealer seat (1 accepted world out of 160k proposals in some rows).
"""
from __future__ import annotations

import numpy as np

from ..cards import FULL_DECK
from .features_vec import hand_features_batch  # noqa: F401  (re-export convenience)

_POW = (np.uint64(1) << np.arange(52, dtype=np.uint64))


def _masks(H: np.ndarray) -> np.ndarray:
    return (H.astype(np.uint64) * _POW).sum(1)


def _draw(avail: np.ndarray, n: int, rng) -> np.ndarray:
    """avail: (R, 52) bool.  One uniform n-subset of the available cards per row -> (R, 52) bool."""
    key = rng.random(avail.shape)
    key[~avail] = 2.0
    idx = np.argpartition(key, n - 1, axis=1)[:, :n] if n < 52 else np.argsort(key, 1)[:, :n]
    out = np.zeros_like(avail)
    np.put_along_axis(out, idx, True, axis=1)
    return out


class SeqPosteriorSampler:
    def __init__(self, info, likelihood, prior_override=None):
        self.info = info
        self.N = N = info.n_players
        self.n, self.trump, self.seat = info.n_cards, info.trump, info.seat
        self.opps = [p for p in range(N) if p != self.seat]
        self.lik = likelihood
        bids = list(prior_override) if prior_override is not None else list(info.bids)
        self.obs = []                    # (seat, pos, prior bids, bid)
        for k, (p, b) in enumerate(bids):
            if p == self.seat:
                continue
            self.obs.append((p, (p - info.dealer - 1) % N, [bb for _, bb in bids[:k]], b))
        hero = np.array([(info.hand >> c) & 1 for c in range(52)], dtype=bool)
        self.free0 = ~hero & np.array([(FULL_DECK >> c) & 1 for c in range(52)], dtype=bool)

    def particles(self, M: int, rng, B: int = 128):
        """-> (hands dict seat -> (M,) uint64 masks, log weights (M,))."""
        n = self.n
        avail = np.repeat(self.free0[None], M, 0)
        lw = np.zeros(M)
        hands = {}
        informed = self.lik is not None
        for p, pos, prior, b in (self.obs if informed else []):
            cand = _draw(np.repeat(avail, B, 0), n, rng)                    # (M*B, 52)
            L = np.exp(self.lik.log_lik(cand, n, self.trump, pos, prior, b)).reshape(M, B)
            Z = L.mean(1)
            with np.errstate(divide="ignore"):
                lw += np.log(Z)
            P = np.where(Z[:, None] > 0, L / np.where(Z > 0, Z * B, 1.0)[:, None], 1.0 / B)
            u = rng.random(M)[:, None]
            pick = np.minimum((P.cumsum(1) < u).sum(1), B - 1)
            H = cand.reshape(M, B, 52)[np.arange(M), pick]
            hands[p] = H
            avail = avail & ~H
        for p in self.opps:
            if p not in hands:
                H = _draw(avail, n, rng)
                hands[p] = H
                avail = avail & ~H
        return {p: _masks(H) for p, H in hands.items()}, lw

    def sample(self, K: int, rng, M: int | None = None, B: int = 128):
        """K worlds (list of {seat: mask}) and stats {ess, M}."""
        if self.lik is None or not self.obs:
            hands, _ = self.particles(K, rng, B)
            return [{p: int(hands[p][i]) for p in self.opps} for i in range(K)], {"ess": float(K), "M": K}
        M = M or 3 * K
        hands, lw = self.particles(M, rng, B)
        if not np.isfinite(lw).any():          # pathological: retry with a bigger batch
            hands, lw = self.particles(M, rng, 8 * B)
        if not np.isfinite(lw).any():          # observation impossible under the likelihood
            lw = np.zeros(M)
        w = np.exp(lw - lw[np.isfinite(lw)].max())
        w[~np.isfinite(lw)] = 0.0
        w /= w.sum()
        ess = float(1.0 / (w * w).sum())
        pos = (rng.random() + np.arange(K)) / K
        idx = np.minimum(np.searchsorted(np.cumsum(w), pos), M - 1)
        return [{p: int(hands[p][i]) for p in self.opps} for i in idx], {"ess": ess, "M": M}


def posterior_worlds(info, likelihood, K: int, rng, max_reject: int = 32000, prior_override=None):
    """K worlds from the bid posterior.  Hard eps=0 likelihood: exact rejection
    (i.i.d.) up to ``max_reject`` proposals, topped up with the sequential IS sampler
    when acceptance is too low.  Soft / eps>0 likelihood: sequential sampler only.
    Returns (worlds, stats)."""
    from .sampler import BidPhaseSampler
    seq = SeqPosteriorSampler(info, likelihood, prior_override)
    if likelihood is None or not seq.obs:
        return seq.sample(K, rng)
    if likelihood.tau <= 0 and likelihood.eps == 0:
        bs = BidPhaseSampler(info, likelihood, prior_override)
        got, tot = [], 0
        while tot < max_reject and sum(len(g) for g in got) < K:
            props = bs.propose(4000, rng)
            tot += 4000
            got.append(props[np.isfinite(bs.log_weights(props))])
        acc = np.concatenate(got)[:K]
        worlds = bs.worlds(acc)
        st = {"n_reject": len(worlds), "proposals": tot}
        if len(worlds) < K:
            w2, s2 = seq.sample(K - len(worlds), rng, M=max(60, 3 * (K - len(worlds))))
            worlds += w2
            st["seq_ess"] = s2["ess"]
        return worlds, st
    return seq.sample(K, rng)

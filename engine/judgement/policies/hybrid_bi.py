"""Hybrid-BI: Hybrid play (TA+ early, PIMC once the seat holds <= ``pimc_cards`` cards) whose
PIMC worlds are drawn from the posterior given the opponents' observed BIDS, not uniformly.

World posterior at a play decision (hero = info.seat):
    p(h | obs) ∝ U(h | own hand, played cards, voids) * prod_{opp j} L_j(b_j | H_j, prior bids, pos_j)
where h are the opponents' current hands, H_j = h_j | cards j already played this round (the hand
j held when bidding) and L_j is the Phase-3 bid likelihood (judgement.bidding2.BidLikelihood on the
Phase-1 TA bid model; hard argmax with eps, hook-legal set for the dealer).  Realised by sampling-
importance-resampling: proposals from the exact uniform HandSampler (which already enforces voids and
played cards), log-weights = sum_j log L_j, systematic resampling to K worlds.  Proposals are drawn in
batches until the effective sample size reaches ``ess_target * K`` or ``m_max`` proposals.
With ModelBidder opponents and eps -> 0 this is the exact posterior (P1: likelihood argmax == actual
bid on 4500/4500 self-play bids).
"""
from __future__ import annotations

import numpy as np

from ..cards import NT, cards_of
from ..sampling import HandSampler
from ..solver.core import NEUTRAL, MaxN
from ..state import state_from_info
from .ta_plus import TAPlusPolicy

_SH = np.arange(52, dtype=np.uint64)


def _bool(masks) -> np.ndarray:
    a = np.asarray(masks, dtype=np.uint64)
    return ((a[:, None] >> _SH) & np.uint64(1)).astype(bool)


class BidInformedWorlds:
    def __init__(self, likelihood, batch: int = 64, m_max: int = 512, ess_target: float = 1.0):
        self.lik, self.batch, self.m_max, self.ess_target = likelihood, batch, m_max, ess_target

    @staticmethod
    def observations(info):
        """[(seat, pos, prior bids, bid, legal bids)] for every opponent bid."""
        N, n = info.n_players, info.n_cards
        out = []
        bids = list(info.bids)
        for k, (p, b) in enumerate(bids):
            if p == info.seat:
                continue
            prior = [bb for _, bb in bids[:k]]
            pos = (p - info.dealer - 1) % N
            legal = list(range(n + 1))
            if pos == N - 1 and info.rules.hook:
                f = n - sum(prior)
                if 0 <= f <= n:
                    legal.remove(f)
            out.append((p, pos, prior, b, legal))
        return out

    def log_weights(self, info, worlds, obs=None):
        obs = self.observations(info) if obs is None else obs
        played = {}
        for p, c in info.history:
            played[p] = played.get(p, 0) | (1 << c)
        lw = np.zeros(len(worlds))
        for p, pos, prior, b, legal in obs:
            H = _bool([w[p] | played.get(p, 0) for w in worlds])
            lw += self.lik.log_lik(H, info.n_cards, info.trump, pos, prior, b, legal=legal)
        return lw

    def sample(self, info, K, rng):
        """-> (K worlds [{seat: current hand}], stats dict)."""
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
        w /= w.sum()
        pos = (rng.random() + np.arange(K)) / K
        idx = np.minimum(np.searchsorted(np.cumsum(w), pos), len(w) - 1)
        thr = lw.max() - 1e-9
        st = {"ess": ess, "M": len(worlds), "n_top": int((lw >= thr).sum()), "lw_max": float(lw.max())}
        return [worlds[i] for i in idx], st


class HybridBIPolicy:
    """Hybrid play with bid-informed PIMC worlds.  ``informed=False`` -> plain uniform PIMC
    (same code path, for CRN-style ablations).  ``log`` collects per-decision sampler stats; with
    ``validate=True`` it also logs, for informed and uniform worlds, the fraction of each opponent's
    true remaining cards placed correctly (reads the true state: diagnostics only, never used to act)."""

    def __init__(self, likelihood, n_samples: int = 16, pimc_cards: int = 5, base=None, informed: bool = True,
                 batch: int = 64, m_max: int = 512, ess_target: float = 1.0, validate: bool = False, name=None):
        self.n_samples, self.pimc_cards, self.informed = n_samples, pimc_cards, informed
        self.base = base or TAPlusPolicy()
        self.worlds = BidInformedWorlds(likelihood, batch, m_max, ess_target)
        self.validate = validate
        self.log: list = []
        self.name = name or f"hybrid_bi_k{n_samples}_c{pimc_cards}" + ("" if informed else "_unif")

    def bid(self, info, legal, rng):
        return self.base.bid(info, legal, rng)

    def _scores(self, info, legal_mask, worlds):
        legal = cards_of(legal_mask)
        tot = {c: 0.0 for c in legal}
        for w in worlds:
            st = state_from_info(info, w)
            mv = MaxN(st.rules, st.trump, st.bids, st.N, NEUTRAL).move_values(st)
            for c in legal:
                tot[c] += mv[c][info.seat]
        return tot

    def play(self, info, legal_mask, rng):
        if legal_mask & (legal_mask - 1) == 0:
            return legal_mask.bit_length() - 1
        if info.hand.bit_count() > self.pimc_cards:
            return self.base.play(info, legal_mask, rng)
        K = self.n_samples
        if self.informed:
            worlds, st = self.worlds.sample(info, K, rng)
        else:
            hs = HandSampler(info)
            worlds, st = [hs.sample(rng) for _ in range(K)], {"ess": float(K), "M": K, "n_top": K, "lw_max": 0.0}
        st.update(n=info.n_cards, left=info.hand.bit_count(), nt=info.trump == NT)
        if self.validate:
            true = {p: info._s.hands[p] for p in range(info.n_players) if p != info.seat}

            def acc(ws):
                return float(np.mean([sum((w[p] & true[p]).bit_count() for p in true) /
                                      max(1, sum(true[p].bit_count() for p in true)) for w in ws]))
            hs = HandSampler(info)
            st["acc_inf"] = acc(worlds)
            st["acc_unif"] = acc([hs.sample(np.random.default_rng(len(self.log))) for _ in range(64)])
            st["true_lw"] = float(self.worlds.log_weights(info, [true])[0])
        self.log.append(st)
        sc = self._scores(info, legal_mask, worlds)
        return max(sorted(sc), key=lambda c: sc[c])


__all__ = ["HybridBIPolicy", "BidInformedWorlds"]

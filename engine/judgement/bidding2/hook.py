"""Hook-setting bidder for the seat that bids just before the dealer.

The dealer may not bid n - S, where S is the sum of the first N-1 bids.  The
bidder before the dealer fixes S.  Rule: among own bids whose model EV is within
``delta`` points of the best, pick the one maximising the dealer's expected
forced loss

    E_dealer-hand[ EV_d(d*) - max_{d != f} EV_d(d) ],   f = n - S(b),  d* = argmax_d EV_d(d)

where dealer hands are drawn from the bid-informed posterior (earlier bids
observed) and EV_d is the Phase-1 bidder model for the dealer given all prior
bids including our own b.  Ties -> higher own EV, then lower bid.
"""
from __future__ import annotations

import numpy as np

from ..bidding.model import info_features, score_matrix
from .sampler import BidLikelihood
from .seqsampler import SeqPosteriorSampler, posterior_worlds


class HookSetter:
    name = "hookset"

    def __init__(self, model, rules, delta: float, K: int = 200, informed: bool = True, seed: int = 0):
        self.model, self.rules, self.delta, self.K, self.informed = model, rules, float(delta), K, informed
        self.lik = BidLikelihood(model, rules, tau=0.0, eps=0.02)
        self.lik0 = BidLikelihood(model, rules, tau=0.0, eps=0.0)
        self.rng = np.random.default_rng(seed)
        self._S: dict = {}
        self.last = None

    def _ev(self, info):
        n = info.n_cards
        S = self._S.get(n)
        if S is None:
            S = self._S[n] = score_matrix(self.rules, n)
        return (self.model.get(n, info.trump).dist_all_bids(info_features(info)) * S).sum(1)

    def dealer_loss(self, info, cands):
        N, n = info.n_players, info.n_cards
        if self.informed:       # exact posterior for a ModelBidder table (rejection + sequential IS top-up)
            worlds, _ = posterior_worlds(info, self.lik0, self.K, self.rng)
        else:
            worlds, _ = SeqPosteriorSampler(info, None).sample(self.K, self.rng)
        d = info.dealer
        Hd = np.zeros((len(worlds), 52), dtype=bool)
        for i, w in enumerate(worlds):
            m = w[d]
            for c in range(52):
                if m >> c & 1:
                    Hd[i, c] = True
        prior = [b for _, b in info.bids]
        out, hookp = {}, {}
        for b in cands:
            pr = prior + [b]
            f = n - sum(pr)
            if not 0 <= f <= n:
                out[b], hookp[b] = 0.0, 0.0
                continue
            ev = self.lik.evs(Hd, n, info.trump, N - 1, pr)
            best = ev.max(1)
            ev2 = ev.copy()
            ev2[:, f] = -np.inf
            out[b] = float((best - ev2.max(1)).mean())
            hookp[b] = float((ev.argmax(1) == f).mean())
        return out, hookp

    def bid(self, info, legal, rng):
        ev = self._ev(info)
        bm = max(legal, key=lambda b: (ev[b], -b))
        pos = (info.seat - info.dealer - 1) % info.n_players
        if pos != info.n_players - 2 or self.delta <= 0:
            self.last = None
            return bm
        cands = [b for b in legal if ev[b] >= ev[bm] - self.delta]
        loss, hookp = self.dealer_loss(info, cands)
        b = max(cands, key=lambda x: (loss[x], ev[x], -x))
        self.last = dict(bm=bm, b=b, loss_bm=loss[bm], loss_b=loss[b], hookp_bm=hookp[bm], hookp_b=hookp[b],
                         own_ev_drop=float(ev[bm] - ev[b]))
        return b

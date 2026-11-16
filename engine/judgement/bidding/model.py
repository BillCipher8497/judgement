"""Heuristic bidder: a conditional-logit model of P(tricks = t | hand features, own bid).

One model per (n_cards, trump/NT).  For class t in 0..n and own bid b:

    logit_t = W_t . x  +  sum_j gamma_j * F_j(t, b)

``W_t . x`` is an ordinary multinomial logit on hand/position features (what the
hand is worth).  The bid-indicator terms F_j capture how target-aware play pulls
the outcome toward the bid:  [t==b], [t==b+1], [t==b-1], [t==b]*b/n,
[t==b]*[b==0], [t>b], [t<b].  ``exp(gamma_0)`` is the odds multiplier on
hitting exactly that the play policy buys.  Under bid-agnostic play (greedy)
``use_bid=False`` and gamma is absent.

Bid = argmax over hook-legal b of sum_t P_b(t) * score(b, t)  (ties -> lower).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize

from ..cards import NT
from .features import D, FEATURE_NAMES, hand_features

N_GAMMA = 7
GAMMA_NAMES = ["hit", "over1", "under1", "hit_x_b/n", "hit_x_b0", "over", "under"]


def bid_indicators(n: int) -> np.ndarray:
    """F[b, j, t] for b, t in 0..n."""
    K = n + 1
    F = np.zeros((K, N_GAMMA, K))
    t = np.arange(K)
    for b in range(K):
        F[b, 0] = t == b
        F[b, 1] = t == b + 1
        F[b, 2] = t == b - 1
        F[b, 3] = (t == b) * (b / n)
        F[b, 4] = (t == b) * (b == 0)
        F[b, 5] = t > b
        F[b, 6] = t < b
    return F


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class CondLogit:
    def __init__(self, n: int, use_bid: bool = True, l2: float = 1e-3):
        self.n, self.K, self.use_bid, self.l2 = n, n + 1, use_bid, l2
        self.mu = np.zeros(D)
        self.sd = np.ones(D)
        self.W = np.zeros((self.K, D))
        self.gamma = np.zeros(N_GAMMA)
        self.F = bid_indicators(n)
        self._G = None

    # ---- fitting ------------------------------------------------------------
    def fit(self, X: np.ndarray, bids: np.ndarray, tricks: np.ndarray, maxiter: int = 400):
        X = np.asarray(X, float)
        self.mu = X.mean(0)
        self.mu[0] = 0.0
        sd = X.std(0)
        sd[sd < 1e-9] = 1.0
        sd[0] = 1.0
        self.sd = sd
        Z = (X - self.mu) / sd
        Nn, K = len(Z), self.K
        Y = np.zeros((Nn, K))
        Y[np.arange(Nn), tricks] = 1.0
        Fi = self.F[bids] if self.use_bid else None   # (N, g, K)
        nw = K * D
        l2 = self.l2

        def f(theta):
            W = theta[:nw].reshape(K, D)
            logits = Z @ W.T
            if self.use_bid:
                g = theta[nw:]
                logits = logits + np.einsum("j,njk->nk", g, Fi)
            logits -= logits.max(1, keepdims=True)
            lse = np.log(np.exp(logits).sum(1))
            nll = (lse - (logits * Y).sum(1)).sum() / Nn
            P = np.exp(logits - lse[:, None])
            R = (P - Y) / Nn
            gW = R.T @ Z + 2 * l2 * W
            val = nll + l2 * (W * W).sum()
            if self.use_bid:
                gg = np.einsum("nk,njk->j", R, Fi)
                return val, np.concatenate([gW.ravel(), gg])
            return val, gW.ravel()

        theta0 = np.zeros(nw + (N_GAMMA if self.use_bid else 0))
        res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        self.W = res.x[:nw].reshape(K, D)
        if self.use_bid:
            self.gamma = res.x[nw:]
        self._G = None
        self.train_nll = float(res.fun)
        return self

    # ---- prediction ---------------------------------------------------------
    def _gmat(self):
        if self._G is None:
            self._G = np.einsum("j,bjk->bk", self.gamma, self.F) if self.use_bid else np.zeros((self.K, self.K))
        return self._G

    def dist_all_bids(self, x: np.ndarray) -> np.ndarray:
        """P[b, t] for every own bid b (rows identical when use_bid=False)."""
        base = self.W @ ((x - self.mu) / self.sd)
        return _softmax(base[None, :] + self._gmat())

    def dist_batch(self, X: np.ndarray, bids: np.ndarray) -> np.ndarray:
        base = ((X - self.mu) / self.sd) @ self.W.T
        return _softmax(base + self._gmat()[bids])

    def to_dict(self):
        return {"n": self.n, "use_bid": self.use_bid, "l2": self.l2, "mu": self.mu.tolist(),
                "sd": self.sd.tolist(), "W": self.W.tolist(), "gamma": self.gamma.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["n"], d["use_bid"], d["l2"])
        m.mu, m.sd, m.W = np.array(d["mu"]), np.array(d["sd"]), np.array(d["W"])
        m.gamma = np.array(d["gamma"])
        return m


class BidModel:
    """Collection of CondLogit models keyed by (n, is_nt)."""

    def __init__(self, models: dict | None = None):
        self.models: dict[tuple[int, bool], CondLogit] = models or {}

    def get(self, n: int, trump: int) -> CondLogit:
        return self.models[(n, trump == NT)]

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({f"{n}|{int(nt)}": m.to_dict() for (n, nt), m in self.models.items()}, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            raw = json.load(fh)
        out = {}
        for k, d in raw.items():
            n, nt = k.split("|")
            out[(int(n), bool(int(nt)))] = CondLogit.from_dict(d)
        return cls(out)


def info_features(info) -> np.ndarray:
    N = info.n_players
    pos = (info.seat - info.dealer - 1) % N
    prior = [b for _, b in info.bids]
    return hand_features(info.hand, info.trump, info.n_cards, pos, prior, N)


def score_matrix(rules, n: int) -> np.ndarray:
    K = n + 1
    return np.array([[rules.score(b, t) for t in range(K)] for b in range(K)], float)


class ModelBidder:
    """bid(info, legal, rng) via argmax_b E_model[score | b]."""

    name = "model_bid"

    def __init__(self, model: BidModel, rules_for_score=None):
        self.model = model
        self.rules_for_score = rules_for_score
        self._S: dict = {}

    def evs(self, info):
        n = info.n_cards
        rules = self.rules_for_score or info.rules
        key = (id(rules), n)
        S = self._S.get(key)
        if S is None:
            S = self._S[key] = score_matrix(rules, n)
        P = self.model.get(n, info.trump).dist_all_bids(info_features(info))
        return (P * S).sum(1), P

    def bid(self, info, legal, rng):
        ev, _ = self.evs(info)
        return max(legal, key=lambda b: (ev[b], -b))


class Agent:
    """Bid with one object, play with another (both follow the Policy protocol)."""

    def __init__(self, bidder, player, name=None):
        self.bidder, self.player = bidder, player
        self.name = name or f"{getattr(bidder, 'name', '?')}+{getattr(player, 'name', '?')}"

    def bid(self, info, legal, rng):
        return self.bidder.bid(info, legal, rng)

    def play(self, info, legal_mask, rng):
        return self.player.play(info, legal_mask, rng)


__all__ = ["CondLogit", "BidModel", "ModelBidder", "Agent", "info_features", "score_matrix",
           "FEATURE_NAMES", "GAMMA_NAMES"]

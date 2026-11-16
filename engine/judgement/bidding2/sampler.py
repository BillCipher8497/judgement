"""Bid-informed sampling of opponents' hidden hands during the bidding phase.

Prior: uniform over deals consistent with the hero's hand (HandSampler's law at
bid time).  Likelihood of an earlier bidder j's observed bid b_j given a
hypothesised hand h_j:

    L_j(b_j | h_j) = softmax_b(EV_j(b | h_j) / tau)[b_j]        (tau > 0, "soft")
    L_j(b_j | h_j) = (1-eps) 1[b_j = argmax_b EV_j(b | h_j)] + eps/(n+1)   (tau = 0, "hard")

where EV_j(b | h) = sum_t P_model(t | features(h, n, trump, pos_j, prior bids), b) * score(b, t)
is the Phase-1 bidder model (model_ta_final.json).  Under a table of ModelBidders the
hard likelihood with eps = 0 is the exact posterior (the bid is a deterministic
function of the hand); eps > 0 / tau > 0 add robustness to opponents who are not
exactly the model.  Posterior = prior x prod_j L_j, realised by importance
weighting of vectorised uniform proposals (rejection when the weights are 0/1).
"""
from __future__ import annotations

import numpy as np

from ..bidding.model import score_matrix
from ..cards import FULL_DECK, cards_of
from .features_vec import hand_features_batch

_POW = (np.uint64(1) << np.arange(52, dtype=np.uint64))


def bool_to_masks(H: np.ndarray) -> np.ndarray:
    return (H.astype(np.uint64) * _POW).sum(1)


class BidLikelihood:
    def __init__(self, model, rules, tau: float = 0.0, eps: float = 0.02):
        self.model, self.rules, self.tau, self.eps = model, rules, float(tau), float(eps)
        self._S: dict = {}

    def evs(self, H: np.ndarray, n: int, trump: int, pos: int, prior) -> np.ndarray:
        """(M, n+1) model EV of each own bid for each hypothesised hand."""
        S = self._S.get(n)
        if S is None:
            S = self._S[n] = score_matrix(self.rules, n)
        cl = self.model.get(n, trump)
        X = hand_features_batch(H, trump, n, pos, prior, self.rules.n_players)
        base = ((X - cl.mu) / cl.sd) @ cl.W.T                      # (M, K)
        z = base[:, None, :] + cl._gmat()[None, :, :]               # (M, b, t)
        z -= z.max(2, keepdims=True)
        e = np.exp(z)
        P = e / e.sum(2, keepdims=True)
        return (P * S[None]).sum(2)

    def log_lik(self, H, n, trump, pos, prior, bid, legal=None) -> np.ndarray:
        ev = self.evs(H, n, trump, pos, prior)
        if legal is not None:
            mask = np.full(n + 1, -np.inf)
            mask[list(legal)] = 0.0
            ev = ev + mask
            nl = len(legal)
        else:
            nl = n + 1
        if self.tau <= 0:
            hit = ev.argmax(1) == bid                      # ties -> lower bid, as ModelBidder
            p = np.where(hit, 1.0 - self.eps, 0.0) + self.eps / nl
            with np.errstate(divide="ignore"):
                return np.log(p)
        z = ev / self.tau
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        return z[:, bid] - lse

    def probs(self, H, n, trump, pos, prior, legal=None) -> np.ndarray:
        """(M, n+1) P(bid | hand) (rows sum to 1 over legal bids)."""
        ev = self.evs(H, n, trump, pos, prior)
        if legal is not None:
            mask = np.full(n + 1, -np.inf)
            mask[list(legal)] = 0.0
            ev = ev + mask
            nl = len(legal)
        else:
            nl = n + 1
        if self.tau <= 0:
            P = np.zeros_like(ev)
            P[np.arange(len(ev)), ev.argmax(1)] = 1.0 - self.eps
            P += self.eps / nl
            if legal is not None:
                P[:, [b for b in range(n + 1) if b not in legal]] = 0.0
            return P
        z = ev / self.tau
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


class BidPhaseSampler:
    """Uniform proposals + bid-likelihood weights for the hero's bidding decision.

    ``prior_override`` (list of (seat, bid) in order) replaces the observed bids
    in the likelihood only (used for the information-only response curve)."""

    def __init__(self, info, likelihood: BidLikelihood | None, prior_override=None):
        self.info = info
        self.N = N = info.n_players
        self.n = info.n_cards
        self.trump = info.trump
        self.seat = info.seat
        self.opps = [p for p in range(N) if p != self.seat]
        self.unknown = np.array(cards_of(FULL_DECK & ~info.hand), dtype=np.int64)
        self.lik = likelihood
        bids = list(prior_override) if prior_override is not None else list(info.bids)
        dealer = info.dealer
        # (opp index, pos, prior bids before it, its bid)
        self.obs = []
        for k, (p, b) in enumerate(bids):
            if p == self.seat:
                continue
            pos = (p - dealer - 1) % N
            self.obs.append((self.opps.index(p), pos, [bb for _, bb in bids[:k]], b))

    def propose(self, M: int, rng) -> np.ndarray:
        """(M, N-1) uint64 opponent hand masks, uniform."""
        n, U = self.n, len(self.unknown)
        perm = np.argsort(rng.random((M, U)), axis=1)[:, :(self.N - 1) * n]
        cards = self.unknown[perm]                     # (M, 3n)
        out = np.zeros((M, self.N - 1), dtype=np.uint64)
        for j in range(self.N - 1):
            H = np.zeros((M, 52), dtype=bool)
            np.put_along_axis(H, cards[:, j * n:(j + 1) * n], True, axis=1)
            out[:, j] = bool_to_masks(H)
        return out

    @staticmethod
    def _bool(masks: np.ndarray) -> np.ndarray:
        return ((masks[:, None] >> np.arange(52, dtype=np.uint64)) & np.uint64(1)).astype(bool)

    def log_weights(self, props: np.ndarray, lik: BidLikelihood | None = None) -> np.ndarray:
        lik = lik or self.lik
        lw = np.zeros(len(props))
        if lik is None:
            return lw
        for j, pos, prior, b in self.obs:
            lw += lik.log_lik(self._bool(props[:, j]), self.n, self.trump, pos, prior, b)
        return lw

    def worlds(self, props: np.ndarray) -> list[dict[int, int]]:
        return [{p: int(props[i, j]) for j, p in enumerate(self.opps)} for i in range(len(props))]

    def sample(self, K: int, rng, informed: bool = True, batch: int = 2000, max_batches: int = 8,
               first: np.ndarray | None = None):
        """K worlds.  uniform: first K proposals.  informed: K drawn from the
        posterior (uniform among accepted when hard weights; SIR otherwise).
        Returns (worlds, stats)."""
        props = first if first is not None else self.propose(max(batch, K), rng)
        if not informed or not self.obs or self.lik is None:
            return self.worlds(props[:K]), {"ess": float(K), "acc": 1.0, "proposals": K}
        lw = self.log_weights(props)
        hard = self.lik.tau <= 0
        nb = 1
        thr = np.log(1.0 - self.lik.eps) * len(self.obs) - 1e-9 if hard else None
        while nb < max_batches:
            if hard and (lw >= thr).sum() >= K:
                break
            if not hard:
                w = np.exp(lw - lw.max())
                if w.sum() ** 2 / (w * w).sum() >= 2 * K:
                    break
            more = self.propose(batch, rng)
            props = np.concatenate([props, more])
            lw = np.concatenate([lw, self.log_weights(more)])
            nb += 1
        w = np.exp(lw - lw.max())
        w /= w.sum()
        ess = float(1.0 / (w * w).sum())
        if hard:
            acc = np.flatnonzero(lw >= thr)
            if len(acc) >= K:
                idx = acc[:K]           # accepted proposals in order: uniform over the exact posterior
            else:
                idx = np.concatenate([acc, rng.choice(len(w), size=K - len(acc), p=w)])
            return self.worlds(props[idx]), {"ess": ess, "acc": float(len(acc) / len(props)),
                                             "proposals": int(len(props))}
        idx = rng.choice(len(w), size=K, p=w)
        return self.worlds(props[idx]), {"ess": ess, "acc": float("nan"), "proposals": int(len(props))}


def exact_posterior_worlds(info, likelihood_hard: BidLikelihood, K: int, rng, batch: int = 4000,
                           max_batches: int = 40):
    """Rejection sampling with the eps=0 hard likelihood: i.i.d. draws from the exact
    posterior when every earlier bidder is a ModelBidder.  May return < K worlds."""
    assert likelihood_hard.tau <= 0
    lik0 = BidLikelihood(likelihood_hard.model, likelihood_hard.rules, tau=0.0, eps=0.0)
    s = BidPhaseSampler(info, lik0)
    got = []
    tot = 0
    for _ in range(max_batches):
        props = s.propose(batch, rng)
        tot += batch
        lw = s.log_weights(props)
        got.append(props[np.isfinite(lw)])
        if sum(len(g) for g in got) >= K:
            break
    acc = np.concatenate(got)[:K]
    return s.worlds(acc), tot

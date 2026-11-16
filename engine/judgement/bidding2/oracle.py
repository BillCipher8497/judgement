"""Monte-Carlo bid oracle over a supplied set of worlds (CRN across candidates)."""
from __future__ import annotations

import numpy as np

from ..engine import run_round
from ..state import BID, state_from_info


def world_scores(info, worlds, candidates, agents, rngs=None, prior_override=None) -> np.ndarray:
    """(len(worlds), len(candidates)) hero score in each world for each candidate bid.
    Later bidders bid and everyone plays with ``agents``.  ``prior_override``
    replaces the earlier bids actually applied in the rollout (table-system effect)."""
    assert info.phase == BID
    hero = info.seat
    rules = info.rules
    N = info.n_players
    rngs = rngs or [np.random.default_rng(0)] * N
    out = np.zeros((len(worlds), len(candidates)))
    bids = list(prior_override) if prior_override is not None else list(info.bids)
    from ..state import RoundState
    for i, w in enumerate(worlds):
        hands = [info.hand if p == hero else w[p] for p in range(N)]
        for k, b in enumerate(candidates):
            st = RoundState(rules, info.n_cards, info.trump, info.dealer, hands)
            for _, pb in bids:
                st.apply_bid(pb)
            st.apply_bid(b)
            run_round(st, agents, rngs, check=False)
            out[i, k] = rules.score(b, st.tricks_won[hero])
    return out


def legal_for(info, prior_bids_sum: int | None = None) -> list[int]:
    n, N = info.n_cards, info.n_players
    sb = prior_bids_sum if prior_bids_sum is not None else sum(b for _, b in info.bids)
    if info.rules.hook and info.seat == info.dealer:
        return [b for b in range(n + 1) if b != n - sb]
    return list(range(n + 1))


def best_of(scores: np.ndarray, candidates) -> tuple[int, np.ndarray]:
    ev = scores.mean(0)
    k = max(range(len(candidates)), key=lambda i: (ev[i], -candidates[i]))
    return candidates[k], ev


class InformedOracleBidder:
    """MC bid oracle: K worlds from the exact bid posterior for a ModelBidder table
    (uniform worlds when ``informed=False``), candidates = model bid ± ``span`` (legal),
    every candidate scored on the same worlds with ``table_agents`` doing the rest."""

    name = "oracle"

    def __init__(self, model, rules, table_agents, K: int = 32, span: int = 2, informed: bool = True, seed: int = 0):
        from ..bidding.model import ModelBidder
        from .sampler import BidLikelihood
        self.mb = ModelBidder(model)
        self.lik = BidLikelihood(model, rules, 0.0, 0.0)
        self.agents, self.K, self.span, self.informed = table_agents, K, span, informed
        self.rng = np.random.default_rng(seed)

    def bid(self, info, legal, rng):
        from .seqsampler import SeqPosteriorSampler, posterior_worlds
        bm = self.mb.bid(info, legal, rng)
        cands = [b for b in legal if abs(b - bm) <= self.span]
        if len(cands) == 1:
            return cands[0]
        if self.informed:
            worlds, _ = posterior_worlds(info, self.lik, self.K, self.rng)
        else:
            worlds, _ = SeqPosteriorSampler(info, None).sample(self.K, self.rng)
        return best_of(world_scores(info, worlds, cands, self.agents), cands)[0]

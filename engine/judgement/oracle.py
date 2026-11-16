"""Monte-Carlo trick distribution and score-optimal bidding."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .engine import run_round
from .sampling import HandSampler
from .state import BID, state_from_info


def mc_trick_distribution(info, play_policies: Sequence, n_samples: int,
                          rng: np.random.Generator, bid_policies: Sequence | None = None) -> np.ndarray:
    """P(seat ends the round with k total tricks), k = 0..n_cards.

    For each sample: deal hidden hands consistent with ``info``, rebuild the full
    state, then play it out.  Bids not yet made are filled by ``bid_policies``
    (default: the play policies' own ``bid``); the seat's own pending bid is
    irrelevant to bid-agnostic play policies such as greedy/random.
    """
    sampler = HandSampler(info)
    N = info.n_players
    pols = list(play_policies)
    if bid_policies is not None and info.phase == BID:
        pols = [_Combined(bid_policies[p], play_policies[p]) for p in range(N)]
    rngs = [rng] * N
    counts = np.zeros(info.n_cards + 1, dtype=np.int64)
    seat = info.seat
    for _ in range(n_samples):
        st = state_from_info(info, sampler.sample(rng))
        run_round(st, pols, rngs, check=False)
        counts[st.tricks_won[seat]] += 1
    return counts / n_samples


class _Combined:
    def __init__(self, bidder, player):
        self._b, self._p = bidder, player
        self.name = f"{getattr(bidder, 'name', '?')}+{getattr(player, 'name', '?')}"

    def bid(self, info, legal, rng):
        return self._b.bid(info, legal, rng)

    def play(self, info, legal_mask, rng):
        return self._p.play(info, legal_mask, rng)


def expected_scores(dist: Sequence[float], rules, legal: Sequence[int]) -> dict[int, float]:
    """E[score | bid b] = sum_t P(t) * score(b, t) for each legal b.

    Under the default miss_penalty="zero" this is exactly (hit_base + b) * P(b)."""
    return {b: float(sum(p * rules.score(b, t) for t, p in enumerate(dist))) for b in legal}


def optimal_bid(info, dist: Sequence[float], legal: Sequence[int] | None = None) -> tuple[int, dict[int, float]]:
    """argmax over hook-legal bids of expected score (ties -> lower bid)."""
    if legal is None:
        legal = _legal_bids_for(info)
    ev = expected_scores(dist, info.rules, legal)
    best = max(legal, key=lambda b: (ev[b], -b))
    return best, ev


def mean_tricks_bid(dist: Sequence[float], legal: Sequence[int] | None = None) -> tuple[int, float]:
    """round(E[tricks]) (half rounds down), snapped to nearest legal bid if given."""
    mean = float(sum(k * p for k, p in enumerate(dist)))
    b = int(np.floor(mean + 0.5 - 1e-12))
    if legal is not None and b not in legal:
        b = min(legal, key=lambda x: (abs(x - mean), x))
    return b, mean


def _legal_bids_for(info) -> list[int]:
    """Hook-legal bids for info.seat given bids already made (works even if it
    is not yet this seat's turn: only the last bidder is constrained)."""
    rules, n, N = info.rules, info.n_cards, info.n_players
    last_bidder = info.dealer
    if rules.hook and info.seat == last_bidder and len(info.bids) == N - 1:
        forbidden = n - sum(b for _, b in info.bids)
        return [b for b in range(n + 1) if b != forbidden]
    return list(range(n + 1))

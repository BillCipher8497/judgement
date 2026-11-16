import numpy as np
import pytest

from helpers import make_state
from judgement.cards import NT
from judgement.engine import deal
from judgement.oracle import expected_scores, mc_trick_distribution, mean_tricks_bid, optimal_bid
from judgement.policies import GreedyPolicy
from judgement.rules import Rules
from judgement.state import RoundState


def test_distribution_sums_to_one_and_deterministic():
    r = Rules()
    st = RoundState(r, 13, 0, 0, deal(r, 13, np.random.default_rng(3)))
    info = st.info_state(1)
    d1 = mc_trick_distribution(info, [GreedyPolicy()] * 4, 300, np.random.default_rng(0))
    d2 = mc_trick_distribution(info, [GreedyPolicy()] * 4, 300, np.random.default_rng(0))
    assert len(d1) == 14 and abs(d1.sum() - 1) < 1e-12
    assert np.array_equal(d1, d2)
    assert st.phase == 0 and st.history == []  # oracle did not mutate the real state


def test_certain_hand():
    # seat 1 holds all four aces + top trumps in a 4-card NT round: wins every trick it leads
    st = make_state(["2S 3S 4S 5S", "AS AH AD AC", "6S 7S 8S 9S", "TS JS QS KS"], trump=NT, dealer=0)
    d = mc_trick_distribution(st.info_state(1), [GreedyPolicy()] * 4, 50, np.random.default_rng(0))
    assert d[4] == 1.0


def test_optimal_bid_formula_and_hook():
    r = Rules()
    dist = [0.1, 0.2, 0.6, 0.1]
    ev = expected_scores(dist, r, range(4))
    assert ev == pytest.approx({0: 1.0, 1: 2.2, 2: 7.2, 3: 1.3})
    # dealer with others bidding 1 in a 3-card round: bid 2 is forbidden
    st = make_state(["AS KS QS", "JS TS 9S", "8S 7S 6S", "5S 4S 3S"], trump=NT, dealer=3, bids=[1, 0, 0])
    info = st.info_state(3)
    b, ev = optimal_bid(info, dist)
    assert 2 not in ev and b == 1  # best legal: 1 (2.2) beats 3 (1.3) and 0 (1.0)

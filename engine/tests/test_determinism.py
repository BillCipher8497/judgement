import json

import numpy as np
import pytest

from judgement.engine import fractional_wins, play_game, play_round
from judgement.policies import GreedyPolicy, RandomPolicy
from judgement.rules import Rules
from judgement.trace import dumps, loads, replay


@pytest.mark.parametrize("N", [3, 4, 7])
def test_same_seed_byte_identical(N):
    r = Rules(n_players=N)
    pols = [RandomPolicy(), GreedyPolicy()] * 4
    a = play_game(r, pols[:N], seed=123).trace
    b = play_game(r, pols[:N], seed=123).trace
    assert dumps(a) == dumps(b)
    c = play_game(r, pols[:N], seed=124).trace
    assert dumps(a) != dumps(c)


@pytest.mark.parametrize("seed", range(5))
def test_replay_reproduces(seed):
    r = Rules(miss_penalty="neg_diff")
    t = play_game(r, [RandomPolicy(), GreedyPolicy(), RandomPolicy(), GreedyPolicy()], seed=seed).trace
    s = dumps(t)
    t2 = loads(s)
    assert replay(t2) == t2
    assert dumps(replay(t2)) == s
    json.loads(s)  # valid JSON


def test_replay_rejects_tampering():
    t = play_game(Rules(), [RandomPolicy()] * 4, seed=9).trace
    bad = loads(dumps(t))
    rt = bad["rounds"][0]
    # swap two plays from different seats -> out-of-turn or illegal
    rt["plays"][0], rt["plays"][1] = rt["plays"][1], rt["plays"][0]
    with pytest.raises(ValueError):
        replay(bad)


def test_round_trace_replay():
    res = play_round(Rules(), [GreedyPolicy()] * 4, seed=7, round_index=4, record_trace=True)
    assert res.trace["trump"] == "NT"
    assert replay(res.trace, Rules()) == res.trace


def test_callable_penalty_requires_rules():
    r = Rules(miss_penalty=lambda b, t: -1)
    t = play_game(r, [RandomPolicy()] * 4, seed=1).trace
    with pytest.raises(ValueError):
        replay(t)
    assert replay(t, rules=r) == t


def test_common_random_numbers():
    """Changing the policy in one seat never changes the deals, nor another
    seat's policy RNG stream."""
    r = Rules()
    a = play_game(r, [RandomPolicy()] * 4, seed=42).trace
    b = play_game(r, [GreedyPolicy(), RandomPolicy(), RandomPolicy(), RandomPolicy()], seed=42).trace
    for ra, rb in zip(a["rounds"], b["rounds"]):
        assert ra["hands"] == rb["hands"]
    # seat 1..3 bids are drawn from their own streams: identical in round 0
    assert a["rounds"][0]["bids"][:3] == b["rounds"][0]["bids"][:3]  # seats 1-3 bid before 0


def test_fractional_wins():
    assert fractional_wins([5, 7, 7, 1]) == [0, 0.5, 0.5, 0]
    assert fractional_wins([3, 3, 3]) == [1 / 3] * 3

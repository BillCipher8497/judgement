"""Tests for experiments/final: TA+3 (lead highest sure loser at target) and the final agents/runner."""
import sys
from pathlib import Path

import numpy as np

from judgement import Rules, play_game
from judgement.cards import SUIT_OF
from judgement.policies.ta_plus import ta_plus_v2
from judgement.policies.ta_plus3 import TAPlus3Policy, ta_plus3
from judgement.policies.target_aware import TargetAwarePolicy, TrickModel, _bits, card_value

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "final"))


class Checker:
    """Plays TA+3 and asserts, at every decision, TA+3 == TA+ v2 unless the
    M9 rule fires (at target, leading, >= 2 sure losers) -> highest-value sure loser."""

    def __init__(self, ll=1):
        self.p3, self.p2 = ta_plus3(lead_loser=ll), ta_plus_v2()
        self.ll, self.fired, self.n = ll, 0, 0

    def bid(self, info, legal, rng):
        return self.p2.bid(info, legal, rng)

    def play(self, info, legal, rng):
        c3, c2 = self.p3.play(info, legal, rng), self.p2.play(info, legal, rng)
        self.n += 1
        bid = dict(info.bids)[info.seat]
        need = bid - info.tricks_won[info.seat]
        leading = info.trick_len == 0
        cards = _bits(legal)
        if len(cards) > 1 and need == 0 and leading:
            tm = TrickModel(info, True)
            losers = [c for c in cards if tm.p_win(c) <= 0.0]
            if self.ll == 2:
                losers = [c for c in losers if SUIT_OF[c] == SUIT_OF[c2]]
            if len(losers) >= 2:
                self.fired += 1
                assert c3 == max(losers, key=lambda c: (card_value(c, info.trump), -c))
                return c3
        assert c3 == c2
        return c3


def test_lead_loser_off_equals_ta_plus_v2():
    for seed in range(3):
        a = play_game(Rules(max_cards=7), [ta_plus3(lead_loser=0)] + [TargetAwarePolicy()] * 3, seed=seed, record_trace=False)
        b = play_game(Rules(max_cards=7), [ta_plus_v2()] + [TargetAwarePolicy()] * 3, seed=seed, record_trace=False)
        assert a.totals == b.totals


def test_rule_fires_only_where_specified():
    for ll in (1, 2):
        ch = Checker(ll)
        for seed in range(4):
            play_game(Rules(), [ch, TargetAwarePolicy(), ch, TargetAwarePolicy()], seed=seed, record_trace=False)
        assert ch.n > 100
        assert ch.fired > 0


def test_defaults_are_ta_plus_v2_components():
    p, q = TAPlus3Policy(), ta_plus_v2()
    for k in ("shed", "theorems", "lead_high", "win_high", "win_margin", "shed_margin", "surplus_duck", "scarce"):
        assert getattr(p, k) == getattr(q, k)
    assert p.lead_loser == 1


def test_final_agents_and_runner_deterministic(tmp_path, monkeypatch):
    import run as frun
    from agents import make_agent
    for k in ("ta_plus3", "ta_plus3:ll=2", "hybrid3:k=2,c=2", "ta"):
        assert make_agent(k) is not None
    monkeypatch.setattr(frun, "RES", tmp_path)
    frun.run("ta", "ta_plus3", 5, 2, "t1", 123)
    frun.run("ta", "ta_plus3", 5, 2, "t2", 123)
    a, b = np.load(tmp_path / "f_t1.npz"), np.load(tmp_path / "f_t2.npz")
    assert np.array_equal(a["totals"], b["totals"]) and list(a["game"]) == [5, 6]

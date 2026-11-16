"""Tests for the experiments/play additions: hybrid NT switch, TA+ with every
component off == TA, and the small-round oracle bidder."""
import sys
from pathlib import Path

import numpy as np

from judgement import Rules, play_game, play_round
from judgement.policies.hybrid import HybridPolicy
from judgement.policies.ta_plus import TAPlusPolicy
from judgement.policies.target_aware import TargetAwarePolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "play"))


def test_hybrid_nt_base_never_searches_in_nt():
    hyb = HybridPolicy(n_samples=4, pimc_cards=5, pimc_in_nt=False)
    play_round(Rules(), [hyb, TargetAwarePolicy(), TargetAwarePolicy(), TargetAwarePolicy()],
               seed=3, n_cards=5, trump=-1, dealer=0)
    assert hyb.n_pimc == 0
    play_round(Rules(), [hyb, TargetAwarePolicy(), TargetAwarePolicy(), TargetAwarePolicy()],
               seed=3, n_cards=5, trump=0, dealer=0)
    assert hyb.n_pimc > 0


def test_ta_plus_all_off_equals_ta():
    off = TAPlusPolicy(shed=False, surplus_duck=False, theorems=False, lead_high=0, scarce=False)
    for seed in range(3):
        a = play_game(Rules(max_cards=6), [off] + [TargetAwarePolicy()] * 3, seed=seed, record_trace=False)
        b = play_game(Rules(max_cards=6), [TargetAwarePolicy()] * 4, seed=seed, record_trace=False)
        assert a.totals == b.totals
        assert [r.tricks for r in a.rounds] == [r.tricks for r in b.rounds]


def test_small_round_oracle_bidder_legal_and_falls_back():
    from common import make_agent
    from judgement.bidding.model import ModelBidder
    from common import M_TA
    ag = make_agent("tap_oracle:K=4,nmax=2", seed=1)
    ta = make_agent("ta")
    res = play_game(Rules(max_cards=3), [ag, ta, ta, ta], seed=5, record_trace=False)
    assert len(res.rounds) == Rules(max_cards=3).n_rounds
    # n > nmax: identical to the model bidder
    from judgement.state import RoundState
    from judgement.engine import deal
    rules = Rules(max_cards=3)
    st = RoundState(rules, 3, 0, 3, deal(rules, 3, np.random.default_rng(0)))
    info = st.info_state(0)
    rng = np.random.default_rng(0)
    assert ag.bid(info, st.legal_bids(), rng) == ModelBidder(M_TA).bid(info, st.legal_bids(), rng)


def test_win_high_dumps_most_dangerous_sure_winner():
    from helpers import make_state
    from judgement.cards import card_str, parse_card as C
    from judgement.policies.ta_plus import ta_plus_v2
    st = make_state(["9S 2D 3D 4D", "5S AS 5D 6D", "6S 7D 8D 9D", "QS KS 2C 3C"], trump=1, dealer=3,
                    bids=[0, 1, 0, 1])
    for c in ("9S", "5S", "6S"):
        st.apply_card(C(c))
    rng = np.random.default_rng(0)
    info = st.info_state(3)
    assert card_str(TargetAwarePolicy().play(info, st.legal_cards(), rng)) == "QS"   # cheapest sure winner
    assert card_str(ta_plus_v2().play(info, st.legal_cards(), rng)) == "KS"          # dump the danger card
    assert ta_plus_v2().name == "ta_plus2" and ta_plus_v2().win_high == 2

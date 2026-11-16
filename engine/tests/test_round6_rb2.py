import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from judgement import Rules, play_game
from judgement.bidding.model import ModelBidder
from judgement.bidding2.sampler import BidLikelihood
from judgement.cards import NT
from judgement.engine import deal
from judgement.policies.human_guide import HumanBidder
from judgement.policies.hybrid_bi import _bool
from judgement.policies.hybrid_rb import BidderTypes
from judgement.policies.hybrid_rb2 import ContTypes, OpponentModel2, rb2_prior, RB_TYPES

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("r6_agents", ROOT / "experiments" / "round6" / "agents.py")
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)
RULES = Rules()
LIK = BidLikelihood(A.M_TA, RULES, eps=0.02)


def _bids(bidder, nd, seed):
    """synthetic sequential bids (hook for the dealer): [(H bool, n, trump, pos, prior, b, legal)]"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nd):
        r = int(rng.integers(RULES.n_rounds)); n = RULES.ladder()[r]; trump = RULES.trump_for_round(r)
        dealer = int(rng.integers(4)); hands = deal(RULES, n, rng); bids = []
        for i in range(4):
            s = (dealer + 1 + i) % 4
            legal = list(range(n + 1))
            if i == 3 and 0 <= n - sum(b for _, b in bids) <= n:
                legal.remove(n - sum(b for _, b in bids))
            info = SimpleNamespace(hand=hands[s], trump=trump, n_cards=n, seat=s, dealer=dealer, bids=list(bids),
                                   n_players=4, rules=RULES, history=[])
            b = bidder.bid(info, legal, rng) if not isinstance(bidder, HumanBidder) else bidder.bid(info, legal)
            out.append((_bool([hands[s]])[0], n, trump, i, [x for _, x in bids], b, legal))
            bids.append((s, b))
    return out


def test_probs_normalised_and_rb_columns_match():
    T = ContTypes(LIK)
    rng = np.random.default_rng(0)
    for n, trump, legal in ((5, 0, [0, 1, 2, 4, 5]), (1, NT, [0, 1]), (9, 2, list(range(10)))):
        H = _bool([deal(RULES, n, rng)[0] for _ in range(7)])
        tot = sum(T.probs(H, n, trump, 3, [1, 0, 1][: 3], b, legal) for b in legal)
        assert np.allclose(tot[:, :-1], 1.0) and np.allclose(tot[:, -1], 1.0)
        rb = BidderTypes(LIK, 0.02).probs(H, n, trump, 3, [1, 0, 1], legal[0], legal)
        p = T.probs(H, n, trump, 3, [1, 0, 1], legal[0], legal)
        assert np.allclose(p[:, :T.n_rb], rb[:, :-1]) and np.allclose(p[:, -1], rb[:, -1])
    assert len(T.names) == T.n_rb + T.n_grid + 1


def _learn(recs):
    T = ContTypes(LIK)
    om = OpponentModel2(T, rb2_prior(T))
    for H, n, trump, pos, prior, b, legal in recs:
        om.update(0, H, n, trump, pos, prior, b, legal)
    return om


def test_posterior_guide_goes_continuous_model_stays_model():
    og = _learn(_bids(HumanBidder(), 30, 1))
    sg = og.summary(0)
    assert sg["cont"] > 0.9 and sg["a"] < 1.0           # guide underbids the model
    om = _learn(_bids(ModelBidder(A.M_TA), 30, 2))
    sm = om.summary(0)
    assert sm["model"] + sm["noisy0.1"] > 0.5 and sm["unknown"] < 0.01
    assert set(RB_TYPES) <= set(sm) and abs(sum(sm[t] for t in RB_TYPES) + sm["cont"] + sm["unknown"] - 1) < 1e-3


def test_guide_agent_bids_with_human_bidder_and_rb2_game_legal():
    g = A.make_agent(A.GUIDE)
    assert isinstance(g.bidder, HumanBidder)
    rules = Rules(max_cards=3)
    rb2 = A.make_agent("hybrid_rb2:k=4,c=3")
    pols = [rb2, A.make_agent(A.GUIDE), A.make_agent("hyb_honour:k=4,c=3"), A.make_agent("hybrid:k=4,c=3")]
    res = play_game(rules, pols, seed=3, record_trace=False)
    assert len(res.rounds) == rules.n_rounds
    assert rb2.player.rounds_seen == rules.n_rounds - 1 and len(rb2.player.post_log) == 3 * (rules.n_rounds - 1)
    assert all(isinstance(x[2], dict) and "cont" in x[2] for x in rb2.player.post_log)

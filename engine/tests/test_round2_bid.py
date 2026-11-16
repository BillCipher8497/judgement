"""Tests for judgement.bidding2 (Phase 3 strategic bidding)."""
from pathlib import Path

import numpy as np
import pytest

from judgement import Rules
from judgement.bidding.features import hand_features
from judgement.bidding.model import BidModel, ModelBidder, info_features
from judgement.bidding2.features_vec import hand_features_batch
from judgement.bidding2.sampler import BidLikelihood, BidPhaseSampler, exact_posterior_worlds
from judgement.cards import NT
from judgement.engine import deal
from judgement.state import RoundState

MODEL = Path(__file__).resolve().parents[1] / "experiments/phase1/results/model_ta_final.json"


@pytest.fixture(scope="module")
def model():
    if not MODEL.exists():
        pytest.skip("model file missing")
    return BidModel.load(MODEL)


def _bool(mask):
    return np.array([(mask >> c) & 1 for c in range(52)], dtype=bool)


@pytest.mark.parametrize("trump", [0, 1, 2, 3, NT])
def test_vector_features_match_scalar(trump):
    rng = np.random.default_rng(trump + 5)
    rules = Rules()
    for n in (1, 2, 5, 8, 13):
        hands = deal(rules, n, rng)
        H = np.stack([_bool(h) for h in hands])
        prior = [int(x) for x in rng.integers(0, n + 1, size=2)]
        Fv = hand_features_batch(H, trump, n, 2, prior)
        for i, h in enumerate(hands):
            np.testing.assert_allclose(Fv[i], hand_features(h, trump, n, 2, prior), atol=1e-12)


def test_hard_likelihood_matches_model_bidder(model):
    rules = Rules()
    rng = np.random.default_rng(1)
    mb = ModelBidder(model)
    lik = BidLikelihood(model, rules, tau=0.0, eps=0.0)
    for n in (2, 5, 9):
        for _ in range(20):
            st = RoundState(rules, n, int(rng.integers(4)), 3, deal(rules, n, rng))
            info = st.info_state(0)
            b = mb.bid(info, st.legal_bids(), rng)
            ll = lik.log_lik(_bool(info.hand)[None], n, info.trump, 0, [], b)
            assert np.isfinite(ll[0])
            P = lik.probs(_bool(info.hand)[None], n, info.trump, 0, [])
            assert P[0].argmax() == b


def test_informed_worlds_reproduce_observed_bids(model):
    rules = Rules()
    rng = np.random.default_rng(2)
    mb = ModelBidder(model)
    lik = BidLikelihood(model, rules, tau=0.0, eps=0.0)
    n = 5
    st = RoundState(rules, n, 1, 3, deal(rules, n, rng))
    for _ in range(3):
        st.apply_bid(mb.bid(st.info_state(st.to_act), st.legal_bids(), rng))
    info = st.info_state(3)
    worlds, tot = exact_posterior_worlds(info, lik, 30, rng)
    assert len(worlds) == 30
    for w in worlds:
        hands = [w.get(p, info.hand) if p != 3 else info.hand for p in range(4)]
        assert bin(hands[0] | hands[1] | hands[2] | hands[3]).count("1") == 4 * n
        s2 = RoundState(rules, n, 1, 3, hands)
        for p in range(3):
            b = mb.bid(s2.info_state(p), s2.legal_bids(), rng)
            assert b == st.bids[p]
            s2.apply_bid(b)


def test_uniform_proposals_are_disjoint_and_sized(model):
    rules = Rules()
    rng = np.random.default_rng(3)
    st = RoundState(rules, 7, 0, 3, deal(rules, 7, rng))
    s = BidPhaseSampler(st.info_state(0), None)
    props = s.propose(200, rng)
    hero = st.hands[0]
    for row in props:
        ms = [int(x) for x in row]
        assert all(bin(m).count("1") == 7 for m in ms)
        assert (ms[0] & ms[1]) == 0 and (ms[0] & ms[2]) == 0 and (ms[1] & ms[2]) == 0
        assert all(m & hero == 0 for m in ms)


def test_soft_likelihood_normalised(model):
    rules = Rules()
    lik = BidLikelihood(model, rules, tau=1.5)
    rng = np.random.default_rng(4)
    hands = deal(rules, 6, rng)
    H = np.stack([_bool(h) for h in hands])
    P = lik.probs(H, 6, NT, 1, [2])
    np.testing.assert_allclose(P.sum(1), 1.0)
    lp = np.stack([lik.log_lik(H, 6, NT, 1, [2], b) for b in range(7)], 1)
    np.testing.assert_allclose(np.exp(lp), P, atol=1e-10)


GAMES = Path(__file__).resolve().parents[1] / "experiments/phase1/results/games_ta.npz"


@pytest.fixture(scope="module")
def bank():
    if not GAMES.exists():
        pytest.skip("games_ta.npz missing")
    from judgement.bidding2.winmax import ScoreBank
    return ScoreBank(GAMES)


def test_scorebank_future_sums(bank):
    g = 7
    for r in (0, 10, 23, 24):
        np.testing.assert_allclose(bank.future[g, r], bank.scores[g, r + 1:].sum(0))
    q = bank.hit_rate(24, 0, 3)
    assert 0.0 < q < 1.0


def test_winmax_far_ahead_plays_model_bid(model, bank):
    from judgement.bidding2.winmax import WinMaxBidder
    rules = Rules()
    rng = np.random.default_rng(5)
    wm = WinMaxBidder(model, rules, bank, k_final=3, L=500, seed=1)
    mb = ModelBidder(model)
    st = RoundState(rules, 13, rules.trump_for_round(24), rules.dealer_for_round(24), deal(rules, 13, rng))
    hero = st.to_act
    info = st.info_state(hero)
    legal = st.legal_bids()
    totals = [0.0] * 4
    totals[hero] = 10_000.0
    wm.ctx = (tuple(totals), 24)
    pw, _ = wm.pwin_by_bid(info, legal)
    assert all(abs(v - 1.0) < 1e-12 for v in pw.values())
    assert wm.bid(info, legal, rng) == mb.bid(info, legal, rng)
    wm.ctx = (tuple(totals), 5)          # not a final round -> model bid, no log
    wm.log.clear()
    assert wm.bid(info, legal, rng) == mb.bid(info, legal, rng) and not wm.log


def test_winmax_far_behind_needs_big_hit(model, bank):
    """Behind by more than any bid can recover in the last round -> P(win)=0 for all bids."""
    from judgement.bidding2.winmax import WinMaxBidder
    rules = Rules()
    rng = np.random.default_rng(6)
    wm = WinMaxBidder(model, rules, bank, k_final=1, L=300, seed=2)
    st = RoundState(rules, 13, rules.trump_for_round(24), rules.dealer_for_round(24), deal(rules, 13, rng))
    hero = st.to_act
    totals = [10_000.0] * 4
    totals[hero] = 0.0
    wm.ctx = (tuple(totals), 24)
    pw, _ = wm.pwin_by_bid(st.info_state(hero), st.legal_bids())
    assert all(v == 0.0 for v in pw.values())


def test_hook_setter_basics(model):
    from judgement.bidding2.hook import HookSetter
    rules = Rules()
    rng = np.random.default_rng(8)
    mb = ModelBidder(model)
    for n in (3, 6):
        st = RoundState(rules, n, 0, 3, deal(rules, n, rng))
        for _ in range(2):
            st.apply_bid(mb.bid(st.info_state(st.to_act), st.legal_bids(), rng))
        info = st.info_state(2)
        legal = st.legal_bids()
        assert HookSetter(model, rules, 0.0).bid(info, legal, rng) == mb.bid(info, legal, rng)
        hs = HookSetter(model, rules, 100.0, K=50, seed=3)
        b = hs.bid(info, legal, rng)
        assert b in legal and hs.last["loss_b"] >= hs.last["loss_bm"] - 1e-12 and hs.last["loss_b"] >= 0
        # not the bidder before the dealer -> model bid
        info0 = RoundState(rules, n, 0, 3, deal(rules, n, rng)).info_state(0)
        assert hs.bid(info0, list(range(n + 1)), rng) == mb.bid(info0, list(range(n + 1)), rng)


def test_onecard_winner_rule():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "experiments/round2_bid/p4_onecard.py"
    import sys
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("p4_onecard", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from judgement.cards import parse_card
    D = np.array([[parse_card(c) for c in row] for row in
                  (["KH", "AH", "2S", "QH"], ["KH", "AH", "3D", "QH"], ["2C", "AH", "3D", "QH"])])
    np.testing.assert_array_equal(mod.winners(D, 0), [2, 1, 0])
    np.testing.assert_array_equal(mod.winners(D, NT), [1, 1, 0])

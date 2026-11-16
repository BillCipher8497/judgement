"""Round 3: Hybrid-BI (bid-informed PIMC worlds) and the ties-as-win win-max bidder."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from judgement import Rules, play_game
from judgement.bidding.model import Agent, BidModel, ModelBidder
from judgement.bidding2.sampler import BidLikelihood
from judgement.engine import deal
from judgement.policies.hybrid import HybridPolicy
from judgement.policies.hybrid_bi import BidInformedWorlds, HybridBIPolicy
from judgement.policies.ta_plus import TAPlusPolicy
from judgement.state import RoundState, state_from_info

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "experiments/phase1/results/model_ta_final.json"


@pytest.fixture(scope="module")
def model():
    if not MODEL.exists():
        pytest.skip("model file missing")
    return BidModel.load(MODEL)


def _midplay(model, n, seed, plays, dealer=1, trump=0):
    """Round with ModelBidder bids, then `plays` cards by TA+; returns (state, info of the seat to act)."""
    rules = Rules()
    rng = np.random.default_rng(seed)
    st = RoundState(rules, n, trump, dealer, deal(rules, n, rng))
    mb, tp = ModelBidder(model), TAPlusPolicy()
    while st.phase == 0:
        st.apply_bid(mb.bid(st.info_state(st.to_act), st.legal_bids(), rng))
    for _ in range(plays):
        p = st.to_act
        st.apply_card(tp.play(st.info_state(p), st.legal_cards(), rng))
    return st, st.info_state(st.to_act)


def test_observations_hook_and_positions(model):
    st, info = _midplay(model, 5, 1, 0, dealer=1)
    obs = BidInformedWorlds.observations(info)
    assert len(obs) == 3 and info.seat not in [o[0] for o in obs]
    for p, pos, prior, b, legal in obs:
        assert pos == (p - 1 - 1) % 4
        assert b in legal
        if pos == 3:                         # dealer: hook-forbidden bid removed
            assert (5 - sum(prior)) not in legal or not (0 <= 5 - sum(prior) <= 5)


@pytest.mark.parametrize("n,plays,seed", [(5, 0, 2), (5, 6, 3), (8, 13, 4), (3, 2, 5)])
def test_true_world_has_max_weight_and_worlds_consistent(model, n, plays, seed):
    """Opponents bid with ModelBidder, so the true deal satisfies every hard-likelihood constraint
    (their hands at bid time = current + played); sampled worlds respect played cards and voids."""
    st, info = _midplay(model, n, seed, plays)
    lik = BidLikelihood(model, Rules(), tau=0.0, eps=0.02)
    bw = BidInformedWorlds(lik)
    true = {p: st.hands[p] for p in range(4) if p != info.seat}
    lw_true = bw.log_weights(info, [true])[0]
    best = sum(np.log(0.98 + 0.02 / len(o[4])) for o in bw.observations(info))
    assert lw_true == pytest.approx(best)
    worlds, s = bw.sample(info, 16, np.random.default_rng(seed))
    assert len(worlds) == 16 and s["ess"] > 0 and s["M"] >= 64
    for w in worlds:
        state_from_info(info, w)             # raises on an inconsistent world


def test_flat_likelihood_gives_full_ess(model):
    st, info = _midplay(model, 5, 7, 3)
    lik = BidLikelihood(model, Rules(), tau=0.0, eps=1.0)     # eps=1: L independent of the hand
    worlds, s = BidInformedWorlds(lik, batch=64).sample(info, 16, np.random.default_rng(0))
    assert s["ess"] == pytest.approx(64) and s["M"] == 64


def test_informed_places_cards_better_than_uniform(model):
    """Posterior worlds put more of the opponents' true cards in the right hand than uniform ones."""
    lik = BidLikelihood(model, Rules(), tau=0.0, eps=0.02)
    bw = BidInformedWorlds(lik)
    from judgement.sampling import HandSampler
    gi, gu = [], []
    for seed in range(30):
        st, info = _midplay(model, 6, 100 + seed, 4)
        true = {p: st.hands[p] for p in range(4) if p != info.seat}
        tot = sum(h.bit_count() for h in true.values())
        rng = np.random.default_rng(seed)
        wi, _ = bw.sample(info, 32, rng)
        hs = HandSampler(info)
        wu = [hs.sample(rng) for _ in range(32)]
        gi.append(np.mean([sum((w[p] & true[p]).bit_count() for p in true) / tot for w in wi]))
        gu.append(np.mean([sum((w[p] & true[p]).bit_count() for p in true) / tot for w in wu]))
    assert np.mean(gi) > np.mean(gu) + 0.02


def test_uniform_mode_equals_hybrid(model):
    """informed=False reproduces HybridPolicy exactly (same RNG draws, same solver)."""
    rules = Rules(max_cards=4)
    lik = BidLikelihood(model, Rules(), tau=0.0, eps=0.02)
    mk = [lambda: Agent(ModelBidder(model), HybridPolicy(n_samples=4, pimc_cards=3, base=TAPlusPolicy())),
          lambda: Agent(ModelBidder(model), HybridBIPolicy(lik, n_samples=4, pimc_cards=3, informed=False))]
    a = play_game(rules, [mk[0]() for _ in range(4)], seed=11, record_trace=False)
    b = play_game(rules, [mk[1]() for _ in range(4)], seed=11, record_trace=False)
    assert a.totals == b.totals


def test_hybrid_bi_full_round_legal(model):
    rules = Rules(max_cards=5)
    lik = BidLikelihood(model, Rules(), tau=0.0, eps=0.02)
    pol = HybridBIPolicy(lik, n_samples=4, pimc_cards=4)
    ags = [Agent(ModelBidder(model), pol)] + [Agent(ModelBidder(model), TAPlusPolicy()) for _ in range(3)]
    res = play_game(rules, ags, seed=3, record_trace=False)     # engine checks legality
    assert len(res.totals) == 4 and pol.log and all(s["ess"] >= 1 for s in pol.log)


def _r2():
    spec = importlib.util.spec_from_file_location("r2_main", ROOT / "experiments/round3_play/r2_main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ties_as_win_objective_dominates_split(model):
    r2 = _r2()
    from judgement.bidding2.winmax import ScoreBank, WinMaxBidder
    gpath = ROOT / "experiments/phase1/results/games_ta.npz"
    if not gpath.exists():
        pytest.skip("score bank missing")
    bank, rules = ScoreBank(gpath), Rules()
    st = RoundState(rules, 2, 3, 23 % 4, deal(rules, 2, np.random.default_rng(5)))
    info = st.info_state(st.to_act)
    legal = st.legal_bids()
    ctx = ((300.0, 300.0, 290.0, 280.0), 23)
    a, b = r2.WinMaxTieBidder(model, rules, bank, 3, L=500, seed=9), WinMaxBidder(model, rules, bank, 3, L=500, seed=9)
    a.ctx = b.ctx = ctx
    pa, _ = a.pwin_by_bid(info, legal)
    pb, _ = b.pwin_by_bid(info, legal)
    for x in legal:
        assert pa[x] >= pb[x] - 1e-12
    assert r2.w_tie([5, 7, 7, 1], 1) == 1.0 and r2.w_split([5, 7, 7, 1], 1) == 0.5 and r2.w_tie([5, 7, 7, 1], 0) == 0.0

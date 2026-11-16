import numpy as np

from helpers import make_state
from judgement import Rules, play_round
from judgement.bidding.features import D, hand_features
from judgement.bidding.model import Agent, BidModel, CondLogit, ModelBidder
from judgement.bidding.sim import bid_conditional_dists, simulate_rounds
from judgement.cards import NT, parse_hand
from judgement.policies.target_aware import TargetAwarePolicy

TA = TargetAwarePolicy()


def test_features_shape_and_values():
    f = hand_features(parse_hand("AS KS QS 2H 3H AD"), 0, 6, 1, [2], 4)
    assert f.shape == (D,)
    assert f[6] == 3 and f[7] == 1 and f[8] == 1 and f[13] == 1     # trump len, tA, tK, side aces
    fnt = hand_features(parse_hand("AS KS QS 2H 3H AD"), NT, 6, 0, [], 4)
    assert fnt[6] == 0 and fnt[13] == 2


def test_condlogit_learns_bid_pull():
    rng = np.random.default_rng(0)
    n = 5
    X = np.column_stack([np.ones(3000), rng.normal(size=(3000, D - 1))])
    bids = rng.integers(0, n + 1, 3000)
    tricks = np.where(rng.random(3000) < 0.7, bids, rng.integers(0, n + 1, 3000))
    m = CondLogit(n).fit(X, bids, tricks)
    assert m.gamma[0] > 1.0
    P = m.dist_all_bids(X[0])
    assert P.shape == (n + 1, n + 1) and np.allclose(P.sum(1), 1)
    assert all(P[b].argmax() == b for b in range(n + 1))


def _tiny_model():
    data = {k: [] for k in ("x", "bid", "tricks", "n", "trump")}
    d = simulate_rounds(Rules(), TA, 3, 300, 0, fallback_bidder=TA, explore_prob=0.5)
    models = {}
    for nt in (False, True):
        m = (d["trump"] == -1) == nt
        models[(3, nt)] = CondLogit(3).fit(d["x"][m], d["bid"][m], d["tricks"][m])
    return BidModel(models), d


def test_model_bidder_respects_hook_and_roundtrips(tmp_path):
    bm, d = _tiny_model()
    assert d["x"].shape[1] == D and set(np.unique(d["pos"])) == {0, 1, 2, 3}
    p = tmp_path / "m.json"
    bm.save(p)
    bm2 = BidModel.load(p)
    x = d["x"][0]
    assert np.allclose(bm.get(3, 0).dist_all_bids(x), bm2.get(3, 0).dist_all_bids(x))
    ag = Agent(ModelBidder(bm), TA)
    for s in range(30):
        rr = play_round(Rules(), [ag] * 4, seed=s, n_cards=3, trump=s % 4 if s % 5 else NT, dealer=s % 4)
        assert sum(rr.bids) != 3                      # hook rule held (engine also checks)


def test_bid_conditional_dists_crn_and_normalised():
    bm, _ = _tiny_model()
    ag = Agent(ModelBidder(bm), TA)
    st = make_state(["AS KS 2H", "QS 3H 4H", "JS 5H 6H", "TS 7H 8H"], trump=0, dealer=3)
    out = bid_conditional_dists(st, 0, [ag] * 4, [0, 1, 2, 3], 50, np.random.default_rng(1))
    for b, p in out.items():
        assert np.isclose(p.sum(), 1.0) and len(p) == 4
    # deterministic given rng seed
    out2 = bid_conditional_dists(st, 0, [ag] * 4, [0, 1, 2, 3], 50, np.random.default_rng(1))
    assert all(np.array_equal(out[b], out2[b]) for b in out)

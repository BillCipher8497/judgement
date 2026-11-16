import importlib.util
from pathlib import Path

import numpy as np

from judgement import Rules, play_game

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("r4_agents", ROOT / "experiments" / "round4" / "agents.py")
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)


class _Rec:
    """wrap a bidder, record (legal, model bid, actual bid)"""
    def __init__(self, nb):
        self.nb, self.rec = nb, []

    def bid(self, info, legal, rng):
        b0 = self.nb.mb.bid(info, legal, rng)
        b = self.nb.bid(info, legal, rng)
        self.rec.append((list(legal), b0, b))
        return b


def test_noisy_bidder_legal_and_rate():
    from judgement.policies.target_aware import TargetAwarePolicy
    from judgement.bidding.model import Agent
    recs = []
    for g in range(12):
        ags = []
        for p in range(4):
            r = _Rec(A.NoisyBidder(A.M_TA, 0.3, seed=1000 * g + p))
            recs.append(r)
            ags.append(Agent(r, TargetAwarePolicy()))
        play_game(Rules(), ags, seed=g, record_trace=False)
    rows = [x for r in recs for x in r.rec]
    assert all(b in legal for legal, _, b in rows)
    assert all(abs(b - b0) <= 1 for _, b0, b in rows)
    frac = np.mean([b != b0 for _, b0, b in rows])
    assert 0.2 < frac < 0.36, frac          # 0.3 minus the rare both-neighbours-illegal case


def test_noisy_bidder_p0_is_model_and_seeded():
    from judgement.bidding.model import ModelBidder
    nb0 = A.NoisyBidder(A.M_TA, 0.0, seed=3)
    a, b = A.NoisyBidder(A.M_TA, 0.3, seed=5), A.NoisyBidder(A.M_TA, 0.3, seed=5)
    assert [a.rng.random() for _ in range(5)] == [b.rng.random() for _ in range(5)]
    r1 = play_game(Rules(), [A.Agent(nb0, A.ta_plus_from({})) for _ in range(4)], seed=7, record_trace=False)
    r2 = play_game(Rules(), [A.Agent(ModelBidder(A.M_TA), A.ta_plus_from({})) for _ in range(4)], seed=7,
                   record_trace=False)
    assert [list(x.bids) for x in r1.rounds] == [list(x.bids) for x in r2.rounds]


def test_pool_kinds_construct():
    for k in ("hyb_noisy:k=4,c=2,p=0.3", "hyb_honour:k=4,c=2", "hybrid_bi:k=4,c=2,eps=0.1", "hybrid:k=4,c=2"):
        assert A.make_agent(k, seed=1) is not None

"""Offline online-learning check: one opponent bids S sequences of 25 rounds (the real ladder order, random seat/dealer/
deal, other seats bid with the same bidder); predictive log-lik of its bid BEFORE each update under Hybrid-RB's
OpponentModel vs Hybrid-RB2's OpponentModel2 (default priors).  python online_study.py S SEED -> results/online_study.json"""
import json
import sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/play"))
from common import M_TA  # noqa
from judgement import Rules  # noqa
from judgement.engine import deal  # noqa
from judgement.bidding.model import ModelBidder  # noqa
from judgement.bidding2.sampler import BidLikelihood  # noqa
from judgement.policies.human_guide import HumanBidder  # noqa
from judgement.policies.hybrid_bi import _bool  # noqa
from judgement.policies.hybrid_rb import BidderTypes, OpponentModel  # noqa
from judgement.policies.hybrid_rb2 import ContTypes, OpponentModel2, rb2_prior  # noqa

rules = Rules(); lik = BidLikelihood(M_TA, rules, eps=0.02)
S, seed = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(seed)
T2 = ContTypes(lik)
out = {}
for nm, bidder in (("guide", HumanBidder()), ("model", ModelBidder(M_TA))):
    LL = np.zeros((S, rules.n_rounds, 2))
    for s in range(S):
        rb, rb2 = OpponentModel(BidderTypes(lik)), OpponentModel2(T2, rb2_prior(T2))
        me = int(rng.integers(4))
        for r in range(rules.n_rounds):
            n, trump, dealer = rules.ladder()[r], rules.trump_for_round(r), rules.dealer_for_round(r)
            hands = deal(rules, n, rng); bids = []
            for i in range(4):
                p = (dealer + 1 + i) % 4
                legal = list(range(n + 1))
                if i == 3 and 0 <= n - sum(b for _, b in bids) <= n:
                    legal.remove(n - sum(b for _, b in bids))
                info = SimpleNamespace(hand=hands[p], trump=trump, n_cards=n, seat=p, dealer=dealer, bids=list(bids),
                                       n_players=4, rules=rules, history=[])
                b = bidder.bid(info, legal, None)
                if p == me:
                    H = _bool([hands[p]]); args = (n, trump, i, [x for _, x in bids], b, legal)
                    LL[s, r] = [rb.log_lik(0, H, *args)[0], rb2.log_lik(0, H, *args)[0]]
                    rb.update(0, H[0], *args); rb2.update(0, H[0], *args)
                bids.append((p, b))
    m = LL.mean(0)
    seg = lambda a, b: [[float(LL[:, a:b, k].sum(1).mean() / (b - a)), float(1.96 * LL[:, a:b, k].mean(1).std(ddof=1) / np.sqrt(S))] for k in (0, 1)]
    d = LL[:, :, 1] - LL[:, :, 0]
    out[nm] = {"S": S, "rb_rb2_ll_per_bid_r1_8": seg(0, 8), "r9_25": seg(8, 25),
               "rb2_minus_rb_r1_8": ci1 if (ci1 := [float(d[:, :8].mean()), float(1.96 * d[:, :8].mean(1).std(ddof=1) / np.sqrt(S))]) else None,
               "rb2_minus_rb_r9_25": [float(d[:, 8:].mean()), float(1.96 * d[:, 8:].mean(1).std(ddof=1) / np.sqrt(S))],
               "P_exact_rb_rb2_r9_25": [float(np.exp(LL[:, 8:, 0]).mean()), float(np.exp(LL[:, 8:, 1]).mean())]}
    print(nm, json.dumps(out[nm]))
(Path(__file__).parent / "results" / "online_study.json").write_text(json.dumps(out, indent=1))

"""Offline: how well does the RB2 continuous family fit guide (HumanBidder) bids?  Synthetic deals over the ladder
(n, trump as in the real schedule), sequential guide bids with hook; per-bid log-lik of each grid component
(centre 'mu' vs 'm') and of the RB types; reports best single component and posterior-mixture log-lik / bid.
python fit_study.py NDEALS SEED"""
import sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/play"))
from common import M_TA  # noqa
from judgement import Rules  # noqa
from judgement.engine import deal  # noqa
from judgement.bidding2.sampler import BidLikelihood  # noqa
from judgement.policies.human_guide import HumanBidder  # noqa
from judgement.policies.hybrid_bi import _bool  # noqa
from judgement.policies.hybrid_rb2 import ContTypes  # noqa

rules = Rules(); hb = HumanBidder(); lik = BidLikelihood(M_TA, rules, eps=0.02)
N = 4
ND, seed = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(seed)
SL = tuple(float(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0.8, 1.0, 1.2)
TT = {c: ContTypes(lik, centre=c, slopes=SL) for c in ("mu", "m")}
LL = {c: [] for c in TT}
for k in range(ND):
    r = int(rng.integers(rules.n_rounds)); n = rules.ladder()[r]
    trump = rules.trump_for_round(r)
    dealer = int(rng.integers(N)); hands = deal(rules, n, rng)
    bids = []
    for i in range(N):
        s = (dealer + 1 + i) % N
        legal = list(range(n + 1))
        if i == N - 1:
            f = n - sum(b for _, b in bids)
            if 0 <= f <= n: legal.remove(f)
        info = SimpleNamespace(hand=hands[s], trump=trump, n_cards=n, seat=s, dealer=dealer, bids=list(bids), n_players=N)
        b = hb.bid(info, legal)
        H = _bool([hands[s]])
        for c, T in TT.items():
            LL[c].append(np.log(T.probs(H, n, trump, i, [x for _, x in bids], b, legal)[0]))
        bids.append((s, b))
for c, T in TT.items():
    X = np.array(LL[c]); tot = X.sum(0); j = tot.argmax()
    lp = np.log(np.full(X.shape[1], 1 / X.shape[1])) + tot; lp -= lp.max()
    print(c, "bids", len(X), "best comp", T.names[j], "ll/bid %.3f" % (tot[j] / len(X)),
          "| model %.3f honour %.3f noisy.5 %.3f unknown %.3f" % tuple(tot[[0, 4, 3, -1]] / len(X)),
          "| P(best bid exact-ish)", np.exp(X[:, j]).mean().round(3))
    o = np.argsort(-tot)[:8]; print("  top:", [(T.names[i], round(tot[i] / len(X), 3)) for i in o])

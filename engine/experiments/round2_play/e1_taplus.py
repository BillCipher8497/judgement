"""E1 robustness: does the greedy-optimal bid change under target-aware play
when play is TA+ instead of TA?  Exactly Phase 1's E1 protocol (e1_shift.run):
n=13, hero seat 1, dealer 0, trump S, seed 5000+n for deals and 5001+n for
sampling, Sg=400 Greedy worlds, St=100 worlds per candidate bid, candidates =
greedy-oracle bid +-2 and TA-model bid +-1.  The first H hands are the same
hands as Phase 1 (TA play), so the comparison is paired.  Neither TA nor TA+
uses its rng, so the Greedy distributions are identical to Phase 1's.

python e1_taplus.py H  -> results/e1_taplus_n13.json
"""
import sys
import time

import numpy as np

from common import M_GR, M_TA, dump, make_agent
from judgement import Rules
from judgement.bidding.model import info_features
from judgement.bidding.sim import bid_conditional_dists, bid_free_dist, model_bid
from judgement.engine import deal
from judgement.state import RoundState

HERO = 1


def ev_of(rules, d, b):
    return float(sum(p * rules.score(b, t) for t, p in enumerate(d)))


def run(n, H, play="ta_plus", S_g=400, S_t=100):
    rules = Rules()
    seed = 5000 + n
    rng_deal = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    ag_t = [make_agent(play)] * 4
    ag_g = [make_agent("greedy")] * 4
    recs = []
    S_cache = {}
    t0 = time.time()
    for i in range(H):
        hands = deal(rules, n, rng_deal)
        st = RoundState(rules, n, 0, 0, list(hands))
        legal = st.legal_bids()
        dg = bid_free_dist(st, HERO, ag_g, S_g, rng)
        evg = {b: ev_of(rules, dg, b) for b in legal}
        bg = max(evg, key=lambda b: (evg[b], -b))
        bm, _ = model_bid(M_TA, rules, info_features(st.info_state(HERO)), n, 0, legal, S_cache)
        cands = sorted({b for b in legal if abs(b - bg) <= 2 or abs(b - bm) <= 1})
        dt = bid_conditional_dists(st, HERO, ag_t, cands, S_t, rng)
        evt = {b: ev_of(rules, dt[b], b) for b in cands}
        bt = max(evt, key=lambda b: (evt[b], -b))
        recs.append({"i": i, "b_greedy": bg, "ev_greedy": evg[bg], "b_model_ta": bm, "b_t": bt, "ev_t": evt[bt],
                     "t_ev_at_greedy_bid": evt.get(bg), "cands": cands,
                     "dists": {int(b): dt[b].tolist() for b in cands}})
        if (i + 1) % 50 == 0:
            print("e1", i + 1, round(time.time() - t0, 1), flush=True)
            dump(recs, f"e1_{play}_n{n}.json")
    dump(recs, f"e1_{play}_n{n}.json")


if __name__ == "__main__":
    run(13, int(sys.argv[1]))

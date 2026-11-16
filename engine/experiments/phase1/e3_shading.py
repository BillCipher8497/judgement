"""E3 (greedy part): bid shading b_oracle - round(E[t]) over a broad grid of
situations (n = 1..13, trump suit vs NT, bidding position 0..3, prior bids from
the greedy-fitted model).  Greedy play is bid-agnostic, so P(t) is well defined.
The TA part is computed from oracle_compare_ta.json and e1_n*.json in analyze.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import info_features  # noqa: E402
from judgement.bidding.sim import bid_free_dist, model_bid  # noqa: E402
from judgement.cards import NT  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.state import RoundState  # noqa: E402

from common import agents, dump, load_model  # noqa: E402


def main(per_cell=40, S=300, seed=4242):
    rules = Rules()
    m = load_model("greedy")
    ag = [agents("greedy", m)] * 4
    rng = np.random.default_rng(seed)
    S_cache = {}
    recs = []
    t0 = time.time()
    for n in range(1, 14):
        for tr in (0, NT):
            for pos in range(4):
                hero = (pos + 1) % 4
                for _ in range(per_cell):
                    st = RoundState(rules, n, tr, 0, deal(rules, n, rng))
                    while st.to_act != hero:
                        info = st.info_state(st.to_act)
                        b, _ = model_bid(m, rules, info_features(info), n, tr, st.legal_bids(), S_cache)
                        st.apply_bid(b)
                    legal = st.legal_bids()
                    d = bid_free_dist(st, hero, ag, S, rng)
                    ev = {b: float(sum(p * rules.score(b, t) for t, p in enumerate(d))) for b in legal}
                    bo = max(ev, key=lambda b: (ev[b], -b))
                    bm, _ = model_bid(m, rules, info_features(st.info_state(hero)), n, tr, legal, S_cache)
                    recs.append({"n": n, "nt": tr == NT, "pos": pos, "dist": d.tolist(), "b_oracle": bo,
                                 "b_model": bm, "prior_sum": sum(b for _, b in st.bid_seq),
                                 "hooked": len(legal) == n})
        print(n, round(time.time() - t0, 1), flush=True)
        dump(recs, "e3_greedy_grid.json")
    dump(recs, "e3_greedy_grid.json")


if __name__ == "__main__":
    main()

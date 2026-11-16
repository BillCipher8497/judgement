"""E1 follow-up: hands whose TA oracle bid landed on an interior edge of the
candidate window are re-solved with the window widened by 2 on that side (all
candidates recomputed on fresh common worlds, same sample size).  Updates
results/e1_n{n}.json in place (keeps the original under 'orig_*')."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.sim import bid_conditional_dists  # noqa: E402
from judgement.engine import deal  # noqa: E402

from common import RES, agents, dump, load_model  # noqa: E402
from e1_shift import HERO, ev_of, hero_state  # noqa: E402


def main(S_t=100):
    rules = Rules()
    m_ta = load_model("ta")
    ag = [agents("ta", m_ta)] * 4
    for n in (13, 8, 5):
        p = RES / f"e1_n{n}.json"
        if not p.exists():
            continue
        R = json.load(open(p))
        rng_deal = np.random.default_rng(5000 + n)
        rng = np.random.default_rng(99000 + n)
        fixed = 0
        for r in R:
            hands = deal(rules, n, rng_deal)
            if not r["edge"]:
                continue
            st = hero_state(rules, n, hands)
            legal = st.legal_bids()
            for _ in range(3):                      # widen up to 3 times
                cands = sorted(int(b) for b in r["ta"])
                lo, hi = cands[0], cands[-1]
                if r["b_ta"] == lo:
                    lo = max(0, lo - 2)
                if r["b_ta"] == hi:
                    hi = min(n, hi + 2)
                new = [b for b in legal if lo <= b <= hi]
                dt = bid_conditional_dists(st, HERO, ag, new, S_t, rng)
                evt = {b: ev_of(rules, dt[b], b) for b in new}
                bt = max(evt, key=lambda b: (evt[b], -b))
                r.setdefault("orig_b_ta", r["b_ta"])
                r["ta"] = {int(b): dt[b].tolist() for b in new}
                r["b_ta"], r["ev_ta"] = bt, evt[bt]
                r["ta_ev_at_greedy_bid"] = evt.get(r["b_greedy"])
                r["edge"] = bt in (new[0], new[-1]) and not (bt == 0 or bt == n)
                if not r["edge"]:
                    break
            fixed += 1
        dump(R, f"e1_n{n}.json")
        print(n, "re-solved", fixed, "still at edge", sum(r["edge"] for r in R), flush=True)


if __name__ == "__main__":
    main()

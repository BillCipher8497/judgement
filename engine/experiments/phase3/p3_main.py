"""P3: hook-setting by the bidder before the dealer (position 2, seat 2; dealer = seat 3).

Real deals (deal rng = default_rng([SEED, n, i])), trump cycles S,H,D,C,NT.  Baseline:
everyone ModelBidder + TA.  Treatment: seat 2 uses HookSetter(delta); everything else
unchanged.  Play is deterministic, so per-deal paired differences are exact CRN and are
nonzero only when seat 2's bid changes.  Reports seat-2 and dealer score change, and
how often the dealer's model-preferred bid is the forbidden one."""
from __future__ import annotations

import argparse
import time

import numpy as np

from common3 import dump, fmt, load_model
from judgement import Rules
from judgement.bidding.model import Agent, ModelBidder
from judgement.bidding2.hook import HookSetter
from judgement.cards import NT
from judgement.engine import deal, run_round
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.state import RoundState

TRUMPS = [0, 1, 2, 3, NT]


def dealer_hooked(rules, mb, n, tr, hands, bids):
    """Was the dealer's unconstrained model-best bid the forbidden one?  Returns
    (hooked, dealer EV loss under the model)."""
    s = RoundState(rules, n, tr, 3, hands)
    for b in bids[:3]:
        s.apply_bid(b)
    ev, _ = mb.evs(s.info_state(3))
    f = n - sum(bids[:3])
    best = int(np.argmax(ev))
    if not 0 <= f <= n or best != f:
        return [0, 0.0]
    ev2 = ev.copy()
    ev2[f] = -np.inf
    return [1, float(ev[best] - ev2.max())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2,3,5,8,13")
    ap.add_argument("--deals", type=int, default=1200)
    ap.add_argument("--deltas", default="1,3")
    ap.add_argument("--K", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rules = Rules()
    model = load_model()
    ta = TargetAwarePolicy()
    mb = ModelBidder(model)
    base = Agent(mb, ta)
    deltas = [float(x) for x in a.deltas.split(",")]
    rows = []
    t0 = time.time()
    rng0 = np.random.default_rng(0)
    for n in [int(x) for x in a.ns.split(",")]:
        for i in range(a.deals):
            rng = np.random.default_rng([a.seed, n, i])
            hands = deal(rules, n, rng)
            tr = TRUMPS[i % 5]
            st = RoundState(rules, n, tr, 3, hands)
            run_round(st, [base] * 4, [rng0] * 4, check=False)
            bsc = st.scores()
            row = dict(n=n, i=i, trump=tr, base_bids=list(st.bids), base_scores=bsc,
                       base_hooked=dealer_hooked(rules, mb, n, tr, hands, st.bids))
            for dlt in deltas:
                hs = HookSetter(model, rules, dlt, K=a.K, seed=int(np.random.SeedSequence([a.seed, n, i, 7]).generate_state(1)[0]))
                ags = [base] * 4
                ags[2] = Agent(hs, ta)
                s2 = RoundState(rules, n, tr, 3, hands)
                run_round(s2, ags, [rng0] * 4, check=False)
                sc = s2.scores()
                row[f"d{dlt:g}"] = dict(bids=list(s2.bids), scores=sc, info=hs.last,
                                        hooked=dealer_hooked(rules, mb, n, tr, hands, s2.bids))
            rows.append(row)
        print(f"n={n} done {time.time()-t0:.0f}s", flush=True)
        dump({"args": vars(a), "rows": rows}, f"p3_raw{a.tag}.json")
    print("total", time.time() - t0)


if __name__ == "__main__":
    main()

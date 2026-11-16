"""M1b: bid-conditional test of the independent-card model under all-TA play.

Under TA the hero's trick law depends on its own bid b. For each hand and each
candidate b (TA-model bid ±2, clipped), run R rollouts on common sampled worlds with
the hero forced to bid b, recording the trick histogram and per-card win counts.
Emp-oracle bid = argmax_b (10+b) P_b(b). PB-oracle bid = argmax_b (10+b) PB(p^(b))(b).
Hands are the first H hands of the m1_sim deal stream (same seeds) for comparability.
Output: math/results/m1_ta_oracle_raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "phase1"))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.bidding.sim import FixedBid  # noqa: E402
from judgement.cards import NT  # noqa: E402
from judgement.engine import deal, run_round  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.sampling import HandSampler  # noqa: E402
from judgement.state import RoundState, state_from_info  # noqa: E402

from common import agents, load_model  # noqa: E402

HERO = 1
OUT = ROOT / "math" / "results" / "m1_ta_oracle_raw.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5,8,13")
    ap.add_argument("--hands", type=int, default=300)
    ap.add_argument("--R", type=int, default=200)
    a = ap.parse_args()
    rules = Rules()
    m = load_model("ta")
    base = [agents("ta", m)] * 4
    recs = json.load(open(OUT)) if OUT.exists() else []
    done = {(r["n"], r["trump"], r["i"]) for r in recs}
    mb = ModelBidder(m)
    for n in map(int, a.ns.split(",")):
        for tr in (0, NT):
            rng_deal = np.random.default_rng(91000 + 10 * n + (tr + 1))
            t0 = time.time()
            for i in range(a.hands):
                hands = deal(rules, n, rng_deal)
                if (n, tr, i) in done:
                    continue
                st = RoundState(rules, n, tr, 0, list(hands))
                info = st.info_state(HERO)
                bm = mb.bid(info, st.legal_bids(), None)
                cands = [b for b in range(bm - 2, bm + 3) if 0 <= b <= n]
                rng = np.random.default_rng([93000, n, tr + 1, i])
                sampler = HandSampler(info)
                cards = [c for c in range(52) if hands[HERO] >> c & 1]
                idx = {c: j for j, c in enumerate(cards)}
                cnt = {b: np.zeros(n + 1, int) for b in cands}
                wins = {b: np.zeros(n, int) for b in cands}
                pols = {b: [base[0], Agent(FixedBid(b), TargetAwarePolicy()), base[0], base[0]] for b in cands}
                for _ in range(a.R):
                    hid = sampler.sample(rng)
                    for b in cands:
                        s2 = state_from_info(info, hid)
                        run_round(s2, pols[b], [rng] * 4, check=False)
                        cnt[b][s2.tricks_won[HERO]] += 1
                        h = s2.history
                        for k, w in enumerate(s2.trick_winners):
                            if w == HERO:
                                for p, c in h[4 * k:4 * k + 4]:
                                    if p == HERO:
                                        wins[b][idx[c]] += 1
                recs.append({"n": n, "trump": tr, "i": i, "b_model": bm, "R": a.R,
                             "counts": {str(b): cnt[b].tolist() for b in cands},
                             "wins": {str(b): wins[b].tolist() for b in cands}})
                if (i + 1) % 25 == 0:
                    json.dump(recs, open(OUT, "w"))
                    print(n, tr, i + 1, round(time.time() - t0), flush=True)
            json.dump(recs, open(OUT, "w"))


if __name__ == "__main__":
    main()

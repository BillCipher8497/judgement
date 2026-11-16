"""M5 data: per-rollout card-win masks for the hero (first bidder), TA and TA+ play.

Same deals and (for TA) the same rollout seeds as m1_sim.py, so TA counts reproduce m1 exactly.
For each hand: R rollouts; each rollout stores the bitmask (over the hero's card indices) of cards
that won a trick.  Output: math/results/m5_raw_n{n}_{S|NT}.npz with
  cards (H, n) int8, masks (P, H, R) int32 (P = policies ta, ta_plus), bids (P, H).
Usage: python math/m5_sim.py --ns 5,8,13 --hands 1000 --R 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, BidModel, ModelBidder  # noqa: E402
from judgement.cards import NT  # noqa: E402
from judgement.engine import deal, run_round  # noqa: E402
from judgement.policies.ta_plus import ta_plus_v2  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.sampling import HandSampler  # noqa: E402
from judgement.state import RoundState, state_from_info  # noqa: E402

HERO = 1
OUT = ROOT / "math" / "results"
POLS = ("ta", "ta_plus")


def make_agents():
    m = BidModel.load(ROOT / "experiments" / "phase1" / "results" / "model_ta_final.json")
    return {"ta": [Agent(ModelBidder(m), TargetAwarePolicy())] * 4,
            "ta_plus": [Agent(ModelBidder(m), ta_plus_v2())] * 4}


def rollouts(st, ags, R, rng):
    info = st.info_state(HERO)
    sampler = HandSampler(info)
    cards = [c for c in range(52) if st.hands[HERO] >> c & 1]
    idx = {c: i for i, c in enumerate(cards)}
    masks = np.zeros(R, np.int32)
    bids = set()
    prngs = [rng] * 4
    for r in range(R):
        s2 = state_from_info(info, sampler.sample(rng))
        run_round(s2, ags, prngs, check=False)
        bids.add(s2.bids[HERO])
        h = s2.history
        m = 0
        for k, w in enumerate(s2.trick_winners):
            if w == HERO:
                for p, c in h[4 * k:4 * k + 4]:
                    if p == HERO:
                        m |= 1 << idx[c]
        masks[r] = m
    assert len(bids) == 1
    return cards, masks, bids.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5,8,13")
    ap.add_argument("--trumps", default="0,-1")
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--R", type=int, default=300)
    a = ap.parse_args()
    rules = Rules()
    ags = make_agents()
    for n in map(int, a.ns.split(",")):
        for tr in map(int, a.trumps.split(",")):
            fn = OUT / f"m5_raw_n{n}_{'NT' if tr == NT else 'S'}.npz"
            rng_deal = np.random.default_rng(91000 + 10 * n + (tr + 1))  # = m1_sim deal stream
            H, R = a.hands, a.R
            C = np.zeros((H, n), np.int8)
            M = np.zeros((len(POLS), H, R), np.int32)
            B = np.zeros((len(POLS), H), np.int8)
            t0 = time.time()
            for i in range(H):
                hands = deal(rules, n, rng_deal)
                for k, pol in enumerate(POLS):
                    hr = np.random.default_rng([92000, n, tr + 1, i, 0 if pol == "ta" else 2])
                    st = RoundState(rules, n, tr, 0, list(hands))
                    cards, masks, b = rollouts(st, ags[pol], R, hr)
                    M[k, i], B[k, i] = masks, b
                C[i] = cards
                if (i + 1) % 100 == 0:
                    print(n, tr, i + 1, round(time.time() - t0), flush=True)
                    np.savez_compressed(fn, cards=C[:i + 1], masks=M[:, :i + 1], bids=B[:, :i + 1], pols=POLS)
            np.savez_compressed(fn, cards=C, masks=M, bids=B, pols=POLS)


if __name__ == "__main__":
    main()

"""M10 (round 3, H2) data: like m5_sim (same deals, same rollout seeds => identical card masks)
but also stores, per rollout, the bitmask over TRICK INDEX of tricks the hero won (bit j = trick j)
and the hero's card index played in each trick (to recover the lead/suit sequence).

Output math/results/m10_raw_n{n}_{S|NT}.npz: cards (H,n), cmask (P,H,R) int32 (= m5 masks),
tmask (P,H,R) int32, order (P,H,R,n) int8 (hero card index played at trick j), bids (P,H).
Usage: python math/m10_sim.py --ns 5,8,13 --trumps -1 --hands 1000 --R 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math"))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.cards import NT  # noqa: E402
from judgement.engine import deal, run_round  # noqa: E402
from judgement.sampling import HandSampler  # noqa: E402
from judgement.state import RoundState, state_from_info  # noqa: E402
from m5_sim import HERO, OUT, POLS, make_agents  # noqa: E402


def rollouts(st, ags, R, rng):
    info = st.info_state(HERO)
    sampler = HandSampler(info)
    n = st.n_cards
    cards = [c for c in range(52) if st.hands[HERO] >> c & 1]
    idx = {c: i for i, c in enumerate(cards)}
    cm = np.zeros(R, np.int32)
    tm = np.zeros(R, np.int32)
    order = np.zeros((R, n), np.int8)
    prngs = [rng] * 4
    for r in range(R):
        s2 = state_from_info(info, sampler.sample(rng))
        run_round(s2, ags, prngs, check=False)
        h = s2.history
        m = t = 0
        for k, w in enumerate(s2.trick_winners):
            for p, c in h[4 * k:4 * k + 4]:
                if p == HERO:
                    order[r, k] = idx[c]
                    if w == HERO:
                        m |= 1 << idx[c]
                        t |= 1 << k
        cm[r], tm[r] = m, t
    return cards, cm, tm, order, s2.bids[HERO]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5,8,13")
    ap.add_argument("--trumps", default="-1")
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--deadline", type=float, default=float("inf"))
    a = ap.parse_args()
    rules = Rules()
    ags = make_agents()
    for n in map(int, a.ns.split(",")):
        for tr in map(int, a.trumps.split(",")):
            fn = OUT / f"m10_raw_n{n}_{'NT' if tr == NT else 'S'}.npz"
            rng_deal = np.random.default_rng(91000 + 10 * n + (tr + 1))  # = m1/m5 deal stream
            H, R = a.hands, a.R
            C = np.zeros((H, n), np.int8)
            CM = np.zeros((len(POLS), H, R), np.int32)
            TM = np.zeros((len(POLS), H, R), np.int32)
            O = np.zeros((len(POLS), H, R, n), np.int8)
            B = np.zeros((len(POLS), H), np.int8)
            t0 = time.time()
            done = 0
            for i in range(H):
                if time.time() > a.deadline:
                    break
                hands = deal(rules, n, rng_deal)
                for k, pol in enumerate(POLS):
                    hr = np.random.default_rng([92000, n, tr + 1, i, 0 if pol == "ta" else 2])
                    st = RoundState(rules, n, tr, 0, list(hands))
                    cards, cm, tm, order, b = rollouts(st, ags[pol], R, hr)
                    CM[k, i], TM[k, i], O[k, i], B[k, i] = cm, tm, order, b
                C[i] = cards
                done = i + 1
                if done % 100 == 0:
                    print(n, tr, done, round(time.time() - t0), flush=True)
                    np.savez_compressed(fn, cards=C[:done], cmask=CM[:, :done], tmask=TM[:, :done],
                                        order=O[:, :done], bids=B[:, :done], pols=POLS)
            np.savez_compressed(fn, cards=C[:done], cmask=CM[:, :done], tmask=TM[:, :done],
                                order=O[:, :done], bids=B[:, :done], pols=POLS)


if __name__ == "__main__":
    main()

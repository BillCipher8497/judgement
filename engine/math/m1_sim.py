"""M1 data: per-card win indicators + hero trick histogram, hero = first bidder.

For each (n, trump) and each dealt hand, run R rollouts under (a) all-TA play with
TA-model bids and (b) all-Greedy play with greedy-model bids, on the SAME deal
(paired).  Hero = seat 1 (left of dealer 0, bids first, leads trick 1).
Records: hero bid used, counts[t], card_wins[i] (# rollouts card i won a trick),
and split-half histograms (rollouts 0..R/2-1 vs R/2..R-1) for a noise floor.
Usage: python math/m1_sim.py --ns 5,8,13 --hands 500 --R 300
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
from judgement.cards import NT, SUIT_MASK  # noqa: E402
from judgement.engine import deal, run_round  # noqa: E402
from judgement.sampling import HandSampler  # noqa: E402
from judgement.state import RoundState, state_from_info  # noqa: E402

from common import agents, load_model  # noqa: E402

HERO = 1
OUT = ROOT / "math" / "results"


def hero_rollouts(st, ags, R, rng):
    info = st.info_state(HERO)
    sampler = HandSampler(info)
    n = st.n_cards
    cards = [c for c in range(52) if st.hands[HERO] >> c & 1]
    idx = {c: i for i, c in enumerate(cards)}
    counts = np.zeros((2, n + 1), np.int32)
    wins = np.zeros(n, np.int32)
    bids = set()
    prngs = [rng] * 4
    for r in range(R):
        s2 = state_from_info(info, sampler.sample(rng))
        run_round(s2, ags, prngs, check=False)
        bids.add(s2.bids[HERO])
        counts[0 if r < R // 2 else 1, s2.tricks_won[HERO]] += 1
        h = s2.history
        for k, w in enumerate(s2.trick_winners):
            if w == HERO:
                for p, c in h[4 * k:4 * k + 4]:
                    if p == HERO:
                        wins[idx[c]] += 1
    assert len(bids) == 1
    return cards, counts, wins, bids.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="5,8,13")
    ap.add_argument("--trumps", default="0,-1")
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--R", type=int, default=300)
    a = ap.parse_args()
    rules = Rules()
    m = {"ta": load_model("ta"), "greedy": load_model("greedy")}
    ags = {k: [agents(k, m[k])] * 4 for k in m}
    for n in map(int, a.ns.split(",")):
        for tr in map(int, a.trumps.split(",")):
            fn = OUT / f"m1_raw_n{n}_{'NT' if tr == NT else 'S'}.json"
            recs = json.load(open(fn)) if fn.exists() else []
            rng_deal = np.random.default_rng(91000 + 10 * n + (tr + 1))
            rng = np.random.default_rng(92000 + 10 * n + (tr + 1))
            # replay deals already done so the seed stream stays aligned
            for _ in range(len(recs)):
                deal(rules, n, rng_deal)
                # rng for rollouts: re-derive per hand to make resume exact
            t0 = time.time()
            for i in range(len(recs), a.hands):
                hands = deal(rules, n, rng_deal)
                rec = {"i": i, "n": n, "trump": tr, "hand": hands[HERO]}
                h = hands[HERO]
                lens = [(h & SUIT_MASK[s]).bit_count() for s in range(4)]
                rec["lens"] = lens
                for k in ("ta", "greedy"):
                    hr = np.random.default_rng([92000, n, tr + 1, i, 0 if k == "ta" else 1])
                    st = RoundState(rules, n, tr, 0, list(hands))
                    cards, counts, wins, b = hero_rollouts(st, ags[k], a.R, hr)
                    rec[k] = {"bid": int(b), "counts": counts.tolist(), "wins": wins.tolist()}
                rec["cards"] = cards
                recs.append(rec)
                if (i + 1) % 50 == 0:
                    json.dump(recs, open(fn, "w"))
                    print(n, tr, i + 1, round(time.time() - t0), flush=True)
            json.dump(recs, open(fn, "w"))


if __name__ == "__main__":
    main()

"""M8 data: 2-card round, trump S, fixed play policy (TA; TA plays on its OWN bid only,
deterministically).  For N random deals and all 81 bid vectors (b0..b3 in {0,1,2}^4; seat 0
bids first and leads, seat 3 = dealer) record each seat's trick count.  Hook is applied later
(play does not depend on it).  Output: math/results/m8_raw_{pol}.npz  hands (N,4) int64,
tricks (N,81) uint8 packed as t0 + 3 t1 + 9 t2 (t3 = 2 - sum).
Usage: python math/m8_sim.py --N 100000 --pol ta
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.engine import deal, run_round  # noqa: E402
from judgement.policies.ta_plus import ta_plus_v2  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.state import RoundState  # noqa: E402

OUT = ROOT / "math" / "results"
BVECS = list(itertools.product(range(3), repeat=4))  # index = b0*27 + b1*9 + b2*3 + b3


class Fixed:
    def __init__(self, b):
        self.b = b

    def bid(self, info, legal, rng):
        return self.b


class Ag:
    def __init__(self, b, pl):
        self.b, self.pl = b, pl

    def bid(self, info, legal, rng):
        return self.b

    def play(self, info, legal, rng):
        return self.pl.play(info, legal, rng)


def play_all(hands, pl, rules, rng):
    out = np.zeros(81, np.uint8)
    for k, bv in enumerate(BVECS):
        st = RoundState(rules, 2, 0, 3, list(hands))
        run_round(st, [Ag(b, pl) for b in bv], [rng] * 4, check=False)
        t = st.tricks_won
        out[k] = t[0] + 3 * t[1] + 9 * t[2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100000)
    ap.add_argument("--pol", default="ta")
    a = ap.parse_args()
    rules = Rules(hook=False)
    pl = TargetAwarePolicy() if a.pol == "ta" else ta_plus_v2()
    rng = np.random.default_rng(8080)
    prng = np.random.default_rng(8081)
    Hs = np.zeros((a.N, 4), np.int64)
    T = np.zeros((a.N, 81), np.uint8)
    t0 = time.time()
    for i in range(a.N):
        h = deal(rules, 2, rng)
        Hs[i] = h
        T[i] = play_all(h, pl, rules, prng)
        if (i + 1) % 10000 == 0:
            print(i + 1, round(time.time() - t0), flush=True)
    np.savez_compressed(OUT / f"m8_raw_{a.pol}.npz", hands=Hs, tricks=T)


if __name__ == "__main__":
    main()

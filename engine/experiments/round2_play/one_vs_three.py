"""1-vs-3: challenger in seat g%4 vs three TA, game g uses seed BASE+g (common
random numbers: identical deals and per-seat rng streams across challengers).
The challenger 'ta' is the all-TA reference table.

python one_vs_three.py KIND G [tag]   -> results/1v3_<tag>.npz (incremental)
KIND uses ';' between options, e.g. ta_plus:shed=1;duck=0
"""
import sys
import time

import numpy as np

from common import RES, make_agent
from judgement import Rules, play_game

BASE = int(__import__("os").environ.get("R2_BASE", 31_000_000))


def run(kind, G, tag, start=0):
    rules = Rules()
    R = rules.n_rounds
    out = RES / f"1v3_{tag}.npz"
    bids = np.zeros((G, R, 4), np.int8)
    tricks = np.zeros((G, R, 4), np.int8)
    scores = np.zeros((G, R, 4), np.float32)
    totals = np.zeros((G, 4))
    wins = np.zeros((G, 4))
    seat = np.arange(G) % 4
    t0 = time.time()
    done = 0
    for g in range(G):
        pols = [make_agent("ta") for _ in range(4)]
        pols[g % 4] = make_agent(kind)
        res = play_game(rules, pols, seed=BASE + g, record_trace=False)
        for r, rr in enumerate(res.rounds):
            bids[g, r] = rr.bids
            tricks[g, r] = rr.tricks
            scores[g, r] = rr.scores
        totals[g] = res.totals
        wins[g] = res.wins
        done = g + 1
        if done % 200 == 0 or done == G:
            np.savez_compressed(out, bids=bids[:done], tricks=tricks[:done], scores=scores[:done],
                                totals=totals[:done], wins=wins[:done], seat=seat[:done], seed_base=BASE,
                                kind=kind)
            print(tag, done, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    kind = sys.argv[1].replace(";", ",")
    run(kind, int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else kind.replace(":", "_").replace(",", "_"))

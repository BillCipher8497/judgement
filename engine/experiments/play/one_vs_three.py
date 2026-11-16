"""1-vs-3: challenger in seat g%4 vs three TA; game g uses seed BASE+g (CRN:
identical deals and per-seat rng streams across challengers).  KIND 'ta' is the
all-TA reference table.  Saves every 100 games (and stops at DEADLINE epoch).

python one_vs_three.py KIND G TAG [DEADLINE_EPOCH]  -> results/1v3_<TAG>.npz
KIND uses ';' between options, e.g. "ta_plus:shed=0;lh=3"
"""
import os
import sys
import time

import numpy as np

from common import RES, make_agent
from judgement import Rules, play_game

BASE = int(os.environ.get("PLAY_BASE", 71_000_000))


def run(kind, G, tag, deadline=float("inf")):
    rules = Rules()
    R = rules.n_rounds
    out = RES / f"1v3_{tag}.npz"
    bids = np.zeros((G, R, 4), np.int8)
    tricks = np.zeros((G, R, 4), np.int8)
    scores = np.zeros((G, R, 4), np.float32)
    totals = np.zeros((G, 4))
    wins = np.zeros((G, 4))
    secs = np.zeros(G)
    seat = np.arange(G) % 4
    t0 = time.time()

    def save(done):
        np.savez_compressed(out, bids=bids[:done], tricks=tricks[:done], scores=scores[:done],
                            totals=totals[:done], wins=wins[:done], seat=seat[:done], secs=secs[:done],
                            seed_base=BASE, kind=kind)
    done = 0
    for g in range(G):
        if time.time() > deadline:
            break
        pols = [make_agent("ta") for _ in range(4)]
        pols[g % 4] = make_agent(kind, seed=int(np.random.SeedSequence([BASE, g, 5]).generate_state(1)[0]))
        t1 = time.time()
        res = play_game(rules, pols, seed=BASE + g, record_trace=False)
        secs[g] = time.time() - t1
        for r, rr in enumerate(res.rounds):
            bids[g, r] = rr.bids
            tricks[g, r] = rr.tricks
            scores[g, r] = rr.scores
        totals[g] = res.totals
        wins[g] = res.wins
        done = g + 1
        if done % 100 == 0:
            save(done)
    save(done)
    print(tag, done, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    kind = sys.argv[1].replace(";", ",")
    run(kind, int(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else float("inf"))

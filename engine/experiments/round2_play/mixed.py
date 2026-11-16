"""Mixed 4-policy tables.  Block b uses seed BASE+b for all 20 tables of the
block: the 5 four-policy subsets of POLICIES x 4 cyclic seat rotations, so every
policy plays every seat equally often within each subset and all tables of a
block share the same deals (common random numbers).

python mixed.py MAX_BLOCKS DEADLINE_EPOCH  -> results/mixed.npz (saved after every block)
"""
import itertools
import sys
import time

import numpy as np

from common import RES, make_agent
from judgement import Rules, play_game

BASE = 51_000_000
POLICIES = ["greedy", "ta", "ta_spite", "ta_plus", "hybrid"]


def main(max_blocks, deadline):
    rules = Rules()
    combos = list(itertools.combinations(range(5), 4))
    rows = {"block": [], "assign": [], "totals": [], "wins": [], "bids": [], "tricks": [], "secs": []}
    t0 = time.time()
    for b in range(max_blocks):
        if time.time() > deadline:
            break
        for ci, combo in enumerate(combos):
            for k in range(4):
                assign = [combo[(s + k) % 4] for s in range(4)]
                pols = [make_agent(POLICIES[i]) for i in assign]
                t1 = time.time()
                res = play_game(rules, pols, seed=BASE + b, record_trace=False)
                rows["secs"].append(time.time() - t1)
                rows["block"].append(b)
                rows["assign"].append(assign)
                rows["totals"].append(res.totals)
                rows["wins"].append(res.wins)
                rows["bids"].append([rr.bids for rr in res.rounds])
                rows["tricks"].append([rr.tricks for rr in res.rounds])
        np.savez_compressed(RES / "mixed.npz", policies=np.array(POLICIES), seed_base=BASE,
                            **{k: np.array(v) for k, v in rows.items()})
        print("block", b + 1, "tables", len(rows["block"]), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]))

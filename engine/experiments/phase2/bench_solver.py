"""Solver speed per n (4p, single process).  Writes results/bench_solver.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement import NT, Rules  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.solver import MODES, MaxN, Paranoid, Social  # noqa: E402
from judgement.solver.bidding import make_play_state, rational_bids  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def main(ns=(1, 2, 3, 4, 5, 6, 7), reps=None, seed=12345):
    R = Rules()
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        k = reps or (30 if n <= 5 else 5 if n == 6 else 3)
        t = {m: [] for m in MODES}
        t["paranoid_x4"] = []
        t["social"] = []
        t["rational_bids"] = []
        for i in range(k):
            hands = deal(R, n, rng)
            trump = [0, 1, 2, 3, NT][i % 5]
            t0 = time.perf_counter()
            if n <= 6:
                bids, _ = rational_bids(R, n, trump, i % 4, hands)
                t["rational_bids"].append(time.perf_counter() - t0)
            else:
                bids = tuple(int(x) for x in rng.integers(0, 3, 4))
                if sum(bids) == n:
                    bids = (bids[0] + 1,) + bids[1:]
            st = make_play_state(R, n, trump, i % 4, hands, bids)
            for m in (MODES if n <= 6 else ("neutral",)):
                t0 = time.perf_counter()
                MaxN(R, trump, bids, 4, m).value(st)
                t[m].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for p in range(4):
                Paranoid(R, trump, bids, p).value(st)
            t["paranoid_x4"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            Social(R, trump, bids).value(st)
            t["social"].append(time.perf_counter() - t0)
        row = {"n": n, "deals": k}
        for key, v in t.items():
            if v:
                row[key + "_mean_s"] = float(np.mean(v))
                row[key + "_max_s"] = float(np.max(v))
        rows.append(row)
        print(json.dumps(row), flush=True)
        (OUT / f"bench_solver_{min(ns)}_{max(ns)}.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    ns = tuple(int(x) for x in sys.argv[1:]) or (1, 2, 3, 4, 5, 6, 7)
    main(ns)

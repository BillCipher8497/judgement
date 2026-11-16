"""Continue the bid/play fixed point with damping, then measure the noise floor.

Iteration k (k > start): fit on the pooled data of the last two iterations
(simulated under model_{k-1} and model_{k-2}): a damped update that removes the
period-2 oscillation seen at large n in the undamped run.  Noise floor: two
independent datasets simulated under the final model, fitted separately; the
bid disagreement between those two fits is what pure sampling noise produces.

python experiments/phase1/fp_continue.py --tag ta --start 5 --iters 4 --rounds 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import BidModel  # noqa: E402

from fixed_point import PLAY, RES, compare, eval_bids, fit_model, gen_data, table_stats  # noqa: E402


def cat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ta")
    ap.add_argument("--play", default="ta")
    ap.add_argument("--miss", default="zero")
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8000)
    ap.add_argument("--explore", type=float, default=0.5)
    a = ap.parse_args()
    rules = Rules(miss_penalty=a.miss)
    play = PLAY[a.play]()
    log_path = RES / f"fixed_point_{a.tag}.json"
    log = json.load(open(log_path))
    model = BidModel.load(RES / f"model_{a.tag}_it{a.start}.json")
    prev_bids = eval_bids(rules, model)
    t0 = time.time()
    old = gen_data(rules, play, model, a.rounds, 50 + a.start, a.explore, None)
    for k in range(a.start + 1, a.start + 1 + a.iters):
        new = gen_data(rules, play, model, a.rounds, 60 + k, a.explore, None) if k > a.start + 1 else old
        pooled = cat(old, new) if new is not old else old
        model = fit_model(pooled, True)
        model.save(RES / f"model_{a.tag}_it{k}.json")
        bids = eval_bids(rules, model)
        rec = {"iter": k, "damped": True, "data_rows": int(len(pooled["n"])),
               "data_hit_rate_unexplored": float(np.mean((new["bid"] == new["tricks"])[~new["explored"]])),
               "table": table_stats(bids), "change_vs_prev": compare(prev_bids, bids),
               "elapsed_s": time.time() - t0}
        prev_bids = bids
        old = new
        log["iters"].append(rec)
        print(k, {kk: v for kk, v in rec.items() if kk not in ("table",)}, flush=True)
        json.dump(log, open(log_path, "w"), indent=1)
    model.save(RES / f"model_{a.tag}_final.json")
    # noise floor
    d1 = gen_data(rules, play, model, a.rounds, 901, a.explore, None)
    d2 = gen_data(rules, play, model, a.rounds, 902, a.explore, None)
    m1, m2 = fit_model(d1, True), fit_model(d2, True)
    nf = compare(eval_bids(rules, m1), eval_bids(rules, m2))
    log["noise_floor_same_policy_two_datasets"] = nf
    nf2 = compare(eval_bids(rules, fit_model(cat(d1, d2), True)), prev_bids)
    log["final_vs_refit_on_fresh_pooled_data"] = nf2
    json.dump(log, open(log_path, "w"), indent=1)
    print("noise floor", nf, "\nfinal vs fresh refit", nf2, "\ndone", time.time() - t0)


if __name__ == "__main__":
    main()

"""Fit the heuristic bidder and iterate the bid/play fixed point.

  iter 0 data : honour-count bids (+exploration), play policy P
  iter k      : fit model_k on iter k-1 data; simulate with model_k bids (+exploration)
Stability: on a fixed set of evaluation deals the bid sequence of model_k vs model_{k-1}.

Usage: python -m experiments.phase1.fixed_point --play ta|greedy|ta_spite [--iters 5] [--rounds 8000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from judgement import Rules
from judgement.bidding.model import BidModel, CondLogit, info_features
from judgement.bidding.sim import model_bid, random_trump, simulate_rounds
from judgement.engine import deal
from judgement.policies import GreedyPolicy
from judgement.policies.target_aware import TargetAwarePolicy, TASpitePolicy
from judgement.state import BID, RoundState

RES = Path(__file__).parent / "results"
PLAY = {"ta": TargetAwarePolicy, "greedy": GreedyPolicy, "ta_spite": TASpitePolicy}


def fit_model(data, use_bid, ns=range(1, 14)):
    models = {}
    for n in ns:
        for nt in (False, True):
            m = (data["n"] == n) & ((data["trump"] == -1) == nt)
            models[(n, nt)] = CondLogit(n, use_bid=use_bid).fit(data["x"][m], data["bid"][m], data["tricks"][m])
    return BidModel(models)


def gen_data(rules, play, model, rounds, seed, explore, fallback, ns=range(1, 14)):
    parts = [simulate_rounds(rules, play, n, rounds, seed * 100 + n, model=model, fallback_bidder=fallback,
                             explore_prob=explore) for n in ns]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def eval_bids(rules, model, n_deals=400, seed=12345):
    """Bid sequences of ``model`` on fixed deals (dealer 0, trump cycle S,H,D,C,NT)."""
    out = {}
    S_cache = {}
    for n in range(1, 14):
        rng = np.random.default_rng(seed + n)
        rows = []
        for i in range(n_deals):
            tr = rules.trump_cycle[i % 5]
            st = RoundState(rules, n, tr, 0, deal(rules, n, rng))
            while st.phase == BID:
                info = st.info_state(st.to_act)
                b, _ = model_bid(model, rules, info_features(info), n, tr, st.legal_bids(), S_cache)
                st.apply_bid(b)
            rows.append(list(st.bids))
        out[n] = np.array(rows)
    return out


def compare(prev, cur):
    allp = np.concatenate([prev[n].ravel() for n in prev])
    allc = np.concatenate([cur[n].ravel() for n in cur])
    per_n = {n: float(np.mean(prev[n] == cur[n])) for n in prev}
    # total-variation distance between bid histograms, pooled over n
    tv = 0.5 * np.abs(np.bincount(allp, minlength=14) - np.bincount(allc, minlength=14)).sum() / len(allp)
    return {"frac_identical": float(np.mean(allp == allc)), "mean_abs_change": float(np.mean(np.abs(allp - allc))),
            "tv_bid_hist": float(tv), "frac_identical_by_n": per_n}


def table_stats(bids):
    return {n: {"mean_bid": float(bids[n].mean()), "mean_sum_minus_n": float(bids[n].sum(1).mean() - n)}
            for n in bids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", default="ta")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=8000)
    ap.add_argument("--explore", type=float, default=0.5)
    ap.add_argument("--miss", default="zero")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or (a.play if a.miss == "zero" else f"{a.play}_{a.miss}")
    rules = Rules(miss_penalty=a.miss)
    play = PLAY[a.play]()
    use_bid = a.play != "greedy"
    log = {"play": a.play, "miss": a.miss, "rounds_per_n": a.rounds, "explore": a.explore, "iters": []}
    t0 = time.time()
    data = gen_data(rules, play, None, a.rounds, 1, a.explore, TargetAwarePolicy())
    prev_bids = None
    model = None
    for k in range(1, a.iters + 1):
        model = fit_model(data, use_bid)
        model.save(RES / f"model_{tag}_it{k}.json")
        bids = eval_bids(rules, model)
        rec = {"iter": k, "data_rows": int(len(data["n"])),
               "data_hit_rate_unexplored": float(np.mean((data["bid"] == data["tricks"])[~data["explored"]])),
               "table": table_stats(bids), "elapsed_s": time.time() - t0}
        if prev_bids is not None:
            rec["change_vs_prev"] = compare(prev_bids, bids)
        prev_bids = bids
        log["iters"].append(rec)
        print(k, {kk: v for kk, v in rec.items() if kk != "table"}, flush=True)
        with open(RES / f"fixed_point_{tag}.json", "w") as fh:
            json.dump(log, fh, indent=1)
        if k < a.iters:
            data = gen_data(rules, play, model, a.rounds, 1 + k, a.explore, None)
    model.save(RES / f"model_{tag}_final.json")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()

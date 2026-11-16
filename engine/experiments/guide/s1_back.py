"""S1 backward elimination.  (1) Joint fit of all rule lines (coordinate descent over the integer grid, best of
R random restarts + the default start), TRAIN set.  (2) Repeatedly remove the line whose removal (others refit)
costs least on TRAIN; remove it if that cost < 0.10 pts/round.  (3) For the final rule report, on the held-out
TEST set, each kept line's value = TEST loss when that line alone is removed (others refit on TRAIN), with
paired 95% CI.  Writes results/s1_back.json.
"""
import json
import sys

import os

import numpy as np

from common import RES, status
from judgement.policies.human_guide import GUIDE_BID
from s1_fit import GRID, fit, load, rule_bids, value

OFFV = dict(t_hon=0, t_guard=1, s_K2=0, s_K1=0, s_Q3=0, short=0, t_sub=99, n_K2=0, n_K1=0, n_Q3=0, n_long=0,
            n_sub=99, low_w=0, s_Q=0, s_J=0, small_cut=0, prior_k=99, hook_up=0, round_up=0)
LINES = {
    "trump honours bonus": ["t_hon", "t_guard"],
    "subtract per cards dealt": ["t_sub", "n_sub"],
    "side kings": ["s_K2", "s_K1", "n_K2", "n_K1"],
    "queens with >=3": ["s_Q3", "n_Q3"],
    "small rounds": ["small_cut", "small_A", "small_K"],
    "short side suits with trumps": ["short"],
    "NT long suits": ["n_long"],
    "earlier bids vs share": ["prior_k"],
    "rounding / hook direction": ["round_up", "hook_up"],
    "low cards (2-7)": ["low_w"],
    "side queens / jacks (any length)": ["s_Q", "s_J"],
}
CORE = ["t_each", "s_A", "n_A"]
THR = 0.10
PFX = os.environ.get("S1PFX", "s1")
ILS = 40


def ci(x):
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def params(lines):
    return CORE + [k for l in lines for k in LINES[l]]


def off(w, line):
    w = dict(w)
    for k in LINES[line]:
        if k in OFFV:
            w[k] = OFFV[k]
    return w


def main(R=10):
    D, meta, cf, _ = load(PFX + "_train.pkl")
    Dt, mt, cft, _ = load(PFX + "_test.pkl")
    vm_te = value(mt["bm"], cft)[0]
    rng = np.random.default_rng(55)
    keys = params(LINES)
    best_w, best = None, -1e9
    starts = [dict(GUIDE_BID)] + [{k: int(rng.choice(v)) for k, v in GRID.items()} for _ in range(R)]
    for i, w0 in enumerate(starts):
        w, b = fit(D, meta, cf, w0, keys, sweeps=8, log=lambda s: None)
        print(f"start {i}: {b:.4f}", flush=True)
        if b > best:
            best_w, best = w, b
    w, cur = best_w, best
    for it in range(ILS):      # iterated local search: perturb 3 weights, refit, keep if better
        w2 = dict(w)
        for k in rng.choice(keys, 3, replace=False):
            w2[k] = int(rng.choice(GRID[k]))
        w2, b2 = fit(D, meta, cf, w2, keys, sweeps=6, log=lambda s: None)
        if b2 > cur + 1e-9:
            w, cur = w2, b2
            print(f"ILS {it}: {cur:.4f}", flush=True)
    lines = list(LINES)
    log = [dict(step="full fit", train=cur, test_loss_vs_mb=ci(vm_te - value(rule_bids(w, Dt, mt), cft)[0]), w=w)]
    print(log[-1], flush=True)
    while lines:
        trials = []
        for l in lines:
            rest = [x for x in lines if x != l]
            w2, b2 = fit(D, meta, cf, off(w, l), params(rest), sweeps=4, log=lambda s: None)
            trials.append((cur - b2, l, w2, b2))
        trials.sort(key=lambda t: t[0])
        for c, l, _, _ in trials:
            print(f"   drop {l:32s} costs {c:+.3f}", flush=True)
        c, l, w2, b2 = trials[0]
        if c >= THR:
            break
        lines.remove(l)
        w, cur = w2, b2
        log.append(dict(step=f"drop {l}", cost_train=c, train=cur,
                        test_loss_vs_mb=ci(vm_te - value(rule_bids(w, Dt, mt), cft)[0]), w=w))
        print(log[-1], flush=True)
    vf = value(rule_bids(w, Dt, mt), cft)[0]
    vals = {}
    for l in lines:
        rest = [x for x in lines if x != l]
        w2, _ = fit(D, meta, cf, off(w, l), params(rest), sweeps=4, log=lambda s: None)
        vals[l] = ci(vf - value(rule_bids(w2, Dt, mt), cft)[0])
    # core line value: drop to "aces only" (t_each=0) and "trumps only" handled by removing all trumps
    for name, chg in [("count every trump", dict(t_each=0)), ("count aces", dict(s_A=0, n_A=0))]:
        w2 = dict(w); w2.update(chg)
        w2, _ = fit(D, meta, cf, w2, [k for k in params(lines) if k not in chg], sweeps=4, log=lambda s: None)
        vals[name] = ci(vf - value(rule_bids(w2, Dt, mt), cft)[0])
    out = dict(log=log, kept=lines, final=w, line_value_test=vals, final_test=ci(vf), mb_test=ci(vm_te),
               final_loss_vs_mb=ci(vm_te - vf))
    json.dump(out, open(RES / f"{PFX}_back.json", "w"), indent=1, default=float)
    print("FINAL", w, "\nvalues", vals)
    status(f"S1 backward elimination done: kept {lines}; final {w}; test loss vs MB {ci(vm_te - vf)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)

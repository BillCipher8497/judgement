"""S1 final: the guide bidding rule (GUIDE_BID defaults) on the held-out TEST set.
Per-line value = TEST loss when the line is removed (a) as is, (b) with the other weights refit on TRAIN.
Writes results/s1_final.json, results/s1_table.txt (+ .json)."""
import json

import os

import numpy as np

from common import RES, status
from judgement.policies.human_guide import GUIDE_BID
from s1_eval import table
from s1_fit import fit, load, rule_bids, value

PFX = os.environ.get("S1PFX", "s1")
LINES = [
    ("1 every trump = 2 pts", dict(t_each=0)),
    ("2 trump A/K/Q +1 pt each", dict(t_hon=0)),
    ("3 side A, side K = 1 pt each", dict(s_A=0, s_K2=0, s_K1=0)),
    ("(rejected) side Q in a 3+ card suit = 1 pt (also NT)", dict(s_Q3=1, n_Q3=1)),
    ("5 NT: A = 2", dict(n_A=0)),
    ("(rejected) NT K = 1", dict(n_K2=1, n_K1=1)),
    ("3a side kings only (trump rounds)", dict(s_K2=0, s_K1=0)),
    ("6 trump rounds: -1 pt per 4 cards dealt", dict(t_sub=99)),
    ("7 earlier bids: -1 per 4 tricks over their share (+1 per 4 under)", dict(prior_k=99)),
    ("8 hooked dealer: odd points -> up, even -> down (vs always down)", dict(hook_up=0, round_up=0, _hookdown=1)),
]


def ci(x):
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    w = dict(GUIDE_BID)
    D, m, cf, _ = load(PFX + "_train.pkl")
    Dt, mt, cft, _ = load(PFX + "_test.pkl")
    vf = value(rule_bids(w, Dt, mt), cft)[0]
    vals = {}
    for name, ch in LINES:
        if "_hookdown" in ch:
            mt2 = dict(mt); mt2["dealer_forb"] = np.full(len(mt["n"]), -1)
            raw = rule_bids(w, Dt, mt2)
            hk = raw == mt["dealer_forb"]
            alt = np.where(hk, np.where(raw - 1 >= 0, raw - 1, raw + 1), raw)
            vals[name] = dict(as_is=ci(vf - value(alt, cft)[0]), hooked_share=float(hk.mean()))
            print(name, vals[name], flush=True)
            continue
        w2 = dict(w); w2.update(ch)
        a = ci(vf - value(rule_bids(w2, Dt, mt), cft)[0])
        keys = [k for k in w if k not in ch and k not in ("small_A", "small_K", "low_w", "s_Q", "s_J")]
        w3, _ = fit(D, m, cf, w2, keys, sweeps=4, log=lambda s: None)
        b = ci(vf - value(rule_bids(w3, Dt, mt), cft)[0])
        vals[name] = dict(as_is=a, refit=b)
        print(name, a, b, flush=True)
    txt, res = table(w, PFX + "_test.pkl", PFX + "_table.txt")
    print(txt)
    json.dump(dict(w=w, line_values=vals, test_ev=ci(vf)), open(RES / f"{PFX}_final.json", "w"), indent=1)
    status(f"S1 final rule evaluated ({PFX}): results/{PFX}_final.json, {PFX}_table.txt; overall loss vs MB "
           f"{res['all']['loss']:+.3f} +- {res['all']['loss_ci']:.3f}/round, match MB {res['all']['match_mb']:.3f}, "
           f"match oracle {res['all']['match_orc']:.3f} (MB itself {res['all']['match_orc_mb']:.3f})")


if __name__ == "__main__":
    main()

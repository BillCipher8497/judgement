"""Audit A1: hidden-information cooperation surplus (H1) recomputed from raw arrays.

Read-only on experiments/surplus_hidden/results. Writes round7_audit/results/a1_surplus.json.
Checks:
  (a) heuristic / PIMC / MC / omni s3 shares on each comparison's own common deal set;
  (b) paired ratio hidden-surplus / omni-surplus with a deal bootstrap CI (the "about a third" claim,
      holding the TA+ hero fixed, instead of comparing against Phase-2's PI max^n hero);
  (c) full-game share and P(win) with ties counted as wins.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "surplus_hidden" / "results"
OUT = HERE.parent / "results" / "a1_surplus.json"
RNG = np.random.default_rng(20261115)
B = 4000


def load(tag):
    z = np.load(SRC / f"rounds_{tag}.npz")
    conds = list(z["conds"])
    out = {}
    for i, n in enumerate(z["ns"]):
        D = int(z["done"][i])
        if D < 2:
            continue
        s = z["scores"][i, :D]
        out[int(n)] = (s[:, conds.index("self"), 0].astype(float), s[:, conds.index("s3"), 0].astype(float))
    return out


def ratio_boot(num, den):
    """mean(num)/mean(den) with paired deal bootstrap 95% percentile CI."""
    D = len(num)
    idx = RNG.integers(0, D, size=(B, D))
    r = num[idx].mean(1) / den[idx].mean(1)
    return [float(num.mean() / den.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    m = np.load(SRC / "rounds_main.npz")
    main_self = {int(n): m["scores"][i, :, 0, 0].astype(float) for i, n in enumerate(m["ns"])}
    main_s3 = {int(n): m["scores"][i, :, 3, 0].astype(float) for i, n in enumerate(m["ns"])}
    kinds = {}
    for k, tags in {"pimc": ["pimc_2_5", "pimc_6", "pimc_7"], "omni": ["omni_2_6", "omni_7", "omni_8"],
                    "mc": ["mc_2_5"]}.items():
        d = {}
        for t in tags:
            if (SRC / f"rounds_{t}.npz").exists():
                d.update(load(t))
        kinds[k] = d
    res = {"per_n": {}}
    for n in range(2, 9):
        row = {}
        if n not in kinds["omni"]:
            continue
        o_self, o_s3 = kinds["omni"][n]
        D_o = len(o_self)
        # CRN check: self condition identical to main on the same deal indices
        row["crn_self_identical_omni"] = bool(np.array_equal(o_self, main_self[n][:D_o]))
        heur_sur = main_self[n][:D_o] - main_s3[n][:D_o]
        omni_sur = o_self - o_s3
        row["deals_omni"] = D_o
        row["share_heur_on_omni_deals"] = ratio_boot(heur_sur, o_self)
        row["share_omni"] = ratio_boot(omni_sur, o_self)
        row["ratio_heur_over_omni"] = ratio_boot(heur_sur, omni_sur)
        for k in ("pimc", "mc"):
            if n in kinds[k]:
                k_self, k_s3 = kinds[k][n]
                D = min(len(k_self), D_o)
                row[f"deals_{k}"] = D
                row[f"crn_self_identical_{k}"] = bool(np.array_equal(k_self[:D], main_self[n][:D]))
                row[f"share_{k}"] = ratio_boot(k_self[:D] - k_s3[:D], k_self[:D])
                row[f"ratio_{k}_over_omni"] = ratio_boot(k_self[:D] - k_s3[:D], omni_sur[:D])
        res["per_n"][n] = row
    # full games: share and P(win) with ties-as-win
    g = np.load(SRC / "games_main.npz")
    sc = g["scores"].sum(2)  # (G, cond, seat)
    hero = g["hero"]
    G = len(hero)
    tot_h = sc[np.arange(G), :, hero]  # (G, cond)
    best = sc.max(2)
    win_tie = (tot_h >= best - 1e-9).astype(float)
    conds = list(g["conds"])
    gr = {"games": G}
    for c in conds:
        j = conds.index(c)
        p = win_tie[:, j].mean()
        gr[f"pwin_ties_{c}"] = [float(p), float(1.96 * np.sqrt(p * (1 - p) / G))]
        gr[f"pwin_split_{c}"] = float(g["wins"][np.arange(G), j, hero].mean())
        if c != "self":
            gr[f"share_{c}"] = ratio_boot(tot_h[:, 0] - tot_h[:, j], tot_h[:, 0])
    res["games"] = gr
    OUT.write_text(json.dumps(res, indent=1))
    for n, row in res["per_n"].items():
        print(n, {k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in row.items()})
    print({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in gr.items()})


if __name__ == "__main__":
    main()

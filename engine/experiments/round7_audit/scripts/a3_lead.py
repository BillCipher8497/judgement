"""Audit A3: confidence intervals for the lead-model "gap closed" fractions (H2 / AGENTS headline
'closes 98%, 82% and 30% of the NT bid gap at n = 5, 8, 13').

Imports math/m10_analyze.analyze_hand unchanged (read-only) and re-runs it on the stored rollouts
(math/results/m10_raw_n{n}_NT.npz), keeping the per-hand cross-fit bid-match indicators that the original
summary averaged away. gap_xmatch(M) = (xmatch_M - xmatch_PB) / (floor - xmatch_PB), paired hand bootstrap.
Per-config raw arrays are checkpointed to results/a3_lead_<cfg>.npz; summary to results/a3_lead.json.
python a3_lead.py [--cfg 5,8,13]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATH = HERE.parents[2] / "math"
sys.path.insert(0, str(MATH))
import m10_analyze as M  # noqa: E402

OUTD = HERE.parent / "results"
KEEP = ("PB", "SB", "LC0", "LC1", "LC2", "LCX", "LCS", "TPB", "LCU")
B = 4000


def gap_ci(xm, pb, fl, rng):
    D = len(pb)
    est = (xm.mean() - pb.mean()) / (fl.mean() - pb.mean())
    idx = rng.integers(0, D, size=(B, D))
    g = (xm[idx].mean(1) - pb[idx].mean(1)) / (fl[idx].mean(1) - pb[idx].mean(1))
    return [float(est), float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="5,8,13")
    a = ap.parse_args()
    summ_fn = OUTD / "a3_lead.json"
    summ = json.loads(summ_fn.read_text()) if summ_fn.exists() else {}
    rng = np.random.default_rng(20261115)
    for n in map(int, a.cfg.split(",")):
        D = np.load(MATH / "results" / f"m10_raw_n{n}_NT.npz")
        for k, pol in enumerate(D["pols"]):
            key = f"{pol}|n={n}|NT"
            raw = OUTD / f"a3_lead_{pol}_n{n}.npz"
            if raw.exists():
                z = np.load(raw)
                arr = {m: z[m] for m in z.files}
            else:
                hs = [M.analyze_hand(D["cards"][i], D["cmask"][k, i], D["tmask"][k, i], n, 2.0)
                      for i in range(len(D["cards"]))]
                hs = [h for h in hs if h is not None]
                arr = {m: np.array([h[m]["xmatch"] for h in hs], float) for m in KEEP}
                arr["floor"] = np.array([h["floor_match"] for h in hs], float)
                np.savez(raw, **arr)
            row = {"hands": int(len(arr["PB"])), "xmatch_PB": float(arr["PB"].mean()),
                   "floor": float(arr["floor"].mean()),
                   "params_vs_pmf_dof": {"pmf_dof": n, "PB": n, "TPB": n, "LC0": 3, "LC1": 2 * n - 1, "LCX": n + 4}}
            for m in KEEP:
                if m == "PB":
                    continue
                row[f"xmatch_{m}"] = float(arr[m].mean())
                row[f"gap_xmatch_{m}"] = gap_ci(arr[m], arr["PB"], arr["floor"], rng)
            summ[key] = row
            summ_fn.write_text(json.dumps(summ, indent=1))
            print(key, {m: np.round(row[f"gap_xmatch_{m}"], 2).tolist() for m in KEEP if m != "PB"}, flush=True)


if __name__ == "__main__":
    main()

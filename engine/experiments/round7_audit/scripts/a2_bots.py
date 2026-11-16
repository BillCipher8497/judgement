"""Audit A2: bot-ladder headline numbers recomputed from raw game files (read-only).

(a) vs 3xTA (seed base 71M, challenger seat g%4): each bot minus the all-TA table on the same game/seat,
    on its full game set AND on the common set g < 1000, plus paired bot-vs-bot differences.
(b) C2 top pool (3x Hybrid-BI K32c6, base 600M): challengers minus pool, split into the first 142 games
    (where the round-4 '+5.8' claim was first made) and the later extension (clean replication).
All CIs: 1.96 SE over games (games are independent seeds). Writes results/a2_bots.json.
"""
import json
from glob import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E = HERE.parents[1]
OUT = HERE.parent / "results" / "a2_bots.json"


def load(files):
    parts = [np.load(f) for f in files]
    if not parts:
        return None
    tot = np.concatenate([p["totals"] for p in parts])
    game = np.concatenate([p["game"] if "game" in p.files else np.arange(len(p["totals"])) for p in parts])
    o = np.argsort(game, kind="stable")
    tot, game = tot[o], game[o]
    # identical replays of the same game id (e.g. round4 ref g142..159 re-run in round6) are dropped;
    # conflicting duplicates would be a data error
    keep = np.ones(len(game), bool)
    dup = np.flatnonzero(game[1:] == game[:-1]) + 1
    for j in dup:
        assert np.array_equal(tot[j], tot[j - 1]), f"conflicting duplicate game {game[j]}"
        keep[j] = False
    tot, game = tot[keep], game[keep]
    s = game % 4
    i = np.arange(len(game))
    return {"game": game, "tot": tot[i, s], "win": (tot[i, s] >= tot.max(1) - 1e-9).astype(float)}


def diff(a, b, gmin=0, gmax=10 ** 9):
    c = np.intersect1d(a["game"], b["game"])
    c = c[(c >= gmin) & (c < gmax)]
    ia, ib = np.searchsorted(a["game"], c), np.searchsorted(b["game"], c)
    d = a["tot"][ia] - b["tot"][ib]
    w = a["win"][ia] - b["win"][ib]
    n = len(c)
    se = lambda x: float(1.96 * x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": int(n), "d": float(d.mean()) if n else None, "ci": se(d), "dwin": float(w.mean()) if n else None,
            "ci_win": se(w)}


def main():
    F, P, R3 = E / "final" / "results", E / "play" / "results", E / "round3_play" / "results"
    ta = {
        "TA (ref)": [F / "f_ta_ta.npz"],
        "TA+3": [F / "f_ta_tap3.npz"],
        "Hybrid c5": [P / "1v3_x_hyb_c5.npz", F / "f_ta_hyb_ext.npz"],
        "Hybrid+oracle": [P / "1v3_x_hyb_orc.npz", F / "f_ta_hybo_ext.npz"],
        "Hybrid K32 c5": [P / "1v3_x_hyb_k32_c5.npz", F / "f_ta_k32_ext.npz"],
        "Hybrid-BI": [R3 / "f_bi_ta.npz", R3 / "f_bi_ta2.npz"],
    }
    D = {k: load([str(f) for f in v if Path(f).exists()]) for k, v in ta.items()}
    res = {"vs_3xTA": {}, "vs_3xTA_g<1000": {}, "pairs_g<1000": {}, "c2": {}}
    for k, d in D.items():
        if k == "TA (ref)" or d is None:
            continue
        res["vs_3xTA"][k] = diff(d, D["TA (ref)"])
        res["vs_3xTA_g<1000"][k] = diff(d, D["TA (ref)"], 0, 1000)
    for a, b in [("Hybrid-BI", "Hybrid c5"), ("Hybrid-BI", "Hybrid+oracle"), ("Hybrid-BI", "Hybrid K32 c5"),
                 ("Hybrid+oracle", "Hybrid c5"), ("Hybrid K32 c5", "Hybrid c5")]:
        res["pairs_g<1000"][f"{a} - {b}"] = diff(D[a], D[b], 0, 1000)

    R4, R6 = E / "round4" / "results", E / "round6" / "results"
    g = lambda pat, d: sorted(glob(str(d / pat)))
    c2 = {"pool": g("f_g2_ref_*.npz", R4) + g("f_c2x_ref_*.npz", R6), "plain": g("f_g2_plain_*.npz", R4),
          "bi7": g("f_g2_c7_*.npz", R4) + g("f_c2x_c7_*.npz", R6),
          "rb7": g("f_c2_rb_*.npz", R6) + g("f_c2x_rb_*.npz", R6)}
    C = {k: load(v) for k, v in c2.items()}
    for a in ("bi7", "rb7", "plain"):
        res["c2"][f"{a} - pool, all"] = diff(C[a], C["pool"])
        res["c2"][f"{a} - pool, g<142"] = diff(C[a], C["pool"], 0, 142)
        res["c2"][f"{a} - pool, g>=142"] = diff(C[a], C["pool"], 142)
    res["c2"]["rb7 - bi7, g>=142"] = diff(C["rb7"], C["bi7"], 142)
    OUT.write_text(json.dumps(res, indent=1))
    for sec, rows in res.items():
        print("==", sec)
        for k, v in rows.items():
            print(f"  {k:32s} n={v['n']:5d}  {v['d']:+7.2f} ± {v['ci']:.2f}   dP(win,ties) {v['dwin']:+.3f} ± {v['ci_win']:.3f}")


if __name__ == "__main__":
    main()

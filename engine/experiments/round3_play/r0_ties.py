"""R0: recompute P(win) with ties-as-win (a tie for first = a win for every tied player)
for the main 1-vs-3 result files.  Paired vs the all-pool reference table (same seed, same seat).
-> results/r0_ties.md / .json"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
E = H.parent
RES = H / "results"; RES.mkdir(exist_ok=True)
F, P, Q = E / "final/results", E / "play/results", E / "phase4/results"
ROWS = {
    "TA (final, 71M)": (F / "f_ta_ta.npz", [("TA+", F / "f_ta_tap.npz"), ("TA+3", F / "f_ta_tap3.npz"),
                                            ("TA+3 ll=2", F / "f_ta_tap3s.npz"), ("Hybrid3", F / "f_ta_hyb3.npz")]),
    "TA (play, 71M)": (P / "1v3_x_ta.npz", [("TA+", P / "1v3_x_ta_plus.npz"), ("Hybrid c5", P / "1v3_x_hyb_c5.npz"),
                                            ("Hybrid c6", P / "1v3_x_hyb_c6.npz"), ("Hybrid K32", P / "1v3_x_hyb_k32_c5.npz"),
                                            ("TA+ + oracle", P / "1v3_x_orc_all.npz"), ("Hybrid + oracle", P / "1v3_x_hyb_orc.npz"),
                                            ("Greedy", P / "1v3_x_greedy.npz")]),
    "TA+ (ladder, 400M)": (Q / "q2_tap_ref.npz", [("TA", Q / "q2_tap_ta.npz"), ("Hybrid", Q / "q2_tap_hyb.npz"),
                                                  ("TA+ + oracle", Q / "q2_tap_tapo.npz"), ("Hybrid + oracle", Q / "q2_tap_hybo.npz")]),
    "Hybrid (ladder, 400M)": (Q / "q2_hyb_ref.npz", [("TA", Q / "q2_hyb_ta.npz"), ("TA+", Q / "q2_hyb_tap.npz"),
                                                     ("TA+ + oracle", Q / "q2_hyb_tapo.npz"), ("Hybrid + oracle", Q / "q2_hyb_hybo.npz")]),
}


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def tiewin(t):
    return (t == t.max(1, keepdims=True)).astype(float)


def main():
    out = {}
    md = ["| pool | challenger | n | P(win) split | P(win) ties-as-win | ΔP split | ΔP ties-as-win | first-place tie rate (chal table) | chal in tie for 1st |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pool, (refp, chals) in ROWS.items():
        ref = np.load(refp)
        rt = ref["totals"]
        out[pool + "|ref"] = dict(n=len(rt), tie_rate=float(((rt == rt.max(1, keepdims=True)).sum(1) > 1).mean()))
        for name, p in chals:
            if not p.exists():
                continue
            d = np.load(p)
            G = min(len(d["totals"]), len(rt)); ar = np.arange(G); s = ar % 4
            assert np.all(d["seat"][:G] == s)
            t, r = d["totals"][:G], rt[:G]
            ws, rs = d["wins"][:G][ar, s], ref["wins"][:G][ar, s]
            wt, rtw = tiewin(t)[ar, s], tiewin(r)[ar, s]
            tie = (t == t.max(1, keepdims=True)).sum(1) > 1
            m1, h1 = ci(ws - rs); m2, h2 = ci(wt - rtw)
            row = dict(n=G, pwin_split=float(ws.mean()), pwin_tie=float(wt.mean()), dp_split=m1, dp_split_ci=h1,
                       dp_tie=m2, dp_tie_ci=h2, tie_rate=float(tie.mean()), chal_tied=float((tie & (wt > 0)).mean()))
            out[f"{pool}|{name}"] = row
            md.append(f"| {pool} | {name} | {G} | {row['pwin_split']:.4f} | {row['pwin_tie']:.4f} | {m1:+.4f} ± {h1:.4f} | "
                      f"{m2:+.4f} ± {h2:.4f} | {row['tie_rate']:.4f} | {row['chal_tied']:.4f} |")
    md.append("")
    md.append("Reference-table first-place tie rates: " + "; ".join(f"{k.split('|')[0]} {v['tie_rate']:.4f} (n={v['n']})"
                                                                  for k, v in out.items() if k.endswith('|ref')))
    (RES / "r0_ties.md").write_text("\n".join(md) + "\n")
    (RES / "r0_ties.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()

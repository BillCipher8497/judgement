"""Q2 ladder table -> results/q2_ladder.md / .json.  Gain = challenger seat total minus the
pool policy's total in the same seat of the all-pool reference game (same seed: CRN).
95% CI = 1.96 SE over games (paired)."""
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).parent
RES = H / "results"
PLAY = H.parent / "play" / "results"
ROWS = {
    "TA": (PLAY / "1v3_x_ta.npz", [("TA+", PLAY / "1v3_x_ta_plus.npz"), ("Hybrid", PLAY / "1v3_x_hyb_c5.npz"),
                                   ("TA+ + oracle bidder", PLAY / "1v3_x_orc_all.npz"),
                                   ("Hybrid + oracle bidder", PLAY / "1v3_x_hyb_orc.npz"),
                                   ("Greedy", PLAY / "1v3_x_greedy.npz")]),
    "TA+": (RES / "q2_tap_ref.npz", [("TA", RES / "q2_tap_ta.npz"), ("Hybrid", RES / "q2_tap_hyb.npz"),
                                     ("TA+ + oracle bidder", RES / "q2_tap_tapo.npz"),
                                     ("Hybrid + oracle bidder", RES / "q2_tap_hybo.npz")]),
    "Hybrid": (RES / "q2_hyb_ref.npz", [("TA", RES / "q2_hyb_ta.npz"), ("TA+", RES / "q2_hyb_tap.npz"),
                                        ("TA+ + oracle bidder", RES / "q2_hyb_tapo.npz"),
                                        ("Hybrid + oracle bidder", RES / "q2_hyb_hybo.npz")]),
}


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), len(x)


def main():
    out, md = {}, ["| pool (3 seats) | challenger | n games | gain pts/game vs pool-in-seat | ΔP(win) | median s/game (whole table) |",
                   "|---|---|---|---|---|---|"]
    for pool, (refp, chals) in ROWS.items():
        if not refp.exists():
            continue
        ref = np.load(refp)
        best = None
        for name, p in chals:
            if not p.exists():
                continue
            d = np.load(p)
            G = min(len(d["totals"]), len(ref["totals"]))
            if G < 2:
                continue
            ar = np.arange(G)
            s = ar % 4
            assert np.all(d["seat"][:G] == s)
            gain = d["totals"][:G][ar, s] - ref["totals"][:G][ar, s]
            dw = d["wins"][:G][ar, s] - ref["wins"][:G][ar, s]
            m, h, n = ci(gain)
            mw, hw, _ = ci(dw)
            out[f"{pool}|{name}"] = dict(gain=m, ci=h, n=n, dpwin=mw, dpwin_ci=hw, secs=float(np.median(d["secs"][:G])))
            md.append(f"| {pool} | {name} | {n} | {m:+.2f} ± {h:.2f} | {mw:+.3f} ± {hw:.3f} | {np.median(d["secs"][:G]):.2f} |")
            if best is None or m > best[1]:
                best = (name, m, h, n)
        if best:
            md.append(f"| **{pool}** | **best: {best[0]}** | {best[3]} | **{best[1]:+.2f} ± {best[2]:.2f}** | | |")
    (RES / "q2_ladder.md").write_text("\n".join(md) + "\n")
    (RES / "q2_ladder.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()

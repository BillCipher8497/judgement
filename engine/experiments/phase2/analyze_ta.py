"""F4b / F5b tables: Phase-1 TargetAwarePolicy vs max^n (perfect info) and vs PIMC."""
from __future__ import annotations

import io
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_pimc

RES = Path(__file__).parent / "results"


def mci(x):
    x = np.asarray(x, dtype=float)
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)), len(x))


def fmt(t, d=3):
    return f"{t[0]:.{d}f} ± {t[1]:.{d}f} (n={t[2]})"


def main():
    md = []
    if (RES / "ta_pi_dec.csv").exists():
        d = pd.read_csv(RES / "ta_pi_dec.csv")
        s = pd.read_csv(RES / "ta_pi.csv")
        rows = []
        for n, g in d.groupby("n"):
            ni = g[g.indiff == 0]
            gs = s[s.n == n]
            tab = gs.groupby(["trump", "deal"]).agg(mx=("maxn", "sum"), ta=("all_ta", "sum"))
            rows.append({"n": n, "decisions": len(g), "TA optimal": fmt(mci(g.ta_opt)), "heur optimal": fmt(mci(g.heur_opt)),
                         "TA opt (non-indiff)": fmt(mci(ni.ta_opt)), "heur opt (non-indiff)": fmt(mci(ni.heur_opt)),
                         "TA 1-step loss": fmt(mci(g.ta_loss)), "heur 1-step loss": fmt(mci(g.heur_loss)),
                         "same card": fmt(mci(g.same)),
                         "1 TA seat loss vs max^n": fmt(mci(gs.maxn - gs.one_ta), 2),
                         "table max^n − all-TA": fmt(mci(tab.mx - tab.ta), 2)})
        cols = list(rows[0])
        md.append("## F4b Phase-1 TargetAwarePolicy vs Phase-2 heuristic under perfect information (scheme O bids, max^n path)")
        md.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
        md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    if (RES / "pimc_rounds_TA.csv").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            analyze_pimc.main("_TA")
        md.append("\n## F5b PIMC vs Phase-1 TargetAwarePolicy (hidden information; 'heur' = TA here)")
        md.append((RES / "summary_pimc_TA.md").read_text().split("\n", 1)[1])
    text = "\n".join(md)
    (RES / "summary_ta.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

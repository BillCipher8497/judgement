"""F5 tables from results/pimc_rounds.csv (paired by deal; seat-level deltas
averaged within deal, CI over deals)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"


def mci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(n), n)


def fmt(t, d=2):
    return f"{t[0]:+.{d}f} ± {t[1]:.{d}f} (n={t[2]})"


def main(tag=""):
    r = pd.read_csv(RES / f"pimc_rounds{tag}.csv")
    r["hit"] = (r["bid"] == r["tricks"]).astype(float)
    idx = ["n", "trump", "deal", "seat"]
    H = r[r.cond == "H"].set_index(idx)
    Pall = r[r.cond == "P"].set_index(idx)
    rows = []
    for (n, tr), g in r.groupby(["n", "trump"]):
        d1, d2, h1, h2 = [], [], [], []
        for s in range(4):
            a = g[(g.cond == f"P@{s}") & (g.seat == s)].set_index(idx)
            b = g[(g.cond == f"H@{s}") & (g.seat == s)].set_index(idx)
            d1.append((a["score"] - H.loc[a.index, "score"]).rename(s))
            h1.append((a["hit"] - H.loc[a.index, "hit"]).rename(s))
            d2.append((b["score"] - Pall.loc[b.index, "score"]).rename(s))
            h2.append((b["hit"] - Pall.loc[b.index, "hit"]).rename(s))
        f = lambda L: pd.concat(L).groupby(level=["n", "trump", "deal"]).mean().values
        gH = g[g.cond == "H"].groupby("deal")["score"].sum()
        gP = g[g.cond == "P"].groupby("deal")["score"].sum()
        rows.append({"n": n, "trump": tr, "deals": g.deal.nunique(),
                     "PIMC seat among heur: Δscore": fmt(mci(f(d1))), "Δhit": fmt(mci(f(h1)), 3),
                     "heur seat among PIMC: Δscore": fmt(mci(f(d2))), "Δhit ": fmt(mci(f(h2)), 3),
                     "table all-heur": f"{gH.mean():.2f}", "table all-PIMC": f"{gP.mean():.2f}",
                     "all-PIMC − all-heur (table)": fmt(mci((gP - gH).values)),
                     "PIMC s/decision (P cond)": f"{g[g.cond == 'P'].time_s.mean() / max(n, 1) / 4:.3f}"})
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(x[c]) for c in cols) + " |" for x in rows]
    text = "## F5 PIMC (K samples, max^n neutral) vs target-aware heuristic, hidden information\n" + "\n".join(out)
    (RES / f"summary_pimc{tag}.md").write_text(text)
    (RES / f"summary_pimc{tag}.json").write_text(json.dumps(rows, indent=1))
    print(text)


def k_sensitivity():
    """K = 4 / 16 / 64 on the same n=4 deals 0-299 (K16 rows taken from the main run)."""
    import contextlib
    import io
    r = pd.read_csv(RES / "pimc_rounds.csv")
    r[(r.n == 4) & (r.deal < 300)].to_csv(RES / "pimc_rounds_K16sub.csv", index=False)
    d = pd.read_csv(RES / "pimc_decisions.csv")
    d[(d.n == 4) & (d.deal < 300)].to_csv(RES / "pimc_decisions_K16sub.csv", index=False)
    rows = []
    for tag, K in (("_K4", 4), ("_K16sub", 16), ("_K64", 64)):
        with contextlib.redirect_stdout(io.StringIO()):
            main(tag)
        res = json.loads((RES / f"summary_pimc{tag}.json").read_text())
        dd = pd.read_csv(RES / f"pimc_decisions{tag}.csv")
        tr = np.array([json.loads(x.mv_true)[str(int(x.chosen))] == max(json.loads(x.mv_true).values())
                       for _, x in dd.iterrows()], float)
        for x in res:
            rows.append({"K": K, "trump": x["trump"], "deals": x["deals"],
                         "PIMC seat among heur Δ": x["PIMC seat among heur: Δscore"],
                         "heur seat among PIMC Δ": x["heur seat among PIMC: Δscore"],
                         "all-PIMC − all-heur": x["all-PIMC − all-heur (table)"]})
        rows.append({"K": K, "trump": "both", "PIMC card truly optimal %":
                     f"{100 * tr.mean():.2f} ± {196 * tr.std(ddof=1) / np.sqrt(len(tr)):.2f} (n={len(tr)})"})
    cols = ["K", "trump", "deals", "PIMC seat among heur Δ", "heur seat among PIMC Δ", "all-PIMC − all-heur",
            "PIMC card truly optimal %"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(x.get(c, "")) for c in cols) + " |" for x in rows]
    t = "## F5 PIMC sample-size sensitivity, n=4, deals 0-299 (paired, same deals and bids)\n" + "\n".join(out)
    (RES / "summary_pimc_K.md").write_text(t)
    print(t)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "K":
        k_sensitivity()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "")

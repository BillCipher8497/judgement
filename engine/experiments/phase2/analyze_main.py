"""Tables for F1-F4 from results/main_*.csv.  Writes results/summary_main.json and
prints markdown tables.  CIs: mean +- 1.96*SE, SE clustered by deal (per-deal
means first) whenever several seats of the same deal are pooled."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"


def mci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, 0)
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return (x.mean(), 1.96 * se, n)


def cl(df, col, by=("n", "trump", "deal")):
    """Deal-clustered mean/CI of col (average within deal first)."""
    g = df.groupby(list(by))[col].mean()
    return mci(g.values)


def fmt(t, d=2):
    m, h, n = t
    return f"{m:.{d}f} ± {h:.{d}f} (n={n})"


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    seat = pd.read_csv(RES / "main_seat.csv")
    dealdf = pd.read_csv(RES / "main_deal.csv", dtype={"R_bids": str, "G_bids": str, "O_bids": str, "RNH_bids": str})
    dec = pd.read_csv(RES / "main_dec.csv")
    seat["surplus"] = seat["maxn_neutral"] - seat["paranoid"]
    seat["surplus_sp"] = seat["maxn_spiteful"] - seat["paranoid"]
    seat["surplus_be"] = seat["maxn_benevolent"] - seat["paranoid"]
    seat["pos_surplus"] = (seat["surplus"] > 0).astype(float)
    seat["hit"] = (seat["maxn_neutral"] > 0).astype(float)
    seat["secure_hit"] = (seat["paranoid"] > 0).astype(float)
    summary = {}
    md = []

    # ---------------- F1 ----------------
    md.append("## F1 cooperation surplus (max^n neutral - paranoid), per seat; deal-clustered 95% CI")
    rows = []
    for (sch, n, tr), g in seat.groupby(["scheme", "n", "trump"]):
        r = {"scheme": sch, "n": n, "trump": tr, "deals": g["deal"].nunique(),
             "maxn": fmt(cl(g, "maxn_neutral")), "paranoid": fmt(cl(g, "paranoid")),
             "surplus": fmt(cl(g, "surplus")), "P(surplus>0)": fmt(cl(g, "pos_surplus"), 3),
             "share": f"{g['surplus'].sum() / max(g['maxn_neutral'].sum(), 1e-9):.3f}",
             "surplus_spite": fmt(cl(g, "surplus_sp")), "surplus_benev": fmt(cl(g, "surplus_be"))}
        rows.append(r)
    summary["F1_by_n"] = rows
    md.append(table(rows, ["scheme", "n", "trump", "deals", "maxn", "paranoid", "surplus", "P(surplus>0)", "share",
                           "surplus_spite", "surplus_benev"]))
    for sch in ("R", "O"):
        g0 = seat[seat.scheme == sch]
        rows = []
        for pos, g in g0.groupby("pos"):
            rows.append({"pos (0=first bidder,3=dealer)": pos, "surplus": fmt(mci(g["surplus"])),
                         "P(surplus>0)": fmt(mci(g["pos_surplus"]), 3), "maxn": fmt(mci(g["maxn_neutral"])),
                         "paranoid": fmt(mci(g["paranoid"]))})
        md.append(f"\n### F1 by bidding position, scheme {sch}, pooled n=1..5, S+NT")
        md.append(table(rows, list(rows[0])))
        summary[f"F1_by_pos_{sch}"] = rows
        rows = []
        for b, g in g0.groupby("bid"):
            if len(g) < 30:
                continue
            rows.append({"bid": b, "obs": len(g), "surplus": fmt(mci(g["surplus"])),
                         "P(surplus>0)": fmt(mci(g["pos_surplus"]), 3),
                         "P(hit) maxn": fmt(mci(g["hit"]), 3), "P(hit) secure": fmt(mci(g["secure_hit"]), 3)})
        md.append(f"\n### F1 by bid (=need at start), scheme {sch}, pooled")
        md.append(table(rows, list(rows[0])))
        summary[f"F1_by_bid_{sch}"] = rows

    # tie-break sensitivity
    md.append("\n## Equilibrium-selection sensitivity (per seat value differences across tie-breaks)")
    rows = []
    seat["d_sp"] = (seat["maxn_spiteful"] != seat["maxn_neutral"]).astype(float)
    seat["d_be"] = (seat["maxn_benevolent"] != seat["maxn_neutral"]).astype(float)
    seat["ad_sp"] = (seat["maxn_spiteful"] - seat["maxn_neutral"]).abs()
    seat["ad_be"] = (seat["maxn_benevolent"] - seat["maxn_neutral"]).abs()
    for (sch, n), g in seat.groupby(["scheme", "n"]):
        dd = g.groupby(["trump", "deal"])
        anyd = dd.apply(lambda x: float((x["d_sp"] + x["d_be"]).sum() > 0), include_groups=False)
        rows.append({"scheme": sch, "n": n, "P(seat val differs) spite": fmt(cl(g, "d_sp"), 3),
                     "benev": fmt(cl(g, "d_be"), 3), "E|diff| spite": fmt(cl(g, "ad_sp")),
                     "E|diff| benev": fmt(cl(g, "ad_be")), "P(any seat differs)": fmt(mci(anyd.values), 3)})
    summary["tiebreak"] = rows
    md.append(table(rows, list(rows[0])))

    # ---------------- F2 ----------------
    md.append("\n## F2 price of anarchy: social optimum vs max^n table total")
    rows = []
    for sch in ("R", "G", "O"):
        for (n, tr), g in dealdf.groupby(["n", "trump"]):
            soc = g[f"{sch}_social"].astype(float)
            r = {"scheme": sch, "n": n, "trump": tr, "deals": len(g), "social": fmt(mci(soc))}
            for m in ("neutral", "spiteful", "benevolent"):
                tot = g[f"{sch}_sum_{m}"].astype(float)
                r[f"gap {m}"] = fmt(mci(soc - tot))
                r[f"hits {m}"] = fmt(mci(g[f"{sch}_hits_{m}"]))
            r["P(gap>0) neutral"] = fmt(mci((soc - g[f"{sch}_sum_neutral"] > 0).astype(float)), 3)
            r["PoA=soc/maxn"] = f"{soc.sum() / g[f'{sch}_sum_neutral'].sum():.3f}"
            rows.append(r)
    summary["F2"] = rows
    md.append(table(rows, list(rows[0])))

    # ---------------- F3 ----------------
    md.append("\n## F3 perfect-info rational bidding (R, hook) vs hook off (RNH) vs naive greedy-trick bids (G)")
    rows = []
    for (n, tr), g in dealdf.groupby(["n", "trump"]):
        sb = g["R_bids"].map(lambda s: sum(map(int, s))) - n
        r = {"n": n, "trump": tr, "deals": len(g), "Σbids-n (R)": fmt(mci(sb)),
             "P(Σbids<n)": fmt(mci((sb < 0).astype(float)), 3), "P(Σbids>n)": fmt(mci((sb > 0).astype(float)), 3),
             "hit rate R": fmt(mci(g["R_hits_neutral"] / 4), 3),
             "table pts R": fmt(mci(g["R_sum_neutral"]))}
        sbg = g["G_bids"].map(lambda s: sum(map(int, s))) - n
        r["Σbids-n (G)"] = fmt(mci(sbg))
        r["hit rate G (maxn play)"] = fmt(mci(g["G_hits_neutral"] / 4), 3)
        r["table pts G"] = fmt(mci(g["G_sum_neutral"]))
        sbo = g["O_bids"].map(lambda s: sum(map(int, s))) - n
        r["Σbids-n (O)"] = fmt(mci(sbo))
        r["hit rate O"] = fmt(mci(g["O_hits_neutral"] / 4), 3)
        nh = g.dropna(subset=["RNH_bids"])
        sbn = nh["RNH_bids"].map(lambda s: sum(map(int, s))) - n
        r["Σbids-n (RNH)"] = fmt(mci(sbn))
        r["P(Σbids=n) RNH"] = fmt(mci((sbn == 0).astype(float)), 3)
        rows.append(r)
    summary["F3"] = rows
    md.append(table(rows, list(rows[0])))
    # seat value by bidding position, hook on vs off
    recs = []
    for _, d in dealdf.dropna(subset=["RNH_vals"]).iterrows():
        nh = json.loads(d["RNH_vals"])
        for p in range(4):
            pos = (p - d["dealer"] - 1) % 4
            recs.append({"n": d["n"], "trump": d["trump"], "deal": d["deal"], "pos": pos, "nohook": nh[p]})
    nhdf = pd.DataFrame(recs)
    rs = seat[seat.scheme == "R"][["n", "trump", "deal", "pos", "maxn_neutral"]]
    mg = rs.merge(nhdf, on=["n", "trump", "deal", "pos"])
    mg["hookcost"] = mg["nohook"] - mg["maxn_neutral"]
    rows = []
    for (n, pos), g in mg.groupby(["n", "pos"]):
        rows.append({"n": n, "pos": pos, "value hook": fmt(mci(g["maxn_neutral"])), "value no-hook": fmt(mci(g["nohook"])),
                     "no-hook − hook": fmt(mci(g["hookcost"]))})
    summary["F3_pos"] = rows
    md.append("\n### F3 seat value by bidding position (3 = dealer), rational bids, hook vs no hook (paired deals)")
    md.append(table(rows, list(rows[0])))

    # ---------------- F4 ----------------
    md.append("\n## F4 target-aware heuristic vs max^n under perfect information")
    rows = []
    for (sch, n), g in dec[dec.path == "maxn"].groupby(["scheme", "n"]):
        agree = (g["v_HEUR"] == g["vmax"]).astype(float)
        ni = g[g.indiff == 0]
        agree_ni = (ni["v_HEUR"] == ni["vmax"]).astype(float)
        rows.append({"scheme": sch, "n": n, "decisions (non-forced)": len(g), "agree all": fmt(mci(agree), 3),
                     "non-indifferent": len(ni), "agree non-indiff": fmt(mci(agree_ni), 3),
                     "1-step loss/decision": fmt(mci(g["vmax"] - g["v_HEUR"]), 3)})
    summary["F4_agree"] = rows
    md.append(table(rows, list(rows[0])))
    rows = []
    for (sch, n), g in seat[seat.scheme.isin(["R", "O"])].groupby(["scheme", "n"]):
        g = g.copy()
        g["loss1"] = g["maxn_neutral"] - g["oneheur"].astype(float)
        dsum = g.groupby(["trump", "deal"]).agg(mx=("maxn_neutral", "sum"), ah=("allheur", "sum"))
        rows.append({"scheme": sch, "n": n, "seats": len(g),
                     "loss: 1 heur seat vs max^n others": fmt(mci(g["loss1"])),
                     "P(loss>0)": fmt(mci((g["loss1"] > 0).astype(float)), 3),
                     "table max^n": fmt(mci(dsum["mx"])), "table all-heur": fmt(mci(dsum["ah"].astype(float))),
                     "table loss": fmt(mci(dsum["mx"] - dsum["ah"].astype(float)))})
    summary["F4_loss"] = rows
    md.append(table(rows, list(rows[0])))

    # ---------------- F1/F2 (a): all hook-legal bid vectors, n <= 3 ----------------
    ef = RES / "enum.csv"
    if ef.exists():
        e = pd.read_csv(ef, dtype={"bids": str})
        e["surplus"] = e["maxn_neutral"] - e["paranoid"]
        e["pos_surplus"] = (e["surplus"] > 0).astype(float)
        e["d_sp"] = (e["maxn_spiteful"] != e["maxn_neutral"]).astype(float)
        e["d_be"] = (e["maxn_benevolent"] != e["maxn_neutral"]).astype(float)
        tv = e.groupby(["n", "trump", "deal", "bids"]).agg(soc=("social_total", "first"), neu=("maxn_neutral", "sum"),
                                                         sp=("maxn_spiteful", "sum"), be=("maxn_benevolent", "sum"))
        rows = []
        for (n, tr), g in e.groupby(["n", "trump"]):
            t = tv.loc[(n, tr)]
            gap = (t["soc"] - t["neu"]).groupby(level="deal").mean().values
            gsp = (t["soc"] - t["sp"]).groupby(level="deal").mean().values
            pg = ((t["soc"] - t["neu"]) > 0).astype(float).groupby(level="deal").mean().values
            rows.append({"n": n, "trump": tr, "deals": g.deal.nunique(), "bid vectors/deal": g.groupby("deal").bids.nunique().iloc[0],
                         "surplus/seat": fmt(cl(g, "surplus")), "P(surplus>0)": fmt(cl(g, "pos_surplus"), 3),
                         "share": f"{g['surplus'].sum() / max(g['maxn_neutral'].sum(), 1e-9):.3f}",
                         "P(val differs) spite": fmt(cl(g, "d_sp"), 3), "benev": fmt(cl(g, "d_be"), 3),
                         "social − Σmax^n (neutral)": fmt(mci(gap)), "social − Σmax^n (spite)": fmt(mci(gsp)),
                         "P(gap>0)": fmt(mci(pg), 3)})
        summary["F1F2_enum"] = rows
        md.append("\n## F1/F2 (a): averaged uniformly over ALL hook-legal bid vectors (n<=3), deal-clustered CI")
        md.append(table(rows, list(rows[0])))

    (RES / "summary_main.json").write_text(json.dumps(summary, indent=1, default=str))
    text = "\n".join(md)
    (RES / "summary_main.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

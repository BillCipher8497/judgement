"""Tables for Phase 3 (round2_bid).  python analyze.py [p1|p2|p3|val|all]"""
from __future__ import annotations

import json
import sys

import numpy as np

from common2 import RES, ci, fmt, fmtu  # noqa: E402


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def p1():
    d = load("p1_raw.json")
    if d is None:
        return ""
    rows = d["rows"]
    out = ["## P1a bid-informed (I) vs uniform (U) sampling in the TA bid oracle",
           f"args: {d['args']}",
           "dEV = exact-posterior evaluation (fresh i.i.d. worlds, rejection eps=0), paired per situation; "
           "real = realised score on the true deal.  M = ModelBidder bid.",
           "",
           "| n | pos | P(b_I != b_U) | dEV I−U | real I−U | dEV U−M | dEV I−M | accept rate (I) |",
           "|---|---|---|---|---|---|---|---|"]
    ns = sorted({r["n"] for r in rows})
    for n in ns + ["all"]:
        for pos in (1, 2, 3, "all"):
            R = [r for r in rows if (n == "all" or r["n"] == n) and (pos == "all" or r["pos"] == pos)]
            if not R or (n != "all" and pos == "all" and False):
                continue
            if n == "all" and pos == "all":
                pass
            diff = [r["bi"] != r["bu"] for r in R]
            out.append(f"| {n} | {pos} | {fmtu(diff)} | {fmt([r['ev_i']-r['ev_u'] for r in R])} | "
                       f"{fmt([r['real_i']-r['real_u'] for r in R])} | {fmt([r['ev_u']-r['ev_m'] for r in R])} | "
                       f"{fmt([r['ev_i']-r['ev_m'] for r in R])} | {np.mean([r['acc'] for r in R]):.3f} |")
    # direction of change
    out += ["", "Direction: mean(b_I − b_U) by pos and by sign of prior excess (Σprior − pos·n/4)", "",
            "| pos | mean b_I−b_U | when prior excess>0 | when <0 |", "|---|---|---|---|"]
    for pos in (1, 2, 3):
        R = [r for r in rows if r["pos"] == pos and r["n"] > 1]
        ex = lambda r: sum(r["prior"]) - pos * r["n"] / 4  # noqa: E731
        out.append(f"| {pos} | {fmt([r['bi']-r['bu'] for r in R],3)} | {fmt([r['bi']-r['bu'] for r in R if ex(r)>0],3)} | "
                   f"{fmt([r['bi']-r['bu'] for r in R if ex(r)<0],3)} |")
    rc = d["rc"]
    if rc:
        out += ["", "## P1b response curve: d(optimal bid)/d(Σprior bids), one earlier bid shifted ±1",
                "slope = (b*(+1) − b*(−1))/2 per situation.  table = uniform worlds + shifted rollouts; "
                "info = informed worlds (shifted likelihood) + true rollouts; both = informed + shifted; model = ModelBidder.",
                "", "| n | pos | situations | table | info | both | both − table | model |", "|---|---|---|---|---|---|---|---|"]
        for n in sorted({r["n"] for r in rc}) + ["all"]:
            for pos in (1, 2, 3, "all"):
                R = [r for r in rc if (n == "all" or r["n"] == n) and (pos == "all" or r["pos"] == pos)]
                if not R:
                    continue
                s = {k: [(r[f"{k}+1"] - r[f"{k}-1"]) / 2 for r in R] for k in ("table", "info", "both", "model")}
                out.append(f"| {n} | {pos} | {len(R)} | {fmt(s['table'],3)} | {fmt(s['info'],3)} | {fmt(s['both'],3)} | "
                           f"{fmt(np.array(s['both'])-np.array(s['table']),3)} | {fmt(s['model'],3)} |")
    return "\n".join(out)


def p2():
    d = load("p2_raw.json")
    if d is None:
        return ""
    out = ["## P2 win-maximiser (1) vs score-maximisers (3), final k rounds",
           f"args: {d['args']}", "",
           "| k | games | ΔP(win) (per game, mean of 4 seats) | Δ mean score (hero) | P(win) base | P(win) treat | bids changed / decisions |",
           "|---|---|---|---|---|---|---|"]
    for k, v in d["res"].items():
        out.append(f"| {k} | {len(v['dwin'])} | {fmt(v['dwin'],4)} | {fmt(v['dscore'])} | {fmtu(v['base_win'],4)} | "
                   f"{fmtu(v['treat_win'],4)} | {v['changed']}/{v['decisions']} ({v['changed']/max(1,v['decisions']):.3f}) |")
    out += ["", "By hero seat (seat 0 deals the final round 24 → dealer):", "", "| k | seat | ΔP(win) |", "|---|---|---|"]
    for k, v in d["res"].items():
        for h in range(4):
            out.append(f"| {k} | {h} | {fmt([x for s, x in v['dwin_seat'] if s == h],4)} |")
    return "\n".join(out)


def p3():
    d = load("p3_raw.json")
    if d is None:
        return ""
    rows = d["rows"]
    deltas = [k for k in rows[0] if k.startswith("d") and k[1:].replace(".", "").isdigit()]
    out = ["## P3 hook-setting by the bidder before the dealer (seat 2; dealer seat 3)",
           f"args: {d['args']}", "",
           "| n | δ | deals | P(bid changed) | Δ seat-2 score | Δ dealer score | Δ seat 0 | Δ seat 1 | P(dealer hooked) base → rule | Δ(seat2 − mean others) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for dk in deltas:
        for n in sorted({r["n"] for r in rows}) + ["all"]:
            R = [r for r in rows if n == "all" or r["n"] == n]
            ch = [r[dk]["bids"][2] != r["base_bids"][2] for r in R]
            ds = np.array([[r[dk]["scores"][p] - r["base_scores"][p] for p in range(4)] for r in R])
            hb = [r["base_hooked"][0] for r in R]
            hr = [r[dk]["hooked"][0] for r in R]
            rel = ds[:, 2] - ds[:, [0, 1, 3]].mean(1)
            out.append(f"| {n} | {dk[1:]} | {len(R)} | {fmtu(ch,3)} | {fmt(ds[:,2])} | {fmt(ds[:,3])} | {fmt(ds[:,0])} | "
                       f"{fmt(ds[:,1])} | {np.mean(hb):.3f} → {np.mean(hr):.3f} | {fmt(rel)} |")
    return "\n".join(out)


def val():
    d = load("p1_validate.json")
    if d is None:
        return ""
    return d.get("table", "")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    parts = []
    for name, fn in (("p1", p1), ("val", val), ("p2", p2), ("p3", p3)):
        if which in (name, "all"):
            parts.append(fn())
    txt = "\n\n".join(p for p in parts if p)
    print(txt)
    if which == "all":
        (RES / "summary_round2_bid.md").write_text(txt + "\n")

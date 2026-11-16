"""Tables for Phase 3.  python analyze3.py [p1|p2|p3|p4]  -> prints and writes results/tables_<p>.txt"""
from __future__ import annotations

import json
import sys

import numpy as np

from common3 import RES, ci, fmt

import os
TAG = os.environ.get("P3TAG", "")


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def slope(rec, c):
    ds = rec["ds"]
    if len(ds) == 3:
        return (rec[f"{c}+1"] - rec[f"{c}-1"]) / 2
    return rec[f"{c}{ds[1]:+d}"] - rec[f"{c}{ds[0]:+d}"]


def p1():
    d = load("p1_raw.json")
    rows, rc = d["rows"], d["rc"]
    L = [f"## P1a bid-informed oracle: ΔEV per situation (points/round), exact-posterior evaluation (KE={d['args']['keval']}), CRN",
         f"args {d['args']}, soft tau={d['tau']:.3f}.  I=exact posterior (hard), S=soft tau, U=uniform, M=ModelBidder.",
         "pos 1 = seat 2 (second bidder), 2 = seat 3, 3 = dealer.", "",
         "| n | pos | P(b_I≠b_U) | ΔEV I−U | ΔEV S−U | ΔEV U−M | ΔEV I−M | real I−U |", "|---|---|---|---|---|---|---|---|"]
    keys = [(n, p) for n in sorted({r["n"] for r in rows}) for p in (1, 2, 3)] + [("all", p) for p in (1, 2, 3)] + [("all", "all")]
    for n, p in keys:
        R = [r for r in rows if (n == "all" or r["n"] == n) and (p == "all" or r["pos"] == p)]
        if not R:
            continue
        L.append(f"| {n} | {p} | {fmt([r['bi'] != r['bu'] for r in R], 2, False)} | {fmt([r['ev_i']-r['ev_u'] for r in R])} | "
                 f"{fmt([r['ev_s']-r['ev_u'] for r in R])} | {fmt([r['ev_u']-r['ev_m'] for r in R])} | "
                 f"{fmt([r['ev_i']-r['ev_m'] for r in R])} | {fmt([r['real_i']-r['real_u'] for r in R])} |")
    L += ["", "## P1b response curve: d(optimal bid)/d(Σ prior bids), one earlier bid shifted by ±1 (CRN worlds)",
          "table = uniform worlds, rollouts see shifted bids (table-system effect); info = posterior from shifted bids, rollouts see true bids "
          "(information effect); both = bid-informed oracle; model = ModelBidder; U-oracle total = table.", "",
          "| n | pos | rows | table | info | both | table+info−both | model |", "|---|---|---|---|---|---|---|---|"]
    out = {}
    keys = [(n, p) for n in sorted({r["n"] for r in rc}) for p in (1, 2, 3)] + [("all", p) for p in (1, 2, 3)] + [(n, "all") for n in sorted({r["n"] for r in rc})] + [("all", "all")]
    for n, p in keys:
        R = [r for r in rc if (n == "all" or r["n"] == n) and (p == "all" or r["pos"] == p)]
        if not R:
            continue
        s = {c: [slope(r, c) for r in R] for c in ("table", "info", "both", "model")}
        inter = [a + b - c for a, b, c in zip(s["table"], s["info"], s["both"])]
        out[f"{n}|{p}"] = {c: ci(v) for c, v in s.items()}
        L.append(f"| {n} | {p} | {len(R)} | {fmt(s['table'])} | {fmt(s['info'])} | {fmt(s['both'])} | {fmt(inter)} | {fmt(s['model'])} |")
    return "\n".join(L), out


def p2():
    d = load("p2_raw" + TAG + ".json")
    L = [f"## P2 win-maximiser (last k rounds) vs score-maximisers; args {d['args']}", "",
         "| k | standing at take-over | units | ΔP(win) | Δ mean score | base P(win) | mean gap to leader |", "|---|---|---|---|---|---|---|"]
    out = {}
    for k, U in d["units"].items():
        U = np.array(U)
        G = U.reshape(-1, 4, 8)
        L.append(f"| {k} | all (game units) | {len(G)} | {fmt((G[:, :, 5]-G[:, :, 4]).mean(1), 4)} | {fmt((G[:, :, 7]-G[:, :, 6]).mean(1))} | "
                 f"{U[:, 4].mean():.3f} | - |")
        out[k] = {"all_dwin": ci((G[:, :, 5]-G[:, :, 4]).mean(1)), "all_dscore": ci((G[:, :, 7]-G[:, :, 6]).mean(1))}
        for rk, nm in zip((1, 2, 3, 4), ("leader", "2nd", "3rd", "last")):
            m = U[:, 2] == rk
            L.append(f"| {k} | {nm} | {m.sum()} | {fmt(U[m, 5]-U[m, 4], 4)} | {fmt(U[m, 7]-U[m, 6])} | {U[m, 4].mean():.3f} | {U[m, 3].mean():.1f} |")
            out[k][nm] = {"dwin": ci(U[m, 5]-U[m, 4]), "dscore": ci(U[m, 7]-U[m, 6])}
    L += ["", "### P2 variance test: decisions where the win-maximiser deviates from the score-max bid, by standing at that decision",
          "sd = model round-score SD = (10+b)·sqrt(p(1−p)), p = P_model(hit | b).  Δ = chosen − score-max bid.", "",
          "| k | standing | decisions | P(deviate) | of deviations: P(higher bid) | Δbid | Δ P(hit) | Δ sd | Δ E[round score] | model ΔP(win) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, D in d["dec"].items():
        D = np.array(D, float)
        for rk, nm in zip((1, 2, 3, 4), ("leader", "2nd", "3rd", "last")):
            m = D[:, 0] == rk
            X = D[m]
            dev = X[:, 4] != X[:, 3]
            Y = X[dev]
            if len(Y) < 2:
                L.append(f"| {k} | {nm} | {m.sum()} | {dev.mean():.3f} | - | - | - | - | - | - |")
                continue
            bm, b, pm, pb = Y[:, 3], Y[:, 4], Y[:, 5], Y[:, 6]
            sdm, sdb = (10 + bm) * np.sqrt(pm * (1 - pm)), (10 + b) * np.sqrt(pb * (1 - pb))
            em, eb = pm * (10 + bm), pb * (10 + b)
            L.append(f"| {k} | {nm} | {m.sum()} | {fmt(dev, 3, False)} | {fmt(b > bm, 2, False)} | {fmt(b-bm)} | {fmt(pb-pm, 3)} | "
                     f"{fmt(sdb-sdm)} | {fmt(eb-em)} | {fmt(Y[:, 8]-Y[:, 7], 4)} |")
    L += ["", "### P2 win-model calibration: predicted P(win) of the chosen bid at the take-over decision vs realised (fractional) win",
          "", "| k | pred bin | units | mean pred | realised ± 95% |", "|---|---|---|---|---|"]
    for k, D in d["dec"].items():
        D = np.array(D, float)
        U = np.array(d["units"][k])
        first = {}
        for row in D:
            first.setdefault(int(row[10]), row)
        idx = np.array(list(first.keys()))
        pred = np.array([first[i][8] for i in idx])
        real = U[idx, 5]
        for lo, hi in ((0, .02), (.02, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .98), (.98, 1.01)):
            m = (pred >= lo) & (pred < hi)
            if m.sum() > 1:
                L.append(f"| {k} | {lo:.2f}-{min(hi, 1):.2f} | {m.sum()} | {pred[m].mean():.3f} | {fmt(real[m], 3, False)} |")
    return "\n".join(L), out


def p3():
    d = load("p3_raw" + TAG + ".json")
    rows = d["rows"]
    deltas = [k for k in rows[0] if k.startswith("d") and k[1:].replace(".", "").isdigit()]
    L = [f"## P3 hook-setting by seat 3 (bids just before the dealer); args {d['args']}",
         "Δ = treatment − baseline on the same deal (exact CRN; deterministic play).  seat3 = hook-setter, dealer = seat 4.", "",
         "| δ | n | deals | P(bid changed) | Δ seat3 score | Δ dealer score | Δ seat1+2 | dealer hooked base→treat | changed only: Δ seat3 | changed only: Δ dealer |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for dl in deltas:
        for n in sorted({r["n"] for r in rows}) + ["all"]:
            R = [r for r in rows if n == "all" or r["n"] == n]
            ch = [r[dl]["bids"][2] != r["base_bids"][2] for r in R]
            d3 = [r[dl]["scores"][2] - r["base_scores"][2] for r in R]
            dd = [r[dl]["scores"][3] - r["base_scores"][3] for r in R]
            d12 = [r[dl]["scores"][0] + r[dl]["scores"][1] - r["base_scores"][0] - r["base_scores"][1] for r in R]
            hb = np.mean([r["base_hooked"][0] for r in R])
            ht = np.mean([r[dl]["hooked"][0] for r in R])
            ddc = [x for x, c in zip(dd, ch) if c]
            out[f"{dl}|{n}"] = {"seat3": ci(d3), "dealer": ci(dd), "changed": float(np.mean(ch))}
            L.append(f"| {dl[1:]} | {n} | {len(R)} | {np.mean(ch):.3f} | {fmt(d3)} | {fmt(dd)} | {fmt(d12)} | {hb:.3f}→{ht:.3f} | "
                     f"{fmt([x for x, c in zip(d3, ch) if c])} | {fmt(ddc)} |")
    return "\n".join(L), out


def p4():
    L = ["## P4 tournament: one challenger vs 3 TA (ModelBidder+TA), seat-rotated, CRN; Δ vs all-TA baseline, unit = game", "",
         "| challenger | games | Δ mean score / game | ΔP(win) (base 0.25) |", "|---|---|---|---|"]
    out = {}
    for arm in ("ta_spite", "greedy", "oracle_u", "oracle_i"):
        d = load(f"p4_{arm}.json")
        if d is None:
            continue
        L.append(f"| {arm} | {len(d['dscore'])} | {fmt(d['dscore'])} | {fmt(d['dwin'], 4)} |")
        out[arm] = {"dscore": ci(d["dscore"]), "dwin": ci(d["dwin"])}
    return "\n".join(L), out


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["p1", "p2", "p3", "p4"]):
        txt, out = globals()[w]()
        (RES / f"tables_{w}.txt").write_text(txt + "\n")
        json.dump(out, open(RES / f"summary_{w}.json", "w"), indent=1)
        print(txt)

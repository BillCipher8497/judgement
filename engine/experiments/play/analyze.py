"""Summaries -> results/summary.md + summary.json.  95% CIs: means +-1.96 SE with
games (1v3) or blocks (mixed) as units; ratio metrics (overshoot share) and
BT/Elo by bootstrap (B=1000 / 500, seed 0)."""
import json
import sys

import numpy as np

from common import RES, mean_ci
from judgement import Rules

rng = np.random.default_rng(0)
OUT, J = [], {}
RULES = Rules()
LAD = np.array(RULES.ladder())
TRUMPS = np.array([RULES.trump_for_round(i) for i in range(RULES.n_rounds)])


def fm(m, h, d=2, sign=True):
    return (f"{m:+.{d}f}" if sign else f"{m:.{d}f}") + f" ± {h:.{d}f}"


def boot_ratio(num, den, B=1000):
    n = len(num)
    r = num.sum() / max(den.sum(), 1)
    idx = rng.integers(0, n, (B, n))
    rs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return float(r), float(lo), float(hi)


def load(tag):
    p = RES / f"1v3_{tag}.npz"
    return np.load(p) if p.exists() else None


def seat_view(d, G):
    s = d["seat"][:G]
    ar = np.arange(G)
    return dict(tot=d["totals"][:G][ar, s], win=d["wins"][:G][ar, s], b=d["bids"][:G][ar, :, s],
                t=d["tricks"][:G][ar, :, s], sc=d["scores"][:G][ar, :, s],
                opp=(d["totals"][:G].sum(1) - d["totals"][:G][ar, s]) / 3,
                secs=d["secs"][:G] if "secs" in d else np.zeros(G))


T1 = [("x_ta", "TA (all-TA reference)"), ("x_ta_plus", "**TA+** (shed + lead-high[trump] + theorems + win-high)"),
      ("x_none", "TA+ all components off (sanity = TA)"),
      ("x_no_shed", "TA+ − shed"), ("x_no_lh", "TA+ − lead-high"), ("x_no_thm", "TA+ − theorems"), ("x_no_wh", "TA+ − win-high (= round-2 TA+)"),
      ("x_shed_only", "shed only"), ("x_lh_only", "lead-high only"), ("x_thm_only", "theorems only"), ("x_wh_only", "win-high only"),
      ("x_lh_all", "TA+ with lead-high in NT too"), ("x_plus_duck", "TA+ + surplus-duck"),
      ("x_plus_duck_last", "TA+ + surplus-duck (last seat only)"), ("x_plus_scarce", "TA+ + scarce-compete")]
T2 = [("x_ta_plus", "TA+ (no search)"), ("x_hyb_c4", "Hybrid K=16, PIMC at ≤4 cards"),
      ("x_hyb_c5", "Hybrid K=16, PIMC at ≤5 cards"), ("x_hyb_c4_nt", "Hybrid K=16 ≤4, TA+ in NT"),
      ("x_hyb_c5_nt", "Hybrid K=16 ≤5, TA+ in NT"), ("x_hyb_k32_c5", "Hybrid K=32, PIMC at ≤5 cards"),
      ("x_hyb_c6", "Hybrid K=16, PIMC at ≤6 cards")]
T3 = [("x_greedy", "Greedy (greedy bidder)"), ("x_ta_spite", "TA-spite"), ("x_ta_plus", "TA+"),
      ("x_hyb_c5", "Hybrid (K=16, ≤5)"), ("x_hyb_c4", "Hybrid (K=16, ≤4)"),
      ("x_orc_n5", "TA+ + bid-informed oracle bidder (n≤5, K=24, ±2)"),
      ("x_orc_all", "TA+ + bid-informed oracle bidder (all rounds, K=24, ±2)"),
      ("x_hyb_orc", "Hybrid (K=16, ≤5) + bid-informed oracle bidder (all rounds)")]


def one_vs_three(title, tags, key):
    ref = load("x_ta")
    tap = load("x_ta_plus")
    rows = []
    OUT.append(f"## {title}\n")
    OUT.append("Challenger in seat g%4 vs 3 TA, seed 71,000,000+g; Δ paired with the all-TA table (same game, same seat).\n")
    OUT.append("| challenger | games | Δscore/game vs TA | ΔP(win) vs TA | P(win) | hit rate | overshoot share of misses | Δ vs TA+ /game | opp. Δ/game (each) | s/game |")
    OUT.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, label in tags:
        d = load(tag)
        if d is None:
            continue
        G = min(len(d["totals"]), len(ref["totals"]))
        x, r0 = seat_view(d, G), seat_view(ref, G)
        R = x["b"].shape[1]
        miss = x["b"] != x["t"]
        over = miss & (x["t"] > x["b"])
        dm = mean_ci(x["tot"] - r0["tot"])
        dw = mean_ci(x["win"] - r0["win"])
        wm = mean_ci(x["win"])
        hm = mean_ci((~miss).sum(1) / R)
        osr = boot_ratio(over.sum(1), miss.sum(1))
        om = mean_ci(x["opp"] - r0["opp"])
        if tap is not None and tag != "x_ta_plus":
            Gp = min(G, len(tap["totals"]))
            vp = mean_ci(x["tot"][:Gp] - seat_view(tap, Gp)["tot"])
            vps = fm(*vp[:2])
        else:
            vp, vps = (0.0, 0.0, G), "—"
        split = {}
        dsr = x["sc"] - r0["sc"]
        for nm, m in (("trump", TRUMPS >= 0), ("nt", TRUMPS < 0), ("n_le5", LAD <= 5), ("n_ge6", LAD >= 6)):
            split[nm] = mean_ci(dsr[:, m].mean(1))[:2]
        rows.append(dict(tag=tag, label=label, games=G, dscore=dm[:2], dwin=dw[:2], pwin=wm[:2], hit=hm[:2],
                         overshoot_share=osr, over_rate=float(over.mean()), under_rate=float((miss & ~over).mean()),
                         d_vs_taplus=vp[:2], opp=om[:2], secs=float(x["secs"].mean()), split=split))
        OUT.append(f"| {label} | {G} | {fm(*dm[:2])} | {fm(*dw[:2], 3)} | {fm(*wm[:2], 3, False)} | {fm(*hm[:2], 4, False)} | "
                   f"{osr[0]:.3f} [{osr[1]:.3f},{osr[2]:.3f}] | {vps} | {fm(*om[:2])} | {x['secs'].mean():.3f} |")
    OUT.append("\nPer-round Δscore vs TA by round type (same games):\n")
    OUT.append("| challenger | trump Δ/round | NT Δ/round | n≤5 Δ/round | n≥6 Δ/round | over / under rate per round |")
    OUT.append("|---|---|---|---|---|---|")
    for r in rows:
        s = r["split"]
        OUT.append(f"| {r['label']} | " + " | ".join(fm(*s[k], 3) for k in ("trump", "nt", "n_le5", "n_ge6"))
                   + f" | {r['over_rate']:.4f} / {r['under_rate']:.4f} |")
    OUT.append("")
    J[key] = rows


def contrasts(pairs):
    OUT.append("Paired contrasts (same games, same seat):\n")
    OUT.append("| A − B | games | Δscore/game | ΔP(win) |")
    OUT.append("|---|---|---|---|")
    res = []
    for a, b in pairs:
        da, db = load(a), load(b)
        if da is None or db is None:
            continue
        G = min(len(da["totals"]), len(db["totals"]))
        xa, xb = seat_view(da, G), seat_view(db, G)
        m, w = mean_ci(xa["tot"] - xb["tot"]), mean_ci(xa["win"] - xb["win"])
        res.append(dict(a=a, b=b, games=G, dscore=m[:2], dwin=w[:2]))
        OUT.append(f"| {a} − {b} | {G} | {fm(*m[:2])} | {fm(*w[:2], 3)} |")
    OUT.append("")
    J.setdefault("contrasts", []).extend(res)


def bt_fit(W, iters=5000):
    k = len(W)
    p = np.ones(k)
    N = W + W.T
    for _ in range(iters):
        den = np.array([sum(N[i, j] / (p[i] + p[j]) for j in range(k) if j != i) for i in range(k)])
        pn = W.sum(1) / den
        pn /= np.exp(np.log(pn).mean())
        if np.max(np.abs(pn - p)) < 1e-10:
            return pn
        p = pn
    return p


def mixed():
    p = RES / "mixed.npz"
    if not p.exists():
        return
    d = np.load(p)
    pols = list(d["policies"])
    P = len(pols)
    A, T, Wn, blk = d["assign"], d["totals"], d["wins"], d["block"]
    bids, tricks = d["bids"], d["tricks"]
    B = int(blk.max()) + 1
    # per block: pairwise W matrix, per-policy score / wins / hits
    Wb = np.zeros((B, P, P))
    S = np.zeros((B, P)); Wi = np.zeros((B, P)); H = np.zeros((B, P)); M = np.zeros((B, P)); O = np.zeros((B, P)); C = np.zeros((B, P))
    for g in range(len(A)):
        b = blk[g]
        for s in range(4):
            i = A[g][s]
            S[b, i] += T[g][s]; Wi[b, i] += Wn[g][s]; C[b, i] += 1
            mi = bids[g][:, s] != tricks[g][:, s]
            H[b, i] += (~mi).sum(); M[b, i] += mi.sum(); O[b, i] += (mi & (tricks[g][:, s] > bids[g][:, s])).sum()
            for s2 in range(4):
                if s2 != s:
                    j = A[g][s2]
                    Wb[b, i, j] += 1.0 if T[g][s] > T[g][s2] else (0.5 if T[g][s] == T[g][s2] else 0.0)
    ita = pols.index("ta")
    elo = lambda pp: 400 * np.log10(pp / pp[ita])  # noqa: E731
    e0 = elo(bt_fit(Wb.sum(0)))
    bs = []
    for _ in range(500):
        idx = rng.integers(0, B, B)
        bs.append(elo(bt_fit(Wb[idx].sum(0), 400)))
    bs = np.array(bs)
    ms = S / C
    R = bids.shape[1]
    OUT.append(f"## Mixed-table round robin ({len(A)} games = {B} blocks × {len(A)//B} tables; all {P}-choose-4 subsets × 4 seat rotations per block; seed 81,000,000+b)\n")
    OUT.append("Pairwise BT outcome: higher game total within a table (tie = ½). CIs: over blocks (scores) / block bootstrap (Elo).\n")
    OUT.append("| policy | mean score/game | Δ vs TA (paired by block) | P(win game) | hit rate | overshoot share | BT Elo vs TA [95% CI] |")
    OUT.append("|---|---|---|---|---|---|---|")
    rows = []
    for i, nm in enumerate(pols):
        sm = mean_ci(ms[:, i]); dm = mean_ci(ms[:, i] - ms[:, ita]); wm = mean_ci(Wi[:, i] / C[:, i])
        hm = mean_ci(H[:, i] / (C[:, i] * R)); osr = boot_ratio(O[:, i], M[:, i])
        lo, hi = np.percentile(bs[:, i], [2.5, 97.5])
        rows.append(dict(policy=nm, score=sm[:2], dscore=dm[:2], pwin=wm[:2], hit=hm[:2], overshoot=osr, elo=float(e0[i]), elo_ci=[float(lo), float(hi)]))
        OUT.append(f"| {nm} | {fm(*sm[:2], 1, False)} | {fm(*dm[:2])} | {fm(*wm[:2], 3, False)} | {fm(*hm[:2], 4, False)} | "
                   f"{osr[0]:.3f} [{osr[1]:.3f},{osr[2]:.3f}] | {e0[i]:+.0f} [{lo:+.0f},{hi:+.0f}] |")
    # head-to-head share matrix
    Wt = Wb.sum(0)
    OUT.append("\nHead-to-head: share of tables where row out-scores column\n")
    OUT.append("| | " + " | ".join(pols) + " |")
    OUT.append("|---" * (P + 1) + "|")
    for i in range(P):
        OUT.append(f"| {pols[i]} | " + " | ".join("—" if i == j else f"{Wt[i, j] / (Wt[i, j] + Wt[j, i]):.3f}" for j in range(P)) + " |")
    OUT.append(f"\nmean s/game: {d['secs'].mean():.3f}\n")
    J["mixed"] = dict(games=int(len(A)), blocks=B, rows=rows)


def pi():
    import csv
    p = RES / "pi_agree_dec.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))
    keys = [k[:-4] for k in rows[0] if k.endswith("_opt")]
    OUT.append(f"## T4 perfect-information agreement with max^n (n=5, Phase-2 F4b protocol, {len(set((r['trump'], r['deal']) for r in rows))} deals)\n")
    OUT.append("| subset | decisions | " + " | ".join(f"{k} optimal" for k in keys) + " | TA+ − TA (paired) | TA 1-step loss | TA+ 1-step loss |")
    OUT.append("|---|---|" + "---|" * len(keys) + "---|---|---|")
    res = {}
    for nm, f in (("all", lambda r: True), ("non-indifferent", lambda r: r["indiff"] == "0"),
                  ("non-indiff, trump S", lambda r: r["indiff"] == "0" and r["trump"] == "S"),
                  ("non-indiff, NT", lambda r: r["indiff"] == "0" and r["trump"] == "NT")):
        sub = [r for r in rows if f(r)]
        cells = []
        res[nm] = {"decisions": len(sub)}
        for k in keys:
            m = mean_ci([int(r[f"{k}_opt"]) for r in sub])
            res[nm][k] = m[:2]
            cells.append(fm(*m[:2], 3, False))
        dd = mean_ci([int(r["tap_opt"]) - int(r["ta_opt"]) for r in sub])
        # deal-clustered bootstrap CI for the paired difference
        keys_d = sorted(set((r["trump"], r["deal"]) for r in sub))
        ix = {k: i for i, k in enumerate(keys_d)}
        num = np.zeros(len(keys_d)); den = np.zeros(len(keys_d))
        for r in sub:
            i = ix[(r["trump"], r["deal"])]
            num[i] += int(r["tap_opt"]) - int(r["ta_opt"]); den[i] += 1
        cb = boot_ratio(num, den)
        res[nm]["tap_minus_ta_cluster_ci"] = cb[1:]
        l0 = mean_ci([float(r["ta_loss"]) for r in sub]); l1 = mean_ci([float(r["tap_loss"]) for r in sub])
        res[nm]["tap_minus_ta"] = dd[:2]
        OUT.append(f"| {nm} | {len(sub)} | " + " | ".join(cells) + f" | {fm(*dd[:2], 3)} (cluster [{cb[1]:+.3f},{cb[2]:+.3f}]) | {fm(*l0[:2], 3, False)} | {fm(*l1[:2], 3, False)} |")
    OUT.append("\n(decisions with ≥2 legal cards on the max^n path; CIs treat decisions as independent — deals cluster them, so they are somewhat optimistic.)\n")
    J["pi"] = res


if __name__ == "__main__":
    one_vs_three("T1 TA+ ablation (1-vs-3 TA)", T1, "t1")
    one_vs_three("T2 Hybrid (1-vs-3 TA)", T2, "t2")
    contrasts([("x_hyb_c5", "x_hyb_c4"), ("x_hyb_c6", "x_hyb_c5"), ("x_hyb_k32_c5", "x_hyb_c5"),
               ("x_hyb_c5", "x_hyb_c5_nt"), ("x_hyb_c4", "x_hyb_c4_nt")])
    one_vs_three("T3 Tournament: each challenger 1-vs-3 TA", T3, "t3")
    contrasts([("x_orc_n5", "x_ta_plus"), ("x_orc_all", "x_ta_plus"), ("x_orc_all", "x_orc_n5"),
               ("x_hyb_orc", "x_hyb_c5"), ("x_hyb_orc", "x_orc_all")])
    mixed()
    pi()
    (RES / "summary.md").write_text("# experiments/play results\n\n" + "\n".join(OUT) + "\n")
    (RES / "summary.json").write_text(json.dumps(J, indent=1))
    if "-q" not in sys.argv:
        print("\n".join(OUT))

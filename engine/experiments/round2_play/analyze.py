"""Summaries -> results/summary.md, summary.json.  All CIs are 95%:
means +-1.96 SE (games / blocks as units); ratios (overshoot share) and ratings
by bootstrap over games / blocks (B=1000/300, seed 0)."""
import json
import sys
from pathlib import Path

import numpy as np

from common import RES, mean_ci

rng = np.random.default_rng(0)
OUT = []
J = {}


def fm(m, h, d=2, sign=False):
    return (f"{m:+.{d}f}" if sign else f"{m:.{d}f}") + f" ± {h:.{d}f}"


def boot_ratio(num, den, B=1000):
    """num, den: per-unit arrays; ratio of sums with bootstrap CI."""
    n = len(num)
    r = num.sum() / max(den.sum(), 1)
    idx = rng.integers(0, n, (B, n))
    rs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r, lo, hi


def seat_stats(b, t):
    """b, t: (G, R) bids/tricks of one seat per game -> per-game hits, misses, overs."""
    miss = b != t
    over = miss & (t > b)
    return (~miss).sum(1), miss.sum(1), over.sum(1)


XTAGS = [("x_ta", "TA (reference, all-TA table)"), ("x_greedy", "Greedy (greedy bidder)"),
         ("x_ta_spite", "TA-spite"), ("x_ta_plus", "**TA+**"), ("x_abl_shed", "TA+ ablation: shed only"),
         ("x_abl_leadhigh", "TA+ ablation: lead-high (trump) only"), ("x_abl_lh_allrounds", "TA+ with lead-high in NT too"),
         ("x_abl_nothm", "TA+ without need=r theorems"), ("x_abl_plusduck", "TA+ + surplus-duck"),
         ("x_hybrid16", "**Hybrid** (PIMC K=16, ≤5 cards)"), ("x_hybrid32", "Hybrid (PIMC K=32, ≤5 cards)")]
FTAGS = [("f_ta", "TA"), ("f_greedy", "Greedy"), ("f_ta_spite", "TA-spite"), ("f_ta_plus", "TA+ (lead-high all rounds)"),
         ("f_abl_shed", "shed only"), ("f_abl_leadhigh", "lead-high (all rounds) only"),
         ("f_abl_plusduck", "TA+(lh all) + surplus-duck")]


def one_vs_three(prefix="x", tags=XTAGS, seeds="61,000,000+g (final; fresh)"):
    ref = np.load(RES / f"1v3_{prefix}_ta.npz")
    rows = []
    OUT.append(f"## 1-vs-3 (challenger in seat g%4 vs three TA; seeds {seeds}; paired vs the all-TA table, same seat)\n")
    OUT.append("| challenger | games | Δscore vs TA /game | P(win game) | hit rate | overshoot share of misses | over / under rate per round | opponents' Δscore /game (each) | TA seat (same games): hit / overshoot share |")
    OUT.append("|---|---|---|---|---|---|---|---|---|")
    for tag, label in tags:
        p = RES / f"1v3_{tag}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        G = min(len(d["totals"]), len(ref["totals"]))
        s = d["seat"][:G]
        ar = np.arange(G)
        diff = d["totals"][:G][ar, s] - ref["totals"][:G][ar, s]
        win = d["wins"][:G][ar, s]
        b, t = d["bids"][:G][ar, :, s], d["tricks"][:G][ar, :, s]
        h, m, o = seat_stats(b, t)
        R = b.shape[1]
        bt, tt = ref["bids"][:G][ar, :, s], ref["tricks"][:G][ar, :, s]
        h0, m0, o0 = seat_stats(bt, tt)
        dm = mean_ci(diff)
        wm = mean_ci(win)
        hm = mean_ci(h / R)
        os_ = boot_ratio(o, m)
        os0 = boot_ratio(o0, m0)
        hm0 = mean_ci(h0 / R)
        opp = (d["totals"][:G].sum(1) - d["totals"][:G][ar, s] - ref["totals"][:G].sum(1) + ref["totals"][:G][ar, s]) / 3
        om = mean_ci(opp)
        rows.append(dict(tag=tag, games=G, opp=om[:2], over_rate=float(o.sum() / (G * R)), under_rate=float((m - o).sum() / (G * R)), diff=dm[:2], pwin=wm[:2], hit=hm[:2], over=os_, ta_hit=hm0[:2], ta_over=os0))
        OUT.append(f"| {label} | {G} | {fm(*dm[:2], sign=True)} | {fm(*wm[:2], 3)} | {fm(*hm[:2], 4)} | "
                   f"{os_[0]:.3f} [{os_[1]:.3f},{os_[2]:.3f}] | {o.sum() / (G * R):.4f} / {(m - o).sum() / (G * R):.4f} | {fm(*om[:2], sign=True)} | {hm0[0]:.4f} / {os0[0]:.3f} [{os0[1]:.3f},{os0[2]:.3f}] |")
    J[f"one_vs_three_{prefix}"] = rows
    OUT.append("")
    # trump vs NT round split for TA+ and hybrid
    OUT.append("Per-round Δscore vs TA split by round type (same games):\n")
    OUT.append("| challenger | trump rounds Δ/round | NT rounds Δ/round | rounds with n≤5 Δ/round | rounds n≥6 Δ/round |")
    OUT.append("|---|---|---|---|---|")
    from judgement import Rules
    rules = Rules()
    lad = np.array(rules.ladder())
    trumps = np.array([rules.trump_for_round(i) for i in range(rules.n_rounds)])
    for tag, label in tags[1:]:
        p = RES / f"1v3_{tag}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        G = min(len(d["totals"]), len(ref["totals"]))
        s = d["seat"][:G]
        ar = np.arange(G)
        ds = d["scores"][:G][ar, :, s] - ref["scores"][:G][ar, :, s]   # (G, R)
        cells = []
        for mask in (trumps >= 0, trumps < 0, lad <= 5, lad >= 6):
            per_game = ds[:, mask].mean(1)
            cells.append(fm(*mean_ci(per_game)[:2], 3, sign=True))
        OUT.append(f"| {label} | " + " | ".join(cells) + " |")
    OUT.append("")


def bt_fit(W, iters=2000):
    """Bradley-Terry by MM; W[i,j] = (fractional) wins of i over j."""
    k = len(W)
    p = np.ones(k)
    N = W + W.T
    for _ in range(iters):
        num = W.sum(1)
        den = np.array([sum(N[i, j] / (p[i] + p[j]) for j in range(k) if j != i) for i in range(k)])
        p_new = num / den
        p_new /= np.exp(np.log(p_new).mean())
        if np.max(np.abs(p_new - p)) < 1e-10:
            p = p_new
            break
        p = p_new
    return p


def mixed():
    p = RES / "mixed.npz"
    if not p.exists():
        return
    d = np.load(p)
    pols = list(d["policies"])
    k = len(pols)
    blocks = d["block"]
    nb = int(blocks.max()) + 1
    full = np.bincount(blocks) == 20
    keep = np.isin(blocks, np.where(full)[0])
    A, T, Wn = d["assign"][keep], d["totals"][keep], d["wins"][keep]
    Bi, Tr, bl = d["bids"][keep], d["tricks"][keep], blocks[keep]
    blist = np.unique(bl)
    nB = len(blist)
    OUT.append(f"## Mixed 4-policy tables (5 subsets × 4 seat rotations per block, all 20 tables of a block share seed 51,000,000+b; {nB} blocks = {len(A)} games)\n")
    # per-block mean score per policy, wins, pairwise
    S = np.zeros((nB, k))
    Pw = np.zeros((nB, k))
    Wb = np.zeros((nB, k, k))
    hit = np.zeros((nB, k))
    miss = np.zeros((nB, k))
    over = np.zeros((nB, k))
    cnt = np.zeros((nB, k))
    bidx = {b: i for i, b in enumerate(blist)}
    for g in range(len(A)):
        i = bidx[bl[g]]
        a = A[g]
        for s in range(4):
            q = a[s]
            S[i, q] += T[g, s]
            Pw[i, q] += Wn[g, s]
            cnt[i, q] += 1
            m = Bi[g, :, s] != Tr[g, :, s]
            miss[i, q] += m.sum()
            over[i, q] += (m & (Tr[g, :, s] > Bi[g, :, s])).sum()
            hit[i, q] += (~m).sum()
            for s2 in range(4):
                if s2 != s:
                    x, y = T[g, s], T[g, s2]
                    Wb[i, q, a[s2]] += 1.0 if x > y else (0.5 if x == y else 0.0)
    S /= cnt
    Pw /= cnt
    R = Bi.shape[1]
    ta = pols.index("ta")
    W = Wb.sum(0)
    pbt = bt_fit(W)
    elo = 400 * np.log10(pbt / pbt[ta])
    B = 300
    elos = np.zeros((B, k))
    for bb in range(B):
        ii = rng.integers(0, nB, nB)
        pb = bt_fit(Wb[ii].sum(0), 500)
        elos[bb] = 400 * np.log10(pb / pb[ta])
    lo, hi = np.percentile(elos, [2.5, 97.5], axis=0)
    secs = d["secs"][keep]
    has_h = np.array([pols.index("hybrid") in a for a in A])
    OUT.append("| policy | mean score (games=16/block) | Δ vs TA (paired by block) | P(win game) | P(win) Δ vs TA | hit rate | overshoot share | BT strength | Elo vs TA [95% CI] |")
    OUT.append("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for q in range(k):
        ms = mean_ci(S[:, q])
        dd = mean_ci(S[:, q] - S[:, ta])
        pw = mean_ci(Pw[:, q])
        pwd = mean_ci(Pw[:, q] - Pw[:, ta])
        hr = mean_ci(hit[:, q] / (cnt[:, q] * R))
        osr = boot_ratio(over[:, q], miss[:, q])
        rows.append(dict(policy=pols[q], score=ms[:2], diff=dd[:2], pwin=pw[:2], hit=hr[:2], over=osr,
                         bt=float(pbt[q]), elo=float(elo[q]), elo_ci=[float(lo[q]), float(hi[q])]))
        OUT.append(f"| {pols[q]} | {fm(*ms[:2])} | {fm(*dd[:2], sign=True)} | {fm(*pw[:2], 3)} | {fm(*pwd[:2], 3, sign=True)} | "
                   f"{fm(*hr[:2], 4)} | {osr[0]:.3f} [{osr[1]:.3f},{osr[2]:.3f}] | {pbt[q]:.3f} | {elo[q]:+.0f} [{lo[q]:+.0f},{hi[q]:+.0f}] |")
    OUT.append("")
    OUT.append("Pairwise finishing order (row finishes above column; ties = ½), fraction over all shared tables:\n")
    OUT.append("| | " + " | ".join(pols) + " |")
    OUT.append("|---|" + "---|" * k)
    for i in range(k):
        OUT.append(f"| {pols[i]} | " + " | ".join("-" if i == j else f"{W[i, j] / (W[i, j] + W[j, i]):.3f}" for j in range(k)) + " |")
    OUT.append(f"\n(pairs per cell: {int(W[0, 1] + W[1, 0])}; per-game wall time: tables with Hybrid {secs[has_h].mean():.2f} s, without {secs[~has_h].mean():.3f} s)\n")
    J["mixed"] = dict(blocks=nB, games=int(len(A)), rows=rows, pairwise=W.tolist())


def e1():
    p = RES / "e1_ta_plus_n13.json"
    if not p.exists():
        return
    recs = json.load(open(p))
    old = json.load(open(Path(__file__).resolve().parents[1] / "phase1" / "results" / "e1_n13.json"))
    n = len(recs)
    old = old[:n]
    assert all(a["b_greedy"] == b["b_greedy"] for a, b in zip(recs, old)), "greedy dists not reproduced"
    ch_p = np.array([r["b_t"] != r["b_greedy"] for r in recs], float)
    ch_t = np.array([r["b_ta"] != r["b_greedy"] for r in old], float)
    same = np.array([r["b_t"] == o["b_ta"] for r, o in zip(recs, old)], float)
    sh_p = np.array([r["b_t"] - r["b_greedy"] for r in recs], float)
    sh_t = np.array([o["b_ta"] - o["b_greedy"] for o in old], float)
    ev_p = np.array([r["ev_t"] for r in recs])
    ev_t = np.array([o["ev_ta"] for o in old])
    gain_p = np.array([r["ev_t"] - r["t_ev_at_greedy_bid"] for r in recs])
    gain_t = np.array([o["ev_ta"] - o["ta_ev_at_greedy_bid"] for o in old])
    from common import prop_ci
    OUT.append(f"## E1 robustness (n=13, trump S, hero seat 1; first {n} Phase-1 E1 hands, identical Greedy distributions; 100 worlds/candidate)\n")
    OUT.append("| quantity | TA play (Phase 1, same hands) | TA+ play | paired Δ (TA+ − TA) |")
    OUT.append("|---|---|---|---|")
    pp, pl, ph = prop_ci(ch_p.sum(), n)
    tp, tl, th = prop_ci(ch_t.sum(), n)
    OUT.append(f"| P(oracle bid ≠ greedy-oracle bid) | {tp:.3f} [{tl:.3f},{th:.3f}] | {pp:.3f} [{pl:.3f},{ph:.3f}] | {fm(*mean_ci(ch_p - ch_t)[:2], 3, sign=True)} |")
    OUT.append(f"| mean |shift| | {fm(*mean_ci(np.abs(sh_t))[:2], 3)} | {fm(*mean_ci(np.abs(sh_p))[:2], 3)} | {fm(*mean_ci(np.abs(sh_p) - np.abs(sh_t))[:2], 3, sign=True)} |")
    OUT.append(f"| mean signed shift | {fm(*mean_ci(sh_t)[:2], 3, sign=True)} | {fm(*mean_ci(sh_p)[:2], 3, sign=True)} | |")
    OUT.append(f"| EV at own oracle bid | {fm(*mean_ci(ev_t)[:2])} | {fm(*mean_ci(ev_p)[:2])} | {fm(*mean_ci(ev_p - ev_t)[:2], sign=True)} |")
    OUT.append(f"| EV(b*) − EV(greedy bid) under that play | {fm(*mean_ci(gain_t)[:2], 3)} | {fm(*mean_ci(gain_p)[:2], 3)} | |")
    sp, sl, sh = prop_ci(same.sum(), n)
    OUT.append(f"| TA+ oracle bid = TA oracle bid | | {sp:.3f} [{sl:.3f},{sh:.3f}] | |\n")
    J["e1"] = dict(n=n, change_ta=float(ch_t.mean()), change_taplus=float(ch_p.mean()))


def pi():
    p = RES / "pi_agree_dec.csv"
    if not p.exists():
        return
    import csv
    rows = list(csv.DictReader(open(p)))
    OUT.append("## Perfect-information agreement with max^n (n=5, Phase-2 F4b protocol: scheme-O bids, decisions on the max^n path)\n")
    OUT.append("| subset | decisions | TA optimal | TA+ optimal | Δ (paired) | TA 1-step loss | TA+ 1-step loss | Δ loss | same card |")
    OUT.append("|---|---|---|---|---|---|---|---|---|")
    from common import prop_ci
    for name, f in (("all", lambda r: True), ("non-indifferent", lambda r: r["indiff"] == "0"),
                    ("non-indiff, trump S", lambda r: r["indiff"] == "0" and r["trump"] == "S"),
                    ("non-indiff, NT", lambda r: r["indiff"] == "0" and r["trump"] == "NT")):
        sub = [r for r in rows if f(r)]
        a = np.array([int(r["ta_opt"]) for r in sub])
        b = np.array([int(r["tap_opt"]) for r in sub])
        la = np.array([float(r["ta_loss"]) for r in sub])
        lb = np.array([float(r["tap_loss"]) for r in sub])
        same = np.array([r["ta_card"] == r["tap_card"] for r in sub], float)
        n = len(sub)
        OUT.append(f"| {name} | {n} | {fm(*mean_ci(a)[:2], 3)} | {fm(*mean_ci(b)[:2], 3)} | {fm(*mean_ci(b - a)[:2], 3, sign=True)} | "
                   f"{fm(*mean_ci(la)[:2], 3)} | {fm(*mean_ci(lb)[:2], 3)} | {fm(*mean_ci(lb - la)[:2], 3, sign=True)} | {same.mean():.3f} |")
    OUT.append("")


if __name__ == "__main__":
    one_vs_three()
    one_vs_three("f", FTAGS, "41,000,000+g (validation; TA+ here = lead-high in all rounds, before the trump-only change)")
    e1()
    pi()
    mixed()
    (RES / "summary.md").write_text("# Round 2 play — results\n\n" + "\n".join(OUT) + "\n")
    (RES / "summary.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(OUT))

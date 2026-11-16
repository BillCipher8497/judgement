"""H1 analysis: hidden-information cooperation surplus.
python analyze.py [TAG]  -> results/summary_<TAG>.md / .json
Paired (CRN) deal-level 95% CIs; share = surplus / score(self) with delta-method CI.
"""
import json
import sys
from pathlib import Path

import numpy as np

RES = Path(__file__).parent / "results"
# Phase-2 perfect-information share (max^n - paranoid)/max^n, scheme R, mean of S and NT (summary_main.md)
PI_SHARE = {1: 0.0, 2: (0.239 + 0.221) / 2, 3: (0.497 + 0.495) / 2, 4: (0.715 + 0.698) / 2,
            5: (0.819 + 0.802) / 2, 6: (0.907 + 0.871) / 2}
PI_SURPLUS = {1: 0.0, 2: (1.89 + 1.75) / 2, 3: (4.03 + 4.01) / 2, 4: (5.94 + 5.81) / 2, 5: (7.01 + 6.85) / 2,
              6: (7.97 + 7.67) / 2}


def mci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))


def ratio_ci(d, s):
    d, s = np.asarray(d, float), np.asarray(s, float)
    N = len(d)
    R = d.mean() / s.mean()
    v = (d.var(ddof=1) - 2 * R * np.cov(d, s)[0, 1] + R * R * s.var(ddof=1)) / (N * s.mean() ** 2)
    return float(R), float(1.96 * np.sqrt(max(v, 0))), N


def f(m, d=2):
    return f"{m[0]:.{d}f} ± {m[1]:.{d}f}"


def rounds_summary(z):
    S, B, T = z["scores"], z["bids"], z["tricks"]
    conds = list(z["conds"])
    ns = list(z["ns"])
    D = int(z["done"].min())
    out = {}
    for i, n in enumerate(ns):
        s = S[i, :D]  # (D, C, 4)
        hero = s[:, :, 0]
        hit = (B[i, :D, :, 0] == T[i, :D, :, 0])
        d = {"deals": D}
        for ci, c in enumerate(conds):
            d[f"hero_{c}"] = mci(hero[:, ci])
            d[f"hit_{c}"] = mci(hit[:, ci])
        base = hero[:, 0]
        for ci, c in enumerate(conds[1:], 1):
            d[f"surplus_{c}"] = mci(base - hero[:, ci])
            d[f"share_{c}"] = ratio_ci(base - hero[:, ci], base)
        # cost of punishment: spoiler seats' own score vs the same seat self-interested (CRN)
        dl = np.arange(D)
        for ci, c in enumerate(conds[1:], 1):
            if c == "s3":
                offs = [(1, 2, 3)] * D
            elif c == "s1":
                offs = [(1 + (x // 4) % 3,) for x in dl]
            else:
                offs = [((1, 2), (1, 3), (2, 3))[(x // 4) % 3] for x in dl]
            sp = np.array([np.mean([s[k, ci, o] for o in offs[k]]) for k in range(D)])
            sp0 = np.array([np.mean([s[k, 0, o] for o in offs[k]]) for k in range(D)])
            d[f"spoiler_own_{c}"] = mci(sp)
            d[f"spoiler_cost_{c}"] = mci(sp0 - sp)
            if c != "s3":
                by = [tuple(o for o in (1, 2, 3) if o not in offs[k]) for k in range(D)]
                bs = np.array([np.mean([s[k, ci, o] for o in by[k]]) for k in range(D)])
                bs0 = np.array([np.mean([s[k, 0, o] for o in by[k]]) for k in range(D)])
                d[f"bystander_gain_{c}"] = mci(bs - bs0)
        # by hero bidding position (dealer = d%4, hero seat 0 => pos = (0 - dealer - 1) % 4)
        pos = (-(dl % 4) - 1) % 4
        d["surplus_s3_by_pos"] = {int(p): mci((base - hero[:, 3])[pos == p]) for p in range(4)}
        d["hero_self_by_pos"] = {int(p): mci(base[pos == p]) for p in range(4)}
        out[int(n)] = d
    return out


def games_summary(z):
    S, W = z["scores"], z["wins"]
    conds = list(z["conds"])
    G = len(S)
    hero = z["hero"]
    idx = np.arange(G)
    tot = S.sum(2)  # (G, C, 4)
    ht = tot[idx, :, hero]  # (G, C)
    hw = W[idx, :, hero]
    d = {"games": G}
    for ci, c in enumerate(conds):
        d[f"hero_total_{c}"] = mci(ht[:, ci])
        d[f"hero_win_{c}"] = mci(hw[:, ci])
    for ci, c in enumerate(conds[1:], 1):
        d[f"surplus_{c}"] = mci(ht[:, 0] - ht[:, ci])
        d[f"share_{c}"] = ratio_ci(ht[:, 0] - ht[:, ci], ht[:, 0])
    # s3: spoilers' own totals vs self-interested
    opp = np.array([[o for o in range(4) if o != h] for h in hero])
    so = np.array([tot[g, 3, opp[g]].mean() for g in range(G)])
    so0 = np.array([tot[g, 0, opp[g]].mean() for g in range(G)])
    d["spoiler_total_s3"] = mci(so)
    d["spoiler_cost_s3"] = mci(so0 - so)
    best_sp = np.array([tot[g, 3, opp[g]].max() for g in range(G)])
    d["P(hero beats every spoiler)_s3"] = mci(ht[:, 3] > best_sp)
    return d


def main(tag="main"):
    summ = {}
    L = []
    fr = RES / f"rounds_{tag}.npz"
    if fr.exists():
        r = rounds_summary(np.load(fr))
        summ["rounds"] = r
        L.append(f"## H1 per-round: hero TA+ (model bid, seat 0, dealer d%4, trump cycle d%5); CRN deals; "
                 f"95% CI (n = deals)")
        L.append("| n | deals | self | 1 spoiler | 2 spoilers | 3 spoilers | surplus (s3) | share s3 | share s1 | share s2 | "
          "PI share | PI surplus | hit self | hit s3 |")
        L.append("|" + "---|" * 15)
        for n, d in r.items():
            L.append(f"| {n} | {d['deals']} | {f(d['hero_self'])} | {f(d['hero_s1'])} | {f(d['hero_s2'])} | "
                     f"{f(d['hero_s3'])} | {f(d['surplus_s3'])} | {f(d['share_s3'], 3)} | {f(d['share_s1'], 3)} | "
                     f"{f(d['share_s2'], 3)} | {PI_SHARE.get(n, float('nan')):.3f} | "
                     f"{PI_SURPLUS.get(n, float('nan')):.2f} | {f(d['hit_self'], 3)} | {f(d['hit_s3'], 3)} |")
        L.append("")
        L.append("Cost of punishment (spoiler's own score: self-interested minus spoiler, same seat, CRN) and "
                 "bystander gain (non-spoiler opponents)")
        L.append("| n | spoiler own s3 | cost s3 | cost s2 | cost s1 | bystander gain s1 | bystander gain s2 |")
        L.append("|---|---|---|---|---|---|---|")
        for n, d in r.items():
            L.append(f"| {n} | {f(d['spoiler_own_s3'])} | {f(d['spoiler_cost_s3'])} | {f(d['spoiler_cost_s2'])} | "
                     f"{f(d['spoiler_cost_s1'])} | {f(d['bystander_gain_s1'])} | {f(d['bystander_gain_s2'])} |")
        L.append("")
        L.append("Surplus (s3) by hero bidding position (0 = first bidder/leader, 3 = dealer)")
        L.append("| n | pos0 | pos1 | pos2 | pos3 |")
        L.append("|---|---|---|---|---|")
        for n, d in r.items():
            L.append(f"| {n} | " + " | ".join(f(d['surplus_s3_by_pos'][p]) for p in range(4)) + " |")
        L.append("")
    fg = RES / f"games_{tag}.npz"
    if fg.exists():
        g = games_summary(np.load(fg))
        summ["games"] = g
        L.append(f"## H1 full games (hero seat g%4, spoilers target hero all game), {g['games']} games, CRN")
        L.append("| cond | hero total | P(win) | surplus | share |")
        L.append("|---|---|---|---|---|")
        for c in ("self", "s1", "s2", "s3"):
            sp = f(g[f'surplus_{c}']) if c != "self" else "–"
            sh = f(g[f'share_{c}'], 3) if c != "self" else "–"
            L.append(f"| {c} | {f(g[f'hero_total_{c}'], 1)} | {f(g[f'hero_win_{c}'], 3)} | {sp} | {sh} |")
        L.append(f"spoilers' own total (s3) {f(g['spoiler_total_s3'], 1)}, cost vs self-interested "
                 f"{f(g['spoiler_cost_s3'], 1)}; P(hero outscores every spoiler) {f(g['P(hero beats every spoiler)_s3'], 3)}")
    (RES / f"summary_{tag}.json").write_text(json.dumps(summ, indent=1))
    (RES / f"summary_{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")

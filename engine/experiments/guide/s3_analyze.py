"""S3 analysis -> results/s3_summary.txt / s3_summary.json.  All comparisons paired by game (CRN; same seeds,
challenger seat g%4), baseline = the all-<pool> table at the same seat.  P(win): ties for first count as wins."""
import json

import numpy as np

from common import RES
from judgement import Rules

rules = Rules()
LAD = rules.ladder()
TR = [rules.trump_for_round(r) for r in range(len(LAD))]


def tag(pool, chal):
    return f"f_{pool}__" + chal.replace(":", "_").replace(",", "_").replace("=", "-")


def load(pool, chal):
    p = RES / f"g_{tag(pool, chal)}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    s = d["seat"]; i = np.arange(len(s))
    tot = d["totals"]
    return dict(game=d["game"], tot=tot[i, s], win=(tot[i, s] >= tot.max(1)).astype(float), sc=d["scores"][i, :, s],
                hit=(d["bids"][i, :, s] == d["tricks"][i, :, s]).astype(float), all_tot=tot,
                all_hit=(d["bids"] == d["tricks"]).astype(float))


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    jobs = [l.split() for l in open(RES.parent / "s3_jobs.txt") if l.strip()]
    out, lines = {}, []
    lines.append("S3: 1 challenger vs 3 pool players; d = challenger total - pool player's total at the same seat in the all-pool table "
                 "(same games); P(win) ties count as wins; 95% CI.")
    lines.append("pool | challenger | games | d/game vs pool | P(win) | dP(win) vs pool seat | hit rate")
    data = {}
    for pool, chal in jobs:
        a, b = load(pool, chal), load(pool, pool)
        if a is None or b is None:
            continue
        m = min(len(a["game"]), len(b["game"]))
        data[(pool, chal)] = (a, b, m)
        d = ci(a["tot"][:m] - b["tot"][:m]); pw = ci(a["win"][:m]); dp = ci(a["win"][:m] - b["win"][:m])
        hr = float(a["hit"][:m].mean())
        out[f"{pool}|{chal}"] = dict(n=m, d=d, pwin=pw, dpwin=dp, hit=hr)
        lines.append(f"{pool} | {chal} | {m} | {d[0]:+.2f} +- {d[1]:.2f} | {pw[0]:.3f} +- {pw[1]:.3f} | {dp[0]:+.3f} +- {dp[1]:.3f} | {hr:.3f}")
    # headline: share of TA+'s edge over greedy kept by the guide (and by each half), bootstrap over games
    lines.append("")
    lines.append("Headline: share kept = (X - greedy) / (TA+3 - greedy), same games, per pool; bootstrap 95% CI (2000 resamples over games)")
    rng = np.random.default_rng(1)
    for pool in ("ta", "tap3"):
        need = [(pool, k) for k in ("greedy", "tap3", "guide", "gbid", "gplay", "ta")]
        if not all(k in data for k in need):
            continue
        m = min(data[k][2] for k in need)
        base = data[(pool, "greedy")][1]["tot"][:m]
        vals = {k[1]: data[k][0]["tot"][:m] - base for k in need}   # vs pool seat; greedy shift cancels
        g, t = vals["greedy"], vals["tap3"]
        for x in ("guide", "gbid", "gplay", "ta"):
            v = vals[x]
            est = (v.mean() - g.mean()) / (t.mean() - g.mean())
            bs = []
            for _ in range(2000):
                ii = rng.integers(0, m, m)
                bs.append((v[ii].mean() - g[ii].mean()) / (t[ii].mean() - g[ii].mean()))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            # same with TA as the floor (share of TA+3's edge over TA)
            vt = vals["ta"]
            est2 = (v.mean() - vt.mean()) / (t.mean() - vt.mean())
            bs2 = []
            for _ in range(2000):
                ii = rng.integers(0, m, m)
                bs2.append((v[ii].mean() - vt[ii].mean()) / (t[ii].mean() - vt[ii].mean()))
            lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
            out[f"share|{pool}|{x}"] = dict(n=m, over_greedy=[est, lo, hi], over_ta=[est2, lo2, hi2])
            lines.append(f"pool {pool}: {x:6s} keeps {est:.3f} [{lo:.3f}, {hi:.3f}] of TA+3's edge over greedy; "
                         f"{est2:+.3f} [{lo2:+.3f}, {hi2:+.3f}] of its edge over TA   (n={m})")
    # all-guide table vs all-TA+3 / all-TA tables (self-play level per player)
    lines.append("")
    lines.append("Self-play tables (mean score per player per game; hit rate):")
    for p in ("ta", "tap3", "guide"):
        a = load(p, p)
        if a is None:
            continue
        x = a["all_tot"].mean(1)
        out[f"table|{p}"] = dict(n=len(x), mean=ci(x), hit=float(a["all_hit"].mean()))
        lines.append(f"all-{p}: {ci(x)[0]:.2f} +- {ci(x)[1]:.2f} per player/game, hit {a['all_hit'].mean():.4f} (n={len(x)})")
    # per-round bidding loss by n and trump/NT
    lines.append("")
    lines.append("Per-round paired loss (score/round, 95% CI) by n and trump/NT: [A] guide bid vs MB, TA+3 play (pool tap3: gbid - tap3);"
                 " [B] guide bid vs MB, guide play (pool tap3: guide - gplay); [C] guide play vs TA+3 play, MB bid (pool tap3: gplay - tap3)")
    defs = dict(A=(("tap3", "gbid"), ("tap3", "tap3")), B=(("tap3", "guide"), ("tap3", "gplay")),
                C=(("tap3", "gplay"), ("tap3", "tap3")))
    per = {}
    for key, (x, y) in defs.items():
        a, b = load(*x), load(*y)
        if a is None or b is None:
            continue
        m = min(len(a["game"]), len(b["game"]))
        dr = a["sc"][:m] - b["sc"][:m]          # (games, rounds)
        per[key] = dr
    if per:
        lines.append("n | trump/NT | rounds/game | " + " | ".join(f"[{k}]" for k in per))
        groups = {}
        for r, n in enumerate(LAD):
            groups.setdefault((n, TR[r] == -1), []).append(r)
        for (n, nt) in sorted(groups):
            rs = groups[(n, nt)]
            cells = []
            for k, dr in per.items():
                v = dr[:, rs].mean(1)
                c = ci(v)
                cells.append(f"{-c[0]:+.3f} +- {c[1]:.3f}")
                out.setdefault(f"perround|{k}", {})[f"{n}|{'NT' if nt else 'T'}"] = [-c[0], c[1]]
            lines.append(f"{n} | {'NT' if nt else 'trump'} | {len(rs)} | " + " | ".join(cells))
        for k, dr in per.items():
            c = ci(dr.mean(1))
            lines.append(f"[{k}] all rounds: loss {-c[0]:+.3f} +- {c[1]:.3f} /round  ({-25 * c[0]:+.2f}/game)")
            for nt in (False, True):
                rs = [r for r in range(len(LAD)) if (TR[r] == -1) == nt]
                c2 = ci(dr[:, rs].mean(1))
                lines.append(f"     {'NT' if nt else 'trump'} rounds: {-c2[0]:+.3f} +- {c2[1]:.3f} /round")
    txt = "\n".join(lines)
    open(RES / "s3_summary.txt", "w").write(txt + "\n")
    json.dump(out, open(RES / "s3_summary.json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()

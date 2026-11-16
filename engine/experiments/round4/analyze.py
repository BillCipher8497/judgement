"""Round-4 tables (G1, G2). gain = challenger seat total − pool policy in the same seat of the all-pool
reference game (same seed, CRN); X − Y = paired difference of two challengers on the same games.
P(win) ties-as-win.  -> results/r4.md / r4.json"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"


def load(prefix):
    fs = sorted(RES.glob(f"f_{prefix}_*.npz"))
    fs = [f for f in fs if not f.name.endswith("_stats.json")]
    if not fs:
        return None
    ps = [np.load(f) for f in fs]
    d = {k: np.concatenate([p[k] for p in ps]) for k in ("totals", "game", "secs")}
    o = np.argsort(d["game"])
    return {k: v[o] for k, v in d.items()}


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def tie(t):
    return (t == t.max(1, keepdims=True)).astype(float)


def row(c, r):
    g = np.intersect1d(c["game"], r["game"])
    ic, ir = np.searchsorted(c["game"], g), np.searchsorted(r["game"], g)
    s, a = g % 4, np.arange(len(g))
    tc, tr = c["totals"][ic], r["totals"][ir]
    return dict(n=int(len(g)), gain=ci(tc[a, s] - tr[a, s]), dp=ci(tie(tc)[a, s] - tie(tr)[a, s]),
                p=float(tie(tc)[a, s].mean()), secs=float(np.median(c["secs"][ic])))


def f(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


def fp(m):
    return f"{m[0]:+.3f} ± {m[1]:.3f}"


def main():
    out, md = {}, []
    hdr = ["| pool | comparison | n | Δ pts/game | ΔP(win) | P(win) first arm | s/game |", "|---|---|---|---|---|---|---|"]
    NAMES = {"plain": "Hybrid", "bi": "Hybrid-BI eps.02", "bit": "Hybrid-BI eps.1", "ref": "pool",
             "c7": "Hybrid-BI k32 c7"}
    for grp, pools, arms, comps in (
            ("G1", {"g1a": "3x Hybrid noisy-bid (±1 w.p. .3)", "g1b": "3x Hybrid honour-count bid"},
             ["ref", "plain", "bi", "bit"], [("plain", "ref"), ("bi", "ref"), ("bit", "ref"), ("bi", "plain"),
                                              ("bit", "plain"), ("bit", "bi")]),
            ("G2", {"g2": "3x Hybrid-BI k32 c6"}, ["ref", "plain", "c7"],
             [("plain", "ref"), ("c7", "ref"), ("c7", "plain")])):
        md += [f"## {grp}", ""] + hdr
        for pk, pname in pools.items():
            D = {a: load(f"{pk}_{a}") for a in arms}
            for x, y in comps:
                if D[x] is None or D[y] is None:
                    continue
                rw = row(D[x], D[y])
                if rw["n"] < 2:
                    continue
                lab = f"{NAMES[x]} vs pool" if y == "ref" else f"{NAMES[x]} − {NAMES[y]}"
                out[f"{pk}:{x}-{y}"] = rw
                md.append(f"| {pname} | {lab} | {rw['n']} | {f(rw['gain'])} | {fp(rw['dp'])} | {rw['p']:.3f} | {rw['secs']:.1f} |")
        md.append("")
    md += ["## Sampler (BI challenger PIMC decisions)", "", "| arm | decisions | median ESS | ESS<4 | M capped |", "|---|---|---|---|---|"]
    for pre in ("g1a_bi", "g1a_bit", "g1b_bi", "g1b_bit", "g2_c7"):
        st = [x for f in sorted(RES.glob(f"f_{pre}_*_stats.json")) for x in json.loads(f.read_text())]
        if st:
            e = np.array([x["ess"] for x in st]); M = np.array([x["M"] for x in st])
            out[f"stats:{pre}"] = dict(n=len(st), ess_med=float(np.median(e)), ess_lt4=float((e < 4).mean()),
                                       cap=float((M >= 512).mean()))
            md.append(f"| {pre} | {len(st)} | {np.median(e):.1f} | {(e < 4).mean():.3f} | {(M >= 512).mean():.3f} |")
    md.append("")
    (RES / "r4.json").write_text(json.dumps(out, indent=1))
    (RES / "r4.md").write_text("\n".join(md) + "\n95% CIs (1.96·SE over games); CRN same seeds; challenger seat = g % 4.\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

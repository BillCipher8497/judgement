"""Round-6 tables -> results/r6.md, results/r6.json.  Paired comparisons on the same games (CRN): challenger-seat total
and ties-as-win indicator, A minus B; 95% CI = 1.96 SE over games.  Reuses round5 analyze.load/ci/pair."""
import importlib.util
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
RES, E = H / "results", H.parent
_s = importlib.util.spec_from_file_location("r5_an", E / "round5_bots" / "analyze.py")
R5 = importlib.util.module_from_spec(_s); _s.loader.exec_module(R5)
load, ci, pair, fmt, g = R5.load, R5.ci, R5.pair, R5.fmt, R5.g
R5R, R4R = E / "round5_bots" / "results", E / "round4" / "results"

ARMS = {
    "guide": {"plain": g("f_a_plain_*.npz", RES), "rb": g("f_a_rb_?.npz", RES), "rb2": g("f_a_rb2_*.npz", RES)},
    "mix": {"plain": R5.ARMS["mix"]["plain"], "bi": R5.ARMS["mix"]["bi"], "rb": R5.ARMS["mix"]["rb"],
            "rb2": g("f_b_rb2_*.npz", RES)},
    "hyb": {"plain": R5.ARMS["hyb"]["plain"], "bi": R5.ARMS["hyb"]["bi"], "rb": R5.ARMS["hyb"]["rb"],
            "rb2": g("f_c_rb2_*.npz", RES)},
    "g2": {"pool": g("f_g2_ref_*.npz", R4R) + g("f_c2x_ref_*.npz", RES), "plain": g("f_g2_plain_*.npz", R4R),
           "bi7": g("f_g2_c7_*.npz", R4R) + g("f_c2x_c7_*.npz", RES), "rb7": g("f_c2_rb_*.npz", RES) + g("f_c2x_rb_*.npz", RES), "rb2_7": g("f_c2_rb2_*.npz", RES)},
}
LABEL = {"guide": "(a) 3x guide-bidding Hybrid", "mix": "(b) mixed (1 of each)", "hyb": "(c) 3x Hybrid (model bids)",
         "g2": "C2: 3x Hybrid-BI K32 c6"}
NAME = {"pool": "pool (BI K32c6)", "plain": "Hybrid", "bi": "Hybrid-BI", "rb": "Hybrid-RB", "rb2": "Hybrid-RB2",
        "bi7": "Hybrid-BI K32c7", "rb7": "Hybrid-RB K32c7", "rb2_7": "Hybrid-RB2 K32c7"}
COMP = [("rb2", "plain"), ("rb2", "rb"), ("rb", "plain"), ("rb2", "bi"), ("rb7", "pool"), ("rb7", "bi7"), ("bi7", "pool"),
        ("rb7", "plain"), ("rb2_7", "pool"), ("rb2_7", "bi7"), ("rb2_7", "rb7")]


def comparisons(D, out, md):
    md += ["| pool | A − B | n | Δ pts/game | ΔP(win) ties-as-win | Δ rounds 1–8 | Δ rounds 9–25 | s/game A |",
           "|---|---|---|---|---|---|---|---|"]
    for p, arms in D.items():
        for a, b in COMP:
            A, B = arms.get(a), arms.get(b)
            if A is None or B is None:
                continue
            ia, ib = pair(A, B)
            if len(ia) < 2:
                continue
            r = {"pts": ci(A["tot"][ia] - B["tot"][ib]), "win": ci(A["win"][ia] - B["win"][ib]),
                 "early": ci(A["rs"][ia, :8].sum(1) - B["rs"][ib, :8].sum(1)),
                 "late": ci(A["rs"][ia, 8:].sum(1) - B["rs"][ib, 8:].sum(1)), "secs": float(A["secs"][ia].mean()),
                 "games": [int(A["game"][ia].min()), int(A["game"][ia].max())]}
            out[f"{p}|{a}-{b}"] = r
            md.append(f"| {LABEL[p]} | {NAME[a]} − {NAME[b]} | {r['pts'][2]} | {fmt(r['pts'])} | "
                      f"{r['win'][0]:+.3f} ± {r['win'][1]:.3f} | {fmt(r['early'])} | {fmt(r['late'])} | {r['secs']:.1f} |")


RB_T = ["model", "noisy0.1", "noisy0.25", "noisy0.5", "honour", "unknown"]
GUIDE, HYB = "hyb_guide:k=16,c=5", "hybrid:k=16,c=5"
RS = (1, 2, 3, 5, 8, 12, 24)


def posteriors(out, md):
    md += ["", "## Posterior convergence (mean over opponent-games, after round r; r = rounds observed)", ""]
    groups = {}
    for f in g("f_*_rb2_*_post.json", RES):
        pool = {"a": "guide", "b": "mix", "c": "hyb"}.get(f.name[2])
        if pool is None:
            continue
        for x in json.load(open(f)):
            groups.setdefault(("RB2", pool, x["kind"]), []).append(x)
    for f in g("f_a_rb_?_post.json", RES) + g("f_b1m_rb_*_post.json", R5R):
        pool = "guide" if f.name.startswith("f_a_") else "mix"
        for x in json.load(open(f)):
            groups.setdefault(("RB", pool, x["kind"]), []).append(x)
    out["posterior"] = {}
    md += ["| agent | pool | opp kind | n opp-games | quantity | " + " | ".join(f"r={r}" for r in RS) + " |",
           "|---|---|---|---|---|" + "---|" * len(RS)]
    for (ag, pool, kind), xs in sorted(groups.items()):
        if ag == "RB" and kind != GUIDE and pool != "guide":
            continue
        by = {}
        for x in xs:
            by.setdefault(x["r"], []).append(x)
        n = len({(x["g"], x["seat"]) for x in xs})
        if ag == "RB2":
            qs = {"P(cont)": lambda x: x["cont"], "P(model+noisy.1)": lambda x: x["model"] + x["noisy0.1"],
                  "P(noisy.25+.5)": lambda x: x["noisy0.25"] + x["noisy0.5"], "P(unknown)": lambda x: x["unknown"],
                  "E[a|cont]": lambda x: x["a"], "E[d|cont]": lambda x: x["d"], "E[T|cont]": lambda x: x["t"],
                  "sd(d|cont)": lambda x: x["d_sd"], "sd(T|cont)": lambda x: x["t_sd"]}
        else:
            qs = {f"P({t})": (lambda i: lambda x: x["post"][i])(i) for i, t in enumerate(RB_T)}
        res = {}
        for qn, fq in qs.items():
            row = {r: float(np.mean([fq(x) for x in by.get(r - 1, [])])) if by.get(r - 1) else float("nan") for r in RS}
            res[qn] = row
            md.append(f"| {ag} | {pool} | {kind.split(':')[0]} | {n} | {qn} | " + " | ".join(f"{row[r]:.2f}" for r in RS) + " |")
        out["posterior"][f"{ag}|{pool}|{kind}"] = {"n": n, **res}


def sampler(out, md):
    md += ["", "## Sampler (challenger PIMC decisions)", "", "| arm | decisions | median ESS | ESS<4 | raw ESS<4 | tempered | median M | M capped |",
           "|---|---|---|---|---|---|---|---|"]
    SS = {"guide Hybrid-RB": g("f_a_rb_?_stats.json", RES), "guide Hybrid-RB2": g("f_a_rb2_*_stats.json", RES),
          "mix Hybrid-RB2": g("f_b_rb2_*_stats.json", RES), "hyb Hybrid-RB2": g("f_c_rb2_*_stats.json", RES),
          "g2 Hybrid-RB K32c7": g("f_c2_rb_*_stats.json", RES) + g("f_c2x_rb_*_stats.json", RES), "g2 Hybrid-RB2 K32c7": g("f_c2_rb2_*_stats.json", RES),
          "g2 Hybrid-BI K32c7 (r4)": g("f_g2_c7_*_stats.json", R4R)}
    out["sampler"] = {}
    for k, fs in SS.items():
        st = []
        for f in fs:
            st += json.load(open(f))
        if not st:
            continue
        ess = np.array([s["ess"] for s in st]); raw = np.array([s.get("ess_raw", s["ess"]) for s in st])
        M = np.array([s["M"] for s in st]); beta = np.array([s.get("beta", 1.0) for s in st])
        r = {"n": len(st), "ess_med": float(np.median(ess)), "ess_lt4": float((ess < 4).mean()), "raw_lt4": float((raw < 4).mean()),
             "tempered": float((beta < 1).mean()), "M_med": float(np.median(M)), "cap": float((M >= 512).mean())}
        out["sampler"][k] = r
        md.append(f"| {k} | {r['n']} | {r['ess_med']:.1f} | {r['ess_lt4']:.3f} | {r['raw_lt4']:.3f} | {r['tempered']:.3f} | "
                  f"{r['M_med']:.0f} | {r['cap']:.3f} |")


def main():
    out, md = {}, ["# Round 6: Hybrid-RB2 (continuous bid model) and the top of the ladder", "",
                   "## Paired comparisons (same games, challenger seat g%4)", ""]
    D = {p: {a: load(f) for a, f in arms.items()} for p, arms in ARMS.items()}
    comparisons(D, out, md)
    posteriors(out, md)
    sampler(out, md)
    md += ["", "95% CIs = 1.96·SE over games; CRN; seeds: guide 900M (new), mix 700M (r5), hyb 400M (r3/r5), g2 600M (r4 G2)."]
    (RES / "r6.md").write_text("\n".join(md) + "\n")
    (RES / "r6.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()

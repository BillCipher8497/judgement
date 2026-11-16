"""Round-5 tables -> results/r5.md, results/r5.json.  Paired comparisons on the same games (CRN): challenger-seat total
and ties-as-win indicator, arm A minus arm B; 95% CI = 1.96 SE over games."""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
RES, E = H / "results", H.parent
R3, R4, Q = E / "round3_play/results", E / "round4/results", E / "phase4/results"
TYPES = ["model", "noisy0.1", "noisy0.25", "noisy0.5", "honour", "unknown"]
TRUE = {"hybrid:k=16,c=5": "model", "hyb_noisy:k=16,c=5,p=0.3": "noisy", "hyb_honour:k=16,c=5": "honour"}


def g(pat, d=RES):
    return sorted(d.glob(pat))


ARMS = {
    "hyb": {"pool": [Q / "q2_hyb_ref.npz", R3 / "f_hyb_ref2.npz"], "plain": [Q / "q2_hyb_ref.npz", R3 / "f_hyb_ref2.npz"],
            "bi": [R3 / "f_bi_hyb.npz", R3 / "f_bi_hyb2.npz"], "rb": g("f_b1h_rb_*.npz"),
            "rbpf": g("f_b2_rbpf_*.npz"), "bipf": g("f_b2_bipf_*.npz")},
    "noisy": {"pool": g("f_g1a_ref_*.npz", R4), "plain": g("f_g1a_plain_*.npz", R4), "bi": g("f_g1a_bi_?.npz", R4),
              "rb": g("f_b1n_rb_*.npz")},
    "honour": {"pool": g("f_g1b_ref_*.npz", R4), "plain": g("f_g1b_plain_*.npz", R4), "bi": g("f_g1b_bi_?.npz", R4),
               "rb": g("f_b1o_rb_*.npz")},
    "mix": {"plain": g("f_b1m_plain_*.npz"), "bi": g("f_b1m_bi_*.npz"), "rb": g("f_b1m_rb_*.npz"), "rbnl": g("f_b1m_rbnl_*.npz")},
}
LABEL = {"hyb": "3x Hybrid (model bids)", "noisy": "3x noisy-bid (±1 w.p. .3)", "honour": "3x honour-count bid",
         "mix": "mixed (1 of each)"}
NAME = {"pool": "pool", "plain": "Hybrid", "bi": "Hybrid-BI", "rb": "Hybrid-RB", "rbpf": "Hybrid-RB+PF",
        "bipf": "Hybrid-BI+PF", "rbnl": "Hybrid-RB no-learning"}


def load(files):
    parts = [np.load(f) for f in files if Path(f).exists()]
    if not parts:
        return None
    d = {}
    for k in ("totals", "scores", "secs"):
        d[k] = np.concatenate([p[k] for p in parts])
    d["game"] = np.concatenate([p["game"] if "game" in p else np.arange(len(p["totals"])) for p in parts])
    o = np.argsort(d["game"], kind="stable")
    d = {k: v[o] for k, v in d.items()}
    s = d["game"] % 4
    i = np.arange(len(s))
    d["tot"] = d["totals"][i, s]
    d["win"] = (d["totals"][i, s] == d["totals"].max(1)).astype(float)
    d["rs"] = d["scores"][i, :, s]                       # (G, 25) challenger per-round scores
    return d


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def pair(a, b):
    c = np.intersect1d(a["game"], b["game"])
    ia, ib = np.searchsorted(a["game"], c), np.searchsorted(b["game"], c)
    return ia, ib


def fmt(m):
    return f"{m[0]:+.2f} ± {m[1]:.2f}"


def main():
    out, md = {}, ["# Round 5 (bots): Hybrid-RB and persistent particles", ""]
    D = {p: {a: load(f) for a, f in arms.items()} for p, arms in ARMS.items()}
    md += ["## Paired comparisons (same games, challenger seat g%4)", "",
           "| pool | A − B | n | Δ pts/game | ΔP(win) ties-as-win | Δ rounds 1–8 | Δ rounds 9–25 | s/game A |",
           "|---|---|---|---|---|---|---|---|"]
    COMP = [("plain", "pool"), ("bi", "pool"), ("rb", "pool"), ("bi", "plain"), ("rb", "plain"), ("rb", "bi"),
            ("rbpf", "rb"), ("bipf", "bi"), ("rbpf", "plain"), ("bipf", "plain"), ("rbpf", "bi"), ("rbnl", "plain"), ("rb", "rbnl")]
    for p, arms in D.items():
        for a, b in COMP:
            if arms.get(a) is None or arms.get(b) is None or (b == "pool" and p == "hyb"):
                continue
            A, B = arms[a], arms[b]
            ia, ib = pair(A, B)
            if a == "bi" and arms.get("rb") is not None:     # restrict to the games Hybrid-RB played (same n)
                keep = np.isin(A["game"][ia], arms["rb"]["game"])
                ia, ib = ia[keep], ib[keep]
            if len(ia) < 2:
                continue
            r = {"pts": ci(A["tot"][ia] - B["tot"][ib]), "win": ci(A["win"][ia] - B["win"][ib]),
                 "early": ci(A["rs"][ia, :8].sum(1) - B["rs"][ib, :8].sum(1)),
                 "late": ci(A["rs"][ia, 8:].sum(1) - B["rs"][ib, 8:].sum(1)), "secs": float(A["secs"][ia].mean())}
            out[f"{p}|{a}-{b}"] = r
            md.append(f"| {LABEL[p]} | {NAME[a]} − {NAME[b]} | {r['pts'][2]} | {fmt(r['pts'])} | "
                      f"{r['win'][0]:+.3f} ± {r['win'][1]:.3f} | {fmt(r['early'])} | {fmt(r['late'])} | {r['secs']:.1f} |")
    # posterior validation
    md += ["", "## Type posterior (Hybrid-RB, all pools): mean posterior mass on the true type after round r", "",
           "true type 'noisy' = total mass on noisy0.1/0.25/0.5 (true p = 0.3 is not in the family)", "",
           "| true type | n opp-games | r=1 | r=2 | r=3 | r=5 | r=8 | r=12 | r=24 | MAP correct r=3 | r=8 | r=24 | median rounds to stay MAP-correct |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    recs = []
    for f in g("f_b1*_rb_*_post.json"):
        for x in json.load(open(f)):
            x["mix"] = "b1m" in f.name
            recs.append(x)
    post = {}
    for x in recs:
        tt = TRUE[x["kind"]]
        pr = np.array(x["post"])
        fam = pr[1:4].sum() if tt == "noisy" else pr[TYPES.index(tt)]
        grp = np.array([pr[0], pr[1:4].sum(), pr[4], pr[5]])
        mapc = int(np.argmax(grp)) == ["model", "noisy", "honour"].index(tt)
        post.setdefault(tt + (" (mixed pool)" if x["mix"] else ""), {}).setdefault((x["g"], x["seat"], x.get("f", "")), {})[x["r"]] = (fam, mapc)
    out["posterior"] = {}
    for tt, dd in post.items():
        rows = list(dd.values())
        def m(r, j):
            v = [row[r][j] for row in rows if r in row]
            return float(np.mean(v)) if v else float("nan")
        stay = []
        for row in rows:
            rr = sorted(row)
            k = len(rr)
            while k > 0 and row[rr[k - 1]][1]:
                k -= 1
            stay.append(rr[k] + 1 if k < len(rr) else 99)
        o = {r: m(r, 0) for r in range(24)}
        out["posterior"][tt] = {"n": len(rows), "mass": o, "map": {r: m(r, 1) for r in range(24)},
                                "stay_median": float(np.median(stay)), "never": float(np.mean(np.array(stay) == 99))}
        md.append(f"| {tt} | {len(rows)} | " + " | ".join(f"{o[r - 1]:.2f}" for r in (1, 2, 3, 5, 8, 12, 24)) + " | "
                  + " | ".join(f"{m(r - 1, 1):.2f}" for r in (3, 8, 24)) + f" | {np.median(stay):.0f} |")
    # sampler stats
    md += ["", "## Sampler (challenger PIMC decisions)", "",
           "| arm | decisions | median ESS | ESS<4 | raw ESS<4 (pre-guard) | tempered | median proposals | M capped (512 new) | kept survivors PF: median / mean / P(>=16) |",
           "|---|---|---|---|---|---|---|---|---|"]
    SS = {"hyb Hybrid-BI (r3)": [R3 / "f_bi_hyb_stats.json"], "noisy Hybrid-BI (r4)": g("f_g1a_bi_?_stats.json", R4),
          "honour Hybrid-BI (r4)": g("f_g1b_bi_?_stats.json", R4)}
    for p, t in (("hyb", "b1h"), ("noisy", "b1n"), ("honour", "b1o"), ("mix", "b1m")):
        SS[f"{p} Hybrid-RB"] = g(f"f_{t}_rb_*_stats.json")
    SS["mix Hybrid-BI"] = g("f_b1m_bi_*_stats.json")
    SS["hyb Hybrid-RB+PF"] = g("f_b2_rbpf_*_stats.json")
    SS["hyb Hybrid-BI+PF"] = g("f_b2_bipf_*_stats.json")
    out["sampler"] = {}
    for k, fs in SS.items():
        st = []
        for f in fs:
            st += json.load(open(f))
        if not st:
            continue
        ess = np.array([s["ess"] for s in st]); raw = np.array([s.get("ess_raw", s["ess"]) for s in st])
        M = np.array([s["M"] for s in st]); beta = np.array([s.get("beta", 1.0) for s in st])
        kept = [s["kept"] for s in st if "kept" in s]
        r = {"n": len(st), "ess_med": float(np.median(ess)), "ess_lt4": float((ess < 4).mean()),
             "raw_lt4": float((raw < 4).mean()), "tempered": float((beta < 1).mean()), "M_med": float(np.median(M)),
             "cap": float((M >= 512).mean()), "kept_med": float(np.median(kept)) if kept else None,
             "kept_mean": float(np.mean(kept)) if kept else None, "kept_geK": float(np.mean(np.array(kept) >= 16)) if kept else None,
             "pool_med": float(np.median([s["pool"] for s in st if "pool" in s])) if kept else None}
        out["sampler"][k] = r
        kp = "" if r["kept_med"] is None else f"{r['kept_med']:.0f} / {r['kept_mean']:.1f} / {r['kept_geK']:.3f}"
        md.append(f"| {k} | {r['n']} | {r['ess_med']:.1f} | {r['ess_lt4']:.3f} | {r['raw_lt4']:.3f} | {r['tempered']:.3f} | "
                  f"{r['M_med']:.0f} | {r['cap']:.3f} | {kp} |")
    md += ["", "95% CIs = 1.96·SE over games; CRN (same seeds, same deals); seeds: hyb 400M (r3), noisy/honour 500M (r4), mix 700M."]
    (RES / "r5.md").write_text("\n".join(md) + "\n")
    (RES / "r5.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()

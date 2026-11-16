"""R1 tables: challenger seat vs pool policy in the same seat of the all-pool reference game (same seed = CRN).
-> results/r1.md / r1.json"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
RES = H / "results"
E = H.parent
Q, F, P = E / "phase4/results", E / "final/results", E / "play/results"
NT_ROUNDS = np.array([r % 5 == 4 for r in range(25)])
HREF = [Q / "q2_hyb_ref.npz", RES / "f_hyb_ref2.npz"]
BI_H = [RES / "f_bi_hyb.npz", RES / "f_bi_hyb2.npz"]
BIG_H = [RES / "f_big_hyb.npz", RES / "f_big_hyb2.npz"]
BI_T = [RES / "f_bi_ta.npz", RES / "f_bi_ta2.npz"]
ROWS = [  # (label, challenger files, reference files)
    ("3xHybrid | Hybrid-BI vs Hybrid", BI_H, HREF),
    ("3xHybrid | Hybrid k32 c6 vs Hybrid", BIG_H, HREF),
    ("3xHybrid | Hybrid-BI k32 c6 vs Hybrid", [RES / "f_bi_big.npz", RES / "f_bi_big2.npz"], HREF),
    ("3xHybrid | Hybrid-BI k32 c6 − Hybrid k32 c6", [RES / "f_bi_big.npz", RES / "f_bi_big2.npz"], BIG_H),
    ("3xHybrid | Hybrid-BI+oracle vs Hybrid", [RES / "f_bio_hyb.npz"], HREF),
    ("3xHybrid | Hybrid+oracle (phase4) vs Hybrid", [Q / "q2_hyb_hybo.npz"], HREF),
    ("3xHybrid | Hybrid-BI − Hybrid k32 c6", BI_H, BIG_H),
    ("3xTA | Hybrid-BI vs TA", BI_T, [F / "f_ta_ta.npz"]),
    ("3xTA | Hybrid (c5,k16) vs TA", [P / "1v3_x_hyb_c5.npz"], [F / "f_ta_ta.npz"]),
    ("3xTA | Hybrid-BI − Hybrid (paired)", BI_T, [P / "1v3_x_hyb_c5.npz"]),
]
KEYS = ("totals", "wins", "scores", "secs")


def load(files):
    """Concatenate result files -> dict of arrays indexed by game number (q2 files: game = row index)."""
    parts = [np.load(f) for f in files if f.exists()]
    if not parts:
        return None
    out = {k: np.concatenate([p[k] for p in parts]) for k in KEYS}
    out["game"] = np.concatenate([p["game"] if "game" in p else np.arange(len(p["totals"])) for p in parts])
    return out


def pair(c, r):
    common = np.intersect1d(c["game"], r["game"])
    ic = np.searchsorted(c["game"], common) if np.all(np.diff(c["game"]) > 0) else None
    ir = np.searchsorted(r["game"], common) if np.all(np.diff(r["game"]) > 0) else None
    assert ic is not None and ir is not None
    return {k: c[k][ic] for k in KEYS}, {k: r[k][ir] for k in KEYS}, common


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def tiewin(t):
    return (t == t.max(1, keepdims=True)).astype(float)


def main():
    out = {}
    md = ["| table | n | gain pts/game | ΔP(win) ties-as-win | ΔP(win) split | gain trump rds/game | gain NT rds/game | P(win) tie chal |",
          "|---|---|---|---|---|---|---|---|"]
    for lab, cp, rp in ROWS:
        c, r = load(cp), load(rp)
        if c is None or r is None:
            continue
        c, r, games = pair(c, r)
        G = len(games)
        if G < 10:
            continue
        ar = np.arange(G); s = games % 4
        gain = c["totals"][:G][ar, s] - r["totals"][:G][ar, s]
        dt = tiewin(c["totals"][:G])[ar, s] - tiewin(r["totals"][:G])[ar, s]
        ds = c["wins"][:G][ar, s] - r["wins"][:G][ar, s]
        sc = c["scores"][:G][ar, :, s] - r["scores"][:G][ar, :, s]          # (G, 25)
        row = dict(n=G, gain=ci(gain), dp_tie=ci(dt), dp_split=ci(ds), trump=ci(sc[:, ~NT_ROUNDS].sum(1)),
                   nt=ci(sc[:, NT_ROUNDS].sum(1)), p_tie=float(tiewin(c["totals"][:G])[ar, s].mean()),
                   secs=float(np.median(c["secs"][:G])))
        lad = np.array([13 - r if r < 13 else r - 11 for r in range(25)])
        for nm, m in (("n<=5", lad <= 5), ("n6-9", (lad >= 6) & (lad <= 9)), ("n10-13", lad >= 10)):
            row[nm] = ci(sc[:, m].sum(1))
        out[lab] = row
        f = lambda v, d=2: f"{v[0]:+.{d}f} ± {v[1]:.{d}f}"
        md.append(f"| {lab} | {G} | {f(row['gain'])} | {f(row['dp_tie'], 3)} | {f(row['dp_split'], 3)} | {f(row['trump'])} | {f(row['nt'])} | {row['p_tie']:.3f} |")
    md += ["", "Gain by round size (pts/game summed over those rounds): " + "; ".join(
        f"{k.split('| ')[1]}: " + ", ".join(f"{nm} {v[nm][0]:+.2f}±{v[nm][1]:.2f}" for nm in ("n<=5", "n6-9", "n10-13"))
        for k, v in out.items() if "n<=5" in v)]
    # sampler validation
    md += ["", "Sampler stats (challenger PIMC decisions):", "",
           "| run | decisions | median ESS | ESS<16 frac | ESS<4 frac | mean M | M capped (512) | card-placement acc informed / uniform | true deal at max weight |",
           "|---|---|---|---|---|---|---|---|---|"]
    for tag in ("bi_hyb", "bi_ta", "bi_big", "bio_hyb"):
        S = []
        for p in (RES / f"f_{tag}_stats.json", RES / f"f_{tag}2_stats.json"):
            if p.exists():
                S += json.load(open(p))
        if not S:
            continue
        ess = np.array([x["ess"] for x in S]); M = np.array([x["M"] for x in S])
        accs = ""
        tl = ""
        if "acc_inf" in S[0]:
            ai = np.array([x["acc_inf"] for x in S]); au = np.array([x["acc_unif"] for x in S])
            accs = f"{ai.mean():.3f} / {au.mean():.3f} (Δ {ci(ai - au)[0]:+.3f} ± {ci(ai - au)[1]:.3f})"
            tl = f"{np.mean([x['true_lw'] >= x['lw_max'] - 1e-6 for x in S]):.4f}"
        out[f"stats|{tag}"] = dict(decisions=len(S), ess_med=float(np.median(ess)), ess_lt_k=float((ess < 16).mean()),
                                   ess_lt4=float((ess < 4).mean()), M=float(M.mean()), capped=float((M >= 512).mean()), acc=accs, true_max=tl)
        md.append(f"| {tag} | {len(S)} | {np.median(ess):.1f} | {(ess < 16).mean():.3f} | {(ess < 4).mean():.3f} | {M.mean():.0f} | {(M >= 512).mean():.3f} | {accs} | {tl} |")
        by = {}
        for x in S:
            by.setdefault(x["left"], []).append(x)
        md.append("  by cards left: " + "; ".join(
            f"{k}: n={len(v)} ESSmed {np.median([x['ess'] for x in v]):.1f}" +
            (f" acc {np.mean([x['acc_inf'] for x in v]):.3f}/{np.mean([x['acc_unif'] for x in v]):.3f}" if "acc_inf" in v[0] else "")
            for k, v in sorted(by.items())))
    (RES / "r1.md").write_text("\n".join(md) + "\n")
    (RES / "r1.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()

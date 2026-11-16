"""M1 analysis: Poisson-binomial (independent card) model vs simulated hero trick
distributions.  Input math/results/m1_raw_*.json (m1_sim.py).  Output
math/results/m1_summary.json and math/results/m1_tables.txt.

Per hand and play policy (ta / greedy):
  p_i = P(card i wins a trick)     (sum p_i = E[T] exactly)
  emp = empirical P(t) (R rollouts);  pb = PB(p)
  disp = Var_emp / Var_PB ;  Var_emp - Var_PB = 2 sum_{i<j} Cov(X_i, X_j) exactly (plug-in).
  b_emp = argmax (10+b) emp(b) ; b_pb = argmax (10+b) pb(b) ; EV loss = EV_emp(b_emp)-EV_emp(b_pb)
  noise floor: argmax on rollout half A vs half B.
Cheap bidder: p_i from a lookup (n, NT, is_trump, rank, min(suit len,6)) with backoff,
fitted on even-index hands, evaluated on odd-index hands.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "experiments" / "phase1"))
from pb import best_bid, ev_vec, kl, mean_var, modes, pb_pmf, tv  # noqa: E402
from common import is_bimodal  # noqa: E402

RES = HERE / "results"


def mci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return [float("nan"), float("nan"), int(len(x))]
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def card_key(c, hand, trump, n):
    s, r = c // 13, c % 13
    ln = ((hand >> (13 * s)) & 0x1FFF).bit_count()
    return (n, trump < 0, s == trump, r, min(ln, 6))


def card_key2(c, hand, trump, n):
    """Richer key: side cards also see own trump length; trumps see 'has a void'."""
    s, r = c // 13, c % 13
    lens = [((hand >> (13 * x)) & 0x1FFF).bit_count() for x in range(4)]
    if trump >= 0 and s == trump:
        extra = int(any(lens[x] == 0 for x in range(4) if x != trump))
        return (n, False, True, r, min(lens[s], 7), extra)
    tl = min(lens[trump], 4) if trump >= 0 else -1
    return (n, trump < 0, False, r, min(lens[s], 6), tl)


def load():
    recs = []
    for f in sorted(RES.glob("m1_raw_n*_*.json")):
        recs += json.load(open(f))
    return recs


def per_hand(recs):
    rows = []
    for rec in recs:
        n, tr = rec["n"], rec["trump"]
        lens = rec["lens"]
        for k in ("ta", "greedy"):
            d = rec[k]
            c = np.array(d["counts"], float)
            R = c.sum()
            emp = c.sum(0) / R
            A, B = c[0] / c[0].sum(), c[1] / c[1].sum()
            p = np.array(d["wins"], float) / R
            pb = pb_pmf(p)
            me, ve = mean_var(emp)
            mp, vp = mean_var(pb)
            be, bp = best_bid(emp), best_bid(pb)
            eve = ev_vec(emp)
            mo = modes(emp)
            rows.append({
                "n": n, "nt": tr < 0, "pol": k, "i": rec["i"], "tlen": lens[0] if tr >= 0 else -1,
                "voids": sum(1 for x in lens if x == 0), "mu": me, "mu_pb": mp, "var": ve, "var_pb": vp,
                "disp": ve / vp if vp > 0.02 else np.nan, "cov_sum": (ve - vp) / 2,
                "tv": tv(emp, pb), "kl": kl(emp, pb), "b_emp": be, "b_pb": bp, "b_played": d["bid"],
                "ev_loss": float(eve[be] - eve[bp]), "ev_opt": float(eve[be]),
                "b_A": best_bid(A), "b_B": best_bid(B),
                "ev_loss_split": float(ev_vec(B)[best_bid(A)] - ev_vec(B)[bp]),
                "bimodal": is_bimodal(emp), "mode_hi": mo[-1], "mode_lo": mo[0],
                "pb_mode": modes(pb)[-1],
                "emp_b_in_mode_rule": mo[0] <= be <= mo[-1] + 1,
                "pb_in_floorceil": np.floor(mp + 1e-9) <= bp <= np.ceil(mp - 1e-9) + (0 if mp % 1 else 0),
                "round_mu": int(np.floor(me + 0.5 - 1e-12)),
            })
    return rows


def cheap_bidder(recs, keyf=card_key):
    out = {}
    for k in ("ta", "greedy"):
        tab = defaultdict(lambda: [0.0, 0])
        tab2 = defaultdict(lambda: [0.0, 0])
        for rec in recs:
            if rec["i"] % 2:
                continue
            R = np.array(rec[k]["counts"]).sum()
            for c, w in zip(rec["cards"], rec[k]["wins"]):
                key = keyf(c, rec["hand"], rec["trump"], rec["n"])
                tab[key][0] += w / R
                tab[key][1] += 1
                tab2[key[:4]][0] += w / R
                tab2[key[:4]][1] += 1
        res = defaultdict(list)
        for rec in recs:
            if rec["i"] % 2 == 0:
                continue
            p = []
            for c in rec["cards"]:
                key = keyf(c, rec["hand"], rec["trump"], rec["n"])
                s, m = tab.get(key, [0, 0])
                if m >= 5:
                    p.append(s / m)
                else:
                    s2, m2 = tab2.get(key[:4], [0, 0])
                    p.append(s2 / m2 if m2 else 0.0)
            c = np.array(rec[k]["counts"], float)
            emp = c.sum(0) / c.sum()
            be = best_bid(emp)
            bl = best_bid(pb_pmf(p))
            bm = rec[k]["bid"]
            ev = ev_vec(emp)
            g = (rec["n"], "NT" if rec["trump"] < 0 else "S")
            for key in (g, "all"):
                res[key].append((bl == be, abs(bl - be) <= 1, ev[be] - ev[bl], bm == be, ev[be] - ev[bm],
                                 sum(p) - float((np.arange(len(emp)) * emp).sum())))
        out[k] = {str(g): {"hands": len(v), "lookup_match": mci([x[0] for x in v]),
                           "lookup_within1": mci([x[1] for x in v]), "lookup_ev_loss": mci([x[2] for x in v]),
                           "model_bid_match": mci([x[3] for x in v]), "model_ev_loss": mci([x[4] for x in v]),
                           "lookup_mean_bias": mci([x[5] for x in v])} for g, v in res.items()}
    return out


def summarize(rows, sel):
    r = [x for x in rows if sel(x)]
    if not r:
        return None
    f = lambda key: mci([x[key] for x in r])  # noqa: E731
    ve = np.array([x["var"] for x in r])
    vp = np.array([x["var_pb"] for x in r])
    return {"hands": len(r), "TV": f("tv"), "KL": f("kl"), "disp_mean": f("disp"),
            "disp_pooled": float(ve.sum() / vp.sum()), "frac_under": mci([x["var"] < x["var_pb"] for x in r]),
            "match": mci([x["b_emp"] == x["b_pb"] for x in r]), "ev_loss": f("ev_loss"),
            "ev_loss_split": f("ev_loss_split"),
            "noise_floor_match_AB": mci([x["b_A"] == x["b_B"] for x in r]),
            "bimodal": mci([x["bimodal"] for x in r]), "mean_tricks": f("mu"), "var_emp": f("var"),
            "var_pb": f("var_pb"), "b_pb_minus_round_mu": mci([x["b_pb"] - x["round_mu"] for x in r]),
            "b_emp_minus_round_mu": mci([x["b_emp"] - x["round_mu"] for x in r]),
            "emp_bid_in_[mode,mode+1]": mci([x["emp_b_in_mode_rule"] for x in r])}


def main():
    recs = load()
    rows = per_hand(recs)
    S = {"n_records": len(recs), "groups": {}, "paired_ta_minus_greedy": {}}
    classes = {
        "all": lambda x: True, "trump": lambda x: not x["nt"], "NT": lambda x: x["nt"],
        "long_trump>=5": lambda x: x["tlen"] >= 5, "trump_len<=2": lambda x: 0 <= x["tlen"] <= 2,
        "void>=1": lambda x: x["voids"] >= 1, "no_void": lambda x: x["voids"] == 0,
        "long_trump>=5&void": lambda x: x["tlen"] >= 5 and x["voids"] >= 1,
    }
    for pol in ("ta", "greedy"):
        for n in (5, 8, 13, "all"):
            for cname, cf in classes.items():
                key = f"{pol}|n={n}|{cname}"
                S["groups"][key] = summarize(rows, lambda x: x["pol"] == pol and (n == "all" or x["n"] == n) and cf(x))
    # paired difference in dispersion and covariance (same hands)
    idx = {(x["n"], x["nt"], x["i"], x["pol"]): x for x in rows}
    for n in (5, 8, 13):
        for nt in (False, True):
            d = [idx[(n, nt, i, "ta")]["cov_sum"] - idx[(n, nt, i, "greedy")]["cov_sum"]
                 for (nn, nnt, i, pol) in idx if nn == n and nnt == nt and pol == "ta"]
            dd = [idx[(n, nt, i, "ta")]["disp"] - idx[(n, nt, i, "greedy")]["disp"]
                  for (nn, nnt, i, pol) in idx if nn == n and nnt == nt and pol == "ta"]
            S["paired_ta_minus_greedy"][f"n={n}|{'NT' if nt else 'S'}"] = {"cov_sum": mci(d), "disp": mci(dd)}
    # shading pattern by frac(mu) (compare to E3 greedy)
    sh = {}
    for pol in ("ta", "greedy"):
        for lo in (0, .25, .5, .75):
            r = [x for x in rows if x["pol"] == pol and lo <= x["mu"] % 1 < lo + .25]
            sh[f"{pol}|frac[{lo},{lo + .25})"] = {"emp": mci([x["b_emp"] - x["round_mu"] for x in r]),
                                                  "pb": mci([x["b_pb"] - x["round_mu"] for x in r])}
    S["shading_by_frac_mu"] = sh
    S["cheap_bidder"] = cheap_bidder(recs)
    S["cheap_bidder_rich"] = cheap_bidder(recs, card_key2)
    json.dump(S, open(RES / "m1_summary.json", "w"), indent=1, default=float)
    # compact text tables
    L = ["M1 tables (± = 95% CI half-width, n = hands)", ""]
    L.append("pol | n | class | hands | TV | KL | disp(mean) | disp(pooled) | frac under | PB-bid match | noise-floor A/B | EV loss | EV loss split")
    for key, g in S["groups"].items():
        if g is None or g["hands"] < 20:
            continue
        pol, n, c = key.split("|")
        L.append(f"{pol} | {n[2:]} | {c} | {g['hands']} | {g['TV'][0]:.3f}±{g['TV'][1]:.3f} | {g['KL'][0]:.3f}±{g['KL'][1]:.3f} | "
                 f"{g['disp_mean'][0]:.3f}±{g['disp_mean'][1]:.3f} | {g['disp_pooled']:.3f} | {g['frac_under'][0]:.3f} | "
                 f"{g['match'][0]:.3f}±{g['match'][1]:.3f} | {g['noise_floor_match_AB'][0]:.3f} | "
                 f"{g['ev_loss'][0]:.3f}±{g['ev_loss'][1]:.3f} | {g['ev_loss_split'][0]:.3f}±{g['ev_loss_split'][1]:.3f}")
    L += ["", "paired TA − Greedy (same hands): Σ_{i<j}Cov and dispersion"]
    for k, v in S["paired_ta_minus_greedy"].items():
        L.append(f"{k} | cov {v['cov_sum'][0]:+.3f}±{v['cov_sum'][1]:.3f} | disp {v['disp'][0]:+.3f}±{v['disp'][1]:.3f} (n={v['disp'][2]})")
    L += ["", "shading b − round(mu) by frac(mu): empirical-optimal vs PB-optimal"]
    for k, v in sh.items():
        L.append(f"{k} | emp {v['emp'][0]:+.3f}±{v['emp'][1]:.3f} | PB {v['pb'][0]:+.3f}±{v['pb'][1]:.3f} (n={v['emp'][2]})")
    L += ["", "cheap lookup bidder (odd hands, fitted on even) vs played model bid, oracle = empirical-optimal at played bid"]
    for pol, d in list(S["cheap_bidder"].items()) + [(k + "(rich key)", v) for k, v in S["cheap_bidder_rich"].items()]:
        for g, v in d.items():
            L.append(f"{pol} | {g} | {v['hands']} | lookup match {v['lookup_match'][0]:.3f}±{v['lookup_match'][1]:.3f} "
                     f"| ≤1 {v['lookup_within1'][0]:.3f} | EVloss {v['lookup_ev_loss'][0]:.3f}±{v['lookup_ev_loss'][1]:.3f} "
                     f"| model match {v['model_bid_match'][0]:.3f}±{v['model_bid_match'][1]:.3f} | model EVloss {v['model_ev_loss'][0]:.3f} "
                     f"| lookup mean bias {v['lookup_mean_bias'][0]:+.3f}")
    open(RES / "m1_tables.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

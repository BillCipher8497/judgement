"""M5 analysis: where does the card-win dependence live?

Per hand (hero = first bidder, model bid), from R rollout masks:
  X (R x n) card-win indicators, p = mean, C = plug-in covariance.
  Var_emp - Var_PB = 2 sum_{i<j} C_ij  (P0), split by pair type:
    trump rounds: TT (both trump), SS (same side suit), TX (trump x side), XX (different side suits)
    NT: same suit, cross suit.
  Each component reported as its share of delta - 1 (= 2 sum_{i<j} cov_type = sum over ordered pairs / Var_PB), per hand then averaged.
Models for the hero's trick law: PB (independent cards), SB (suit-block: per-suit empirical law of T_s,
suits independent).  Var_SB = Var_PB + within-suit cov exactly, so dispersion error of SB = cross-suit cov.
Fit metrics: TV, KL(emp||model) (full data), cross-fit TV (model from rollouts A, emp from B) vs noise floor
TV(emp_A, emp_B); bid = argmax (10+b) P(b); match with emp-optimal; EV loss under emp.
Output: results/m5_summary.json, m5_tables.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
H = 10
CFG = [(n, t) for n in (5, 8, 13) for t in ("S", "NT")]


def pb_pmf(p):
    f = np.array([1.0])
    for q in p:
        g = np.zeros(len(f) + 1)
        g[:-1] += f * (1 - q)
        g[1:] += f * q
        f = g
    return f


def conv_all(pmfs, n):
    f = np.array([1.0])
    for g in pmfs:
        f = np.convolve(f, g)
    out = np.zeros(n + 1)
    out[:len(f)] = f[:n + 1]
    return out


def hist(T, n):
    return np.bincount(T, minlength=n + 1)[:n + 1] / len(T)


def bid(P):
    ev = (H + np.arange(len(P))) * P
    return int(np.argmax(ev - 1e-12 * np.arange(len(P))))


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def kl(a, b):
    m = a > 0
    return float(np.sum(a[m] * np.log(a[m] / np.maximum(b[m], 1e-12))))


def models(X, suit, n):
    p = X.mean(0)
    P_pb = pb_pmf(p)
    blocks = []
    for s in np.unique(suit):
        Ts = X[:, suit == s].sum(1).astype(int)
        blocks.append(hist(Ts, int((suit == s).sum())))
    P_sb = conv_all(blocks, n)
    return P_pb, P_sb


def analyze_hand(cards, masks, n, trump):
    R = len(masks)
    X = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
    suit = np.asarray(cards) // 13
    p = X.mean(0)
    C = X.T @ X / R - np.outer(p, p)
    vpb = float((p * (1 - p)).sum())
    T = X.sum(1).astype(int)
    vemp = float(T.var())
    same = suit[:, None] == suit[None, :]
    off = ~np.eye(n, dtype=bool)
    istr = suit == trump if trump >= 0 else np.zeros(n, bool)
    comp = {}
    if trump >= 0:
        tt = istr[:, None] & istr[None, :]
        tx = istr[:, None] ^ istr[None, :]
        comp["TT"] = C[off & tt].sum()
        comp["SS"] = C[off & same & ~tt].sum()
        comp["TX"] = C[off & tx].sum()
        comp["XX"] = C[off & ~same & ~tx].sum()
    else:
        comp["same"] = C[off & same].sum()
        comp["cross"] = C[off & ~same].sum()
    within = C[off & same].sum()
    # per-suit within contributions by suit length
    per_suit = []
    for s in np.unique(suit):
        m = suit == s
        k = int(m.sum())
        if k >= 2:
            per_suit.append((k, bool(s == trump), float(C[np.ix_(m, m)][~np.eye(k, dtype=bool)].sum())))
    P_emp = hist(T, n)
    P_pb, P_sb = models(X, suit, n)
    A, B = X[:R // 2], X[R // 2:]
    PA_pb, PA_sb = models(A, suit, n)
    PB_emp = hist(B.sum(1).astype(int), n)
    PA_emp = hist(A.sum(1).astype(int), n)
    be = bid(P_emp)
    ev = (H + np.arange(n + 1)) * P_emp
    out = {"vpb": vpb, "vemp": vemp, "vsb": vpb + within, "comp": comp, "per_suit": per_suit,
           "tv_pb": tv(P_emp, P_pb), "tv_sb": tv(P_emp, P_sb), "kl_pb": kl(P_emp, P_pb), "kl_sb": kl(P_emp, P_sb),
           "xtv_pb": tv(PB_emp, PA_pb), "xtv_sb": tv(PB_emp, PA_sb), "xtv_floor": tv(PB_emp, PA_emp),
           "b_emp": be, "b_pb": bid(P_pb), "b_sb": bid(P_sb), "ab_match": bid(PA_emp) == bid(PB_emp),
           "loss_pb": float(ev[be] - ev[bid(P_pb)]), "loss_sb": float(ev[be] - ev[bid(P_sb)]),
           "trump_len": int(istr.sum())}
    return out


def mci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def f(m, d=3):
    return f"{m[0]:.{d}f} ± {m[1]:.{d}f}"


def main():
    summary = {}
    lines = []
    suit_rows = []
    for n, t in CFG:
        fn = RES / f"m5_raw_n{n}_{t}.npz"
        if not fn.exists():
            continue
        D = np.load(fn)
        trump = 0 if t == "S" else -1
        for k, pol in enumerate(D["pols"]):
            hs = [analyze_hand(D["cards"][i], D["masks"][k, i], n, trump) for i in range(len(D["cards"]))]
            hs = [h for h in hs if h["vpb"] > 1e-9]
            key = f"{pol}|n={n}|{t}"
            d = {"hands": len(hs)}
            d["delta"] = mci([h["vemp"] / h["vpb"] for h in hs])
            d["delta_SB"] = mci([h["vsb"] / h["vpb"] for h in hs])
            d["SB_disp_error"] = mci([(h["vemp"] - h["vsb"]) / h["vpb"] for h in hs])   # = cross share
            for c in hs[0]["comp"]:
                d["share_" + c] = mci([h["comp"][c] / h["vpb"] for h in hs])
            d["sumcov_abs"] = {c: mci([h["comp"][c] for h in hs]) for c in hs[0]["comp"]}
            for m in ("tv_pb", "tv_sb", "kl_pb", "kl_sb", "xtv_pb", "xtv_sb", "xtv_floor", "loss_pb", "loss_sb"):
                d[m] = mci([h[m] for h in hs])
            d["match_pb"] = mci([h["b_pb"] == h["b_emp"] for h in hs])
            d["match_sb"] = mci([h["b_sb"] == h["b_emp"] for h in hs])
            d["ab_floor"] = mci([h["ab_match"] for h in hs])
            summary[key] = d
            for h in hs:
                for (L, ist, v) in h["per_suit"]:
                    suit_rows.append((str(pol), t, n, L, ist, v, h["vpb"]))
            print(key, f(d["delta"]), {c: f(d["share_" + c]) for c in hs[0]["comp"]}, flush=True)
    # suit-length breakdown of within-suit covariance (2 sum cov within the suit), pooled n
    sl = {}
    for pol in ("ta", "ta_plus"):
        for t in ("S", "NT"):
            for ist in ((True, False) if t == "S" else (False,)):
                for lo, hi in ((2, 2), (3, 3), (4, 4), (5, 13)):
                    v = [r[5] for r in suit_rows if r[0] == pol and r[1] == t and r[4] == ist and lo <= r[3] <= hi]
                    if len(v) > 5:
                        sl[f"{pol}|{t}|{'trump' if ist else 'side'}|len{lo}-{hi}"] = mci(v)
    summary["within_suit_by_length(2sumcov per suit)"] = sl
    json.dump(summary, open(RES / "m5_summary.json", "w"), indent=1)

    L = lines.append
    L("M5 dependence decomposition (hero = first bidder, model bid; 1000 hands x R=300 per cell)")
    L("share_X = 2*sum cov over pair type X / Var_PB (per hand, mean ± 95% CI); delta - 1 = sum of shares")
    L("| play | n | type | hands | δ | within-suit share | cross-suit share | TT | SS | TX | XX |")
    L("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, d in summary.items():
        if "|" not in key or "within" in key:
            continue
        pol, nn, t = key.split("|")
        if t == "S":
            w = [d["share_TT"][0] + d["share_SS"][0]]
            c = d["share_TX"][0] + d["share_XX"][0]
            L(f"| {pol} | {nn[2:]} | S | {d['hands']} | {f(d['delta'])} | {w[0]:.3f} | {c:.3f} | {f(d['share_TT'])} | "
              f"{f(d['share_SS'])} | {f(d['share_TX'])} | {f(d['share_XX'])} |")
        else:
            L(f"| {pol} | {nn[2:]} | NT | {d['hands']} | {f(d['delta'])} | {f(d['share_same'])} | {f(d['share_cross'])} | – | – | – | – |")
    L("")
    L("Models: PB (independent cards) vs SB (suit-block). Dispersion error = δ_model − δ_emp; TV/KL full-data;")
    L("xTV = model from rollouts 1-150 vs empirical 151-300 (floor = emp A vs emp B); bid match vs emp-optimal; EV loss pts/round")
    L("| play | n | type | δ_PB err | δ_SB err | TV PB | TV SB | KL PB | KL SB | xTV PB | xTV SB | xTV floor | match PB | match SB | A/B floor | loss PB | loss SB |")
    L("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, d in summary.items():
        if "|" not in key or "within" in key:
            continue
        pol, nn, t = key.split("|")
        L(f"| {pol} | {nn[2:]} | {t} | {1 - d['delta'][0]:+.3f} | {-d['SB_disp_error'][0]:+.3f} ± {d['SB_disp_error'][1]:.3f} | "
          f"{f(d['tv_pb'])} | {f(d['tv_sb'])} | {d['kl_pb'][0]:.3f} | {d['kl_sb'][0]:.3f} | {d['xtv_pb'][0]:.3f} | {d['xtv_sb'][0]:.3f} | "
          f"{d['xtv_floor'][0]:.3f} | {f(d['match_pb'])} | {f(d['match_sb'])} | {d['ab_floor'][0]:.2f} | {f(d['loss_pb'], 2)} | {f(d['loss_sb'], 2)} |")
    L("")
    L("Within-suit 2Σcov per suit by suit length (pooled n; absolute, tricks²):")
    for k, v in sl.items():
        L(f"  {k}: {f(v)} (suits {v[2]})")
    open(RES / "m5_tables.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

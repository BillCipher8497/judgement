"""M7: recalibrated hook-cost formula.

Same dealer situations as m3_hook.py (Phase-1 all-TA games, games_ta.npz vs games_ta_nohook.npz, CRN).
In hooked rounds (f == b*) the dealer bids b_nh = b* without the hook and b_h != f with it.
A1:  C(n) = E[ 1{f=b*} ( (10+b_nh) P_nh - (10+b_h) P_h ) ],  P from the TA bidder model.
Recalibration: P(hit) multiplied by an empirical factor k_cell = realized hit / model P(hit),
cell = direction (natural b* | forced up | forced down) x n-bucket, FIT ON EVEN GAMES ONLY,
evaluated on odd games against the measured CRN cost on the same odd games.
Variants: raw (k=1), k by direction pooled over n, k by direction x n-bucket, k by direction x n.
Output: math/results/m7_hookcal.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "phase1"))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.features import hand_features  # noqa: E402
from judgement.bidding.model import score_matrix  # noqa: E402
from judgement.engine import deal, make_rngs  # noqa: E402

from common import load_model  # noqa: E402

RES1 = ROOT / "experiments" / "phase1" / "results"
E5 = {1: 5.41, 2: 2.84, 3: 1.85, 4: 1.15, 5: 0.70, 6: 0.51, 7: 0.42, 8: 0.39, 9: 0.35, 10: 0.34,
      11: 0.36, 12: 0.32, 13: 0.27}


def bucket(n):
    return 0 if n <= 2 else 1 if n <= 4 else 2 if n <= 8 else 3


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def collect(games):
    rules = Rules()
    model = load_model("ta")
    G = dict(np.load(RES1 / "games_ta.npz"))
    Gn = dict(np.load(RES1 / "games_ta_nohook.npz"))
    base = int(G["seed_base"])
    lad = rules.ladder()
    S = {n: score_matrix(rules, n) for n in set(lad)}
    rows = []
    for g in range(games):
        drng, _ = make_rngs(base + g, 4)
        for r, n in enumerate(lad):
            hands = deal(rules, n, drng)
            tr = rules.trump_for_round(r)
            dl = rules.dealer_for_round(r)
            bids = G["bids"][g, r].astype(int)
            prior = []
            for k in range(1, 4):
                prior.append(int(bids[(dl + k) % 4]))
            x = hand_features(hands[dl], tr, n, 3, prior)
            P = model.get(n, tr).dist_all_bids(x)
            ev = (P * S[n]).sum(1)
            bstar = int(max(range(n + 1), key=lambda b: (ev[b], -b)))
            f = n - sum(prior)
            bh, bnh = int(bids[dl]), int(Gn["bids"][g, r, dl])
            hit_h = float(G["tricks"][g, r, dl] == bh)
            hit_nh = float(Gn["tricks"][g, r, dl] == bnh)
            real = float(Gn["scores"][g, r, dl] - G["scores"][g, r, dl])
            rows.append((g, n, int(f == bstar), bstar, bh, bnh, float(P[bh, bh]), float(P[bnh, bnh]),
                         hit_h, hit_nh, real))
    return np.array(rows, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    a = ap.parse_args()
    R = collect(a.games)
    g, n, hk, bs, bh, bnh, Ph, Pnh, hh, hnh, real = R.T
    up = hk.astype(bool) & (bh > bs)
    dn = hk.astype(bool) & (bh < bs)
    hkb = hk.astype(bool)
    train, test = (g % 2 == 0), (g % 2 == 1)
    nb = np.array([bucket(int(v)) for v in n])
    out = {"games": a.games, "rows": int(len(R)), "check_bnh_eq_bstar_when_hooked": float((bnh[hkb] == bs[hkb]).mean())}

    # calibration tables (train): realized / model, by direction and n
    def k_of(mask_dir, mask_cell, realized, pred):
        m = train & mask_dir & mask_cell
        return float(realized[m].mean() / pred[m].mean()) if m.sum() > 20 else 1.0

    cal = {}
    for nm, md, re_, pr in (("natural(b*,no hook)", hkb, hnh, Pnh), ("forced_up", up, hh, Ph), ("forced_down", dn, hh, Ph)):
        cal[nm] = {}
        for v in range(1, 14):
            m = md & (n == v)
            cal[nm][v] = {"model_P": ci(pr[m])[:2] + [int(m.sum())], "realized": ci(re_[m])[:2]}
    out["calibration_all_games"] = cal

    variants = {
        "raw": lambda: (np.ones_like(n), np.ones_like(n)),
    }

    def factors(level):
        knat, kh = np.ones_like(n), np.ones_like(n)
        for i in range(len(n)):
            pass
        cells = {"pooled": lambda v: 0, "bucket": bucket, "per_n": lambda v: v}[level]
        keys = np.array([cells(int(v)) for v in n])
        table = {}
        for key in np.unique(keys):
            cm = keys == key
            table[int(key)] = (k_of(hkb, cm, hnh, Pnh), k_of(up, cm, hh, Ph), k_of(dn, cm, hh, Ph))
        for i in range(len(n)):
            kn_, ku, kd = table[int(keys[i])]
            knat[i] = kn_
            kh[i] = ku if up[i] else kd if dn[i] else 1.0
        return knat, kh, table

    res = {}
    for level in ("raw", "pooled", "bucket", "per_n"):
        if level == "raw":
            knat, kh, table = np.ones_like(n), np.ones_like(n), None
        else:
            knat, kh, table = factors(level)
        pred = np.where(hkb, (10 + bnh) * np.minimum(1, Pnh * knat) - (10 + bh) * np.minimum(1, Ph * kh), 0.0)
        # also a component split: only forced-direction corrected / only natural corrected
        res[level] = {"factors(nat,up,down)": table, "by_n": {}}
        for v in range(1, 14):
            m = test & (n == v)
            res[level]["by_n"][v] = {"pred": ci(pred[m]), "measured": ci(real[m]), "E5": E5[v]}
        mm = test
        res[level]["pooled_ratio_meas_over_pred"] = float(real[mm].mean() / pred[mm].mean())
        res[level]["pred_minus_meas_by_n"] = {v: res[level]["by_n"][v]["pred"][0] - res[level]["by_n"][v]["measured"][0]
                                              for v in range(1, 14)}
    out["variants"] = res
    # component split at pooled level
    knat, kh, _ = factors("pooled")
    comp = {}
    for nm, kn_, kh_ in (("fix_forced_only", np.ones_like(n), kh), ("fix_natural_only", knat, np.ones_like(n))):
        pred = np.where(hkb, (10 + bnh) * Pnh * kn_ - (10 + bh) * Ph * kh_, 0.0)
        comp[nm] = {v: ci(pred[test & (n == v)])[:2] for v in range(1, 14)}
    out["component_split_pooled"] = comp
    json.dump(out, open(ROOT / "math" / "results" / "m7_hookcal.json", "w"), indent=1)
    for level in res:
        print(level, res[level]["factors(nat,up,down)"])
        print("  ", " ".join(f"{v}:{res[level]['by_n'][v]['pred'][0]:.2f}/{res[level]['by_n'][v]['measured'][0]:.2f}"
                             for v in range(1, 14)), "ratio %.3f" % res[level]["pooled_ratio_meas_over_pred"])


if __name__ == "__main__":
    main()

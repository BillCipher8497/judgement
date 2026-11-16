"""M3(b): predicted dealer hook cost from the TA bidder model on dealer situations
re-created from the Phase-1 all-TA self-play games (games_ta.npz, seeds BASE+g).

For each (game, round): regenerate the deal (deal stream depends only on the seed),
check the stored non-dealer bids equal the model's bids (consistency), then with
the dealer's model distributions P_b(t) (TA: depends on own bid):
   EV(b) = (10+b) P_b(b),  b* = argmax_all EV,  f = n - sum(prior bids)
   gap = 1{f == b*} (EV(b*) - max_{b != f} EV(b)).
C_pred(n) = E[gap] = P(f == b*) * E[gap | f == b*].
Compared with the measured CRN hook cost (games_ta_nohook - games_ta, dealer).
Output: math/results/m3_hook.json
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


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    a = ap.parse_args()
    rules = Rules()
    model = load_model("ta")
    G = dict(np.load(RES1 / "games_ta.npz"))
    Gn = dict(np.load(RES1 / "games_ta_nohook.npz"))
    base = int(G["seed_base"])
    lad = rules.ladder()
    S = {n: score_matrix(rules, n) for n in set(lad)}
    rows = []  # n, trump, gap, f==b*, realized dealer delta, mismatch flag, sd, Pmax
    mism = 0
    tot = 0
    for g in range(a.games):
        drng, _ = make_rngs(base + g, 4)
        for r, n in enumerate(lad):
            hands = deal(rules, n, drng)
            tr = rules.trump_for_round(r)
            dl = rules.dealer_for_round(r)
            bids = G["bids"][g, r].astype(int)
            prior = []
            for k in range(1, 4):
                p = (dl + k) % 4
                x = hand_features(hands[p], tr, n, k - 1, prior)
                if k == 1 and g < 200:  # spot-check model consistency on first bidder
                    P = model.get(n, tr).dist_all_bids(x)
                    ev = (P * S[n]).sum(1)
                    bm = max(range(n + 1), key=lambda b: (ev[b], -b))
                    tot += 1
                    mism += bm != bids[p]
                prior.append(int(bids[p]))
            x = hand_features(hands[dl], tr, n, 3, prior)
            P = model.get(n, tr).dist_all_bids(x)
            ev = (P * S[n]).sum(1)
            bstar = int(max(range(n + 1), key=lambda b: (ev[b], -b)))
            f = n - sum(prior)
            hit = f == bstar
            gap = float(ev[bstar] - max(ev[b] for b in range(n + 1) if b != f)) if hit else 0.0
            Pb = P[bstar]
            t = np.arange(n + 1)
            mu = (Pb * t).sum()
            sd = float(np.sqrt((Pb * (t - mu) ** 2).sum()))
            real = float(Gn["scores"][g, r, dl] - G["scores"][g, r, dl])
            bd = int(bids[dl])
            hd = float(G["tricks"][g, r, dl] == bd)
            rows.append((n, tr, gap, hit, real, sd, float(P[bstar, bstar]), f, bstar, bd, hd, sum(prior) + bd - n,
                         float(P[bd, bd])))
    R = np.array(rows, float)
    out = {"games": a.games, "first_bidder_model_consistency": [int(tot - mism), int(tot)], "by_n": {}}
    E5 = {1: 5.41, 2: 2.84, 3: 1.85, 4: 1.15, 5: 0.70, 6: 0.51, 7: 0.42, 8: 0.39, 9: 0.35, 10: 0.34,
          11: 0.36, 12: 0.32, 13: 0.27}
    for n in range(1, 14):
        m = R[:, 0] == n
        h = m & (R[:, 3] == 1)
        out["by_n"][n] = {
            "C_pred": ci(R[m, 2]), "measured_crn": ci(R[m, 4]), "E5_table": E5[n],
            "P(f==b*)": ci(R[m, 3]), "E[gap|f==b*]": ci(R[h, 2]) if h.sum() > 1 else None,
            "mean_sd_dealer": float(R[m, 5].mean()), "mean_Pmax": float(R[m, 6].mean()),
            "P(f out of range)": float(((R[m, 7] < 0) | (R[m, 7] > n)).mean()),
        }
    # calibration: realized CRN delta vs predicted gap (pooled)
    bins = [0, 1e-9, 1, 2, 4, 8, 100]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (R[:, 2] >= lo) & (R[:, 2] < hi) if lo > 0 else (R[:, 2] == 0)
        if m.sum() > 1:
            cal.append({"gap_bin": [lo, hi], "rows": int(m.sum()), "mean_pred": float(R[m, 2].mean()),
                        "realized": ci(R[m, 4])})
    out["calibration"] = cal
    # dealer hit rate by direction of hook displacement (pooled n), and table sign
    dirs = {}
    for name, m in (("hooked_forced_up", (R[:, 3] == 1) & (R[:, 9] > R[:, 8])),
                    ("hooked_forced_down", (R[:, 3] == 1) & (R[:, 9] < R[:, 8])),
                    ("not_hooked", R[:, 3] == 0)):
        dirs[name] = {"dealer_hit": ci(R[m, 10]), "mean_sum_minus_n": float(R[m, 11].mean()),
                      "model_P_hit_at_actual_bid": ci(R[m, 12])}
        for sgn in (-1, 1):
            mm = m & (R[:, 11] == sgn)
            dirs[name][f"share_of_rounds_with_sum-n={sgn:+d}"] = int(mm.sum())
    for sgn in (-1, 1):
        m = R[:, 11] == sgn
        dirs[f"sum-n={sgn:+d}"] = {"frac_hooked": ci(R[m, 3]), "dealer_hit": ci(R[m, 10])}
    out["hook_direction"] = dirs
    json.dump(out, open(ROOT / "math" / "results" / "m3_hook.json", "w"), indent=1)
    for n, d in out["by_n"].items():
        print(n, "pred %.2f±%.2f" % tuple(d["C_pred"][:2]), "meas %.2f±%.2f" % tuple(d["measured_crn"][:2]),
              "P(f=b*) %.3f" % d["P(f==b*)"][0], "gap|hit %.2f" % (d["E[gap|f==b*]"] or [0])[0],
              "sd %.2f" % d["mean_sd_dealer"])
    print(out["first_bidder_model_consistency"])


if __name__ == "__main__":
    main()

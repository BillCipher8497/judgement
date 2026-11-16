"""Bidder accuracy: calibration on held-out self-play rounds and agreement / EV loss
vs a bid-conditional Monte-Carlo oracle on sampled bidding situations.

python experiments/phase1/bidder_eval.py --play ta --sits 390 --samples 160
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import info_features  # noqa: E402
from judgement.bidding.sim import (bid_conditional_dists, bid_free_dist, model_bid,  # noqa: E402
                                   simulate_rounds)
from judgement.engine import deal  # noqa: E402
from judgement.policies import GreedyPolicy  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.state import BID, RoundState  # noqa: E402

from common import RES, agents, dump, load_model, mean_se  # noqa: E402


def calibration(rules, model, play, rounds_per_n=1000):
    rows = []
    for n in range(1, 14):
        d = simulate_rounds(rules, play, n, rounds_per_n, 9000 + n, model=model)
        for nt in (False, True):
            m = (d["trump"] == -1) == nt
            if not m.any():
                continue
            P = model.get(n, -1 if nt else 0).dist_batch(d["x"][m], d["bid"][m])
            t = d["tricks"][m]
            b = d["bid"][m]
            p_hit = P[np.arange(len(b)), b]
            ll = -np.log(np.clip(P[np.arange(len(t)), t], 1e-12, None))
            marg = np.bincount(t, minlength=n + 1) / len(t)
            ll0 = -np.log(np.clip(marg[t], 1e-12, None))
            Et = (P * np.arange(n + 1)).sum(1)
            for i in range(len(t)):
                rows.append((n, nt, p_hit[i], b[i] == t[i], ll[i], ll0[i], Et[i], t[i]))
    a = np.array(rows, float)
    out = {"n_rows": len(a), "logloss_model": mean_se(a[:, 4]), "logloss_marginal_by_n": mean_se(a[:, 5]),
           "mean_pred_hit": mean_se(a[:, 2]), "mean_real_hit": mean_se(a[:, 3]),
           "mean_pred_Et": mean_se(a[:, 6]), "mean_real_t": mean_se(a[:, 7])}
    bins = np.linspace(0, 1, 11)
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (a[:, 2] >= lo) & (a[:, 2] < hi + (hi == 1))
        if m.sum():
            cal.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "pred": float(a[m, 2].mean()),
                        "real": float(a[m, 3].mean()), "se": float(a[m, 3].std() / np.sqrt(m.sum()))})
    out["reliability"] = cal
    byn = []
    for n in range(1, 14):
        m = a[:, 0] == n
        byn.append({"n": n, "pred_hit": float(a[m, 2].mean()), "real_hit": float(a[m, 3].mean()),
                    "se": float(a[m, 3].std() / np.sqrt(m.sum())), "pred_Et": float(a[m, 6].mean()),
                    "real_t": float(a[m, 7].mean())})
    out["by_n"] = byn
    return out


def oracle_compare(rules, model, kind, n_sits, S, seed=777):
    ag = [agents(kind, model)] * 4
    rng = np.random.default_rng(seed)
    S_cache = {}
    recs = []
    t0 = time.time()
    for i in range(n_sits):
        n = 1 + i % 13
        tr = rules.trump_cycle[(i // 13) % 5]
        pos = (i // 65) % 4
        hero = (pos + 1) % 4
        st = RoundState(rules, n, tr, 0, deal(rules, n, rng))
        while st.to_act != hero:
            info = st.info_state(st.to_act)
            b, _ = model_bid(model, rules, info_features(info), n, tr, st.legal_bids(), S_cache)
            st.apply_bid(b)
        info = st.info_state(hero)
        legal = st.legal_bids()
        bm, ev_model = model_bid(model, rules, info_features(info), n, tr, legal, S_cache)
        if kind == "greedy":
            d = bid_free_dist(st, hero, ag, S, rng)
            dists = {b: d for b in legal}
            dA = bid_free_dist(st, hero, ag, S, rng)
            distsA, distsB = {b: dA for b in legal}, dists
        else:
            cands = [b for b in legal if abs(b - bm) <= 3]
            half = S // 2
            dA = bid_conditional_dists(st, hero, ag, cands, half, rng)
            dB = bid_conditional_dists(st, hero, ag, cands, S - half, rng)
            dists = {b: (dA[b] * half + dB[b] * (S - half)) / S for b in cands}
            distsA, distsB = dA, dB
        sc = lambda d, b: float(sum(p * rules.score(b, t) for t, p in enumerate(d)))  # noqa: E731
        ev = {b: sc(dists[b], b) for b in dists}
        b_or = max(ev, key=lambda b: (ev[b], -b))
        evA = {b: sc(distsA[b], b) for b in distsA}
        b_A = max(evA, key=lambda b: (evA[b], -b))
        evB = {b: sc(distsB[b], b) for b in distsB}
        Et = float(np.dot(dists[b_or], np.arange(n + 1)))
        var = float(np.dot(dists[b_or], (np.arange(n + 1) - Et) ** 2))
        recs.append({"i": i, "n": n, "trump": tr, "pos": pos, "b_model": bm, "b_oracle": b_or,
                     "ev_oracle": ev[b_or], "ev_model_bid": ev[bm], "b_oracle_A": b_A,
                     "ev_heldout_oracleA": evB[b_A], "ev_heldout_model": evB[bm],
                     "model_ev_pred": float(ev_model[bm]), "Et_at_oracle": Et, "var_at_oracle": var,
                     "edge": b_or in (min(dists), max(dists)) and kind != "greedy" and len(dists) < len(legal),
                     "dists": {int(b): dists[b].tolist() for b in dists}})
        if (i + 1) % 50 == 0:
            print(kind, i + 1, time.time() - t0, flush=True)
            dump(recs, f"oracle_compare_{kind}.json")
    dump(recs, f"oracle_compare_{kind}.json")
    return recs


def summarize(recs):
    match = np.array([r["b_model"] == r["b_oracle"] for r in recs], float)
    loss = np.array([r["ev_oracle"] - r["ev_model_bid"] for r in recs])
    loss_ho = np.array([r["ev_heldout_oracleA"] - r["ev_heldout_model"] for r in recs])
    within1 = np.array([abs(r["b_model"] - r["b_oracle"]) <= 1 for r in recs], float)
    return {"n_situations": len(recs), "match_rate": mean_se(match), "within1": mean_se(within1),
            "ev_loss_insample": mean_se(loss), "ev_loss_heldout_split": mean_se(loss_ho),
            "mean_ev_oracle": mean_se([r["ev_oracle"] for r in recs]),
            "frac_oracle_at_window_edge": float(np.mean([r["edge"] for r in recs]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--play", default="ta")
    ap.add_argument("--sits", type=int, default=390)
    ap.add_argument("--samples", type=int, default=160)
    ap.add_argument("--cal_rounds", type=int, default=1000)
    a = ap.parse_args()
    rules = Rules()
    model = load_model(a.play)
    play = TargetAwarePolicy() if a.play != "greedy" else GreedyPolicy()
    t0 = time.time()
    cal = calibration(rules, model, play, a.cal_rounds)
    dump(cal, f"calibration_{a.play}.json")
    print("calibration done", time.time() - t0, flush=True)
    recs = oracle_compare(rules, model, a.play, a.sits, a.samples)
    s = summarize(recs)
    dump({"calibration": {k: v for k, v in cal.items()}, "oracle": s, "samples": a.samples},
         f"bidder_eval_{a.play}.json")
    print(s)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()

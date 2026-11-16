"""Phase 2 main perfect-information experiments (F1, F2, F3, F4 + F6 decision dataset).

Per deal (4 players, n cards, trump S or NT; dealer rotates with the deal index):
  bid schemes   R   perfect-info rational bids (backward induction, neutral max^n), hook on
                RNH same with hook disabled (F3 only: bids + values)
                G   naive: tricks won under all-greedy perfect-info play (dealer hook-snapped)
                O   realistic: each seat's oracle bid from its own InfoState
                    (MC trick distribution under greedy play, 100 samples)
  for R, G, O:  max^n value vector under neutral/spiteful/benevolent, paranoid value
                per seat, social optimum, hits along the max^n (neutral) path.
  F4 (R, O):    heuristic agreement along the max^n path; one-seat-heuristic vs
                max^n others; all-heuristic table.
  F6:           every non-forced decision on the max^n path and the all-heuristic
                path (R, O): features + per-card own max^n values.

Seeds: deal i of (n, trump) uses SeedSequence([20261110, n, trump_code, i]).
Usage: python exp_main.py N_DEALS_PER_TRUMP n1 n2 ...   (appends to results/main_*.csv)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement import NT, Rules, RoundState  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.solver import MODES, NEUTRAL, MaxN, Paranoid, Social  # noqa: E402
from judgement.solver.bidding import bid_order, greedy_trick_bids, make_play_state, rational_bids  # noqa: E402
from judgement.solver.features import decision_record  # noqa: E402
from judgement.solver.heuristic import OracleGreedyBidder, heuristic_card  # noqa: E402
from judgement.state import PLAY  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
R_HOOK = Rules()
R_NOHOOK = Rules(hook=False)
TRUMPS = {"S": 0, "NT": NT}


def oracle_bids(rules, n, trump, dealer, hands, rng, n_samples=100):
    st = RoundState(rules, n, trump, dealer, list(hands))
    bidder = OracleGreedyBidder(n_samples)
    while st.phase == 0:
        p = st.to_act
        st.apply_bid(bidder.bid(st.info_state(p), st.legal_bids(), rng))
    return tuple(st.bids)


def maxn_path(st, solver, rec_cb=None):
    """Play out with max^n neutral; optionally record decisions."""
    s = st.copy()
    while s.phase == PLAY:
        mv = solver.move_values(s)
        c = solver.best_move(s)
        if rec_cb:
            rec_cb(s, mv, c)
        s.apply_card(c)
    return s


def mixed_path(st, solver, heur_seats, rec_cb=None):
    s = st.copy()
    while s.phase == PLAY:
        p = s.to_act
        if p in heur_seats:
            c = heuristic_card(s.info_state(p), s.legal_cards())
            if rec_cb:
                rec_cb(s, None, c)
        else:
            c = solver.best_move(s)
        s.apply_card(c)
    return s


def run(n_deals, ns):
    files = {k: open(OUT / f"main_{k}.csv", "a", newline="") for k in ("seat", "deal", "dec")}
    writers = {}
    t_start = time.time()
    for n in ns:
        only = __import__('os').environ.get('PH2_TRUMPS')
        for tname, trump in TRUMPS.items():
            if only and tname not in only.split(','):
                continue
            t0 = time.time()
            start = int(__import__('os').environ.get('PH2_START', '0'))
            for i in range(start, start + n_deals):
                ss = np.random.SeedSequence([20261110, n, 0 if trump == 0 else 9, i])
                drng, brng = [np.random.default_rng(c) for c in ss.spawn(2)]
                hands = deal(R_HOOK, n, drng)
                dealer = i % 4
                order = bid_order(4, dealer)
                pos_of = {p: k for k, p in enumerate(order)}   # 0 = first bidder, 3 = dealer
                schemes = {}
                tb = time.perf_counter()
                schemes["R"] = rational_bids(R_HOOK, n, trump, dealer, hands)[0]
                t_rat = time.perf_counter() - tb
                do_nh = n <= 4 or i % 5 < 3
                if do_nh:
                    rnh_bids, rnh_val = rational_bids(R_NOHOOK, n, trump, dealer, hands)
                schemes["G"] = greedy_trick_bids(R_HOOK, n, trump, dealer, hands)
                schemes["O"] = oracle_bids(R_HOOK, n, trump, dealer, hands, brng)
                deal_row = {"n": n, "trump": tname, "deal": i, "dealer": dealer, "t_rational_s": round(t_rat, 4)}
                if do_nh:
                    deal_row["RNH_bids"] = "".join(map(str, rnh_bids))
                    deal_row["RNH_vals"] = json.dumps(rnh_val)
                for sch, bids in schemes.items():
                    st = make_play_state(R_HOOK, n, trump, dealer, hands, bids)
                    vals = {m: MaxN(R_HOOK, trump, bids, 4, m).value(st) for m in MODES}
                    neu = MaxN(R_HOOK, trump, bids, 4, NEUTRAL)
                    para = [Paranoid(R_HOOK, trump, bids, p).value(st) for p in range(4)]
                    soc = Social(R_HOOK, trump, bids).value(st)
                    deal_row[f"{sch}_bids"] = "".join(map(str, bids))
                    deal_row[f"{sch}_social"] = soc
                    for m in MODES:
                        deal_row[f"{sch}_sum_{m}"] = sum(vals[m])
                        deal_row[f"{sch}_hits_{m}"] = sum(1 for p in range(4) if vals[m][p] > 0)
                    if sch in ("R", "O"):
                        # F4: agreement along the max^n path + F6 records
                        def cb(s, mv, c, tag="maxn"):
                            rec = decision_record(s, mv, heuristic_card(s.info_state(s.to_act), s.legal_cards()))
                            if rec is None:
                                return
                            rec.update(n=n, trump=tname, deal=i, scheme=sch, path=tag, chosen=c)
                            if "dec" not in writers:
                                writers["dec"] = csv.DictWriter(files["dec"], fieldnames=list(rec))
                                if files["dec"].tell() == 0:
                                    writers["dec"].writeheader()
                            writers["dec"].writerow(rec)
                        maxn_path(st, neu, cb)
                        # all-heuristic path (states off the equilibrium path)
                        def cb2(s, _mv, c):
                            cb(s, neu.move_values(s), c, tag="heur")
                        allh = mixed_path(st, neu, {0, 1, 2, 3}, cb2)
                        deal_row[f"{sch}_allheur_sum"] = sum(allh.scores())
                        deal_row[f"{sch}_allheur_hits"] = sum(1 for x in allh.scores() if x > 0)
                        one_h = [mixed_path(st, neu, {p}).scores()[p] for p in range(4)]
                    for p in range(4):
                        row = {"n": n, "trump": tname, "deal": i, "scheme": sch, "seat": p, "pos": pos_of[p],
                               "bid": bids[p], "sum_bids_minus_n": sum(bids) - n,
                               "paranoid": para[p], "social_total": soc,
                               **{f"maxn_{m}": vals[m][p] for m in MODES}}
                        if sch in ("R", "O"):
                            row["oneheur"] = one_h[p]
                            row["allheur"] = allh.scores()[p]
                        else:
                            row["oneheur"] = row["allheur"] = ""
                        if "seat" not in writers:
                            writers["seat"] = csv.DictWriter(files["seat"], fieldnames=list(row))
                            if files["seat"].tell() == 0:
                                writers["seat"].writeheader()
                        writers["seat"].writerow(row)
                if "deal" not in writers:
                    keys = list(deal_row)
                    if "RNH_bids" not in keys:
                        keys[5:5] = ["RNH_bids", "RNH_vals"]
                    writers["deal"] = csv.DictWriter(files["deal"], fieldnames=keys, restval="")
                    if files["deal"].tell() == 0:
                        writers["deal"].writeheader()
                writers["deal"].writerow(deal_row)
            for f in files.values():
                f.flush()
            print(f"n={n} {tname}: {n_deals} deals in {time.time() - t0:.1f}s (total {time.time() - t_start:.0f}s)",
                  flush=True)
    for f in files.values():
        f.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), [int(x) for x in sys.argv[2:]])

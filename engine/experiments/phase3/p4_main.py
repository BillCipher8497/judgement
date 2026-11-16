"""P4: small tournament.  One challenger vs three TA agents (ModelBidder(model_ta_final)
+ TA play), seat-rotated: for every game g the challenger sits at each seat h in 0..3,
and the all-TA baseline game is played once; all deals from default_rng([seed, g, r])
(CRN).  Challengers:
  ta_spite : ModelBidder(ta model) + TASpitePolicy
  greedy   : ModelBidder(greedy model) + GreedyPolicy
  oracle_i : InformedOracleBidder (K worlds from the exact bid posterior, model bid ±2) + TA
Δ = challenger seat's (total, fractional win) minus the same seat's in the baseline game;
unit for CIs = game (mean over the 4 seats).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common3 import ROOT, dump, fmt, load_model
from judgement import Rules
from judgement.bidding.model import Agent, BidModel, ModelBidder
from judgement.bidding2.oracle import InformedOracleBidder
from judgement.engine import deal, fractional_wins, run_round
from judgement.policies import GreedyPolicy
from judgement.policies.target_aware import TargetAwarePolicy, TASpitePolicy
from judgement.state import RoundState


def play(rules, agents, seed, g):
    tot = [0.0] * 4
    rng0 = np.random.default_rng(0)
    for r, n in enumerate(rules.ladder()):
        rng = np.random.default_rng([seed, g, r])
        st = RoundState(rules, n, rules.trump_for_round(r), rules.dealer_for_round(r), deal(rules, n, rng))
        run_round(st, agents, [rng0] * 4, check=False)
        sc = st.scores()
        for p in range(4):
            tot[p] += sc[p]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--span", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rules = Rules()
    model = load_model()
    ta = TargetAwarePolicy()
    base = Agent(ModelBidder(model), ta)
    if a.arm == "ta_spite":
        mk = lambda g, h: Agent(ModelBidder(model), TASpitePolicy())  # noqa: E731
    elif a.arm == "greedy":
        mg = BidModel.load(ROOT / "experiments/phase1/results/model_greedy_final.json")
        mk = lambda g, h: Agent(ModelBidder(mg), GreedyPolicy())  # noqa: E731
    elif a.arm in ("oracle_i", "oracle_u"):
        mk = lambda g, h: Agent(InformedOracleBidder(  # noqa: E731
            model, rules, [base] * 4, K=a.K, span=a.span, informed=a.arm == "oracle_i",
            seed=int(np.random.SeedSequence([a.seed, g, h, 5]).generate_state(1)[0])), ta)
    else:
        raise ValueError(a.arm)
    ds, dw, bw = [], [], []
    t0 = time.time()
    for g in range(a.games):
        bt = play(rules, [base] * 4, a.seed, g)
        bwin = fractional_wins(bt)
        s_, w_ = [], []
        for h in range(4):
            ags = [base] * 4
            ags[h] = mk(g, h)
            tt = play(rules, ags, a.seed, g)
            s_.append(tt[h] - bt[h])
            w_.append(fractional_wins(tt)[h] - bwin[h])
        ds.append(float(np.mean(s_)))
        dw.append(float(np.mean(w_)))
        bw.append(float(np.mean(bwin)))
        if (g + 1) % (10 if a.arm.startswith("oracle") else 100) == 0 or g + 1 == a.games:
            print(f"{a.arm} g={g+1} {time.time()-t0:.0f}s dscore {fmt(ds)} dP(win) {fmt(dw, 4)}", flush=True)
            dump({"args": vars(a), "dscore": ds, "dwin": dw}, f"p4_{a.arm}{a.tag}.json")


if __name__ == "__main__":
    main()

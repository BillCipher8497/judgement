"""P2: score-maximising vs win-maximising bidder (one win-maximiser vs three
score-maximisers, ModelBidder + TA everywhere else).

Game g: round r deals from default_rng([SEED, g, r]); trump/dealer/n from the rules.
Baseline = all ModelBidder.  Because TA play and ModelBidder are deterministic, the
treatment game with hero seat h is identical to the baseline until round 25-k, so we
branch from the baseline standings at round 25-k for each h in 0..3 (seat rotation
within every game, CRN on all deals).  Unit of analysis for CIs = game (mean over the
four hero seats)."""
from __future__ import annotations

import argparse
import time

import numpy as np

from common2 import GAMES_TA, ci, dump, fmt, load_model, RES  # noqa: E402
from judgement import Rules
from judgement.bidding.model import Agent, ModelBidder
from judgement.bidding2.winmax import ScoreBank, WinMaxBidder
from judgement.engine import deal, fractional_wins, run_round
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.state import BID, RoundState


def play_rounds(rules, agents, seed, g, r0, totals, winmax=None):
    totals = list(totals)
    rng0 = np.random.default_rng(0)
    for r, n in list(enumerate(rules.ladder()))[r0:]:
        rng = np.random.default_rng([seed, g, r])
        st = RoundState(rules, n, rules.trump_for_round(r), rules.dealer_for_round(r), deal(rules, n, rng))
        if winmax is not None:
            winmax.ctx = (tuple(totals), r)
        run_round(st, agents, [rng0] * 4, check=False)
        sc = st.scores()
        for p in range(4):
            totals[p] += sc[p]
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--ks", default="3,6")
    ap.add_argument("--L", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rules = Rules()
    model = load_model()
    bank = ScoreBank(GAMES_TA)
    ta = TargetAwarePolicy()
    base_ag = Agent(ModelBidder(model), ta)
    ks = [int(x) for x in a.ks.split(",")]
    R = len(rules.ladder())
    res = {k: {"dwin": [], "dscore": [], "dwin_seat": [], "base_win": [], "treat_win": [],
               "dscore_seat": [], "changed": 0, "decisions": 0} for k in ks}
    t0 = time.time()
    for g in range(a.games):
        base_tot_at = {}
        totals = [0.0] * 4
        for k in ks:
            pass
        # baseline, recording standings before each candidate branch round
        branch = {R - k for k in ks}
        rng0 = np.random.default_rng(0)
        for r, n in enumerate(rules.ladder()):
            if r in branch:
                base_tot_at[r] = list(totals)
            rng = np.random.default_rng([a.seed, g, r])
            st = RoundState(rules, n, rules.trump_for_round(r), rules.dealer_for_round(r), deal(rules, n, rng))
            run_round(st, [base_ag] * 4, [rng0] * 4, check=False)
            sc = st.scores()
            for p in range(4):
                totals[p] += sc[p]
        base_w = fractional_wins(totals)
        for k in ks:
            dws, dss = [], []
            for h in range(4):
                wm = WinMaxBidder(model, rules, bank, k, L=a.L, seed=hash((a.seed, g, h, k)) % (2**32))
                ags = [base_ag] * 4
                ags[h] = Agent(wm, ta)
                tt = play_rounds(rules, ags, a.seed, g, R - k, base_tot_at[R - k], winmax=wm)
                tw = fractional_wins(tt)
                dws.append(tw[h] - base_w[h])
                dss.append(tt[h] - totals[h])
                res[k]["dwin_seat"].append((h, tw[h] - base_w[h]))
                res[k]["base_win"].append(base_w[h])
                res[k]["treat_win"].append(tw[h])
                res[k]["changed"] += sum(1 for x in wm.log if x[1] != x[2])
                res[k]["decisions"] += len(wm.log)
            res[k]["dwin"].append(float(np.mean(dws)))
            res[k]["dscore"].append(float(np.mean(dss)))
        if (g + 1) % 250 == 0:
            msg = "  ".join(f"k={k}: dP(win) {fmt(res[k]['dwin'], 4)}  dscore {fmt(res[k]['dscore'])}" for k in ks)
            print(f"g={g+1} {time.time()-t0:.0f}s  {msg}", flush=True)
            dump({"args": vars(a), "res": {str(k): v for k, v in res.items()}}, f"p2_raw{a.tag}.json")
    dump({"args": vars(a), "res": {str(k): v for k, v in res.items()}}, f"p2_raw{a.tag}.json")
    print("total", time.time() - t0)


if __name__ == "__main__":
    main()

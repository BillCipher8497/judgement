"""P2: one standings-aware win-maximiser vs three score-maximisers.

All seats: ModelBidder(model_ta_final) + TA play.  Treatment seat h uses WinMaxBidder in
the last k rounds (k in {3,6}): bid = argmax_b P(win game | standings, b), P(win) by MC
(L draws) over the rest of the game with a cheap score model (judgement/bidding2/winmax.py:
hero this round hits w.p. P_model(t=b|hand,b); opponents who already bid hit at the
empirical TA self-play rate by (round, position, bid); everything else = whole score rows
of a random TA self-play game (games_ta.npz, 10,000 games), aligned by round index so n,
trump and seat/position are matched).

Round r of game g deals from default_rng([seed, g, r]).  Play and ModelBidder are
deterministic (verified in tests/test_phase3_bidding.py), so the treatment game equals the
baseline until round 25-k: we branch from the baseline standings at 25-k for each h in 0..3
(seat rotation within each game; all deals shared = CRN).  Unit for overall CIs = game
(mean over the 4 hero seats); by-standing rows use (game, seat) units.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common3 import GAMES_TA, ci, dump, fmt, load_model, status
from judgement import Rules
from judgement.bidding.model import Agent, ModelBidder
from judgement.bidding2.winmax import ScoreBank, WinMaxBidder
from judgement.engine import deal, fractional_wins, run_round
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.state import RoundState


def play_rounds(rules, agents, seed, g, r0, totals, winmax=None, until=None):
    totals = list(totals)
    rng0 = np.random.default_rng(0)
    snaps = {}
    for r, n in list(enumerate(rules.ladder()))[r0:]:
        snaps[r] = list(totals)
        rng = np.random.default_rng([seed, g, r])
        st = RoundState(rules, n, rules.trump_for_round(r), rules.dealer_for_round(r), deal(rules, n, rng))
        if winmax is not None:
            winmax.ctx = (tuple(totals), r)
        run_round(st, agents, [rng0] * 4, check=False)
        sc = st.scores()
        for p in range(4):
            totals[p] += sc[p]
    return totals, snaps


def rank_of(tot, h):
    return 1 + sum(1 for p in range(4) if tot[p] > tot[h])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--ks", default="3,6")
    ap.add_argument("--L", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rules = Rules()
    model = load_model()
    bank = ScoreBank(GAMES_TA)
    ta = TargetAwarePolicy()
    base_ag = Agent(ModelBidder(model), ta)
    ks = [int(x) for x in a.ks.split(",")]
    R = len(rules.ladder())
    units = {k: [] for k in ks}       # (g, h, rank, gap, base_win, treat_win, base_score, treat_score)
    dec = {k: [] for k in ks}         # decisions: (rank_now, gap_now, n, bm, b, phit_bm, phit_b, pw_bm, pw_b, rounds_left, unit index)
    t0 = time.time()
    for g in range(a.games):
        totals, snaps = play_rounds(rules, [base_ag] * 4, a.seed, g, 0, [0.0] * 4)
        base_w = fractional_wins(totals)
        for k in ks:
            S = snaps[R - k]
            for h in range(4):
                ss = np.random.SeedSequence([a.seed, g, h, k, 99])
                wm = WinMaxBidder(model, rules, bank, k, L=a.L, seed=int(ss.generate_state(1)[0]))
                ags = [base_ag] * 4
                ags[h] = Agent(wm, ta)
                tt, _ = play_rounds(rules, ags, a.seed, g, R - k, S, winmax=wm)
                tw = fractional_wins(tt)
                units[k].append((g, h, rank_of(S, h), max(S) - S[h], base_w[h], tw[h], totals[h], tt[h]))
                for d in wm.detail:
                    T = d["totals"]
                    dec[k].append((rank_of(T, h), max(T) - T[h], d["n"], d["bm"], d["b"], d["phit"][d["bm"]],
                                   d["phit"][d["b"]], d["pw"][d["bm"]], d["pw"][d["b"]], R - d["r"], len(units[k]) - 1))
        if (g + 1) % 250 == 0 or g + 1 == a.games:
            msg = []
            for k in ks:
                U = np.array(units[k]).reshape(-1, 4, 8)
                msg.append(f"k={k}: dP(win) {fmt((U[:, :, 5] - U[:, :, 4]).mean(1), 4)} "
                           f"dscore {fmt((U[:, :, 7] - U[:, :, 6]).mean(1))}")
            print(f"g={g+1} {time.time()-t0:.0f}s  " + "  ".join(msg), flush=True)
            dump({"args": vars(a), "units": {str(k): v for k, v in units.items()},
                  "dec": {str(k): v for k, v in dec.items()}}, f"p2_raw{a.tag}.json")
    print("total", time.time() - t0)


if __name__ == "__main__":
    main()

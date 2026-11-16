"""M9 empirical check of the two exchange propositions (NOTES M9).

Variants (hero seat only, rest of table = base policy, paired CRN on the same deal/seed):
  W  ("keep-low at need 1"): following, need == 1, >= 2 sure winners -> win with the HIGHEST sure winner
     (Prop. W: weakly dominant vs Markov opponents).  Base TA (TA+ already does this via win_high).
  L  ("highest sure loser on lead at target"): leading, need == 0, >= 2 sure losers (pw == 0)
     -> lead the highest-value sure loser instead of TA's lowest (Prop. L': dominant within a suit under
     forced-response coupling; heuristic across suits).  L_same: only within the suit of TA's choice.
     Tested on base TA and base TA+.
Rounds: round index uniform over the 25-round ladder (n, trump, dealer from the schedule), seat of the hero
uniform.  Output: results/m9_play.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, BidModel, ModelBidder  # noqa: E402
from judgement.cards import SUIT_OF  # noqa: E402
from judgement.engine import deal, make_rngs, run_round  # noqa: E402
from judgement.policies.ta_plus import TAPlusPolicy, ta_plus_v2  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402


def with_variants(base_cls, W=False, L=0):
    class V(base_cls):
        changed = 0

        def _win(self, info, tm, cards, pw, need, r, trump, val):
            if W and tm.wc >= 0 and need == 1 and r > 1:
                sure = [c for c in cards if pw[c] >= 1.0]
                if len(sure) >= 2:
                    c = max(sure, key=val)
                    d = super()._win(info, tm, cards, pw, need, r, trump, val)
                    V.changed += c != d
                    return c
            return super()._win(info, tm, cards, pw, need, r, trump, val)

        def _lose(self, tm, cards, pw, val):
            d = super()._lose(tm, cards, pw, val)
            if L and tm.wc < 0:
                losers = [c for c in cards if pw[c] <= 0.0]
                if L == 2:
                    losers = [c for c in losers if SUIT_OF[c] == SUIT_OF[d]]
                if len(losers) >= 2:
                    c = max(losers, key=val)
                    V.changed += c != d
                    return c
            return d
    return V


def main(rounds=30000, seed=9090, only=None):
    rules = Rules()
    M = BidModel.load(ROOT / "experiments" / "phase1" / "results" / "model_ta_final.json")
    lad = rules.ladder()
    tests = {
        "W_on_TA": (lambda: TargetAwarePolicy(), lambda: with_variants(TargetAwarePolicy, W=True)()),
        "L_on_TA": (lambda: TargetAwarePolicy(), lambda: with_variants(TargetAwarePolicy, L=1)()),
        "Lsame_on_TA": (lambda: TargetAwarePolicy(), lambda: with_variants(TargetAwarePolicy, L=2)()),
        "L_on_TA+": (lambda: ta_plus_v2(), lambda: with_variants(TAPlusPolicy, L=1)(win_high=2, win_margin=-0.5)),
        "Lsame_on_TA+": (lambda: ta_plus_v2(), lambda: with_variants(TAPlusPolicy, L=2)(win_high=2, win_margin=-0.5)),
    }
    out = {}
    for name, (mk_base, mk_var) in tests.items():
        if only and name not in only:
            continue
        base = Agent(ModelBidder(M), mk_base())
        varp = mk_var()
        var = Agent(ModelBidder(M), varp)
        type(varp).changed = 0
        d, dn, dh = [], {}, []
        for g in range(rounds):
            r = g % len(lad)
            n, tr, dl = lad[r], rules.trump_for_round(r), rules.dealer_for_round(r)
            hero = (g // len(lad)) % 4
            res = []
            for ag in (base, var):
                drng, prngs = make_rngs(np.random.SeedSequence([seed, g]), 4)  # fresh -> identical streams
                hands = deal(rules, n, drng)
                from judgement.state import RoundState
                st = RoundState(rules, n, tr, dl, list(hands))
                agents = [base] * 4
                agents[hero] = ag
                run_round(st, agents, prngs, check=False)
                b, t = st.bids[hero], st.tricks_won[hero]
                res.append((rules.score(b, t), b == t))
            d.append(res[1][0] - res[0][0])
            dh.append(float(res[1][1]) - float(res[0][1]))
            dn.setdefault(n, []).append(res[1][0] - res[0][0])
        d = np.array(d, float)
        o = {"rounds": rounds, "d_score_per_round": [float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))],
             "d_hit": [float(np.mean(dh)), float(1.96 * np.std(dh, ddof=1) / np.sqrt(len(dh)))],
             "frac_rounds_changed_outcome": float((d != 0).mean()), "decisions_changed": int(type(varp).changed),
             "per_game_equiv(x25)": float(25 * d.mean())}
        o["by_n"] = {int(k): [float(np.mean(v)), float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v)))] for k, v in sorted(dn.items())}
        out[name] = o
        print(name, o["d_score_per_round"], o["d_hit"], o["decisions_changed"], flush=True)
    json.dump(out, open(ROOT / "math" / "results" / ("m9_play.json" if seed == 9090 else f"m9_play_seed{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000, int(sys.argv[2]) if len(sys.argv) > 2 else 9090,
         sys.argv[3].split(",") if len(sys.argv) > 3 else None)

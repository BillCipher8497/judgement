"""M9: count rounds where each variant does better / worse than its base (paired CRN, same seeds as m9_play.py). Output: results/m9_signs.json"""
import sys, json
sys.path.insert(0, '/home/claude/judgement-engine'); sys.path.insert(0, '/home/claude/judgement-engine/math')
import numpy as np
import m9_play as m
from judgement import Rules
from judgement.bidding.model import Agent, BidModel, ModelBidder
from judgement.engine import deal, make_rngs, run_round
from judgement.state import RoundState
from judgement.policies.target_aware import TargetAwarePolicy
rules = Rules(); M = BidModel.load(m.ROOT / "experiments/phase1/results/model_ta_final.json"); lad = rules.ladder()
out = {}
for name, kw in (("W_on_TA", dict(W=True)), ("Lsame_on_TA", dict(L=2)), ("L_on_TA", dict(L=1))):
    base = Agent(ModelBidder(M), TargetAwarePolicy()); var = Agent(ModelBidder(M), m.with_variants(TargetAwarePolicy, **kw)())
    pos = neg = 0
    for g in range(30000):
        r = g % len(lad); n, tr, dl = lad[r], rules.trump_for_round(r), rules.dealer_for_round(r); hero = (g // len(lad)) % 4
        sc = []
        for ag in (base, var):
            drng, prngs = make_rngs(np.random.SeedSequence([9090, g]), 4)
            st = RoundState(rules, n, tr, dl, list(deal(rules, n, drng))); A = [base] * 4; A[hero] = ag
            run_round(st, A, prngs, check=False); sc.append(rules.score(st.bids[hero], st.tricks_won[hero]))
        pos += sc[1] > sc[0]; neg += sc[1] < sc[0]
    out[name] = {"rounds_better": int(pos), "rounds_worse": int(neg)}; print(name, pos, neg, flush=True)
json.dump(out, open('/home/claude/judgement-engine/math/results/m9_signs.json', 'w'), indent=1)

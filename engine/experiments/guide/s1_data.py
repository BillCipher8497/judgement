"""S1 dataset: bidding situations with counterfactual scores for every candidate bid.

Situation i: rng = default_rng([SEED, i]); round index r = i % 25 (ladder n, trump, dealer as in a game);
hero = bidding position pos = rng.integers(4); deal from rng; earlier seats bid with ModelBidder(M_TA).
For each candidate hero bid b (all hook-legal b if n <= 6, else MB bid +-4), later seats bid with MB and
all four seats play TA+3 (deterministic) on the TRUE deal -> cf[b] = hero score.  Mean over situations of
cf[rule(situation)] is an unbiased estimate of the rule's EV/round (deal drawn from the real law, earlier
bids by MB on that deal), paired across rules (CRN).
U-oracle (first NORC situations): K uniform worlds for the hidden hands, same candidate playouts;
oracle bid = argmax mean (ties -> lower).
python s1_data.py SEED N NORC K OUT [HERO_PLAY]   HERO_PLAY = tap3 (default) | guide (HumanPlayPolicy for the hero seat)
"""
import pickle
import sys

import numpy as np

from common import M_TA, RES, Agent, ModelBidder, ta_plus3
from judgement import Rules
from judgement.engine import run_round
from judgement.state import RoundState

rules = Rules()
LAD = rules.ladder()
MB = ModelBidder(M_TA)
AG = Agent(MB, ta_plus3())
POLS = [AG] * 4
RNGS = [np.random.default_rng(0)] * 4   # policies are deterministic
HERO = [AG]


def playout(n, trump, dealer, hands, prior, b, hero):
    st = RoundState(rules, n, trump, dealer, list(hands))
    for x in prior:
        st.apply_bid(x)
    st.apply_bid(b)
    pols = list(POLS)
    pols[hero] = HERO[0]
    run_round(st, pols, RNGS, check=False)
    return st.scores()[hero]


def situation(seed, i, norc, K):
    rng = np.random.default_rng([seed, i])
    r = i % len(LAD)
    n, trump, dealer = LAD[r], rules.trump_for_round(r), rules.dealer_for_round(r)
    pos = int(rng.integers(4))
    hero = (dealer + 1 + pos) % 4
    perm = rng.permutation(52)
    hands = [0] * 4
    for p in range(4):
        for c in perm[p * n:(p + 1) * n]:
            hands[p] |= 1 << int(c)
    st = RoundState(rules, n, trump, dealer, list(hands))
    prior = []
    while st.to_act != hero:
        b = MB.bid(st.info_state(st.to_act), st.legal_bids(), None)
        st.apply_bid(b); prior.append(b)
    legal = st.legal_bids()
    info = st.info_state(hero)
    ev, _ = MB.evs(info)
    bm = max(legal, key=lambda b: (ev[b], -b))
    cand = legal if n <= 6 else [b for b in legal if abs(b - bm) <= 4]
    cf = {b: playout(n, trump, dealer, hands, prior, b, hero) for b in cand}
    row = dict(i=i, n=n, trump=trump, dealer=dealer, pos=pos, hero=hero, hand=hands[hero], prior=prior,
               legal=legal, bm=bm, ev_m=[float(x) for x in ev], cf=cf)
    if i < norc:
        rest = [c for c in range(52) if not hands[hero] >> c & 1]
        acc = {b: 0.0 for b in cand}
        for k in range(K):
            w = rng.permutation(rest)
            hs = list(hands)
            j = 0
            for p in range(4):
                if p == hero:
                    continue
                m = 0
                for c in w[j:j + n]:
                    m |= 1 << int(c)
                hs[p] = m; j += n
            # earlier bids are taken as given (uniform oracle ignores what they imply)
            for b in cand:
                acc[b] += playout(n, trump, dealer, hs, prior, b, hero)
        row["orc"] = {b: acc[b] / K for b in cand}
        row["bo"] = max(cand, key=lambda b: (acc[b], -b))
    return row


if __name__ == "__main__":
    seed, N, norc, K, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    if len(sys.argv) > 6 and sys.argv[6] == "guide":
        from judgement.policies.human_guide import HumanPlayPolicy
        HERO[0] = Agent(MB, HumanPlayPolicy())
    rows = [situation(seed, i, norc, K) for i in range(N)]
    with open(RES / out, "wb") as fh:
        pickle.dump(rows, fh)
    print("done", out, len(rows))

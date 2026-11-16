"""1-vs-3 / all-same tables, seed-compatible with experiments/final/run.py (BASE 71,000,000):
game g = play_game seed BASE+g, challenger in seat g%4.  All guide/TA policies are deterministic,
so the table with challenger == pool is the CRN baseline for every challenger (paired by game g).

kinds: ta | tap3 (MB + TA+3) | greedy | guide[:k=v,..] | gbid[:..] (guide bid + TA+3 play)
       | gplay[:..] (MB bid + guide play) | gbid_ta (guide bid + TA play)
       k=v sets HumanBidder params (b.<name>) or HumanPlayPolicy params (p.<name>), e.g. guide:p.r_shed=0
python run.py POOL CHAL START G TAG [BASE] [DEADLINE_EPOCH] -> results/g_<TAG>.npz
"""
import sys
import time

import numpy as np

from common import M_GR, M_TA, RES, Agent, GreedyPolicy, ModelBidder, TargetAwarePolicy, ta_plus3
from judgement import Rules, play_game
from judgement.policies.human_guide import GuideAgent, HumanBidder, HumanPlayPolicy


def _kw(kind):
    b, p = {}, {}
    if ":" in kind:
        for x in kind.split(":", 1)[1].split(","):
            k, v = x.split("=")
            (b if k.startswith("b.") else p)[k[2:]] = int(v)
    return b, p


def make_agent(kind):
    base = kind.split(":", 1)[0]
    bk, pk = _kw(kind)
    if base == "ta":
        return Agent(ModelBidder(M_TA), TargetAwarePolicy(), name=kind)
    if base == "tap3":
        return Agent(ModelBidder(M_TA), ta_plus3(), name=kind)
    if base == "greedy":
        return Agent(ModelBidder(M_GR), GreedyPolicy(), name=kind)
    if base == "guide":
        return GuideAgent(HumanBidder(**bk), HumanPlayPolicy(**pk), name=kind)
    if base == "gbid":
        return Agent(HumanBidder(**bk), ta_plus3(), name=kind)
    if base == "gbid_ta":
        return Agent(HumanBidder(**bk), TargetAwarePolicy(), name=kind)
    if base == "gplay":
        return Agent(ModelBidder(M_TA), HumanPlayPolicy(**pk), name=kind)
    raise ValueError(kind)


def run(pool, chal, start, G, tag, base=71_000_000, deadline=float("inf")):
    rules = Rules()
    R = rules.n_rounds
    bids = np.zeros((G, R, 4), np.int8); tricks = np.zeros((G, R, 4), np.int8)
    scores = np.zeros((G, R, 4), np.float32); totals = np.zeros((G, 4))
    games = np.arange(start, start + G)
    P = make_agent(pool)
    C = make_agent(chal)
    done = 0
    t0 = time.time()
    for i, g in enumerate(games):
        if time.time() > deadline:
            break
        g = int(g)
        pols = [P] * 4
        pols[g % 4] = C
        res = play_game(rules, pols, seed=base + g, record_trace=False)
        for r, rr in enumerate(res.rounds):
            bids[i, r] = rr.bids; tricks[i, r] = rr.tricks; scores[i, r] = rr.scores
        totals[i] = res.totals
        done = i + 1
    np.savez_compressed(RES / f"g_{tag}.npz", bids=bids[:done], tricks=tricks[:done], scores=scores[:done],
                        totals=totals[:done], seat=games[:done] % 4, game=games[:done], seed_base=base,
                        pool=pool, chal=chal)
    print(tag, pool, chal, done, round(time.time() - t0, 1), flush=True)
    return done


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], int(a[2]), int(a[3]), a[4], int(a[5]) if len(a) > 5 else 71_000_000,
        float(a[6]) if len(a) > 6 else float("inf"))

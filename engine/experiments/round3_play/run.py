"""(round3_play copy of experiments/final/run.py; adds challenger sampler-stat logging)
1-vs-3 runner, seed-compatible with experiments/play/one_vs_three.py (pool 'ta')
and experiments/phase4/q2_ladder.py (pool 'hybrid:k=16,c=5', BASE 400,000,000):
game g uses play_game seed BASE+g, challenger in seat g%4 with agent seed
SeedSequence([BASE,g,5]), pool seat p agent seed SeedSequence([BASE,g,7,p]).
Games g in [START, START+G).  Saves every 60 s; stops at DEADLINE epoch.

python run.py POOL CHAL START G TAG BASE [DEADLINE] -> results/f_<TAG>.npz
"""
import sys
import time
from pathlib import Path

import numpy as np

from agents import make_agent  # round3_play/agents.py
from judgement import Rules, play_game

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)


def run(pool, chal, start, G, tag, base, deadline=float("inf")):
    rules = Rules()
    R = rules.n_rounds
    out = RES / f"f_{tag}.npz"
    bids = np.zeros((G, R, 4), np.int8); tricks = np.zeros((G, R, 4), np.int8)
    scores = np.zeros((G, R, 4), np.float32); totals = np.zeros((G, 4)); wins = np.zeros((G, 4))
    secs = np.zeros(G); games = np.arange(start, start + G)
    done = 0
    stats = []

    def save():
        np.savez_compressed(out, bids=bids[:done], tricks=tricks[:done], scores=scores[:done], totals=totals[:done],
                            wins=wins[:done], seat=games[:done] % 4, game=games[:done], secs=secs[:done],
                            seed_base=base, pool=pool, chal=chal)
        if stats:
            import json
            (RES / f"f_{tag}_stats.json").write_text(json.dumps(stats))
    t0 = last = time.time()
    for i, g in enumerate(games):
        if time.time() > deadline:
            break
        g = int(g)
        pols = [make_agent(pool, seed=int(np.random.SeedSequence([base, g, 7, p]).generate_state(1)[0])) for p in range(4)]
        pols[g % 4] = make_agent(chal, seed=int(np.random.SeedSequence([base, g, 5]).generate_state(1)[0]))
        t1 = time.time()
        res = play_game(rules, pols, seed=base + g, record_trace=False)
        secs[i] = time.time() - t1
        for r, rr in enumerate(res.rounds):
            bids[i, r] = rr.bids; tricks[i, r] = rr.tricks; scores[i, r] = rr.scores
        totals[i] = res.totals; wins[i] = res.wins
        lg = getattr(getattr(pols[g % 4], "player", None), "log", None)
        if lg is not None:
            for st in lg:
                st["g"] = g
            stats.extend(lg); lg.clear()
        done = i + 1
        if time.time() - last > 60:
            save(); last = time.time()
    save()
    print(tag, done, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], int(a[2]), int(a[3]), a[4], int(a[5]), float(a[6]) if len(a) > 6 else float("inf"))

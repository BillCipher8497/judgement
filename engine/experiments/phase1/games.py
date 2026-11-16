"""Full 25-round games under several configurations, common random numbers
(game g uses seed BASE+g in every configuration, so deals are identical).

python experiments/phase1/games.py --configs ta,spite1,spite_all,greedy,ta_nohook --games 4000
Saves results/games_<config>.npz with bids/tricks/scores (G,25,4), totals, wins, spite_seat.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules, play_game  # noqa: E402

from common import RES, agents, load_model  # noqa: E402

BASE = 1_000_000


def config(name):
    """-> (rules, policies_fn(g) -> list, model tag)"""
    if name.startswith("ta_") and name[3:] in ("neg_bid", "neg_10", "neg_diff"):
        mode = name[3:]
        m = load_model(f"ta_{mode}")
        return Rules(miss_penalty=mode), lambda g: [agents("ta", m)] * 4
    m_ta = load_model("ta")
    if name == "ta":
        return Rules(), lambda g: [agents("ta", m_ta)] * 4
    if name == "ta_nohook":
        return Rules(hook=False), lambda g: [agents("ta", m_ta)] * 4
    if name == "spite_all":
        return Rules(), lambda g: [agents("ta_spite", m_ta)] * 4
    if name == "spite1":
        def pols(g):
            p = [agents("ta", m_ta)] * 4
            p[g % 4] = agents("ta_spite", m_ta)
            return p
        return Rules(), pols
    if name == "greedy":
        m_gr = load_model("greedy")
        return Rules(), lambda g: [agents("greedy", m_gr)] * 4
    if name == "greedy_play_ta_bids":
        return Rules(), lambda g: [agents("greedy", m_ta)] * 4
    raise ValueError(name)


def run(name, G):
    rules, pols = config(name)
    R = rules.n_rounds
    bids = np.zeros((G, R, 4), np.int8)
    tricks = np.zeros((G, R, 4), np.int8)
    scores = np.zeros((G, R, 4), np.float32)
    totals = np.zeros((G, 4))
    wins = np.zeros((G, 4))
    t0 = time.time()
    for g in range(G):
        res = play_game(rules, pols(g), seed=BASE + g, record_trace=False)
        for r, rr in enumerate(res.rounds):
            bids[g, r] = rr.bids
            tricks[g, r] = rr.tricks
            scores[g, r] = rr.scores
        totals[g] = res.totals
        wins[g] = res.wins
        if (g + 1) % 500 == 0:
            print(name, g + 1, round(time.time() - t0, 1), flush=True)
            np.savez_compressed(RES / f"games_{name}.npz", bids=bids[:g + 1], tricks=tricks[:g + 1],
                                scores=scores[:g + 1], totals=totals[:g + 1], wins=wins[:g + 1],
                                spite_seat=np.arange(g + 1) % 4, seed_base=BASE)
    np.savez_compressed(RES / f"games_{name}.npz", bids=bids, tricks=tricks, scores=scores, totals=totals,
                        wins=wins, spite_seat=np.arange(G) % 4, seed_base=BASE)
    print(name, "done", round(time.time() - t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="ta,spite1,spite_all,greedy,ta_nohook")
    ap.add_argument("--games", type=int, default=4000)
    a = ap.parse_args()
    for c in a.configs.split(","):
        run(c, a.games)


if __name__ == "__main__":
    main()

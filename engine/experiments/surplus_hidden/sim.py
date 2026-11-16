"""H1: hidden-information cooperation surplus.

Hero (seat 0 in round mode; seat g%4 in game mode) = TA+ (ta_plus_v2) with the TA bid
model.  Opponents: self-interested TA+ (same bidder), or SpoilerPolicy(target=hero)
with the same bidder.  Conditions per deal (CRN: same deal + same per-seat rng streams):
  self : 0 spoilers
  s1   : 1 spoiler at hero offset 1+(d//4)%3
  s2   : 2 spoilers, offsets PAIRS[(d//4)%3]
  s3   : 3 spoilers
Round mode: n fixed, deal d: seed SeedSequence([BASE, n, d]); trump cycle[d%5]; dealer d%4.
Game mode : seed BASE_G+g, full ladder.

python sim.py rounds N_LIST D TAG [DEADLINE]   -> results/rounds_<TAG>.npz
python sim.py games G TAG [DEADLINE]           -> results/games_<TAG>.npz
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, BidModel, ModelBidder  # noqa: E402
from judgement.engine import make_rngs, play_game, play_round  # noqa: E402
from judgement.policies.spoiler import MCSpoilerPolicy, OmniscientSpoilerPolicy, PIMCSpoilerPolicy, SpoilerPolicy  # noqa: E402,E501
import os  # noqa: E402
from judgement.policies.ta_plus import ta_plus_v2  # noqa: E402

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
M_TA = BidModel.load(ROOT / "experiments" / "phase1" / "results" / "model_ta_final.json")
BASE_R = 131_000_000
BASE_G = 133_000_000
CONDS = ("self", "s1", "s2", "s3")
PAIRS = ((1, 2), (1, 3), (2, 3))


def spoiler_offsets(cond, d):
    if cond == "self":
        return ()
    if cond == "s1":
        return (1 + (d // 4) % 3,)
    if cond == "s2":
        return PAIRS[(d // 4) % 3]
    return (1, 2, 3)


def table(hero, offs, kind="spoiler"):
    pols = []
    for p in range(4):
        off = (p - hero) % 4
        if off in offs:
            pl = SpoilerPolicy(hero) if kind == "spoiler" else kind(hero)
            pols.append(Agent(ModelBidder(M_TA), pl, name="spoiler"))
        else:
            pols.append(Agent(ModelBidder(M_TA), ta_plus_v2(), name="ta_plus2"))
    return pols


def run_rounds(ns, D, tag, deadline=float("inf"), kind="spoiler", conds=CONDS):
    rules = Rules()
    out = RES / f"rounds_{tag}.npz"
    C = len(conds)
    shape = (len(ns), D, C, 4)
    bids = np.full(shape, -1, np.int8)
    tricks = np.full(shape, -1, np.int8)
    scores = np.full(shape, np.nan, np.float32)
    done = np.zeros(len(ns), int)
    t0 = time.time()

    def save():
        np.savez_compressed(out, bids=bids, tricks=tricks, scores=scores, done=done, ns=np.array(ns),
                            conds=np.array(conds), base=BASE_R)
    for d in range(D):
        if time.time() > deadline:
            break
        for i, n in enumerate(ns):
            trump = rules.trump_cycle[d % 5]
            dealer = d % 4
            for ci, cond in enumerate(conds):
                pols = table(0, spoiler_offsets(cond, d), kind)
                deal_rng, prngs = make_rngs(np.random.SeedSequence([BASE_R, n, d]), 4)
                rr = play_round(rules, pols, n_cards=n, trump=trump, dealer=dealer,
                                deal_rng=deal_rng, policy_rngs=prngs)
                bids[i, d, ci] = rr.bids
                tricks[i, d, ci] = rr.tricks
                scores[i, d, ci] = rr.scores
            done[i] = d + 1
        if (d + 1) % 100 == 0:
            save()
            print(tag, d + 1, round(time.time() - t0, 1), flush=True)
    save()
    print(tag, "DONE", int(done.min()), round(time.time() - t0, 1), flush=True)


def run_games(G, tag, deadline=float("inf"), kind="spoiler", conds=CONDS):
    rules = Rules()
    R = rules.n_rounds
    C = len(conds)
    out = RES / f"games_{tag}.npz"
    scores = np.full((G, C, R, 4), np.nan, np.float32)
    bids = np.full((G, C, R, 4), -1, np.int8)
    tricks = np.full((G, C, R, 4), -1, np.int8)
    wins = np.full((G, C, 4), np.nan)
    hero = np.arange(G) % 4
    t0 = time.time()
    done = 0

    def save():
        np.savez_compressed(out, scores=scores[:done], bids=bids[:done], tricks=tricks[:done],
                            wins=wins[:done], hero=hero[:done], conds=np.array(conds), base=BASE_G)
    for g in range(G):
        if time.time() > deadline:
            break
        for ci, cond in enumerate(conds):
            pols = table(g % 4, spoiler_offsets(cond, g), kind)
            res = play_game(rules, pols, seed=BASE_G + g, record_trace=False)
            for r, rr in enumerate(res.rounds):
                scores[g, ci, r] = rr.scores
                bids[g, ci, r] = rr.bids
                tricks[g, ci, r] = rr.tricks
            wins[g, ci] = res.wins
        done = g + 1
        if done % 50 == 0:
            save()
            print(tag, done, round(time.time() - t0, 1), flush=True)
    save()
    print(tag, "DONE", done, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    # SPOILER=pimc:K -> PIMCSpoilerPolicy(K samples); CONDS=self,s3 restricts conditions
    sk = os.environ.get("SPOILER", "spoiler")
    if sk == "spoiler":
        kind = "spoiler"
    elif sk == "omni":
        kind = OmniscientSpoilerPolicy
    elif sk.startswith("mc:"):
        kind = (lambda h, K=int(sk.split(":")[1]): MCSpoilerPolicy(h, n_samples=K))
    else:
        kind = (lambda h, K=int(sk.split(":")[1]): PIMCSpoilerPolicy(h, n_samples=K))
    conds = tuple(os.environ.get("CONDS", ",".join(CONDS)).split(","))
    if mode == "rounds":
        ns = [int(x) for x in sys.argv[2].split(",")]
        run_rounds(ns, int(sys.argv[3]), sys.argv[4], float(sys.argv[5]) if len(sys.argv) > 5 else float("inf"),
                   kind=kind, conds=conds)
    else:
        run_games(int(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else float("inf"))

"""P1: bid-informed vs uniform opponent-hand sampling inside the TA bid oracle.

Situation: dealer = seat 3, hero at bidding position pos in {1,2,3} (seat 2, seat 3,
dealer); earlier seats bid with ModelBidder(model_ta_final); later seats and all play
are ModelBidder + TA (deterministic, so CRN on play is automatic).  Trump cycles
S,H,D,C,NT over situations.  Deal + earlier bids from default_rng([seed, n, pos, i]).

(a) Oracle bids from K worlds each: U = uniform worlds, I = exact bid posterior
    (hard likelihood eps=0, the exact law for this table), S = soft likelihood
    softmax(EV/tau), tau fitted in p1_calib.py (robustness arm).  Candidates = model
    bid ±2.  Evaluation: KE fresh worlds from the exact posterior, every candidate
    scored on the same worlds (CRN) -> paired ΔEV per situation.  Also realised score
    on the true deal.
(b) Response curve (first rc_per situations per (n,pos)): shift one earlier bid by +1
    and -1 and recompute the oracle bid under
      table: U worlds, rollouts use the shifted bids   (table-system effect)
      info : I worlds from the shifted bids, rollouts use the true bids  (information)
      both : I worlds + shifted rollouts               (total, bid-informed oracle)
      model: ModelBidder bid given shifted bids
    shift d in {-1,0,+1} of ONE earlier bid (both directions when feasible, else the
    feasible one-sided pair); slope = finite difference of b*(d).  Same uniform worlds
    for all table(d) (CRN); info/both use the posterior given the shifted bids.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common3 import RES, dump, load_model, status, table_agent
from judgement import Rules
from judgement.bidding.model import ModelBidder
from judgement.bidding2.oracle import best_of, world_scores
from judgement.bidding2.sampler import BidLikelihood
from judgement.bidding2.seqsampler import SeqPosteriorSampler, posterior_worlds
from judgement.cards import NT
from judgement.engine import deal, run_round
from judgement.state import RoundState

TRUMPS = [0, 1, 2, 3, NT]


def legal_given(n, seat, dealer, sb):
    return [b for b in range(n + 1) if not (seat == dealer and b == n - sb)]


def model_bid_shift(mb, st, hero, pr2, rng):
    s2 = RoundState(st.rules, st.n_cards, st.trump, st.dealer, st.hands)
    for _, b in pr2:
        s2.apply_bid(b)
    return mb.bid(s2.info_state(hero), s2.legal_bids(), rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2,3,5,8,13")
    ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--keval", type=int, default=150)
    ap.add_argument("--rc_per", type=int, default=50)
    ap.add_argument("--rcK", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rules = Rules()
    model = load_model()
    agents = [table_agent(model)] * 4
    mb = ModelBidder(model)
    tau = json.load(open(RES / "p1_calib.json"))["oracle"]["tau"]
    hard = BidLikelihood(model, rules, 0.0, 0.0)
    soft = BidLikelihood(model, rules, tau)
    rows, rc_rows = [], []
    t0 = time.time()
    for n in [int(x) for x in a.ns.split(",")]:
        for pos in (1, 2, 3):
            for i in range(a.per):
                rng = np.random.default_rng([a.seed, n, pos, i])
                tr = TRUMPS[i % 5]
                hands = deal(rules, n, rng)
                st = RoundState(rules, n, tr, 3, hands)
                for _ in range(pos):
                    st.apply_bid(mb.bid(st.info_state(st.to_act), st.legal_bids(), rng))
                hero = pos
                info = st.info_state(hero)
                legal = st.legal_bids()
                bm = mb.bid(info, legal, rng)
                cands = [b for b in range(max(0, bm - 2), min(n, bm + 2) + 1) if b in legal]
                wu, _ = SeqPosteriorSampler(info, None).sample(a.K, rng)
                wi, sti = posterior_worlds(info, hard, a.K, rng)
                ws, sts = SeqPosteriorSampler(info, soft).sample(a.K, rng)
                bu, evu = best_of(world_scores(info, wu, cands, agents), cands)
                bi, evi = best_of(world_scores(info, wi, cands, agents), cands)
                bs, evs = best_of(world_scores(info, ws, cands, agents), cands)
                ev_set = sorted({bm, bu, bi, bs})
                we, ste = posterior_worlds(info, hard, a.keval, rng)
                E = dict(zip(ev_set, world_scores(info, we, ev_set, agents).mean(0)))
                real = {}
                for b in ev_set:
                    s2 = st.copy()
                    s2.apply_bid(b)
                    run_round(s2, agents, [rng] * 4, check=False)
                    real[b] = rules.score(b, s2.tricks_won[hero])
                rows.append(dict(n=n, pos=pos, i=i, trump=tr, hand=str(hands[hero]), prior=[b for _, b in info.bids],
                                 bm=bm, bu=bu, bi=bi, bs=bs, ev_m=E[bm], ev_u=E[bu], ev_i=E[bi], ev_s=E[bs],
                                 real_m=real[bm], real_u=real[bu], real_i=real[bi], real_s=real[bs],
                                 stat_i=sti, stat_s=sts, stat_e=ste))
                if i < a.rc_per:
                    prior = list(info.bids)
                    both_ok = [k for k, (_, b) in enumerate(prior) if 0 <= b - 1 and b + 1 <= n]
                    up_ok = [k for k, (_, b) in enumerate(prior) if b + 1 <= n]
                    dn_ok = [k for k, (_, b) in enumerate(prior) if b - 1 >= 0]
                    if both_ok:
                        k, ds = both_ok[int(rng.integers(len(both_ok)))], (-1, 0, 1)
                    elif up_ok:
                        k, ds = up_ok[int(rng.integers(len(up_ok)))], (0, 1)
                    else:
                        k, ds = dn_ok[int(rng.integers(len(dn_ok)))], (-1, 0)
                    if True:
                        rec = dict(n=n, pos=pos, i=i, base_u=bu, base_i=bi, base_m=bm, shifted=k, ds=list(ds))
                        rc = list(range(max(0, bm - 3), min(n, bm + 3) + 1))
                        sb0 = sum(b for _, b in prior)
                        c0 = [b for b in rc if b in legal_given(n, hero, 3, sb0)]
                        wu2, _ = SeqPosteriorSampler(info, None).sample(a.rcK, rng)
                        for d in ds:
                            pr2 = list(prior)
                            pr2[k] = (prior[k][0], prior[k][1] + d)
                            sb2 = sum(b for _, b in pr2)
                            csh = [b for b in rc if b in legal_given(n, hero, 3, sb2)]
                            wi2, _ = posterior_worlds(info, hard, a.rcK, rng, prior_override=pr2)
                            rec[f"table{d:+d}"] = best_of(world_scores(info, wu2, csh, agents, prior_override=pr2), csh)[0]
                            rec[f"info{d:+d}"] = best_of(world_scores(info, wi2, c0, agents), c0)[0]
                            rec[f"both{d:+d}"] = best_of(world_scores(info, wi2, csh, agents, prior_override=pr2), csh)[0]
                            rec[f"model{d:+d}"] = model_bid_shift(mb, st, hero, pr2, rng)
                        rc_rows.append(rec)
            print(f"n={n} pos={pos} rows={len(rows)} rc={len(rc_rows)} {time.time()-t0:.0f}s", flush=True)
            dump({"args": vars(a), "tau": tau, "rows": rows, "rc": rc_rows}, f"p1_raw{a.tag}.json")
    print("total", time.time() - t0)


if __name__ == "__main__":
    main()

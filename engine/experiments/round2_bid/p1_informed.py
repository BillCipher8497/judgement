"""P1: bid-informed vs uniform sampling inside the TA bid oracle.

Situations: dealer = seat 3, hero at bidding position pos in {1,2,3} (= seat pos),
earlier seats bid with the Phase-1 ModelBidder; later seats, and all play, are
ModelBidder + TA.  Trump cycles S,H,D,C,NT over situations.

(a) Oracle bids b_u (uniform worlds) and b_i (bid-informed, hard likelihood eps=0.02),
    K worlds each from a shared proposal pool (CRN), candidates = model bid +-2.
    Evaluation: K_EVAL fresh i.i.d. worlds from the EXACT posterior (rejection, eps=0),
    every candidate scored on the same worlds (CRN) -> paired dEV per situation.
    Also the realised score on the true deal (unbiased, noisy).
(b) Response curve on a subset: shift one earlier bid by +1 / -1 and recompute the
    oracle bid under three configs:
      table : uniform worlds, rollouts use the shifted bids   (table-system effect)
      info  : informed worlds w/ shifted bids in the likelihood, rollouts use the true bids
      both  : informed + shifted rollouts                      (what a real bidder sees)
    slope = (b*(+1) - b*(-1)) / 2 per situation.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common2 import RES, dump, load_model, table_agent  # noqa: E402
from judgement import Rules
from judgement.bidding.model import ModelBidder
from judgement.bidding2.oracle import best_of, world_scores
from judgement.bidding2.sampler import BidLikelihood, BidPhaseSampler, exact_posterior_worlds
from judgement.cards import NT
from judgement.engine import deal, run_round
from judgement.state import RoundState

TRUMPS = [0, 1, 2, 3, NT]


def legal_given(n, seat, dealer, sb):
    return [b for b in range(n + 1) if not (seat == dealer and b == n - sb)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="1,2,3,5,8,13")
    ap.add_argument("--per", type=int, default=120)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--keval", type=int, default=150)
    ap.add_argument("--rc_ns", default="3,5,8")
    ap.add_argument("--rc_per", type=int, default=40)
    ap.add_argument("--rcK", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rules = Rules()
    model = load_model()
    ag = table_agent(model)
    agents = [ag] * 4
    mb = ModelBidder(model)
    lik = BidLikelihood(model, rules, tau=0.0, eps=0.02)
    rc_ns = {int(x) for x in a.rc_ns.split(",") if x}
    rows, rc_rows = [], []
    t0 = time.time()
    for n in [int(x) for x in a.ns.split(",")]:
        for pos in (1, 2, 3):
            for i in range(a.per):
                seed = [a.seed, n, pos, i]
                rng = np.random.default_rng(seed)
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
                samp = BidPhaseSampler(info, lik)
                pool = samp.propose(2000, rng)
                wu, _ = samp.sample(a.K, rng, informed=False, first=pool)
                wi, stt = samp.sample(a.K, rng, informed=True, first=pool)
                su = world_scores(info, wu, cands, agents)
                si = world_scores(info, wi, cands, agents)
                bu, evu = best_of(su, cands)
                bi, evi = best_of(si, cands)
                ev_set = sorted({bm, bu, bi})
                we, tot = exact_posterior_worlds(info, lik, a.keval, rng)
                se = world_scores(info, we, ev_set, agents).mean(0)
                E = dict(zip(ev_set, se))
                real = {}
                for b in ev_set:
                    s2 = st.copy()
                    s2.apply_bid(b)
                    run_round(s2, agents, [rng] * 4, check=False)
                    real[b] = rules.score(b, s2.tricks_won[hero])
                rows.append(dict(n=n, pos=pos, i=i, trump=tr, prior=[b for _, b in info.bids], bm=bm, bu=bu, bi=bi,
                                 ev_m=E[bm], ev_u=E[bu], ev_i=E[bi], real_m=real[bm], real_u=real[bu], real_i=real[bi],
                                 in_u=float(evu[cands.index(bu)]), in_i=float(evi[cands.index(bi)]),
                                 acc=stt["acc"], ess=stt["ess"], n_eval=len(we), eval_props=tot))
                # ---- (b) response curve ----
                if n in rc_ns and i < a.rc_per:
                    prior = list(info.bids)
                    sh = {}
                    for d in (+1, -1):
                        ok = [k for k, (_, b) in enumerate(prior) if 0 <= b + d <= n]
                        if not ok:
                            break
                        k = ok[int(rng.integers(len(ok)))]
                        pr2 = list(prior)
                        pr2[k] = (prior[k][0], prior[k][1] + d)
                        sh[d] = pr2
                    if len(sh) == 2:
                        rec = dict(n=n, pos=pos, i=i, base_u=bu, base_i=bi)
                        rcands = list(range(max(0, bm - 3), min(n, bm + 3) + 1))
                        pool2 = samp.propose(2000, rng)
                        for d, pr2 in sh.items():
                            sb2 = sum(b for _, b in pr2)
                            sb0 = sum(b for _, b in prior)
                            s_sh = BidPhaseSampler(info, lik, prior_override=pr2)
                            w_u, _ = s_sh.sample(a.rcK, rng, informed=False, first=pool2)
                            w_i, _ = s_sh.sample(a.rcK, rng, informed=True, first=pool2)
                            c_sh = [b for b in rcands if b in legal_given(n, hero, 3, sb2)]
                            c_0 = [b for b in rcands if b in legal_given(n, hero, 3, sb0)]
                            rec[f"table{d:+d}"] = best_of(world_scores(info, w_u, c_sh, agents, prior_override=pr2), c_sh)[0]
                            rec[f"info{d:+d}"] = best_of(world_scores(info, w_i, c_0, agents), c_0)[0]
                            rec[f"both{d:+d}"] = best_of(world_scores(info, w_i, c_sh, agents, prior_override=pr2), c_sh)[0]
                            rec[f"model{d:+d}"] = _model_bid_shift(mb, st, hero, pr2, rng)
                        rc_rows.append(rec)
            print(f"n={n} pos={pos} done  rows={len(rows)} rc={len(rc_rows)}  {time.time()-t0:.0f}s", flush=True)
        dump({"args": vars(a), "rows": rows, "rc": rc_rows}, f"p1_raw{a.tag}.json")
    print("total", time.time() - t0)


def _model_bid_shift(mb, st, hero, pr2, rng):
    s2 = RoundState(st.rules, st.n_cards, st.trump, st.dealer, st.hands)
    for _, b in pr2:
        s2.apply_bid(b)
    return mb.bid(s2.info_state(hero), s2.legal_bids(), rng)


if __name__ == "__main__":
    main()

"""E1/E2: hero trick distributions under all-Greedy vs all-TA play on fixed hero hands.

Hero = seat 1 (first bidder, left of dealer 0), trump S.  Greedy play ignores bids,
so its distribution is bid-free.  Under TA the hero's distribution depends on its
own bid b; we compute P_b(t) for a candidate window (greedy-oracle bid +-2 and the
TA-model bid +-1) on common sampled worlds and take the TA oracle bid
b*_TA = argmax_b EV_b.  Opponents: model bids + same play policy.

Stage 2 (E2 decomposition) on long-trump(>=6)+void hands: hero-greedy/opp-TA and
hero-TA(b*)/opp-greedy, to locate the source of greedy's odd/even dips.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder, info_features  # noqa: E402
from judgement.bidding.sim import bid_conditional_dists, bid_free_dist, model_bid  # noqa: E402
from judgement.cards import SUIT_MASK  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.policies import GreedyPolicy  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.state import RoundState  # noqa: E402

from common import RES, agents, dump, load_model  # noqa: E402

HERO = 1


def hero_state(rules, n, hands):
    return RoundState(rules, n, 0, 0, list(hands))


def ev_of(rules, d, b):
    return float(sum(p * rules.score(b, t) for t, p in enumerate(d)))


def run(n, n_hands, S_g, S_t, seed, rules, m_ta, m_gr):
    rng_deal = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    ag_ta = [agents("ta", m_ta)] * 4
    ag_gr = [agents("greedy", m_gr)] * 4
    recs = []
    t0 = time.time()
    S_cache = {}
    for i in range(n_hands):
        hands = deal(rules, n, rng_deal)
        st = hero_state(rules, n, hands)
        legal = st.legal_bids()
        dg = bid_free_dist(st, HERO, ag_gr, S_g, rng)
        evg = {b: ev_of(rules, dg, b) for b in legal}
        bg = max(evg, key=lambda b: (evg[b], -b))
        bm, _ = model_bid(m_ta, rules, info_features(st.info_state(HERO)), n, 0, legal, S_cache)
        cands = sorted({b for b in legal if abs(b - bg) <= 2 or abs(b - bm) <= 1})
        dt = bid_conditional_dists(st, HERO, ag_ta, cands, S_t, rng)
        evt = {b: ev_of(rules, dt[b], b) for b in cands}
        bt = max(evt, key=lambda b: (evt[b], -b))
        h = hands[HERO]
        lens = [(h & SUIT_MASK[s]).bit_count() for s in range(4)]
        recs.append({"i": i, "hand": h, "tlen": lens[0], "voids": sum(1 for x in lens[1:] if x == 0),
                     "greedy": dg.tolist(), "b_greedy": bg, "ev_greedy": evg[bg], "b_model_ta": bm,
                     "ta": {int(b): dt[b].tolist() for b in cands}, "b_ta": bt, "ev_ta": evt[bt],
                     "ta_ev_at_greedy_bid": evt.get(bg), "edge": bt in (cands[0], cands[-1]) and
                     not (bt == 0 or bt == n)})
        if (i + 1) % 100 == 0:
            print(n, i + 1, round(time.time() - t0, 1), flush=True)
            dump(recs, f"e1_n{n}.json")
    dump(recs, f"e1_n{n}.json")
    return recs


def decompose(n, recs, S, seed, rules, m_ta, m_gr, max_hands=300):
    """Hands with long trump (>=6) and a void: 4 play configurations."""
    rng_deal = np.random.default_rng(seed)
    sel = []
    for i in range(len(recs)):
        hands = deal(rules, n, rng_deal)
        r = recs[i]
        if r["tlen"] >= 6 and r["voids"] >= 1:
            sel.append((i, hands))
    sel = sel[:max_hands]
    rng = np.random.default_rng(seed + 7)
    gr = GreedyPolicy()
    ta = TargetAwarePolicy()
    out = []
    for i, hands in sel:
        r = recs[i]
        st = hero_state(rules, n, hands)
        bstar = r["b_ta"]
        # (a) all greedy, more samples
        a = bid_free_dist(st, HERO, [agents("greedy", m_gr)] * 4, S, rng)
        # (b) hero greedy, opponents TA (hero bids with the TA model so opponents see a bid)
        hero_b = Agent(ModelBidder(m_ta), gr)
        pol_b = [agents("ta", m_ta)] * 4
        pol_b[HERO] = hero_b
        b_ = bid_free_dist(st, HERO, pol_b, S, rng)
        # (c) hero TA bidding b*, opponents greedy
        pol_c = [agents("greedy", m_gr)] * 4
        pol_c[HERO] = Agent(ModelBidder(m_ta), ta)
        c = bid_conditional_dists(st, HERO, pol_c, [bstar], S, rng)[bstar]
        # (d) all TA, hero bids b*
        d = bid_conditional_dists(st, HERO, [agents("ta", m_ta)] * 4, [bstar], S, rng)[bstar]
        out.append({"i": i, "b_ta": bstar, "b_greedy": r["b_greedy"], "all_greedy": a.tolist(),
                    "hero_greedy_opp_ta": b_.tolist(), "hero_ta_opp_greedy": c.tolist(), "all_ta": d.tolist()})
    dump(out, f"e2_decomp_n{n}.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="13,8,5")
    ap.add_argument("--hands", default="2000,1000,1000")
    ap.add_argument("--Sg", type=int, default=400)
    ap.add_argument("--St", type=int, default=100)
    ap.add_argument("--Sd", type=int, default=400)
    ap.add_argument("--stage", default="both")
    a = ap.parse_args()
    rules = Rules()
    m_ta, m_gr = load_model("ta"), load_model("greedy")
    for n, H in zip(map(int, a.ns.split(",")), map(int, a.hands.split(","))):
        seed = 5000 + n
        if a.stage in ("both", "e1"):
            recs = run(n, H, a.Sg, a.St, seed, rules, m_ta, m_gr)
        else:
            import json
            recs = json.load(open(RES / f"e1_n{n}.json"))
        if a.stage in ("both", "decomp") and n >= 8:
            decompose(n, recs, a.Sd, seed, rules, m_ta, m_gr)


if __name__ == "__main__":
    main()

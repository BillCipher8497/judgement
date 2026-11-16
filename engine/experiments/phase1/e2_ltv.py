"""E2 targeted: long-trump + void hands (>=6 trumps, >=1 void; n=13, trump S, hero
seat 1) found by rejection sampling, 4 play configurations at 400 worlds each:
all-greedy, hero-greedy/opp-TA, hero-TA(b*)/opp-greedy, all-TA(b*).  b* = TA
oracle bid over greedy-oracle +-2 (100 CRN worlds per candidate)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.bidding.sim import bid_conditional_dists, bid_free_dist  # noqa: E402
from judgement.cards import SUIT_MASK  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.policies import GreedyPolicy  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402

from common import agents, dump, load_model  # noqa: E402
from e1_shift import HERO, ev_of, hero_state  # noqa: E402


def main(n=13, H=300, S=400, St=100, seed=777000, min_trump=6):
    rules = Rules()
    m_ta, m_gr = load_model("ta"), load_model("greedy")
    rng_deal = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1)
    out = []
    t0 = time.time()
    tried = 0
    while len(out) < H:
        hands = deal(rules, n, rng_deal)
        tried += 1
        h = hands[HERO]
        lens = [(h & SUIT_MASK[s]).bit_count() for s in range(4)]
        if lens[0] < min_trump or min(lens[1:]) > 0:
            continue
        st = hero_state(rules, n, hands)
        legal = st.legal_bids()
        g = bid_free_dist(st, HERO, [agents("greedy", m_gr)] * 4, S, rng)
        evg = {b: ev_of(rules, g, b) for b in legal}
        bg = max(evg, key=lambda b: (evg[b], -b))
        cands = [b for b in legal if abs(b - bg) <= 2]
        dt = bid_conditional_dists(st, HERO, [agents("ta", m_ta)] * 4, cands, St, rng)
        evt = {b: ev_of(rules, dt[b], b) for b in cands}
        bstar = max(evt, key=lambda b: (evt[b], -b))
        pol_b = [agents("ta", m_ta)] * 4
        pol_b[HERO] = Agent(ModelBidder(m_ta), GreedyPolicy())
        hg_ota = bid_free_dist(st, HERO, pol_b, S, rng)
        pol_c = [agents("greedy", m_gr)] * 4
        pol_c[HERO] = Agent(ModelBidder(m_ta), TargetAwarePolicy())
        hta_og = bid_conditional_dists(st, HERO, pol_c, [bstar], S, rng)[bstar]
        ata = bid_conditional_dists(st, HERO, [agents("ta", m_ta)] * 4, [bstar], S, rng)[bstar]
        out.append({"hand": h, "tlen": lens[0], "voids": sum(1 for x in lens[1:] if x == 0), "b_greedy": bg,
                    "b_ta": bstar, "ev_greedy": evg[bg], "ev_ta": evt[bstar],
                    "all_greedy": g.tolist(), "hero_greedy_opp_ta": hg_ota.tolist(),
                    "hero_ta_opp_greedy": hta_og.tolist(), "all_ta": ata.tolist()})
        if len(out) % 25 == 0:
            print(len(out), tried, round(time.time() - t0, 1), flush=True)
            dump({"tried": tried, "recs": out}, "e2_ltv.json")
    dump({"tried": tried, "recs": out}, "e2_ltv.json")


if __name__ == "__main__":
    main()

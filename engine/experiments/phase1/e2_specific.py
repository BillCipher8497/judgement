"""E2 targeted check: the RESULTS.md greedy-baseline hands (4p, n=13, trump S,
hero seat 1) under all-Greedy vs all-TA (hero bidding each b in a window),
plus the two mixed configurations, with many samples."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.bidding.sim import bid_conditional_dists, bid_free_dist  # noqa: E402
from judgement.cards import FULL_DECK, cards_of, parse_hand  # noqa: E402
from judgement.policies import GreedyPolicy  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.state import RoundState  # noqa: E402

from common import agents, dump, load_model  # noqa: E402

HANDS = {
    "h3_KQT632.985.J876.-": "KS QS TS 6S 3S 2S 9H 8H 5H JD 8D 7D 6D",
    "h4_KQJ9843.7.-.AK986": "KS QS JS 9S 8S 4S 3S 7H AC KC 9C 8C 6C",
    "h5_QT86532.Q.-.AKQJT": "QS TS 8S 6S 5S 3S 2S QH AC KC QC JC TC",
}


def main(S=2000, seed=31):
    rules = Rules()
    m_ta, m_gr = load_model("ta"), load_model("greedy")
    rng = np.random.default_rng(seed)
    out = {}
    for name, hs in HANDS.items():
        hero = parse_hand(hs)
        rest = cards_of(FULL_DECK & ~hero)
        rng.shuffle(rest)
        others = [sum(1 << c for c in rest[13 * i:13 * i + 13]) for i in range(3)]
        hands = [others[0], hero, others[1], others[2]]     # hidden hands are resampled anyway
        st = RoundState(rules, 13, 0, 0, hands)
        g = bid_free_dist(st, 1, [agents("greedy", m_gr)] * 4, S, rng)
        pol_b = [agents("ta", m_ta)] * 4
        pol_b[1] = Agent(ModelBidder(m_ta), GreedyPolicy())
        hg_ota = bid_free_dist(st, 1, pol_b, S, rng)
        mode = int(np.argmax(g))
        cands = list(range(max(0, mode - 3), min(13, mode + 3) + 1))
        ta = bid_conditional_dists(st, 1, [agents("ta", m_ta)] * 4, cands, S // 2, rng)
        pol_c = [agents("greedy", m_gr)] * 4
        pol_c[1] = Agent(ModelBidder(m_ta), TargetAwarePolicy())
        hta_og = bid_conditional_dists(st, 1, pol_c, cands, S // 2, rng)
        out[name] = {"samples_greedy": S, "samples_ta": S // 2, "all_greedy": g.tolist(),
                     "hero_greedy_opp_ta": hg_ota.tolist(),
                     "all_ta_by_bid": {b: ta[b].tolist() for b in cands},
                     "hero_ta_opp_greedy_by_bid": {b: hta_og[b].tolist() for b in cands}}
        print(name, flush=True)
        dump(out, "e2_specific.json")


if __name__ == "__main__":
    main()

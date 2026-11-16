"""Round 3 (H1): SpoilerPolicy / PIMCSpoilerPolicy behaviour and legality."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import make_state  # noqa: E402
from judgement import Rules  # noqa: E402
from judgement.cards import NT, parse_card  # noqa: E402
from judgement.engine import play_round  # noqa: E402
from judgement.policies.spoiler import PIMCSpoilerPolicy, SpoilerPolicy  # noqa: E402
from judgement.policies.ta_plus import ta_plus_v2  # noqa: E402


def _round(pols, seed, n=5, trump=0, dealer=0):
    return play_round(Rules(), pols, seed=seed, n_cards=n, trump=trump, dealer=dealer)


def test_modes():
    # dealer 0 -> seat 1 bids first; bids (seat1..seat0) = 1,0,0,0 ; target = seat 1
    st = make_state(["2S 3S", "AS KS", "4S 5S", "6S 7S"], NT, bids=[1, 0, 0, 0])
    sp = SpoilerPolicy(target=1)
    assert sp.mode(st.info_state(2)) == -1          # target needs 1 -> deny
    st.apply_card(parse_card("AS"))                 # seat 1 leads AS
    st.apply_card(parse_card("4S"))
    st.apply_card(parse_card("6S"))
    st.apply_card(parse_card("2S"))                 # target wins trick 1: need 0
    assert sp.mode(st.info_state(2)) == 1           # give
    st2 = make_state(["2S 3S", "AS KS", "4S 5S", "6S 7S"], NT, bids=[2, 0, 0, 1])
    # seat 1 bid 2 with 2 tricks to play: live, needs tricks -> deny
    assert SpoilerPolicy(target=1).mode(st2.info_state(2)) == -1


def test_deny_overtakes_target():
    # trump NT, seat 1 (target, bid 1) leads QS; seat 2 holds AS and 2S -> must play AS (deny)
    st = make_state(["3H 4H", "QS 5H", "AS 2S", "6H 7H"], NT, bids=[1, 0, 0, 0])
    st.apply_card(parse_card("QS"))
    sp = SpoilerPolicy(target=1)
    c = sp.play(st.info_state(2), st.legal_cards(), np.random.default_rng(0))
    assert c == parse_card("AS")


def test_give_ducks_under_target():
    # target seat 1 at need 0 (bid 0) leads QS; seat 2 holds AS, 2S -> plays 2S (give)
    st = make_state(["3H 4H", "QS 5H", "AS 2S", "6H 7H"], NT, bids=[0, 0, 0, 1])
    st.apply_card(parse_card("QS"))
    c = SpoilerPolicy(target=1).play(st.info_state(2), st.legal_cards(), np.random.default_rng(0))
    assert c == parse_card("2S")


def test_spoilers_legal_deterministic_and_hurt():
    from judgement.bidding.model import Agent, BidModel, ModelBidder
    m = BidModel.load(ROOT / "experiments" / "phase1" / "results" / "model_ta_final.json")
    tot_self = tot_sp = 0.0
    for d in range(40):
        self_t = [Agent(ModelBidder(m), ta_plus_v2()) for _ in range(4)]
        sp_t = [self_t[0]] + [Agent(ModelBidder(m), SpoilerPolicy(0)) for _ in range(3)]
        a = _round(self_t, d, n=6, trump=d % 5 if d % 5 < 4 else NT, dealer=d % 4)
        b = _round(sp_t, d, n=6, trump=a.trump, dealer=d % 4)
        b2 = _round(sp_t, d, n=6, trump=a.trump, dealer=d % 4)
        assert b.scores == b2.scores and b.bids == b2.bids
        assert a.bids == b.bids                        # same bidder, same info -> same bids
        tot_self += a.scores[0]
        tot_sp += b.scores[0]
    assert tot_sp <= tot_self


def test_pimc_spoiler_legal_small():
    pols = [ta_plus_v2()] + [PIMCSpoilerPolicy(0, n_samples=3) for _ in range(3)]
    for d in range(3):
        r = _round(pols, 100 + d, n=3, trump=0, dealer=d % 4)
        assert sum(r.tricks) == 3


def test_mc_and_omni_spoilers_legal_small():
    from judgement.policies.spoiler import MCSpoilerPolicy, OmniscientSpoilerPolicy
    for cls in (lambda: MCSpoilerPolicy(0, n_samples=2), lambda: OmniscientSpoilerPolicy(0)):
        pols = [ta_plus_v2()] + [cls() for _ in range(3)]
        for d in range(3):
            r = _round(pols, 200 + d, n=3, trump=NT, dealer=d % 4)
            assert sum(r.tricks) == 3

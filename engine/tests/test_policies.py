import numpy as np

from helpers import bid_through, make_state
from judgement.cards import NT, card_str, parse_card as C
from judgement.policies import GreedyPolicy, Policy, RandomPolicy

S, H = 0, 1
g = GreedyPolicy()
rng = np.random.default_rng(0)


def greedy_card(st):
    return card_str(g.play(st.info_state(st.to_act), st.legal_cards(), rng))


def test_protocol():
    assert isinstance(GreedyPolicy(), Policy) and isinstance(RandomPolicy(), Policy)


def test_greedy_leads_strongest():
    st = bid_through(make_state(["AS 3H KD", "2S 4H 2D", "3S 5H 3D", "4S 6H 4D"], trump=H, dealer=3))
    assert greedy_card(st) == "3H"           # holds trump -> highest trump
    st = bid_through(make_state(["AS 3H KD", "2S 4H 2D", "3S 5H 3D", "4S 6H 4D"], trump=NT, dealer=3))
    assert greedy_card(st) == "AS"           # NT -> highest rank


def test_greedy_follow_win_or_dump():
    st = bid_through(make_state(["TS 2D", "AS 9S", "3S 3D", "4S 4D"], trump=H, dealer=3))
    st.apply_card(C("TS"))
    assert greedy_card(st) == "AS"           # can beat -> highest winner
    st = bid_through(make_state(["TS 2D", "5S 9S", "3S 3D", "4S 4D"], trump=H, dealer=3))
    st.apply_card(C("TS"))
    assert greedy_card(st) == "5S"           # cannot beat -> lowest
    st = bid_through(make_state(["TS 2D", "5H 9H", "3S 3D", "4S 4D"], trump=H, dealer=3))
    st.apply_card(C("TS"))
    assert greedy_card(st) == "9H"           # void -> ruff high
    st = bid_through(make_state(["TS 2D", "5H 9D", "3S 3D", "4S 4D"], trump=H, dealer=3))
    st.apply_card(C("TS")); st.apply_card(C("5H"))
    assert greedy_card(st) == "3S"           # trick already trumped, must follow low


def test_greedy_bid_legal_and_deterministic():
    st = make_state(["AS KS QS", "AH 2C 3C", "AD 4C 5C", "AC 6C 7C"], trump=S, dealer=3)
    info = st.info_state(0)
    assert g.bid(info, st.legal_bids(), rng) == 3

import numpy as np

from helpers import make_state
from judgement import Rules, play_game, play_round
from judgement.cards import NT, card_str, parse_card as C
from judgement.policies.hybrid import HybridPolicy
from judgement.policies.ta_plus import TAPlusPolicy
from judgement.policies.target_aware import TargetAwarePolicy

H = 1
rng = np.random.default_rng(0)
TA, TAP = TargetAwarePolicy(), TAPlusPolicy()


def card(pol, st):
    return card_str(pol.play(st.info_state(st.to_act), st.legal_cards(), rng))


def _void_discard(hand1):
    # NT; seat 0 leads KS, seat 1 (bid 1, void in spades) cannot win this trick.
    st = make_state(["KS 3S 4H 5H", hand1, "QS 6H 7H 8H", "JS 9H TH JH"], trump=NT, dealer=3, bids=[1, 1, 0, 0])
    st.apply_card(C("KS"))
    return st


def test_sheds_surplus_winner_when_cannot_win():
    st = _void_discard("AD AC KC 2D")           # three bosses, needs one
    assert card(TA, st) == "2D"                 # TA: cheapest discard
    assert card(TAP, st) in ("AD", "AC")        # TA+: throw a spare boss


def test_keeps_only_winner_when_no_surplus():
    st = _void_discard("AD 2C 3C 2D")           # one boss, needs one: must keep it
    assert card(TAP, st) != "AD"


def test_shed_off_matches_ta():
    st = _void_discard("AD AC KC 2D")
    off = TAPlusPolicy(shed=False, lead_high=0, theorems=False)
    assert card(off, st) == card(TA, st)


def test_lead_high_without_sure_winner():
    # Trump H, seat 0 needs a trick and holds no sure winner: lead the highest side card.
    hands = ["KS 3S 4D 5D", "AS 2D 3D 6D", "QS 6H 7H 8H", "JS 9H TH JH"]
    st = make_state(hands, trump=H, dealer=3, bids=[1, 1, 0, 0])
    assert card(TAP, st) == "KS"
    # lead_high=3 is trump-rounds only: in NT TA+ leads as TA does
    st = make_state(hands, trump=NT, dealer=3, bids=[1, 1, 0, 0])
    assert card(TAP, st) == card(TA, st)
    assert card(TAPlusPolicy(lead_high=2), st) == "KS"


def test_leads_sure_winner_when_held():
    st = make_state(["AS 2H 3H", "KS 4H 5H", "QS 6H 7H", "JS 8H 9H"], trump=NT, dealer=3, bids=[1, 0, 0, 0])
    assert card(TAP, st) == "AS"


def test_need_equals_remaining_follows_with_strongest():
    # seat 1 must win both tricks; it cannot be sure of this one: play the strongest heart.
    st = make_state(["5H 2C", "QH 6H", "AH 3C", "4S 4C"], trump=NT, dealer=3, bids=[0, 2, 1, 0])
    st.apply_card(C("5H"))
    assert card(TAP, st) == "QH"


def test_at_target_rules_unchanged():
    # R1: at target, following -> highest sure loser; R2: lead low.
    st = make_state(["KS 2D 3D", "9S 3S 4D", "5S 6D 7D", "8S 2H 9D"], trump=H, dealer=3, bids=[1, 0, 0, 1])
    st.apply_card(C("KS"))
    assert card(TAP, st) == "9S"
    st = make_state(["AS 2H 3H", "KS 4H 5H", "QS 6H 7H", "JS 8H 9H"], trump=NT, dealer=3, bids=[0, 0, 0, 0])
    assert card(TAP, st) == "2H"


def test_needs_trick_last_to_play_takes_it_cheaply():
    st = make_state(["9S 2D 3D", "5S 4D 5D", "6S 6D 7D", "QS KS 2C"], trump=H, dealer=3, bids=[0, 0, 0, 1])
    for c in ("9S", "5S", "6S"):
        st.apply_card(C(c))
    assert card(TAP, st) == "QS"


def test_table_slack():
    st = make_state(["KS 3S 4H 5H", "AD AC KC 2D", "QS 6H 7H 8H", "JS 9H TH JH"], trump=NT, dealer=3,
                    bids=[1, 1, 0, 0])
    assert TAPlusPolicy.table_slack(st.info_state(0)) == 2


def test_lead_pw_bosses():
    st = _void_discard("AD AC KC 2D")
    lp = TAPlusPolicy.lead_pw(st.info_state(1))
    assert lp[C("AD")] == 1.0 and lp[C("AC")] == 1.0 and lp[C("KC")] == 1.0
    assert lp[C("2D")] < 1.0


def test_hybrid_switches_to_pimc_and_is_legal():
    hyb = HybridPolicy(n_samples=4, pimc_cards=3)
    res = play_round(Rules(), [hyb, TAP, TAP, TAP], seed=3, n_cards=5, trump=0, dealer=0)
    assert sum(res.tricks) == 5
    assert hyb.n_pimc > 0


def test_hybrid_without_pimc_equals_ta_plus():
    st = _void_discard("AD AC KC 2D")
    assert card(HybridPolicy(pimc_cards=0), st) == card(TAP, st)


def test_ta_plus_full_game_deterministic():
    a = play_game(Rules(max_cards=5), [TAPlusPolicy() for _ in range(4)], seed=11, record_trace=False)
    b = play_game(Rules(max_cards=5), [TAPlusPolicy() for _ in range(4)], seed=11, record_trace=False)
    assert a.totals == b.totals

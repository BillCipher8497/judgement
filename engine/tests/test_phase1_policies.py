import numpy as np
import pytest

from helpers import make_state
from judgement import Rules, play_round
from judgement.cards import NT, card_str, parse_card as C
from judgement.policies.target_aware import TargetAwarePolicy, TASpitePolicy, TrickModel

S, H = 0, 1
rng = np.random.default_rng(0)
TA, SP = TargetAwarePolicy(), TASpitePolicy()


def card(pol, st):
    return card_str(pol.play(st.info_state(st.to_act), st.legal_cards(), rng))


def test_at_target_follows_with_highest_sure_loser():
    st = make_state(["KS 2D 3D", "9S 3S 4D", "5S 6D 7D", "8S 2H 9D"], trump=H, dealer=3, bids=[1, 0, 0, 1])
    st.apply_card(C("KS"))
    assert card(TA, st) == "9S"


def test_at_target_void_sheds_most_dangerous_card():
    st = make_state(["5S 2D 3D", "AD 2C 3C", "6S 6D 7D", "8S 2H 9D"], trump=H, dealer=3, bids=[1, 0, 0, 1])
    st.apply_card(C("5S"))
    assert card(TA, st) == "AD"


def test_needs_trick_last_to_play_takes_it_cheaply():
    st = make_state(["9S 2D 3D", "5S 4D 5D", "6S 6D 7D", "QS KS 2C"], trump=H, dealer=3, bids=[0, 0, 0, 1])
    for c in ("9S", "5S", "6S"):
        st.apply_card(C(c))
    assert card(TA, st) == "QS"


def test_needs_trick_leads_sure_winner_nt():
    st = make_state(["AS 2H 3H", "KS 4H 5H", "QS 6H 7H", "JS 8H 9H"], trump=NT, dealer=3, bids=[1, 0, 0, 0])
    assert card(TA, st) == "AS"


def test_at_target_leads_low_nt():
    st = make_state(["AS 2H 3H", "KS 4H 5H", "QS 6H 7H", "JS 8H 9H"], trump=NT, dealer=3, bids=[0, 0, 0, 0])
    assert card(TA, st) == "2H"


def _missed_position():
    # s1 bids 3 of 3 and loses trick 1 -> missed; s0 (needs 1) then leads KD.
    st = make_state(["AS KD 2C", "KS 2H 3C", "QS 2D 4C", "JS 3D 5C"], trump=H, dealer=3, bids=[2, 3, 0, 0])
    for c in ("AS", "KS", "QS", "JS", "KD"):
        st.apply_card(C(c))
    return st


def test_missed_ta_plays_neutral_lowest():
    assert card(TA, _missed_position()) == "3C"


def test_spite_ruffs_to_deny_needy_leader():
    # s0 still needs a trick, s2/s3 sit at target: spite ruffs to take the trick away.
    assert card(SP, _missed_position()) == "2H"


def test_spite_equals_ta_while_open():
    st = make_state(["KS 2D 3D", "9S 3S 4D", "5S 6D 7D", "8S 2H 9D"], trump=H, dealer=3, bids=[1, 0, 0, 1])
    st.apply_card(C("KS"))
    assert card(SP, st) == card(TA, st)


def test_trickmodel_sure_winner_after_ace_played():
    # NT, ace of spades already gone in trick 1, the KS is then a sure winner when led.
    st = make_state(["AS 2H 3H", "KS 4H 5H", "QS 6H 7H", "JS 8H 9H"], trump=NT, dealer=3, bids=[1, 1, 0, 0])
    for c in ("AS", "KS", "QS", "JS"):
        st.apply_card(C(c))
    tm = TrickModel(st.info_state(0))
    assert tm.p_win(C("3H")) < 1.0
    st2 = make_state(["AS KD 3H", "2S 4H 5H", "QS 6H 7H", "JS 8H 9H"], trump=NT, dealer=3, bids=[1, 1, 0, 0])
    for c in ("AS", "2S", "QS", "JS"):
        st2.apply_card(C(c))
    tm2 = TrickModel(st2.info_state(0))
    assert tm2.p_win(C("KD")) < 1.0          # AD unseen
    # a card that cannot beat the current winner is a sure loser
    st3 = make_state(["KS 2D 3D", "9S 3S 4D", "5S 6D 7D", "8S 2H 9D"], trump=H, dealer=3, bids=[1, 0, 0, 1])
    st3.apply_card(C("KS"))
    tm3 = TrickModel(st3.info_state(1))
    assert tm3.p_win(C("9S")) == 0.0


def test_trickmodel_last_player_exact():
    st = make_state(["9S 2D 3D", "5S 4D 5D", "6S 6D 7D", "QS KS 2C"], trump=H, dealer=3, bids=[0, 0, 0, 1])
    for c in ("9S", "5S", "6S"):
        st.apply_card(C(c))
    tm = TrickModel(st.info_state(3))
    assert tm.p_win(C("QS")) == 1.0 and tm.p_win(C("KS")) == 1.0


@pytest.mark.parametrize("N", [3, 4, 5, 7])
@pytest.mark.parametrize("pol", [TA, SP])
def test_legal_play_random_rounds(N, pol, ):
    rules = Rules(n_players=N)
    for i, n in enumerate(rules.ladder()):
        for miss in ("zero", "neg_diff"):
            r = Rules(n_players=N, miss_penalty=miss)
            rr = play_round(r, [pol] * N, seed=1000 * N + i, round_index=i)   # engine checks legality
            assert sum(rr.tricks) == rr.n_cards


def test_ta_hits_more_than_greedy_with_same_bids():
    from judgement.policies import GreedyPolicy
    from judgement.bidding.model import Agent
    hits = {}
    for name, play in (("ta", TA), ("greedy", GreedyPolicy())):
        h = 0
        for s in range(150):
            rr = play_round(Rules(), [Agent(TA, play)] * 4, seed=s, n_cards=7, trump=S, dealer=0)
            h += sum(b == t for b, t in zip(rr.bids, rr.tricks))
        hits[name] = h
    assert hits["ta"] > hits["greedy"]


def _full_deal_state():
    from judgement.cards import FULL_DECK, cards_of, mask_of, parse_hand
    from judgement.state import RoundState
    h0 = parse_hand("KS QS 2H 3H 4H 5H 6H 7H 8H 9H TH JH QH")
    rest = cards_of(FULL_DECK & ~h0)
    hands = [h0] + [mask_of(rest[13 * i:13 * i + 13]) for i in range(3)]
    st = RoundState(Rules(), 13, NT, 3, hands)
    for b in (1, 0, 0, 0):
        st.apply_bid(b)
    return st


def test_joint_model_knows_unseen_ace_is_held():
    st = _full_deal_state()        # no stock: the AS is certainly with an opponent
    assert TrickModel(st.info_state(0)).p_win(C("KS")) == 0.0
    assert TrickModel(st.info_state(0), joint=False).p_win(C("KS")) > 0.2   # v0 approximation
    assert TargetAwarePolicy(joint=False).name == "ta_v0"

import pytest

from helpers import bid_through, make_state
from judgement.cards import NT, cards_of, parse_card as C, parse_hand
from judgement.rules import Rules
from judgement.state import IllegalAction, RoundState

S, H, D, Cl = 0, 1, 2, 3


def legal_set(st):
    return {c for c in cards_of(st.legal_cards())}


def play(st, *cards):
    for c in cards:
        st.apply_card(C(c))


# ------------------------------------------------------------------ follow suit
def test_must_follow_suit():
    st = bid_through(make_state(["AS 2H", "KS 3H", "QD 4H", "JC 5H"], trump=H, dealer=3))
    assert st.to_act == 0
    assert legal_set(st) == {C("AS"), C("2H")}  # leader: anything
    play(st, "AS")
    assert legal_set(st) == {C("KS")}            # holds spade -> must follow
    with pytest.raises(IllegalAction):
        st.apply_card(C("3H"))
    play(st, "KS")
    assert legal_set(st) == {C("QD"), C("4H")}  # void -> anything, trump optional
    play(st, "QD")                               # discard instead of trumping is legal
    play(st, "JC")
    assert st.trick_winners == [0]


def test_trump_beats_led_suit():
    st = bid_through(make_state(["AS", "2H", "KS", "QS"], trump=H, dealer=3))
    play(st, "AS", "2H", "KS", "QS")
    assert st.trick_winners == [1]


def test_higher_trump_wins():
    st = bid_through(make_state(["AS", "2H", "KH", "QS"], trump=H, dealer=3))
    play(st, "AS", "2H", "KH", "QS")
    assert st.trick_winners == [2]


def test_nt_highest_of_led():
    st = bid_through(make_state(["5D", "AS", "9D", "KD"], trump=NT, dealer=3))
    play(st, "5D", "AS", "9D", "KD")
    assert st.trick_winners == [3]


def test_offsuit_discard_never_wins():
    # led 2C, everyone else discards higher off-suit non-trump cards
    st = bid_through(make_state(["2C", "AS", "AD", "AH"], trump=NT, dealer=3))
    play(st, "2C", "AS", "AD", "AH")
    assert st.trick_winners == [0]
    st = bid_through(make_state(["2C", "AS", "AD", "KH"], trump=S, dealer=3))
    play(st, "2C", "AS", "AD", "KH")
    assert st.trick_winners == [1]  # the trump, not the off-suit aces


def test_led_trump_resolution():
    st = bid_through(make_state(["5S", "AH", "9S", "2S"], trump=S, dealer=3))
    play(st, "5S", "AH", "9S", "2S")
    assert st.trick_winners == [2]


def test_first_lead_left_of_dealer_and_winner_leads():
    st = bid_through(make_state(["AS 2D", "KS 3D", "QS 4D", "JS 5D"], trump=NT, dealer=1))
    assert st.to_act == 2
    play(st, "QS", "JS", "AS", "KS")
    assert st.trick_winners == [0] and st.to_act == 0


def test_bidding_order_dealer_last():
    st = make_state(["AS", "KS", "QS", "JS"], trump=NT, dealer=2)
    order = []
    while st.phase == 0:
        order.append(st.to_act)
        st.apply_bid(st.legal_bids()[0])
    assert order == [3, 0, 1, 2]


# ------------------------------------------------------------------ hook rule
def test_hook_n1():
    st = make_state(["AS", "KS", "QS", "JS"], trump=NT, dealer=3)
    for _ in range(3):
        assert st.legal_bids() == [0, 1]
        st.apply_bid(0)
    assert st.legal_bids() == [0]          # 0+0+0+1 == 1 forbidden
    with pytest.raises(IllegalAction):
        st.apply_bid(1)
    st = make_state(["AS", "KS", "QS", "JS"], trump=NT, dealer=3, bids=[1, 0, 0])
    assert st.legal_bids() == [1]          # 1+0+0+0 == 1 forbidden


def test_hook_edge_sums():
    hands = ["AS KS QS", "JS TS 9S", "8S 7S 6S", "5S 4S 3S"]
    # others already exceed n -> nothing forbidden
    st = make_state(hands, trump=NT, dealer=3, bids=[3, 3, 3])
    assert st.legal_bids() == [0, 1, 2, 3]
    # others sum exactly n -> dealer may not bid 0
    st = make_state(hands, trump=NT, dealer=3, bids=[1, 1, 1])
    assert st.legal_bids() == [1, 2, 3]
    # others sum 0 -> dealer may not bid n
    st = make_state(hands, trump=NT, dealer=3, bids=[0, 0, 0])
    assert st.legal_bids() == [0, 1, 2]
    # non-dealers are unconstrained
    st = make_state(hands, trump=NT, dealer=3, bids=[3, 0])
    assert st.legal_bids() == [0, 1, 2, 3]


def test_hook_disabled():
    st = make_state(["AS", "KS", "QS", "JS"], trump=NT, dealer=3, bids=[0, 0, 0],
                    rules=Rules(hook=False))
    assert st.legal_bids() == [0, 1]


# ------------------------------------------------------------------ copy / undo / errors
def test_undo_restores_exactly():
    import numpy as np
    from judgement.engine import deal
    r = Rules()
    st = bid_through(RoundState(r, 13, S, 0, deal(r, 13, np.random.default_rng(5))))
    snaps = []
    rng = np.random.default_rng(1)
    while not st.is_terminal:
        snaps.append(st.copy())
        legal = cards_of(st.legal_cards())
        st.apply_card(legal[int(rng.integers(len(legal)))])
    final = st.copy()
    for snap in reversed(snaps):
        st.undo_card()
        for f in ("hands", "bids", "tricks_won", "leader", "to_act", "trick", "best_card",
                  "best_seat", "history", "voids", "trick_winners", "phase"):
            assert getattr(st, f) == getattr(snap, f), f
    assert final.is_terminal and sum(final.tricks_won) == 13


def test_copy_independent():
    st = bid_through(make_state(["AS 2D", "KS 3D", "QS 4D", "JS 5D"], trump=NT))
    cp = st.copy()
    play(st, "KS")
    assert cp.history == [] and cp.hands[1] == parse_hand("KS 3D")


def test_bad_construction():
    r = Rules()
    with pytest.raises(ValueError):
        RoundState(r, 1, S, 0, [parse_hand("AS")] * 4)  # overlapping
    with pytest.raises(ValueError):
        RoundState(r, 2, S, 0, [parse_hand(h) for h in ["AS", "KS", "QS", "JS"]])


def test_play_before_bids_illegal():
    st = make_state(["AS", "KS", "QS", "JS"], trump=NT)
    with pytest.raises(IllegalAction):
        st.apply_card(C("KS"))


# ------------------------------------------------------------------ info state
def test_info_state_fields_and_no_leak():
    st = make_state(["AS 2D", "KS 3D", "QS 4D", "JS 5D"], trump=H, dealer=3, bids=[1, 0, 0])
    st.apply_bid(st.legal_bids()[0])
    play(st, "AS", "KS")
    info = st.info_state(2)
    assert info.seat == 2 and info.dealer == 3 and info.n_cards == 2 and info.trump == H
    assert info.hand == parse_hand("QS 4D")
    assert info.bids == ((0, 1), (1, 0), (2, 0), (3, st.bids[3]))
    assert info.current_trick == ((0, C("AS")), (1, C("KS")))
    assert info.tricks_won == (0, 0, 0, 0)
    assert info.voids == (0, 0, 0, 0)
    assert info.cards_left(0) == 1
    snap = info.snapshot()
    # detached snapshot contains no other seat's cards anywhere
    others = parse_hand("2D 3D 5D")
    def walk(o):
        if isinstance(o, int) and not isinstance(o, bool) and o > 51:
            assert o & others == 0
        elif isinstance(o, (tuple, list)):
            for x in o:
                walk(x)
    for v in vars(snap).values() if hasattr(snap, "__dict__") else []:
        walk(v)
    for name in snap.__dataclass_fields__:
        walk(getattr(snap, name))
    # public API of the view never yields hidden cards
    public = [a for a in dir(info) if not a.startswith("_")]
    assert "hands" not in public
    for a in public:
        v = getattr(info, a)
        if not callable(v):
            walk(v)


def test_voids_observed():
    st = bid_through(make_state(["AS 2D", "KS 3D", "QD 4D", "JS 5D"], trump=NT, dealer=3))
    play(st, "AS", "KS", "QD")
    assert st.voids[2] == 1 << S and st.voids[1] == 0

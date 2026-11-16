"""Independent engine witnesses for the welfare theorem and H1 accounting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from judgement import Rules, RoundState
from judgement.cards import NT

spec = importlib.util.spec_from_file_location(
    "round7_welfare", Path(__file__).resolve().parents[1]/"math"/"m11_welfare.py")
welfare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(welfare)


@pytest.mark.parametrize("n", [1, 5, 13])
@pytest.mark.parametrize("trump", [0, NT])
def test_same_deal_attains_hook_ceiling_and_zero(n, trump):
    hands = [sum(1 << (13*s+r) for r in range(n)) for s in range(4)]
    for bids, expected in [([n, 0, 0, 1], 30+n), ([n-1, 1, 1, 1], 0)]:
        rules = Rules()
        st = RoundState(rules, n, trump, 3, hands)
        for b in bids:
            st.apply_bid(b)
        while not st.is_terminal:
            legal = st.legal_cards()
            st.apply_card((legal & -legal).bit_length()-1)
        assert st.tricks_won == [n, 0, 0, 0]
        scores = [rules.score(b, t) for b, t in zip(st.bids, st.tricks_won)]
        assert sum(scores) == expected
        w, _, _ = welfare.welfare_parts(st.bids, st.tricks_won, n, scores)
        assert w == expected
    rules = Rules(hook=False)
    st = RoundState(rules, n, trump, 3, hands)
    for b in [n, 0, 0, 0]:
        st.apply_bid(b)
    while not st.is_terminal:
        legal = st.legal_cards()
        st.apply_card((legal & -legal).bit_length()-1)
    assert sum(rules.score(b, t) for b, t in zip(st.bids, st.tricks_won)) == 40+n


def test_fixed_bids_play_changes_total_score():
    # Opponent's discard on AS decides whether it wins the second trick.
    from judgement.cards import parse_hand, parse_card
    totals = []
    for discard in ["KH", "3C"]:
        st = RoundState(Rules(), 2, NT, 3,
                        [parse_hand(h) for h in ["AS 2H", "KH 3C", "4D 5D", "6D 7D"]])
        for b in [1, 0, 0, 0]:
            st.apply_bid(b)
        st.apply_card(parse_card("AS"))
        st.apply_card(parse_card(discard))
        while not st.is_terminal:
            legal = st.legal_cards()
            st.apply_card((legal & -legal).bit_length()-1)
        totals.append(sum(st.rules.score(b, t) for b, t in zip(st.bids, st.tricks_won)))
    assert totals == [30, 31]


def test_reject_invalid_and_partial_data():
    for b, t, n in [([1, 0, 0, 0], [1, 0, 0, 0], 1),
                    ([-1, 0, 0, 0], [1, 0, 0, 0], 1),
                    ([0, 0, 0, 0], [0, 0, 0, 0], 1)]:
        with pytest.raises(ValueError):
            welfare.welfare_parts(b, t, n)
    with pytest.raises(ValueError):
        welfare.welfare_parts([0]*4, [1, 0, 0, 0], 1, [0]*4)


def test_ties_as_win_and_game_clustering():
    scores = np.array([[[11, 11, 0, 0], [0, 10, 10, 0]],
                       [[0, 11, 11, 0], [0, 0, 10, 10]]])
    misses = np.array([[2, 2], [2, 2]])
    missed = np.array([[0, 2], [0, 2]])
    out, _ = welfare.comparisons(scores, misses, missed, [0, 1], ["self", "s3"])
    assert out["s3"]["hero_win_self"]["mean"] == 1
    assert out["s3"]["hero_win_condition"]["mean"] == 0
    assert out["s3"]["table_loss"]["mean"] == 2
    assert out["s3"]["table_loss"]["n"] == 2


def test_ratio_retains_pairing():
    result = welfare.ratio_ci([2, 4, 6], [1, 2, 3])
    assert result["mean"] == 2
    assert result["half95"] == 0
    with pytest.raises(ValueError):
        welfare.ratio_ci([1, 2], [0, 0])

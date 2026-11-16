"""Round 3 (H3): counterexample showing the information sandwich is strict on a deal pair.

NT, n = 2, dealer 0, hero = seat 1 (leads), bids hero 1, others 0.  Seat 2 holds KH 3C in both deals;
the hero holds AS 2H (deal A) or AS 2C (deal B); seats 3 and 0 hold diamonds.  Perfect-information
paranoid value is a hero MISS (0) in both deals, but the spoiling discard on the AS lead is KH in deal A
and 3C in deal B, and seat 2's information (own hand, bids, AS led) is identical in the two deals.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import make_state  # noqa: E402
from judgement.cards import NT, parse_card  # noqa: E402
from judgement.solver.core import Paranoid  # noqa: E402

DEALS = {"A": ["6D 7D", "AS 2H", "KH 3C", "4D 5D"], "B": ["6D 7D", "AS 2C", "KH 3C", "4D 5D"]}
BIDS = [1, 0, 0, 0]          # seats 1, 2, 3, 0


def _par(st):
    return Paranoid(st.rules, st.trump, st.bids, 1, st.N).value(st)


def _after_discard(deal, discard):
    st = make_state(DEALS[deal], NT, bids=BIDS)
    st.apply_card(parse_card("AS"))
    st.apply_card(parse_card(discard))
    return _par(st)


def test_pi_paranoid_value_is_miss_in_both():
    for d in DEALS:
        assert _par(make_state(DEALS[d], NT, bids=BIDS)) == 0


def test_correct_discard_differs_between_indistinguishable_deals():
    assert _after_discard("A", "KH") == 0 and _after_discard("A", "3C") == 11
    assert _after_discard("B", "3C") == 0 and _after_discard("B", "KH") == 11
    # seat 2's view is identical at its decision in A and B
    va = make_state(DEALS["A"], NT, bids=BIDS)
    vb = make_state(DEALS["B"], NT, bids=BIDS)
    for st in (va, vb):
        st.apply_card(parse_card("AS"))
    ia, ib = va.info_state(2), vb.info_state(2)
    assert ia.hand == ib.hand and ia.history == ib.history and ia.bids == ib.bids

"""Tests for M2 (1-card round closed forms) and M3(a) (hook bound)."""
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math"))

from judgement import Rules  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.state import RoundState  # noqa: E402
from m2_onecard import leader_pwin, winners  # noqa: E402


def test_leader_threshold():
    t = Fraction(10, 21)
    assert leader_pwin(1) < t < leader_pwin(2)          # '3' < 10/21 < '4'
    assert abs(float(leader_pwin(2)) - 0.512) < 1e-3
    assert abs(float(leader_pwin(1)) - 0.474) < 1e-3
    assert Fraction(comb(38, 3), comb(51, 3)) < t       # off-trump ace never bids 1


def test_leader_value_closed_form():
    a = [Fraction(comb(39 + r, 3), comb(51, 3)) for r in range(13)]
    b = [Fraction(comb(26 + r, 3), comb(51, 3)) for r in range(13)]
    v = (sum(max(11 * x, 10 * (1 - x)) for x in a) + 3 * sum(10 * (1 - x) for x in b)) / 52
    assert v == Fraction(1638607, 216580)
    vnt = sum(max(11 * x, 10 * (1 - x)) for x in a) / 13
    assert vnt == Fraction(418063, 54145)
    PL = Fraction(comb(52, 4) - comb(39, 4), 13 * comb(51, 3))
    assert 10 * (1 - (1 - PL) / 3) == Fraction(37432, 4165)


def test_winner_vectorised_matches_engine():
    rules = Rules()
    rng = np.random.default_rng(0)
    for tr in (0, -1):
        rows = []
        exp = []
        for _ in range(300):
            hands = deal(rules, 1, rng)
            cards = [h.bit_length() - 1 for h in hands]
            # dealer 0 -> seat 1 leads; pos j = seat (1+j)%4
            order = [cards[(1 + j) % 4] for j in range(4)]
            st = RoundState(rules, 1, tr, 0, hands)
            for _ in range(4):
                st.apply_bid(0) if st.legal_bids()[0] == 0 else st.apply_bid(st.legal_bids()[0])
            for c in order:
                st.apply_card(c)
            rows.append(order)
            exp.append((st.trick_winners[0] - 1) % 4)
        assert (winners(np.array(rows), tr) == np.array(exp)).all()


def test_hook_at_most_N_minus_1_hit():
    G = np.load(ROOT / "experiments/phase1/results/games_ta.npz")
    hits = (G["bids"] == G["tricks"]).sum(2)
    assert hits.max() <= 3
    lad = np.array(list(range(13, 0, -1)) + list(range(2, 14)))
    assert (G["bids"].astype(int).sum(2) != lad[None, :]).all()
    assert (G["tricks"].astype(int).sum(2) == lad[None, :]).all()

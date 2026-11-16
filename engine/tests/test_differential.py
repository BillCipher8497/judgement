"""Differential test against the deliberately naive reference (crossval/naive_ref.py).

Covers what OpenSpiel cannot: no-trump rounds, 13-card rounds at 4 players,
and every N in 3..7 with full-size hands."""
import numpy as np
import pytest

from crossval import naive_ref
from judgement.cards import NT, SUITS, card_str, cards_of
from judgement.engine import deal
from judgement.rules import Rules
from judgement.state import RoundState


@pytest.mark.parametrize("N", range(3, 8))
def test_vs_naive(N):
    rules = Rules(n_players=N)
    M = 52 // N
    root = np.random.SeedSequence(1000 + N)
    for child in root.spawn(300):
        rng = np.random.default_rng(child)
        n = 1 + int(rng.integers(M))
        trump = [0, 1, 2, 3, NT][int(rng.integers(5))]
        dealer = int(rng.integers(N))
        hands = deal(rules, n, rng)
        st = RoundState(rules, n, trump, dealer, hands)
        nv = naive_ref.NaiveRound([[card_str(c) for c in cards_of(h)] for h in hands],
                                  None if trump == NT else SUITS[trump], dealer)
        while st.phase == 0:
            assert st.to_act == nv.next_bidder()
            assert st.legal_bids() == nv.legal_bids()
            lb = st.legal_bids()
            b = lb[int(rng.integers(len(lb)))]
            st.apply_bid(b); nv.bid(b)
        while not st.is_terminal:
            assert st.to_act == nv.next_player()
            legal = cards_of(st.legal_cards())
            assert {card_str(c) for c in legal} == nv.legal_cards()
            c = legal[int(rng.integers(len(legal)))]
            k = len(st.trick_winners)
            st.apply_card(c)
            w = nv.play(card_str(c))
            if len(st.trick_winners) > k:
                assert w == st.trick_winners[-1]
        assert st.tricks_won == nv.tricks
        assert st.scores() == [naive_ref.score(b, t) for b, t in zip(st.bids, st.tricks_won)]

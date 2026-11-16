"""Property-based invariants over random rounds for N = 3..7."""
import numpy as np
from hypothesis import given, settings, strategies as st_

from judgement.cards import FULL_DECK, NT, cards_of
from judgement.engine import deal
from judgement.policies import GreedyPolicy, RandomPolicy
from judgement.rules import Rules
from judgement.state import BID, DONE, InfoState, RoundState


@st_.composite
def round_setup(draw):
    N = draw(st_.integers(3, 7))
    M = 52 // N
    n = draw(st_.integers(1, M))
    trump = draw(st_.sampled_from([0, 1, 2, 3, NT]))
    dealer = draw(st_.integers(0, N - 1))
    seed = draw(st_.integers(0, 2**32 - 1))
    hook = draw(st_.booleans())
    mix = draw(st_.lists(st_.booleans(), min_size=N, max_size=N))
    return N, n, trump, dealer, seed, hook, mix


@settings(max_examples=400, deadline=None)
@given(round_setup())
def test_round_invariants(setup):
    N, n, trump, dealer, seed, hook, mix = setup
    rules = Rules(n_players=N, hook=hook)
    rng = np.random.default_rng(seed)
    hands = deal(rules, n, rng)
    st = RoundState(rules, n, trump, dealer, hands)
    pols = [GreedyPolicy() if m else RandomPolicy() for m in mix]
    infos = [InfoState(st, p) for p in range(N)]

    while st.phase == BID:
        legal = st.legal_bids()
        assert legal, "every bidder (incl. dealer) has a legal bid"
        if len(st.bid_seq) == N - 1:
            assert st.to_act == dealer
        b = pols[st.to_act].bid(infos[st.to_act], legal, rng)
        st.apply_bid(b)
    if hook:
        assert sum(st.bids) != n

    prev_winner = None
    trick_start = True
    while st.phase != DONE:
        if trick_start and prev_winner is not None:
            assert st.to_act == prev_winner, "trick winner leads next"
        if trick_start and prev_winner is None:
            assert st.to_act == (dealer + 1) % N
        legal = st.legal_cards()
        assert legal != 0, "legal_cards never empty"
        assert legal & ~st.hands[st.to_act] == 0
        p = st.to_act
        c = pols[p].play(infos[p], legal, rng)
        n_w = len(st.trick_winners)
        st.apply_card(c)
        trick_start = len(st.trick_winners) > n_w
        if trick_start:
            prev_winner = st.trick_winners[-1]

    played = [c for _, c in st.history]
    assert len(played) == len(set(played)) == n * N, "every card played exactly once"
    dealt = 0
    for h in hands:
        dealt |= h
    assert sum(1 << c for c in played) == dealt
    assert all(h == 0 for h in st.hands)
    assert sum(st.tricks_won) == n
    assert len(st.trick_winners) == n
    # each player played exactly one card per trick
    for t in range(n):
        seats = sorted(p for p, _ in st.history[t * N:(t + 1) * N])
        assert seats == list(range(N))


@settings(max_examples=60, deadline=None)
@given(st_.integers(3, 7), st_.integers(0, 2**32 - 1))
def test_full_game_invariants(N, seed):
    from judgement.engine import play_game
    rules = Rules(n_players=N)
    g = play_game(rules, [RandomPolicy()] * N, seed=seed, record_trace=False)
    assert len(g.rounds) == rules.n_rounds
    for i, rr in enumerate(g.rounds):
        assert rr.n_cards == rules.ladder()[i]
        assert rr.trump == rules.trump_for_round(i)
        assert rr.dealer == i % N
        assert sum(rr.tricks) == rr.n_cards
        assert sum(rr.bids) != rr.n_cards
    assert abs(sum(g.wins) - 1.0) < 1e-12
    assert g.totals == [sum(rr.scores[p] for rr in g.rounds) for p in range(N)]

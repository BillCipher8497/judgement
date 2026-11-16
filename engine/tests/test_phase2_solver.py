"""Phase 2: exact perfect-information solvers (max^n / paranoid / social), bidding, PIMC."""
import numpy as np
import pytest

from judgement import NT, Rules, RoundState, parse_hand
from judgement.cards import cards_of, parse_card
from judgement.engine import deal
from judgement.solver import (BENEVOLENT, MODES, NEUTRAL, SPITEFUL, MaxN, Paranoid, Social, maxn_value,
                              move_classes, paranoid_value, social_value)
from judgement.solver.bidding import greedy_trick_bids, legal_bid_vectors, make_play_state, rational_bids
from judgement.solver.bruteforce import bf_maxn, bf_paranoid, bf_social
from judgement.solver.heuristic import TargetHeuristic, heuristic_card
from judgement.solver.pimc import PIMCPolicy
from judgement.state import PLAY

R4 = Rules()


def state(hands, trump, dealer, bids, rules=R4):
    masks = [parse_hand(h) for h in hands]
    st = RoundState(rules, masks[0].bit_count(), trump, dealer, masks)
    order = [(dealer + 1 + k) % rules.n_players for k in range(rules.n_players)]
    for p in order:
        st.apply_bid(bids[p])
    return st


def random_position(rng, n, rules=R4, partial=True):
    trump = [0, 1, 2, 3, NT][int(rng.integers(5))]
    st = RoundState(rules, n, trump, int(rng.integers(rules.n_players)), deal(rules, n, rng))
    while st.phase == 0:
        L = st.legal_bids()
        st.apply_bid(L[int(rng.integers(len(L)))])
    if partial:
        for _ in range(int(rng.integers(0, rules.n_players * n))):
            L = cards_of(st.legal_cards())
            st.apply_card(L[int(rng.integers(len(L)))])
            if st.phase != PLAY:
                break
    return st


# ---- hand-constructed positions with known answers --------------------------------
def test_top_cards_secure():
    # seat 1 leads with AS KS (trumps) and bids 2: hits whatever others do.
    st = state(["2H 3H", "AS KS", "4H 5H", "6H 7H"], trump=0, dealer=0, bids=[0, 2, 0, 1])
    assert paranoid_value(st, 1) == 12
    assert maxn_value(st)[1] == 12


def test_one_card_forced():
    # NT, leader seat 1 leads 2H; seat 2's 3H wins (others void in H).
    st = state(["AS", "2H", "3H", "KS"], trump=NT, dealer=0, bids=[1, 0, 1, 0])
    for m in MODES:
        assert maxn_value(st, m) == (0, 10, 11, 10)
    assert social_value(st) == 31
    assert [paranoid_value(st, p) for p in range(4)] == [0, 10, 11, 10]


def test_dead_player_value():
    # seat 0 bid 2 with two deuces of a non-trump suit led into by aces: cannot hit.
    st = state(["2H 3H", "AH KH", "QH JH", "TH 9H"], trump=0, dealer=3, bids=[2, 2, 0, 1])
    assert maxn_value(st)[0] == 0 and paranoid_value(st, 0) == 0


def test_gift_position():
    """NT, dealer 2, seat 3 leads holding 3S 2H with bid 1.  Seat 0 (bid 0) holds
    2S AS: on a 3S lead it ducks with 2S out of self-interest, gifting seat 3 its
    trick.  Paranoid: seat 0 overtakes with AS and also wins the 2S -> seat 3 gets 0.
    max^n (neutral): seats 0, 1, 3 hit; seat 2 (bid 1, hook) is dead.  (10, 11, 0, 11)."""
    st = state(["2S AS", "KH QH", "JH TH", "3S 2H"], trump=NT, dealer=2, bids=[0, 1, 1, 1])
    assert paranoid_value(st, 3) == 0 == bf_paranoid(st.copy(), 3)
    assert maxn_value(st) == (10, 11, 0, 11) == bf_maxn(st.copy())
    assert social_value(st) == 32


def test_move_classes():
    # own 5S 6S 8S, others hold 7S -> classes {5,6} {8}
    own = parse_hand("5S 6S 8S")
    cls = move_classes(own, parse_hand("7S 2H"))
    assert cls == [[parse_card("5S"), parse_card("6S")], [parse_card("8S")]]


# ---- brute-force cross-check (exhaustive tree, engine legality) -------------------
@pytest.mark.parametrize("seed", range(6))
def test_bruteforce_crosscheck(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(40):
        st = random_position(rng, int(rng.integers(1, 4)))
        if st.phase != PLAY:
            continue
        for m in MODES:
            assert maxn_value(st, m) == bf_maxn(st.copy(), m)
        for t in range(4):
            assert paranoid_value(st, t) == bf_paranoid(st.copy(), t)
        assert social_value(st) == bf_social(st.copy())


@pytest.mark.parametrize("miss", ["neg_bid", "neg_10", "neg_diff"])
def test_bruteforce_other_scoring(miss):
    rules = Rules(miss_penalty=miss)
    rng = np.random.default_rng(7)
    for _ in range(40):
        st = random_position(rng, int(rng.integers(1, 4)), rules)
        if st.phase != PLAY:
            continue
        for m in MODES:
            assert maxn_value(st, m) == bf_maxn(st.copy(), m)
        for t in range(4):
            assert paranoid_value(st, t) == bf_paranoid(st.copy(), t)
        assert social_value(st) == bf_social(st.copy())


def test_three_players():
    rules = Rules(n_players=3)
    rng = np.random.default_rng(3)
    for _ in range(40):
        st = random_position(rng, int(rng.integers(1, 4)), rules)
        if st.phase != PLAY:
            continue
        assert maxn_value(st) == bf_maxn(st.copy())
        assert paranoid_value(st, 0) == bf_paranoid(st.copy(), 0)


def test_invariants_and_move_values():
    rng = np.random.default_rng(11)
    for _ in range(60):
        st = random_position(rng, int(rng.integers(2, 6)))
        if st.phase != PLAY:
            continue
        for m in MODES:
            s = MaxN(st.rules, st.trump, st.bids, 4, m)
            v = s.value(st)
            mv = s.move_values(st)
            assert set(mv) == set(cards_of(st.legal_cards()))
            assert mv[s.best_move(st)] == v
            assert max(x[st.to_act] for x in mv.values()) == v[st.to_act]
            assert all(v[t] >= paranoid_value(st, t) for t in range(4))   # max^n >= security
            assert social_value(st) >= sum(v)
        # playing out max^n moves reproduces the value
        s = MaxN(st.rules, st.trump, st.bids, 4)
        v = s.value(st)
        st2 = st.copy()
        while st2.phase == PLAY:
            st2.apply_card(s.best_move(st2))
        assert tuple(st2.scores()) == v


# ---- bidding ---------------------------------------------------------------------
def _bf_rational(rules, n, trump, dealer, hands):
    N = rules.n_players
    order = [(dealer + 1 + k) % N for k in range(N)]

    def rec(seq):
        if len(seq) == N:
            v = [0] * N
            for p, b in zip(order, seq):
                v[p] = b
            return tuple(bf_maxn(make_play_state(rules, n, trump, dealer, hands, v))), seq
        p = order[len(seq)]
        best = None
        for b in range(n + 1):
            if len(seq) == N - 1 and rules.hook and sum(seq) + b == n:
                continue
            r = rec(seq + (b,))
            if best is None or r[0][p] > best[0][p]:
                best = r
        return best
    return rec(())[0]


def test_rational_bids_bruteforce():
    rng = np.random.default_rng(5)
    for i in range(12):
        n = 1 + i % 2
        hands = deal(R4, n, rng)
        bids, val = rational_bids(R4, n, i % 5 - 1 if i % 5 == 0 else i % 4, i % 4, hands)
        tr = i % 5 - 1 if i % 5 == 0 else i % 4
        assert val == _bf_rational(R4, n, tr, i % 4, hands)
        assert sum(bids) != n
        st = make_play_state(R4, n, tr, i % 4, hands, bids)
        assert maxn_value(st) == val


def test_legal_bid_vectors_and_greedy_bids():
    v = legal_bid_vectors(R4, 2, 0)
    assert len(v) == 3 ** 4 - sum(1 for a in range(3) for b in range(3) for c in range(3) if 0 <= 2 - a - b - c <= 2)
    assert all(sum(x) != 2 for x in v)
    rng = np.random.default_rng(0)
    hands = deal(R4, 4, rng)
    b = greedy_trick_bids(R4, 4, 0, 1, hands)
    assert sum(b) != 4 and all(0 <= x <= 4 for x in b)


# ---- policies ----------------------------------------------------------------------
def test_heuristic_and_pimc_play_legal_rounds():
    from judgement.engine import make_rngs, run_round
    rng = np.random.default_rng(2)
    for n in (1, 3, 5):
        st = RoundState(R4, n, 1, 0, deal(R4, n, rng))
        _, rngs = make_rngs(9, 4)
        pols = [PIMCPolicy(n_samples=4), TargetHeuristic(), TargetHeuristic(), TargetHeuristic()]
        run_round(st, pols, rngs, check=True)
        assert sum(st.tricks_won) == n


def test_heuristic_need_zero_sheds_highest_loser():
    # seat 2 at target follows a 9H lead (winning), holding 8H 2H TH: highest loser = 8H
    st = state(["3S 4S", "9H 5S", "8H 2H", "6S 7S"], trump=0, dealer=0, bids=[1, 1, 0, 1])
    st.apply_card(parse_card("9H"))
    assert heuristic_card(st.info_state(2), st.legal_cards()) == parse_card("8H")


def test_rational_bids_bruteforce_n3():
    rng = np.random.default_rng(55)
    for i in range(4):
        hands = deal(R4, 3, rng)
        tr = [0, NT, 2, 3][i]
        bids, val = rational_bids(R4, 3, tr, i, hands)
        assert val == _bf_rational(R4, 3, tr, i, hands)


def _bf_tricks(st, p, agg):
    if st.phase != PLAY:
        return st.tricks_won[p]
    vals = []
    for c in cards_of(st.legal_cards()):
        st.apply_card(c)
        vals.append(_bf_tricks(st, p, agg))
        st.undo_card()
    return agg(vals)


def test_trick_range():
    from judgement.solver.core import TrickRange
    rng = np.random.default_rng(8)
    for _ in range(30):
        st = random_position(rng, 3, partial=False)
        tr = TrickRange(4, st.trump)
        for p in range(4):
            assert tr.range(st.hands, st.leader, p, 3) == (_bf_tricks(st.copy(), p, min), _bf_tricks(st.copy(), p, max))


def test_abstract_actions_and_record():
    from judgement.solver.features import abstract_actions, decision_record
    # seat 2 follows a 9H lead (trump S) holding TH 8H 2H: CHEAPWIN=TH, HIGHLOSE=8H, LOWLOSE=2H
    st = state(["3S 4S 5D", "9H 5S 6D", "TH 8H 2H", "6S 7S 7D"], trump=0, dealer=0, bids=[1, 1, 0, 0])
    st.apply_card(parse_card("9H"))
    a = abstract_actions(st.hands[2], st.legal_cards(), parse_hand("9H"), st.best_card, st.trump)
    assert a["CHEAPWIN"] == parse_card("TH") and a["HIGHLOSE"] == parse_card("8H") and a["LOWLOSE"] == parse_card("2H")
    mv = MaxN(st.rules, st.trump, st.bids, 4).move_values(st)
    rec = decision_record(st, mv, heuristic_card(st.info_state(2), st.legal_cards()))
    assert rec["status"] == "zero" and rec["can_win"] == 1 and rec["vmax"] == max(v[2] for v in mv.values())

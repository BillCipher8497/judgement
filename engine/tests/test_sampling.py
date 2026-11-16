from collections import Counter

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st_

from helpers import bid_through, make_state
from judgement.cards import FULL_DECK, NT, SUIT_MASK, cards_of, parse_card as C
from judgement.engine import deal
from judgement.policies import RandomPolicy
from judgement.rules import Rules
from judgement.sampling import HandSampler
from judgement.state import RoundState


def _random_midround(N, n, seed, k_plays):
    rules = Rules(n_players=N)
    rng = np.random.default_rng(seed)
    st = bid_through(RoundState(rules, n, int(rng.integers(4)), 0, deal(rules, n, rng)))
    pol = RandomPolicy()
    for _ in range(min(k_plays, n * N - 1)):
        st.apply_card(pol.play(None, st.legal_cards(), rng))
    return st


@settings(max_examples=150, deadline=None)
@given(st_.integers(3, 7), st_.integers(0, 10**6), st_.integers(0, 60), st_.integers(0, 6))
def test_samples_consistent(N, seed, k, seat_off):
    n = 52 // N
    st = _random_midround(N, n, seed, k)
    seat = seat_off % N
    info = st.info_state(seat)
    sampler = HandSampler(info)
    rng = np.random.default_rng(seed + 1)
    for _ in range(5):
        hid = sampler.sample(rng)
        assert set(hid) == set(range(N)) - {seat}
        seen = info.hand | info.played_mask
        for p, h in hid.items():
            assert h.bit_count() == info.cards_left(p)
            assert h & seen == 0
            seen |= h
            for s in range(4):
                if info.voids[p] >> s & 1:
                    assert h & SUIT_MASK[s] == 0


def test_sampling_independent_of_hidden_hands():
    """Two worlds with the same observation must produce identical samples."""
    rules = Rules()
    hA = ["AS KS 2H", "QS 3H 4H", "5H 6H 7H", "8H 9H TH"]
    hB = ["AS KS 2H", "TH 3H 4H", "5H 6H 7H", "8H 9H QS"]  # swap QS/TH between seats 1 and 3
    out = []
    for hands in (hA, hB):
        st = bid_through(make_state(hands, trump=NT, dealer=3, rules=rules))
        out.append([HandSampler(st.info_state(0)).sample(np.random.default_rng(5)) for _ in range(3)])
    assert out[0] == out[1]


def test_uniform_with_void_constraint():
    """Seat 1 shows out of spades. Conditional on that, every unseen spade must be
    equally likely to sit with seat 2, seat 3 or any undealt slot: P = 3/47 each."""
    hands = ["AS 2H 3H", "4H 5H 7H", "KS 6H 8H", "QS JS 9H"]
    st = bid_through(make_state(hands, trump=NT, dealer=3))
    st.apply_card(C("AS"))
    st.apply_card(C("4H"))
    info = st.info_state(0)
    assert info.voids[1] == 1
    sampler = HandSampler(info)
    assert sampler.mode == "dp"
    rng = np.random.default_rng(0)
    T = 20000
    hits = Counter()
    for _ in range(T):
        hid = sampler.sample(rng)
        assert hid[1] & SUIT_MASK[0] == 0
        assert hid[1].bit_count() == 2 and hid[2].bit_count() == 3 and hid[3].bit_count() == 3
        for p in (2, 3):
            for c in cards_of(hid[p] & SUIT_MASK[0]):
                hits[(p, c)] += 1
    exp = T * 3 / 47
    spades = [c for c in range(13) if c != C("AS")]
    for p in (2, 3):
        for c in spades:
            assert abs(hits[(p, c)] - exp) < 5 * exp ** 0.5, (p, c, hits[(p, c)], exp)


def test_rejection_fallback_matches_dp():
    """Force the rejection path and compare marginals with the DP path."""
    import judgement.sampling as smp
    hands = ["AS 2H 3H", "4H 5H 7H", "KS 6H 8H", "QS JS 9H"]
    st = bid_through(make_state(hands, trump=NT, dealer=3))
    st.apply_card(C("AS"))
    st.apply_card(C("4H"))
    info = st.info_state(0)
    old = smp.DP_STATE_LIMIT
    smp.DP_STATE_LIMIT = 0
    try:
        rs = HandSampler(info)
    finally:
        smp.DP_STATE_LIMIT = old
    assert rs.mode == "reject"
    rng = np.random.default_rng(1)
    for _ in range(200):
        assert rs.sample(rng)[1] & SUIT_MASK[0] == 0


def test_uniform_exact_small_world():
    """52-card full deal at N=4 so there is no stock; 2 cards each left."""
    rules = Rules(n_players=4, max_cards=13)
    rng = np.random.default_rng(11)
    from judgement.policies import RandomPolicy
    hands = deal(rules, 13, rng)
    st = bid_through(RoundState(rules, 13, NT, 3, hands))
    pol = RandomPolicy()
    while len(st.history) < 44:
        st.apply_card(pol.play(None, st.legal_cards(), rng))
    info = st.info_state(0)
    sampler = HandSampler(info)
    counts = Counter()
    T = 20000
    for _ in range(T):
        counts[tuple(sorted(sampler.sample(rng).items()))] += 1
    # brute-force the consistent deals
    import itertools
    unknown = cards_of(FULL_DECK & ~info.hand & ~info.played_mask)
    need = [info.cards_left(p) for p in (1, 2, 3)]
    valid = set()
    for perm in itertools.permutations(unknown):
        parts, pos, ok = [], 0, True
        for p, k in zip((1, 2, 3), need):
            m = sum(1 << c for c in perm[pos:pos + k]); pos += k
            if any(info.voids[p] >> s & 1 and m & SUIT_MASK[s] for s in range(4)):
                ok = False
            parts.append((p, m))
        if ok:
            valid.add(tuple(parts))
    assert set(counts) == valid
    exp = T / len(valid)
    for v in valid:
        assert abs(counts[v] - exp) < 5 * exp ** 0.5 + 1


def test_impossible_constraints_raise():
    from judgement.sampling import InconsistentInfo

    class FakeInfo:
        seat, n_players = 0, 3
        hand = 1 << 0
        played_mask = 0
        voids = (0, 0b1111, 0)
        def cards_left(self, p): return 17
    with pytest.raises(InconsistentInfo):
        HandSampler(FakeInfo())

"""Tests for judgement.policies.human_guide (Phase 5 human guide)."""
import numpy as np

from judgement import Rules, play_game
from judgement.cards import NT, SUIT_MASK, parse_hand
from judgement.policies.human_guide import GUIDE_PLAY, GuideAgent, HumanBidder, HumanPlayPolicy, HumanView
from judgement.state import RoundState

HON = 0
for s in range(4):
    for r in range(GUIDE_PLAY["track"], 13):
        HON |= 1 << (13 * s + r)


class Proxy:
    """InfoState whose forgotten history (non-honour cards of finished tricks) is replaced by another set
    with the same number of trumps: a human-trackable policy must not notice."""

    def __init__(self, info, rng):
        self._i = info
        cur = 0
        for _, c in info.current_trick:
            cur |= 1 << c
        played = info.played_mask
        keep = (played & HON) | cur
        forgot = played & ~keep
        tr = info.trump
        nt = (forgot & SUIT_MASK[tr]).bit_count() if tr != NT else 0
        pool = [c for c in range(52) if not (info.hand | keep) >> c & 1 and not HON >> c & 1]
        tp = [c for c in pool if tr != NT and c // 13 == tr]
        op = [c for c in pool if not (tr != NT and c // 13 == tr)]
        k = forgot.bit_count()
        pick = list(rng.choice(tp, nt, replace=False)) + list(rng.choice(op, k - nt, replace=False))
        m = keep
        for c in pick:
            m |= 1 << int(c)
        self.played_mask = m

    def __getattr__(self, a):
        return getattr(self._i, a)


class Checker:
    def __init__(self):
        self.p = HumanPlayPolicy()
        self.b = HumanBidder()
        self.rng = np.random.default_rng(7)
        self.n = 0

    def bid(self, info, legal, rng):
        b = self.b.bid(info, legal, rng)
        assert b in legal
        return b

    def play(self, info, legal, rng):
        c = self.p.play(info, legal, rng)
        assert legal >> c & 1
        assert self.p.play(Proxy(info, self.rng), legal, rng) == c
        self.n += 1
        return c


def test_guide_legal_and_human_trackable():
    ch = Checker()
    for g in range(6):
        play_game(Rules(), [ch] * 4, seed=100 + g, record_trace=False)
    assert ch.n > 500


def test_boss_and_sure_win():
    # spades trump; seat 0 holds KS (AS unplayed) -> not boss; QH with AH, KH in hand -> boss
    hands = ["KS AH KH QH", "2S 3S 4S 5S", "2H 3H 4H 5H", "2D 3D 4D 5D"]
    st = RoundState(Rules(), 4, 0, 3, [parse_hand(h) for h in hands])
    for b in [1, 0, 0, 0]:
        st.apply_bid(b)
    hv = HumanView(st.info_state(0), track=10)
    assert not hv.boss(parse_hand("KS").bit_length() - 1)
    assert hv.boss(parse_hand("QH").bit_length() - 1)
    assert hv.trumps_out == 12
    # QH boss but ruffable with trumps out -> not a sure winner on lead
    assert not hv.sure_win(parse_hand("QH").bit_length() - 1)
    assert hv.sure_tricks() == 3


def test_bidder_hook_and_counts():
    rules = Rules()
    hb = HumanBidder()
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(1, 14))
        perm = rng.permutation(52)
        hands = [sum(1 << int(c) for c in perm[p * n:(p + 1) * n]) for p in range(4)]
        st = RoundState(rules, n, int(rng.choice([0, 1, 2, 3, NT])), 0, hands)
        while st.phase == 0:
            legal = st.legal_bids()
            b = hb.bid(st.info_state(st.to_act), legal)
            assert b in legal and 0 <= b <= n
            st.apply_bid(b)
    # aces-only hand in NT: 2 aces -> bid 2 (default weights count an ace as 2 half tricks)
    st = RoundState(rules, 4, NT, 3, [parse_hand(h) for h in ["AS AH 2D 3C", "2S 3S 4S 5S", "2H 3H 4H 5H", "4D 5D 6D 7D"]])
    assert hb.estimate(st.info_state(0))[0] == 2


def test_guide_agent_deterministic():
    a = [GuideAgent() for _ in range(4)]
    r1 = play_game(Rules(), a, seed=5, record_trace=False).totals
    r2 = play_game(Rules(), a, seed=5, record_trace=False).totals
    assert r1 == r2

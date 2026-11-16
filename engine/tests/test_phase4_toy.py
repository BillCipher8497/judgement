"""Phase 4: toy-game logic, exact best response / NashConv, CFR+, tabular heuristics."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judgement import Rules, RoundState  # noqa: E402
from judgement.cfr.heuristics import _pwin_cached, myopic_bid, profile, trick_dist_greedy  # noqa: E402
from judgement.cfr.solve import CFRPlus, best_response, nash_conv, profile_value, values  # noqa: E402
from judgement.cfr.toy import NT, ToyGame, Tree  # noqa: E402


def real(g, c):
    """toy card -> engine card (same suit, top ranks)."""
    return g.suit(c) * 13 + (13 - g.R) + c % g.R


@pytest.mark.parametrize("N,S,R,n,trump", [(3, 2, 4, 2, 0), (4, 2, 4, 2, NT), (4, 3, 3, 2, 1), (3, 3, 4, 3, NT)])
def test_rules_match_engine(N, S, R, n, trump):
    g = ToyGame(N=N, S=S, R=R, n=n, trump=trump)
    rng = np.random.default_rng(1)
    rules = Rules(n_players=N)
    for _ in range(300):
        deck = rng.permutation(g.deck)
        hands = [sorted(deck[i * n:(i + 1) * n].tolist()) for i in range(N)]
        st = RoundState(rules, n, trump, N - 1, [sum(1 << real(g, c) for c in h) for h in hands])
        bids = []
        for p in range(N):
            lb = g.legal_bids(bids)
            assert lb == st.legal_bids()
            b = int(rng.choice(lb))
            bids.append(b)
            st.apply_bid(b)
        left = [list(h) for h in hands]
        trick, leader, tricks = [], 0, [0] * N
        for _ in range(N * n):
            p = (leader + len(trick)) % N
            assert st.current_player == p
            lc = g.legal_cards(left[p], trick)
            assert sum(1 << real(g, c) for c in lc) == st.legal_cards()
            c = int(rng.choice(lc))
            left[p].remove(c)
            trick.append((p, c))
            st.apply_card(real(g, c))
            if len(trick) == N:
                leader = g.trick_winner(trick)
                tricks[leader] += 1
                trick = []
        assert st.is_terminal
        assert list(st.scores()) == g.scores(bids, tricks)


def _eval_direct(g, key2is, sigma):
    """Independent recursive evaluation (no tree arrays, no forced-move compression)."""
    N = g.N
    deals = g.deals()
    tot = np.zeros(N)

    def rec(hands, left, bids, trick, leader, tricks, hist):
        L = len(hist)
        if L < N:
            legal = g.legal_bids(bids)
            if len(legal) == 1:
                return rec(hands, left, bids + [legal[0]], trick, leader, tricks, hist + (legal[0],))
            s = key2is[(L, hands[L], hist)]
            return sum(sigma[s, b] * rec(hands, left, bids + [b], trick, leader, tricks, hist + (b,))
                       for b in legal if sigma[s, b] > 0)
        if L == N + N * g.n:
            return np.array(g.utility(bids, tricks), float)
        p = (leader + len(trick)) % N
        legal = g.legal_cards(left[p], trick)
        out = np.zeros(N)
        for c in legal:
            pr = 1.0 if len(legal) == 1 else sigma[key2is[(p, hands[p], hist)], hands[p].index(c)]
            if pr == 0:
                continue
            l2 = [list(x) for x in left]
            l2[p].remove(c)
            t2, ld, tr = trick + [(p, c)], leader, list(tricks)
            if len(t2) == N:
                ld = g.trick_winner(t2)
                tr[ld] += 1
                t2 = []
            out = out + pr * rec(hands, l2, bids, t2, ld, tr, hist + (c,))
        return out
    for h in deals:
        tot += rec(h, [list(x) for x in h], [], [], 0, [0] * N, ())
    return tot / len(deals)


def _random_profile(t, rng):
    x = rng.random(t.is_valid.shape) * t.is_valid
    return x / x.sum(1, keepdims=True)


@pytest.mark.parametrize("g", [ToyGame(N=3, S=2, R=3, n=2, trump=0), ToyGame(N=4, S=2, R=3, n=1, trump=NT),
                               ToyGame(N=2, S=2, R=3, n=2, trump=1)])
def test_values_match_direct(g):
    t = Tree(g)
    key2is = {k: i for i, k in enumerate(t.is_key)}
    sig = _random_profile(t, np.random.default_rng(0))
    np.testing.assert_allclose(profile_value(t, sig), _eval_direct(g, key2is, sig), atol=1e-10)


def test_br_bruteforce_pure():
    g = ToyGame(N=3, S=2, R=2, n=1, trump=0)
    t = Tree(g)
    rng = np.random.default_rng(2)
    sig = _random_profile(t, rng)
    for p in range(3):
        mine = np.flatnonzero(t.is_pl == p)
        acts = [np.flatnonzero(t.is_valid[s]) for s in mine]
        best = -np.inf
        for combo in itertools.product(*acts):
            s2 = sig.copy()
            s2[mine] = 0
            s2[mine, combo] = 1
            best = max(best, profile_value(t, s2)[p])
        assert best == pytest.approx(best_response(t, sig, p), abs=1e-10)


@pytest.mark.parametrize("g", [ToyGame(N=3, S=2, R=3, n=2), ToyGame(N=4, S=2, R=3, n=1, trump=NT)])
def test_br_policy_consistent(g):
    t = Tree(g)
    rng = np.random.default_rng(3)
    sig = _random_profile(t, rng)
    for p in range(g.N):
        v, pol = best_response(t, sig, p, return_policy=True)
        assert profile_value(t, pol)[p] == pytest.approx(v, abs=1e-10)
        assert v >= profile_value(t, sig)[p] - 1e-12
        for _ in range(20):   # no random mixed deviation beats the BR
            s2 = sig.copy()
            m = t.is_pl == p
            s2[m] = _random_profile(t, rng)[m]
            assert profile_value(t, s2)[p] <= v + 1e-10
        # restricted BRs are bounded by the full BR and sum-compatible
        vb = best_response(t, sig, p, free=t.is_bid)
        vp = best_response(t, sig, p, free=~t.is_bid)
        assert max(vb, vp) <= v + 1e-10


def test_cfr_2p_zero_sum_converges():
    t = Tree(ToyGame(N=2, S=2, R=4, n=2, util="diff"))
    c = CFRPlus(t).iterate(200)
    nc, gains, vals = nash_conv(t, c.average())
    assert nc < 0.01
    assert vals.sum() == pytest.approx(0, abs=1e-9)


def test_myopic_threshold_n1():
    g = ToyGame(N=4, S=2, R=4, n=1, trump=0)
    for c in g.deck:
        p = trick_dist_greedy(g, 0, (c,))[1]
        assert myopic_bid(g, 0, (c,), ()) == int(p > 10 / 21)
    # leader with the top trump always wins
    assert trick_dist_greedy(g, 0, (3,))[1] == 1.0


def test_heuristic_profiles_legal_and_pwin():
    g = ToyGame(N=3, S=2, R=3, n=2, trump=0)
    t = Tree(g)
    for play in ("greedy", "ta"):
        s = profile(t, play=play)
        assert np.all(s.sum(1) == 1) and np.all((s > 0) <= t.is_valid)
    # top trump led -> sure winner; lowest off-suit card led with the suit's top unseen -> not sure
    assert _pwin_cached(g, 0, (2, 5), ())[2] == 1.0
    assert _pwin_cached(g, 0, (0, 1), ())[0] < 1.0

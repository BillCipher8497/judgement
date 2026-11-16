"""M5 decomposition identities, M8 game helpers, M9 variants."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "math"))

import m5_analyze as m5  # noqa: E402
import m8_eq as m8  # noqa: E402


def _fake_hand(rng, n, trump):
    cards = sorted(rng.choice(52, n, replace=False).tolist())
    masks = rng.integers(0, 2 ** n, 300)
    # inject dependence: same-suit cards move together
    return cards, masks


def test_m5_shares_sum_to_delta_minus_one():
    rng = np.random.default_rng(5)
    for trump in (0, -1):
        for n in (5, 8):
            cards, masks = _fake_hand(rng, n, trump)
            h = m5.analyze_hand(cards, masks, n, trump)
            tot = sum(h["comp"].values()) / h["vpb"]  # comp sums ordered pairs = 2 sum_{i<j}
            assert abs((h["vemp"] - h["vpb"]) / h["vpb"] - tot) < 1e-9
            # suit-block variance identity: Var_SB = Var_PB + within-suit covariance (exact for plug-ins)
            X = ((np.asarray(masks)[:, None] >> np.arange(n)) & 1).astype(float)
            suit = np.asarray(cards) // 13
            vsb = sum(X[:, suit == s].sum(1).var() for s in np.unique(suit))
            assert abs(vsb - h["vsb"]) < 1e-9


def test_m5_suit_block_exact_when_suits_independent():
    # one suit only -> SB law equals empirical law
    cards = [0, 1, 2, 3, 4]
    rng = np.random.default_rng(6)
    masks = rng.integers(0, 32, 300)
    h = m5.analyze_hand(cards, masks, 5, 0)
    assert h["tv_sb"] < 1e-12


def test_m8_hand_key_side_suit_symmetry():
    # 2-card hands: permuting the side suits (H,D,C = 1,2,3) leaves the class unchanged
    rng = np.random.default_rng(7)
    for _ in range(200):
        a, b = rng.choice(52, 2, replace=False)
        perm = rng.permutation([1, 2, 3])
        mp = lambda c: c if c // 13 == 0 else 13 * perm[c // 13 - 1] + c % 13  # noqa: E731
        h1 = (1 << int(a)) | (1 << int(b))
        h2 = (1 << int(mp(a))) | (1 << int(mp(b)))
        assert m8.hand_key(h1) == m8.hand_key(h2)


def test_m8_dealer_never_hits_hook():
    rng = np.random.default_rng(8)
    N, K = 500, 6
    cls = rng.integers(0, K, (N, 4)).astype(np.int32)
    tr = np.zeros((N, 81, 4), np.int8)
    tr[:, :, 0] = rng.integers(0, 3, (N, 81))
    tr[:, :, 1] = 2 - tr[:, :, 0]
    g = m8.Game(tr, cls, K, hook=True)
    S, _ = m8.solve(g, m8.init_profile(K, True), sweeps=4)
    b = g.bids(S)
    assert (b.sum(1) != 2).all()


def test_m8_lnds_mass():
    assert m8.lnds_mass([0, 1, 2], [0, 1, 2], [1, 1, 1]) == 3
    assert m8.lnds_mass([0, 1, 2], [2, 1, 0], [1, 1, 5]) == 5


def test_m9_variant_picks_highest_sure_winner():
    import m9_play as m9
    from judgement.policies.target_aware import TargetAwarePolicy

    V = m9.with_variants(TargetAwarePolicy, W=True)
    v = V()

    class TM:
        wc = 0
    cards = [5, 9, 12]
    pw = {5: 1.0, 9: 1.0, 12: 1.0}
    assert v._win(None, TM(), cards, pw, 1, 3, 0, lambda c: c) == 12

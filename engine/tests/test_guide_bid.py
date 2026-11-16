"""experiments/guide/s1_fit.rule_bids (vectorised, used for fitting) == HumanBidder.bid on random situations."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

GUIDE_DIR = str(Path(__file__).resolve().parents[1] / "experiments" / "guide")

from judgement.cards import NT  # noqa: E402
from judgement.policies.human_guide import GUIDE_BID, HumanBidder, bid_counts  # noqa: E402


def _situations(k, rng):
    out = []
    for _ in range(k):
        n = int(rng.integers(1, 14))
        trump = int(rng.choice([0, 1, 2, 3, NT]))
        hand = 0
        for c in rng.permutation(52)[:n]:
            hand |= 1 << int(c)
        pos = int(rng.integers(4))
        prior = [int(rng.integers(0, n + 1)) for _ in range(pos)]
        legal = [b for b in range(n + 1) if not (pos == 3 and b == n - sum(prior))]
        out.append(dict(n=n, trump=trump, hand=hand, pos=pos, prior=prior, legal=legal, dealer=0))
    return out


def _import_s1_fit():
    """Import experiments/guide/s1_fit without leaving its generic module names (common, s1_fit) behind,
    so other tests that import experiments/*/common.py are unaffected."""
    import importlib
    saved = {k: sys.modules.pop(k) for k in ("common", "s1_fit") if k in sys.modules}
    sys.path.insert(0, GUIDE_DIR)
    try:
        mod = importlib.import_module("s1_fit")
    finally:
        sys.path.remove(GUIDE_DIR)
        for k in ("common", "s1_fit"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return mod


def test_vectorised_rule_matches_bidder():
    m = _import_s1_fit()
    GRID, KEYS, rule_bids = m.GRID, m.KEYS, m.rule_bids
    rng = np.random.default_rng(11)
    S = _situations(1500, rng)
    D = {k: np.array([bid_counts(s["hand"], s["trump"], s["n"])[k] for s in S]) for k in KEYS}
    meta = dict(n=np.array([s["n"] for s in S]), nt=np.array([s["trump"] == NT for s in S]).astype(int),
                pos=np.array([s["pos"] for s in S]), prior=np.array([sum(s["prior"]) for s in S]),
                dealer_forb=np.array([s["n"] - sum(s["prior"]) if s["pos"] == 3 else -1 for s in S]))
    for t in range(8):
        w = dict(GUIDE_BID) if t == 0 else {k: int(rng.choice(v)) for k, v in GRID.items()}
        vb = rule_bids(w, D, meta)
        hb = HumanBidder(**w)
        for i, s in enumerate(S):
            info = NS(hand=s["hand"], trump=s["trump"], n_cards=s["n"], seat=(1 + s["pos"]) % 4, dealer=0, n_players=4,
                      bids=tuple(((1 + j) % 4, x) for j, x in enumerate(s["prior"])))
            b = hb.bid(info, s["legal"])
            assert b == vb[i]
            assert b in s["legal"]

"""Vectorised version of judgement.bidding.features.hand_features.

Hands are given as a boolean (M, 52) matrix; the prior-bid / position block is
shared by all rows.  Equality with the scalar version is tested in
tests/test_round2_bid.py.
"""
from __future__ import annotations

import numpy as np

from ..bidding.features import D
from ..cards import NT


def masks_to_bool(masks) -> np.ndarray:
    m = np.asarray([int(x) for x in masks], dtype=object)
    out = np.zeros((len(m), 52), dtype=bool)
    for c in range(52):
        out[:, c] = [(int(v) >> c) & 1 for v in m]
    return out


def hand_features_batch(H: np.ndarray, trump: int, n: int, pos: int, prior_bids,
                        n_players: int = 4) -> np.ndarray:
    """H: (M, 52) bool.  Returns (M, D) features identical to hand_features."""
    M = H.shape[0]
    f = np.zeros((M, D))
    f[:, 0] = 1.0
    f[:, 1] = pos / (n_players - 1)
    sp = float(sum(prior_bids))
    f[:, 2] = sp / n
    f[:, 3] = (sp - pos * n / n_players) / n
    f[:, 4] = (max(prior_bids) / n) if len(prior_bids) else 0.0
    f[:, 5] = 1.0 if pos == n_players - 1 else 0.0
    S = H.reshape(M, 4, 13).astype(np.int64)        # [m, suit, rank]
    ln = S.sum(2)                                   # (M, 4)
    A, K, Q, J, T = S[:, :, 12], S[:, :, 11], S[:, :, 10], S[:, :, 9], S[:, :, 8]
    low = S[:, :, :6].sum((1, 2))
    hcp = (4 * A + 3 * K + 2 * Q + J).sum(1)
    side = np.ones(4, dtype=bool)
    tlen = np.zeros(M)
    if trump != NT:
        side[trump] = False
        tl = ln[:, trump]
        tlen = tl.astype(float)
        f[:, 6] = tl
        f[:, 7] = A[:, trump]
        f[:, 8] = K[:, trump]
        f[:, 9] = Q[:, trump]
        f[:, 10] = J[:, trump]
        f[:, 11] = T[:, trump]
        f[:, 12] = np.maximum(0, tl - 3)
    ls = ln[:, side]
    f[:, 13] = A[:, side].sum(1)
    f[:, 14] = K[:, side].sum(1)
    f[:, 15] = (K[:, side] * (ls >= 2)).sum(1)
    f[:, 16] = Q[:, side].sum(1)
    f[:, 17] = (Q[:, side] * (ls >= 3)).sum(1)
    f[:, 18] = J[:, side].sum(1)
    voids = (ls == 0).sum(1)
    singles = (ls == 1).sum(1)
    f[:, 19] = voids
    f[:, 20] = singles
    f[:, 21] = (ls == 2).sum(1)
    f[:, 22] = ls.max(1)
    f[:, 23] = np.maximum(0, ls - 4).sum(1)
    f[:, 24] = low
    f[:, 25] = hcp / 10.0
    if trump != NT:
        f[:, 26] = voids * tlen
        f[:, 27] = singles * tlen
    return f

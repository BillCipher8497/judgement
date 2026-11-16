"""Cheap, interpretable hand features for the heuristic bidder."""
from __future__ import annotations

import numpy as np

from ..cards import NT

FEATURE_NAMES = [
    "const", "pos", "sum_prior_frac", "prior_excess", "max_prior_frac", "is_last",
    "t_len", "t_A", "t_K", "t_Q", "t_J", "t_T", "t_len_over3",
    "s_aces", "s_kings", "s_kings_guarded", "s_queens", "s_queens_guarded", "s_jacks",
    "s_voids", "s_singletons", "s_doubletons", "s_longest", "s_long_excess",
    "low_cards", "hcp10", "voids_x_tlen", "singles_x_tlen",
]
D = len(FEATURE_NAMES)


def hand_features(hand: int, trump: int, n: int, pos: int, prior_bids, n_players: int = 4) -> np.ndarray:
    """``pos`` = bidding position 0..N-1 (N-1 = dealer, bids last).
    ``prior_bids`` = bids already made this round, in order."""
    f = np.zeros(D)
    f[0] = 1.0
    f[1] = pos / (n_players - 1)
    sp = float(sum(prior_bids))
    f[2] = sp / n
    f[3] = (sp - pos * n / n_players) / n
    f[4] = (max(prior_bids) / n) if prior_bids else 0.0
    f[5] = 1.0 if pos == n_players - 1 else 0.0
    aces = kings = kg = queens = qg = jacks = voids = singles = doubles = longest = longx = low = 0
    hcp = 0
    tlen = 0
    for s in range(4):
        m = (hand >> (13 * s)) & 0x1FFF
        ln = m.bit_count()
        low += (m & 0x3F).bit_count()          # ranks 2..7
        hcp += 4 * (m >> 12 & 1) + 3 * (m >> 11 & 1) + 2 * (m >> 10 & 1) + (m >> 9 & 1)
        if s == trump:
            tlen = ln
            f[6] = ln
            f[7] = m >> 12 & 1
            f[8] = m >> 11 & 1
            f[9] = m >> 10 & 1
            f[10] = m >> 9 & 1
            f[11] = m >> 8 & 1
            f[12] = max(0, ln - 3)
            continue
        aces += m >> 12 & 1
        k = m >> 11 & 1
        q = m >> 10 & 1
        kings += k
        kg += k and ln >= 2
        queens += q
        qg += q and ln >= 3
        jacks += m >> 9 & 1
        voids += ln == 0
        singles += ln == 1
        doubles += ln == 2
        longest = max(longest, ln)
        longx += max(0, ln - 4)
    f[13:24] = (aces, kings, kg, queens, qg, jacks, voids, singles, doubles, longest, longx)
    f[24] = low
    f[25] = hcp / 10.0
    if trump != NT:
        f[26] = voids * tlen
        f[27] = singles * tlen
    return f

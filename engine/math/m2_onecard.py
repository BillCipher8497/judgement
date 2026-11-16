"""M2: exact solution of the 1-card round (4 players, hook on dealer) by full
enumeration of all 52*51*50*49 ordered deals, in exact integer arithmetic.

pos 0 = left of dealer (bids first, leads), pos 3 = dealer.
Each player's payoff depends only on (own bid, own win) and win is fixed by the
deal (play is forced), so the unique perfect-Bayesian strategy is myopic:
bid 1 iff 21*W > 10*T in the information cell (P(win) > 10/21), dealer hooked.
Outputs math/results/m2_onecard.json.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "math" / "results"
RANKS = "23456789TJQKA"
SUITS = "SHDC"


def all_deals():
    c = np.arange(52, dtype=np.int16)
    a = np.repeat(c, 51 * 50 * 49)
    rest = []
    # build sequentially with numpy: positions 1..3 drawn without replacement
    D = np.empty((52 * 51 * 50 * 49, 4), np.int16)
    i = 0
    for c0 in range(52):
        r0 = np.delete(c, c0)
        for c1 in r0:
            r1 = r0[r0 != c1]
            g2, g3 = np.meshgrid(r1, r1, indexing="ij")
            m = g2 != g3
            k = int(m.sum())
            D[i:i + k, 0] = c0
            D[i:i + k, 1] = c1
            D[i:i + k, 2] = g2[m]
            D[i:i + k, 3] = g3[m]
            i += k
    assert i == len(D)
    return D


def winners(D, trump):
    suit = D // 13
    led = suit[:, 0:1]
    # key: trump cards 200+rank, led-suit 100+rank, else rank-100 (never wins)
    key = np.where(suit == led, 100 + D % 13, -100).astype(np.int32)
    if trump >= 0:
        key = np.where(suit == trump, 200 + D % 13, key)
    return key.argmax(1)


def solve(D, trump, H=10, hook=True):
    M = len(D)
    win = winners(D, trump)
    bids = np.zeros((M, 4), np.int8)
    strat = {}
    for pos in range(4):
        W = (win == pos)
        if pos == 0:
            hist_code = np.zeros(M, np.int32)
        else:
            hist_code = (bids[:, :pos].astype(np.int32) * (1 << np.arange(pos))).sum(1)
        key = hist_code * 52 + D[:, pos]
        T = np.bincount(key, minlength=(1 << pos) * 52)
        Wc = np.bincount(key, weights=W, minlength=(1 << pos) * 52).astype(np.int64)
        s_prior = bids[:, :pos].sum(1) if pos else np.zeros(M, int)
        # bid-1 decision per cell: EV1 = (H+1) P, EV0 = H (1-P)  ->  (2H+1) W > H T
        dec = ((2 * H + 1) * Wc > H * T).astype(np.int8)
        b = dec[key]
        if pos == 3 and hook:  # hook: sum(bids) != 1
            b = np.where(s_prior == 0, 0, np.where(s_prior == 1, 1, b)).astype(np.int8)
        bids[:, pos] = b
        st = {}
        for h in range(1 << pos):
            hs = "".join(str(h >> j & 1) for j in range(pos)) or "-"
            cells = [(c, int(T[h * 52 + c]), int(Wc[h * 52 + c])) for c in range(52) if T[h * 52 + c] > 0]
            if not cells:
                continue
            ties = [c for c, t, w in cells if (2 * H + 1) * w == H * t]
            if pos == 3 and hook:
                sp = bin(h).count("1")
                if sp == 0:
                    bid1 = []
                elif sp == 1:
                    bid1 = [c for c, _, _ in cells]
                else:
                    bid1 = [c for c, t, w in cells if (2 * H + 1) * w > H * t]
            else:
                bid1 = [c for c, t, w in cells if (2 * H + 1) * w > H * t]
            # monotonicity of P(win | card, history) in rank within each suit
            mono = True
            for s in range(4):
                ps = [Fraction(w, t) for c, t, w in cells if c // 13 == s]
                if any(ps[j + 1] < ps[j] for j in range(len(ps) - 1)):
                    mono = False
            by_suit = {SUITS[s]: "".join(RANKS[c % 13] for c in bid1 if c // 13 == s) for s in range(4)}
            by_suit = {k: v for k, v in by_suit.items() if v}
            pmax = max(Fraction(w, t) for c, t, w in cells)
            st[hs] = {"bid1": by_suit or "none", "monotone_in_rank": mono, "ties": len(ties),
                      "P(hist)": float(sum(t for _, t, _ in cells) / M), "max_P_win": float(pmax)}
        strat[pos] = st
    score = np.where(bids == (np.arange(4)[None, :] == win[:, None]).astype(np.int8), H + bids, 0)
    vals = [Fraction(int(score[:, p].sum()), M) for p in range(4)]
    hits = [float((bids[:, p] == (win == p)).mean()) for p in range(4)]
    sb = bids.sum(1)
    return {"values_by_pos": [float(v) for v in vals], "values_exact": [str(v) for v in vals],
            "hit_rate_by_pos": hits, "table_value": float(sum(vals)),
            "P(sum_bids=0)": float((sb == 0).mean()), "P(sum_bids>=2)": float((sb >= 2).mean()),
            "strategy": strat}


def leader_pwin(r, trump_suit=True):
    """Closed form, leader with rank index r (0='2'..12='A') of the trump suit / NT."""
    return Fraction(comb(51 - (12 - r), 3), comb(51, 3))


def main():
    D = all_deals()
    res = {}
    for tr, name in ((0, "S"), (-1, "NT")):
        res[name] = solve(D, tr)
        res[name + "_nohook"] = solve(D, tr, hook=False)
        print(name, res[name]["values_by_pos"], res[name + "_nohook"]["values_by_pos"], flush=True)
        res[name]["dealer_hook_cost_exact"] = res[name + "_nohook"]["values_by_pos"][3] - res[name]["values_by_pos"][3]
    res["leader_closed_form"] = {RANKS[r]: float(leader_pwin(r)) for r in range(13)}
    res["leader_offsuit_trump_ace"] = float(Fraction(comb(51 - 13, 3), comb(51, 3)))
    res["threshold"] = 10 / 21
    # compare with Phase 2 exact solve
    p2 = {d["trump"]: d for d in json.load(open(ROOT / "experiments/phase2/results/onecard.json"))}
    cmp = {}
    for name in ("S", "NT"):
        a, b = res[name], p2[name]
        cmp[name] = {"max_abs_value_diff": max(abs(x - y) for x, y in zip(a["values_by_pos"], b["values_by_pos"])),
                     "strategy_match": all(
                         a["strategy"][int(pos)][h]["bid1"] == b["bid1_cards_by_pos_and_history"][pos][h]
                         for pos in b["bid1_cards_by_pos_and_history"]
                         for h in b["bid1_cards_by_pos_and_history"][pos])}
    res["phase2_compare"] = cmp
    json.dump(res, open(OUT / "m2_onecard.json", "w"), indent=1)
    print(json.dumps(cmp))


if __name__ == "__main__":
    main()

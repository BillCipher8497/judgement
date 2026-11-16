"""M5 mechanism check: split each card win into 'won on the hero's own lead' (L) vs 'won while following /
discarding / ruffing' (F), X_i = L_i + F_i, and decompose the cross-suit covariance
  2 sum_{i<j, suit_i != suit_j} Cov(X_i, X_j) = LL + (LF + FL) + FF.
If the dependence is 'winning gives the lead, the lead cashes/establishes another suit', the cross-suit
positive covariance should sit in the L-involving terms.  Also: P(card wins | hero held the lead at
some trick after trick 1) vs not.  TA play, first bidder, model bid, n=8 {S, NT}, 300 hands x R=300
(same deals/seeds as m5_sim for the first 300 hands).  Output: results/m5_lead.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402
from judgement.engine import deal, run_round  # noqa: E402
from judgement.sampling import HandSampler  # noqa: E402
from judgement.state import RoundState, state_from_info  # noqa: E402

from m5_sim import HERO, make_agents  # noqa: E402


def rollouts(st, ags, R, rng):
    info = st.info_state(HERO)
    sampler = HandSampler(info)
    n = st.n_cards
    cards = [c for c in range(52) if st.hands[HERO] >> c & 1]
    idx = {c: i for i, c in enumerate(cards)}
    Lm = np.zeros((R, n))
    Fm = np.zeros((R, n))
    for r in range(R):
        s2 = state_from_info(info, sampler.sample(rng))
        run_round(s2, ags, [rng] * 4, check=False)
        h = s2.history
        for k, w in enumerate(s2.trick_winners):
            if w == HERO:
                tr = h[4 * k:4 * k + 4]
                led = tr[0][0] == HERO
                for p, c in tr:
                    if p == HERO:
                        (Lm if led else Fm)[r, idx[c]] = 1
    return np.array(cards), Lm, Fm


def main(hands=300, R=300, ns=(8,), trumps=(0, -1), tag=""):
    rules = Rules()
    ags = make_agents()["ta"]
    out = {}
    for n in ns:
        for tr in trumps:
            rng_deal = np.random.default_rng(91000 + 10 * n + (tr + 1))
            acc = {"LL": [], "LF": [], "FF": [], "cross_total": [], "within_total": [], "vpb": [],
                   "share_lead_wins": []}
            for i in range(hands):
                hs = deal(rules, n, rng_deal)
                hr = np.random.default_rng([92000, n, tr + 1, i, 0])
                cards, Lm, Fm = rollouts(RoundState(rules, n, tr, 0, list(hs)), ags, R, hr)
                suit = cards // 13
                cross = suit[:, None] != suit[None, :]
                X = Lm + Fm
                p = X.mean(0)
                vpb = (p * (1 - p)).sum()
                if vpb < 1e-9:
                    continue

                def cov(A, B):
                    return A.T @ B / R - np.outer(A.mean(0), B.mean(0))
                CLL, CLF, CFF = cov(Lm, Lm), cov(Lm, Fm), cov(Fm, Fm)
                CX = cov(X, X)
                acc["LL"].append(CLL[cross].sum() / vpb)
                acc["LF"].append((CLF[cross].sum() + CLF.T[cross].sum()) / vpb)
                acc["FF"].append(CFF[cross].sum() / vpb)
                acc["cross_total"].append(CX[cross].sum() / vpb)
                off = ~np.eye(n, dtype=bool) & ~cross
                acc["within_total"].append(CX[off].sum() / vpb)
                acc["vpb"].append(vpb)
                acc["share_lead_wins"].append(Lm.sum() / max(X.sum(), 1))
            res = {}
            for k, v in acc.items():
                v = np.asarray(v)
                res[k] = [float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), int(len(v))]
            out[f"n={n}|{'NT' if tr < 0 else 'S'}"] = res
            print(n, tr, {k: round(v[0], 3) for k, v in res.items()}, flush=True)
    json.dump(out, open(ROOT / "math" / "results" / f"m5_lead{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # e.g. 5,13 -1 _extra
        main(ns=tuple(map(int, sys.argv[1].split(","))), trumps=tuple(map(int, sys.argv[2].split(","))), tag=sys.argv[3])
    else:
        main()

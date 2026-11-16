"""F4b: Phase-1 TargetAwarePolicy (judgement.policies.target_aware) vs max^n under
perfect information, side by side with the Phase-2 minimal heuristic.

Same deals and realistic bids (scheme O) as exp_main.py: deal i of (n, trump)
uses SeedSequence([20261110, n, trump_code, i]).  Per deal: decisions along the
max^n (neutral) path -> is the TA card / heuristic card max^n-optimal, 1-step
loss; one TA seat vs max^n others; all-TA table.
Usage: python exp_ta_pi.py N_DEALS n1 n2 ...   -> results/ta_pi.csv, ta_pi_dec.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement import NT, Rules  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.solver import MaxN  # noqa: E402
from judgement.solver.bidding import make_play_state  # noqa: E402
from judgement.solver.heuristic import heuristic_card  # noqa: E402
from judgement.state import PLAY  # noqa: E402

from exp_main import oracle_bids  # noqa: E402

OUT = Path(__file__).parent / "results"
R = Rules()
TA = TargetAwarePolicy()


def ta_card(s, rng):
    return TA.play(s.info_state(s.to_act), s.legal_cards(), rng)


def run(n_deals, ns):
    f1 = open(OUT / "ta_pi.csv", "a", newline="")
    f2 = open(OUT / "ta_pi_dec.csv", "a", newline="")
    w1 = w2 = None
    for n in ns:
        for tname, trump in (("S", 0), ("NT", NT)):
            t0 = time.time()
            for i in range(n_deals):
                ss = np.random.SeedSequence([20261110, n, 0 if trump == 0 else 9, i])
                drng, brng = [np.random.default_rng(c) for c in ss.spawn(2)]
                hands = deal(R, n, drng)
                dealer = i % 4
                bids = oracle_bids(R, n, trump, dealer, hands, brng)
                st = make_play_state(R, n, trump, dealer, hands, bids)
                sol = MaxN(R, trump, bids, 4)
                val = sol.value(st)
                rng = np.random.default_rng(0)
                s = st.copy()
                while s.phase == PLAY:
                    p = s.to_act
                    if s.legal_cards().bit_count() > 1:
                        mv = sol.move_values(s)
                        own = {c: v[p] for c, v in mv.items()}
                        vmax = max(own.values())
                        ct = ta_card(s, rng)
                        ch = heuristic_card(s.info_state(p), s.legal_cards())
                        rec = {"n": n, "trump": tname, "deal": i, "indiff": int(vmax == min(own.values())),
                               "ta_opt": int(own[ct] == vmax), "heur_opt": int(own[ch] == vmax),
                               "ta_loss": vmax - own[ct], "heur_loss": vmax - own[ch], "same": int(ct == ch)}
                        if w2 is None:
                            w2 = csv.DictWriter(f2, fieldnames=list(rec))
                            if f2.tell() == 0:
                                w2.writeheader()
                        w2.writerow(rec)
                    s.apply_card(sol.best_move(s))
                for p in range(4):
                    s = st.copy()
                    while s.phase == PLAY:
                        s.apply_card(ta_card(s, rng) if s.to_act == p else sol.best_move(s))
                    one = s.scores()[p]
                    if p == 0:
                        a = st.copy()
                        while a.phase == PLAY:
                            a.apply_card(ta_card(a, rng))
                        allta = a.scores()
                    row = {"n": n, "trump": tname, "deal": i, "seat": p, "bid": bids[p], "maxn": val[p],
                           "one_ta": one, "all_ta": allta[p]}
                    if w1 is None:
                        w1 = csv.DictWriter(f1, fieldnames=list(row))
                        if f1.tell() == 0:
                            w1.writeheader()
                    w1.writerow(row)
            f1.flush()
            f2.flush()
            print(f"n={n} {tname}: {n_deals} deals {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), [int(x) for x in sys.argv[2:]])

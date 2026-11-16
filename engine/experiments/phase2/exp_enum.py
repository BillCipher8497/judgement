"""F1/F2 (a): every hook-legal bid vector on a sample of deals, n <= 3.

Per (deal, bid vector): max^n value (3 tie-breaks), paranoid value per seat,
social optimum.  Seeds: SeedSequence([20261113, n, trump_code, i]).
Usage: python exp_enum.py N_DEALS n1 n2 ...   -> results/enum.csv
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
from judgement.solver import MODES, MaxN, Paranoid, Social  # noqa: E402
from judgement.solver.bidding import bid_order, legal_bid_vectors, make_play_state  # noqa: E402

OUT = Path(__file__).parent / "results"
R = Rules()


def run(n_deals, ns):
    f = open(OUT / "enum.csv", "a", newline="")
    w = None
    for n in ns:
        for tname, trump in (("S", 0), ("NT", NT)):
            t0 = time.time()
            for i in range(n_deals):
                rng = np.random.default_rng(np.random.SeedSequence([20261113, n, 0 if trump == 0 else 9, i]))
                hands = deal(R, n, rng)
                dealer = i % 4
                order = bid_order(4, dealer)
                pmemo = [dict() for _ in range(4)]
                nmemo = {}
                for bids in legal_bid_vectors(R, n, dealer):
                    st = make_play_state(R, n, trump, dealer, hands, bids)
                    vals = {m: MaxN(R, trump, bids, 4, m, memo=nmemo if m == "neutral" else None).value(st) for m in MODES}
                    para = [Paranoid(R, trump, bids, p, memo=pmemo[p]).value(st) for p in range(4)]
                    soc = Social(R, trump, bids).value(st)
                    for p in range(4):
                        row = {"n": n, "trump": tname, "deal": i, "bids": "".join(map(str, bids)), "seat": p,
                               "pos": order.index(p), "bid": bids[p], "sum_bids_minus_n": sum(bids) - n,
                               "paranoid": para[p], "social_total": soc,
                               **{f"maxn_{m}": vals[m][p] for m in MODES}}
                        if w is None:
                            w = csv.DictWriter(f, fieldnames=list(row))
                            if f.tell() == 0:
                                w.writeheader()
                        w.writerow(row)
            f.flush()
            print(f"n={n} {tname}: {n_deals} deals {time.time() - t0:.1f}s", flush=True)
    f.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), [int(x) for x in sys.argv[2:]])

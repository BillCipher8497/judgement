"""F5: imperfect-information play, PIMC vs the target-aware heuristic.

Per deal (4p, n cards, trump S or NT, dealer = i mod 4): every seat bids with the
same oracle bidder (MC trick distribution under greedy play, 100 samples) so
bids are identical across conditions.  Then the round is played under:
  H      all four seats heuristic
  P@s    seat s PIMC, others heuristic            (s = 0..3)
  H@s    seat s heuristic, others PIMC            (s = 0..3)
  P      all four PIMC
PIMC: n_samples consistent deals per decision, each solved exactly by max^n
(neutral), best average own score.  All PIMC decisions in condition P are also
logged with features, the PIMC average scores and the true (perfect-info) max^n
move values, for the F6 rule check.

Seeds: deal / bid stream SeedSequence([20261111, n, trump_code, i]); PIMC rng for
(condition, seat) = SeedSequence([20261112, n, trump_code, i, cond_idx, seat]).
Usage: python exp_pimc.py N_DEALS K n1 n2 ...
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement import NT, Rules, RoundState  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.solver import MaxN  # noqa: E402
from judgement.solver.features import decision_record  # noqa: E402
from judgement.solver.heuristic import OracleGreedyBidder, heuristic_card  # noqa: E402
from judgement.solver.pimc import pimc_move_scores  # noqa: E402
from judgement.state import PLAY  # noqa: E402

OUT = Path(__file__).parent / "results"
R = Rules()
TRUMPS = {"S": 0, "NT": NT}


_HEUR = None


def other_card(st, p, rng):
    """The non-PIMC seats' policy: Phase-2 heuristic, or Phase-1 TargetAwarePolicy if PH2_HEUR=ta."""
    if _HEUR is None:
        return heuristic_card(st.info_state(p), st.legal_cards())
    return _HEUR.play(st.info_state(p), st.legal_cards(), rng)


def play(st0, pimc_seats, K, rngs, log=None):
    st = st0.copy()
    while st.phase == PLAY:
        p = st.to_act
        legal = st.legal_cards()
        if p in pimc_seats and legal.bit_count() > 1:
            info = st.info_state(p)
            sc = pimc_move_scores(info, legal, K, rngs[p])
            c = max(sorted(sc), key=lambda x: sc[x])
            if log is not None:
                mv_fake = {cc: {p: v} for cc, v in sc.items()}
                rec = decision_record(st, mv_fake, heuristic_card(info, legal))
                true = MaxN(st.rules, st.trump, st.bids, st.N).move_values(st)
                rec["mv_true"] = json.dumps({str(cc): v[p] for cc, v in sorted(true.items())})
                rec["chosen"] = c
                log.append(rec)
        elif p in pimc_seats:
            c = legal.bit_length() - 1
        else:
            c = other_card(st, p, rngs[p])
        st.apply_card(c)
    return st


def run(n_deals, K, ns):
    import os
    tag = os.environ.get("PH2_TAG", "")
    global _HEUR
    if os.environ.get("PH2_HEUR") == "ta":
        from judgement.policies.target_aware import TargetAwarePolicy
        _HEUR = TargetAwarePolicy()
    start = int(os.environ.get("PH2_START", "0"))
    fr = open(OUT / f"pimc_rounds{tag}.csv", "a", newline="")
    fd = open(OUT / f"pimc_decisions{tag}.csv", "a", newline="")
    wr = wd = None
    t_start = time.time()
    for n in ns:
        for tname, trump in TRUMPS.items():
            t0 = time.time()
            for i in range(start, start + n_deals):
                tc = 0 if trump == 0 else 9
                drng, brng = [np.random.default_rng(c) for c in np.random.SeedSequence([20261111, n, tc, i]).spawn(2)]
                hands = deal(R, n, drng)
                st = RoundState(R, n, trump, i % 4, hands)
                bidder = OracleGreedyBidder(100)
                while st.phase == 0:
                    st.apply_bid(bidder.bid(st.info_state(st.to_act), st.legal_bids(), brng))
                conds = [("H", set())] + [(f"P@{s}", {s}) for s in range(4)] + \
                        [(f"H@{s}", {0, 1, 2, 3} - {s}) for s in range(4)] + [("P", {0, 1, 2, 3})]
                for ci, (cname, ps) in enumerate(conds):
                    rngs = [np.random.default_rng(np.random.SeedSequence([20261112, n, tc, i, ci, s])) for s in range(4)]
                    log = [] if cname == "P" else None
                    t1 = time.perf_counter()
                    end = play(st, ps, K, rngs, log)
                    dt = time.perf_counter() - t1
                    for s in range(4):
                        row = {"n": n, "trump": tname, "deal": i, "cond": cname, "seat": s,
                               "is_pimc": int(s in ps), "bid": st.bids[s], "tricks": end.tricks_won[s],
                               "score": end.scores()[s], "sum_bids_minus_n": sum(st.bids) - n,
                               "time_s": round(dt, 4), "K": K}
                        if wr is None:
                            wr = csv.DictWriter(fr, fieldnames=list(row))
                            if fr.tell() == 0:
                                wr.writeheader()
                        wr.writerow(row)
                    for rec in log or []:
                        rec.update(n=n, trump=tname, deal=i)
                        if wd is None:
                            wd = csv.DictWriter(fd, fieldnames=list(rec))
                            if fd.tell() == 0:
                                wd.writeheader()
                        wd.writerow(rec)
                fr.flush()
                fd.flush()
            print(f"n={n} {tname}: {n_deals} deals in {time.time() - t0:.1f}s (total {time.time() - t_start:.0f}s)",
                  flush=True)
    fr.close()
    fd.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3:]])

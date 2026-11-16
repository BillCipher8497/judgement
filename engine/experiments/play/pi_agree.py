"""Per-decision agreement with exact max^n under perfect information (Phase 2
F4b protocol, as experiments/phase2/exp_ta_pi.py and round2_play/pi_agree.py):
deal i for (n, trump) from SeedSequence([20261110, n, trump_code, i]), scheme-O
oracle bids, decisions along the max^n (neutral) path.  All policies are scored
on the same decisions.

python pi_agree.py N_DEALS n  -> results/pi_agree_dec.csv
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

from common import ROOT, ta_plus_from

sys.path.insert(0, str(ROOT / "experiments" / "phase2"))
from exp_main import oracle_bids  # noqa: E402

from judgement import NT, Rules  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.policies.ta_plus import TAPlusPolicy  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.solver import MaxN  # noqa: E402
from judgement.solver.bidding import make_play_state  # noqa: E402
from judgement.state import PLAY  # noqa: E402

OUT = Path(__file__).parent / "results"
R = Rules()
POLS = {"ta": TargetAwarePolicy(), "tap": ta_plus_from({}), "tap_r2": TAPlusPolicy(), "tap_noshed": ta_plus_from({"shed": 0}),
        "tap_nolh": ta_plus_from({"lh": 0}), "tap_nothm": ta_plus_from({"thm": 0}),
        "tap_none": ta_plus_from({"shed": 0, "lh": 0, "thm": 0, "wh": 0})}


def run(n_deals, n):
    f = open(OUT / "pi_agree_dec.csv", "w", newline="")
    w = None
    rng = np.random.default_rng(0)
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
            s = st.copy()
            while s.phase == PLAY:
                p = s.to_act
                if s.legal_cards().bit_count() > 1:
                    mv = sol.move_values(s)
                    own = {c: v[p] for c, v in mv.items()}
                    vmax = max(own.values())
                    rec = {"n": n, "trump": tname, "deal": i, "indiff": int(vmax == min(own.values()))}
                    info = s.info_state(p)
                    for k, pol in POLS.items():
                        c = pol.play(info, s.legal_cards(), rng)
                        rec[f"{k}_opt"] = int(own[c] == vmax)
                        rec[f"{k}_loss"] = vmax - own[c]
                        rec[f"{k}_card"] = c
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(rec))
                        w.writeheader()
                    w.writerow(rec)
                s.apply_card(sol.best_move(s))
        f.flush()
        print(f"pi n={n} {tname}: {n_deals} deals {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), int(sys.argv[2]))

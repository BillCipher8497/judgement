"""Q1: how the bid-only best response to TA+myopic differs from the myopic bid (per seat;
weighted by P(reach infoset) under the BR-deviated profile).  -> results/q1_brbids.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement.cfr.heuristics import profile  # noqa: E402
from judgement.cfr.solve import best_response, profile_value  # noqa: E402
from judgement.cfr.toy import Tree  # noqa: E402
from q1_toy import GAMES, RES, is_reach  # noqa: E402

out = {}
for tag in sys.argv[1:]:
    g, _ = GAMES[tag]
    t = Tree(g)
    ta = profile(t, "ta")
    rows = []
    for p in range(g.N):
        v, pol = best_response(t, ta, p, free=t.is_bid, return_policy=True)
        rw = is_reach(t, pol)
        m = t.is_bid & (t.is_pl == p)
        bb, mb = pol[m].argmax(1), ta[m].argmax(1)
        w = rw[m] / rw[m].sum()
        rows.append(dict(seat=p, gain=v - profile_value(t, ta)[p], changed=float(w @ (bb != mb)),
                         up=float(w @ (bb > mb)), down=float(w @ (bb < mb)),
                         mean_br=float(w @ bb), mean_myopic=float(w @ mb)))
        print(tag, rows[-1], flush=True)
    out[tag] = rows
(RES / "q1_brbids.json").write_text(json.dumps(out, indent=1))

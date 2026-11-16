"""P2 follow-up: ΔP(win) of the win-maximiser by its standing when it takes over
(rank among the 4 at round 25-k, baseline standings are identical to the treatment's).
Re-simulates only the baseline prefix (same seeds as p2_winmax.py)."""
from __future__ import annotations

import json

import numpy as np

from common2 import RES, dump, fmt, load_model  # noqa: E402
from judgement import Rules
from judgement.bidding.model import Agent, ModelBidder
from judgement.engine import deal, run_round
from judgement.policies.target_aware import TargetAwarePolicy
from judgement.state import RoundState


def main():
    d = json.load(open(RES / "p2_raw.json"))
    a = d["args"]
    rules = Rules()
    model = load_model()
    ag = Agent(ModelBidder(model), TargetAwarePolicy())
    ks = [int(k) for k in d["res"]]
    R = len(rules.ladder())
    G = a["games"]
    rng0 = np.random.default_rng(0)
    stand = {k: np.zeros((G, 4)) for k in ks}
    for g in range(G):
        tot = np.zeros(4)
        for r, n in enumerate(rules.ladder()):
            for k in ks:
                if r == R - k:
                    stand[k][g] = tot
            if r >= R - min(ks):
                break
            rng = np.random.default_rng([a["seed"], g, r])
            st = RoundState(rules, n, rules.trump_for_round(r), rules.dealer_for_round(r), deal(rules, n, rng))
            run_round(st, [ag] * 4, [rng0] * 4, check=False)
            tot += st.scores()
    lines = ["## P2 by standing at take-over (rank 1 = leader; ties share the better rank)", "",
             "| k | rank | n | ΔP(win) | gap to leader (mean) |", "|---|---|---|---|---|"]
    out = {}
    for k in ks:
        v = d["res"][str(k)]
        dw = np.array([x for _, x in v["dwin_seat"]]).reshape(G, 4)
        S = stand[k]
        rank = 1 + (S[:, None, :] > S[:, :, None]).sum(2)    # rank[g, h]
        gap = S.max(1, keepdims=True) - S
        for rk in (1, 2, 3, 4):
            m = rank == rk
            lines.append(f"| {k} | {rk} | {m.sum()} | {fmt(dw[m], 4)} | {gap[m].mean():.1f} |")
        out[k] = {"rank": rank.tolist()}
    txt = "\n".join(lines)
    dump({"table": txt}, "p2_standings.json")
    print(txt)


if __name__ == "__main__":
    main()

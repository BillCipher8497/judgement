"""Q1 equilibrium selection: CFR+ from different starting profiles (regret warm start
R0[I] = 1 pt x chance mass of I x profile, so iteration 1 plays that profile and later
regrets can overturn it).  -> results/q1_eqsel.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement.cfr.heuristics import profile  # noqa: E402
from judgement.cfr.solve import CFRPlus, nash_conv  # noqa: E402
from q1_toy import GAMES, RES, is_reach  # noqa: E402


def run(tag, T):
    g, _ = GAMES[tag]
    from judgement.cfr.toy import Tree
    t = Tree(g)
    ref = np.load(RES / f"q1_{tag}_cfr_avg.npy")
    rng = np.random.default_rng(12345)
    inits = {"ta": profile(t, "ta")} if g.N == 4 else {"greedy": profile(t, "greedy"), "ta": profile(t, "ta")}
    for k in range(2 if g.N < 4 else 1):
        x = rng.dirichlet(np.ones(g.A), t.n_is) * t.is_valid
        inits[f"random{k}"] = x / x.sum(1, keepdims=True)
    out = {}
    rw = is_reach(t, ref)
    for name, s0 in inits.items():
        t0 = time.time()
        c = CFRPlus(t)
        dec = t.node_pl >= 0
        mass = np.bincount(t.node_is[dec], weights=t.node_c[dec], minlength=t.n_is)
        c.R = mass[:, None] * s0 * t.is_valid
        c.iterate(T)
        avg = c.average()
        nc, gains, vals = nash_conv(t, avg)
        diff = (avg.argmax(1) != ref.argmax(1))
        out[name] = dict(T=T, nashconv=nc, value=vals.tolist(),
                         modal_action_diff_reach_ref=float(rw @ diff / rw.sum()),
                         l1_reach_ref=float(rw @ np.abs(avg - ref).sum(1) / rw.sum()))
        print(tag, name, f"NC {nc:.5f}", vals.round(3), f"{out[name]['modal_action_diff_reach_ref']:.3f}",
              f"{time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for tag, T in [("3p_t", 500), ("3p_nt", 500), ("4p_t", 200), ("4p_nt", 200)]:
        res[tag] = run(tag, T)
        (RES / "q1_eqsel.json").write_text(json.dumps(res, indent=1))

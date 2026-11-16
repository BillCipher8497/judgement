"""Paired comparison of g_<tag>.npz files: challenger seat vs the baseline table's same seat (same games)."""
import sys

import numpy as np

from common import RES


def load(tag):
    return np.load(RES / f"g_{tag}.npz")


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def seatvals(d):
    s = d["seat"]
    idx = np.arange(len(s))
    tot = d["totals"]
    win = (tot[idx, s] >= tot.max(1)).astype(float)       # ties for first count as wins
    return tot[idx, s], win, d["scores"][idx, :, s]


def compare(tag, base_tag):
    a, b = load(tag), load(base_tag)
    m = min(len(a["game"]), len(b["game"]))
    assert (a["game"][:m] == b["game"][:m]).all()
    ta, wa, ra = seatvals(a)
    tb, wb, rb = seatvals(b)
    d, dci = ci(ta[:m] - tb[:m])
    pw, pwci = ci(wa[:m])
    dw, dwci = ci(wa[:m] - wb[:m])
    return dict(n=m, d=d, dci=dci, pwin=pw, pwci=pwci, dpw=dw, dpwci=dwci, rounds=(ra[:m] - rb[:m]))


if __name__ == "__main__":
    for t in sys.argv[2:]:
        r = compare(t, sys.argv[1])
        print(f"{t:28s} n={r['n']:5d}  d/game {r['d']:+7.2f} +- {r['dci']:.2f}  P(win) {r['pwin']:.3f} +- {r['pwci']:.3f}"
              f"  dP {r['dpw']:+.3f} +- {r['dpwci']:.3f}")

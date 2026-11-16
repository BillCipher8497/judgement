"""Paired diff of challenger seat score vs the all-TA table (same seed, same seat)."""
import sys

import numpy as np

from common import RES, mean_ci

ref = np.load(RES / "1v3_ta.npz")
for tag in sys.argv[1:]:
    d = np.load(RES / f"1v3_{tag}.npz")
    G = min(len(d["totals"]), len(ref["totals"]))
    s = d["seat"][:G]
    x = d["totals"][np.arange(G), s] - ref["totals"][:G][np.arange(G), s]
    m, h, n = mean_ci(x)
    b = d["bids"][:G][np.arange(G), :, s]
    t = d["tricks"][:G][np.arange(G), :, s]
    miss = b != t
    over = (t > b) & miss
    print(f"{tag:40s} diff {m:+.2f} ± {h:.2f} (n={n})  hit {1 - miss.mean():.4f}  overshare {over.sum() / miss.sum():.3f}")

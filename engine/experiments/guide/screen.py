"""Play-rule screen on SCREEN seeds (BASE 91,000,000; final measurements use 71,000,000):
python screen.py POOL G kind1 kind2 ...  -> paired vs the first kind."""
import sys

from pair import compare
from run import run

BASE = 91_000_000
pool, G = sys.argv[1], int(sys.argv[2])
kinds = sys.argv[3:]
tags = []
for k in kinds:
    tag = "s_" + pool + "_" + k.replace(":", "_").replace(",", "_").replace("=", "")
    run(pool, k, 0, G, tag, BASE)
    tags.append(tag)
for k, t in zip(kinds, tags):
    r = compare(t, tags[0])
    print(f"{k:40s} d/game vs {kinds[0]}: {r['d']:+6.2f} +- {r['dci']:.2f}   P(win) {r['pwin']:.3f}", flush=True)

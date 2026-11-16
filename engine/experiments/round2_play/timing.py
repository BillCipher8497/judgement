"""Quick per-game timing of each policy in one seat vs 3 TA."""
import sys
import time

from common import make_agent
from judgement import Rules, play_game

kinds = sys.argv[1].split(",")
G = int(sys.argv[2])
for k in kinds:
    t0 = time.time()
    for g in range(G):
        pols = [make_agent("ta") for _ in range(4)]
        pols[g % 4] = make_agent(k)
        play_game(Rules(), pols, seed=7_000_000 + g, record_trace=False)
    print(k, f"{(time.time() - t0) / G:.3f} s/game", flush=True)

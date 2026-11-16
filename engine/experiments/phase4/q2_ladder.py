"""Q2 full-game exploitability ladder (1-vs-3, CRN).

Each config = (pool kind, challenger kind).  Game g: seed BASE+g, challenger in seat g%4
(challenger == pool kind gives the all-pool reference table).  Configs advance
round-robin in CHUNK-game chunks so every config has comparable n at any stop time.
Results: results/q2_<tag>.npz (same layout as experiments/play one_vs_three).

python q2_ladder.py DEADLINE_EPOCH
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "play"))
from common import make_agent  # noqa: E402
from judgement import Rules, play_game  # noqa: E402

RES = Path(__file__).parent / "results"
BASE = 400_000_000
CHUNK = 25
CHUNK_CHEAP, CHEAP = 100, {"tap_ref", "tap_ta"}   # TA+/TA-only tables (~0.02 s/game)
TAP, HYB = "ta_plus", "hybrid:k=16,c=5"
TAPO, HYBO = "tap_oracle:K=24,nmax=13", "hyb_oracle:k=16,c=5,K=24,nmax=13"
# tag, pool, challenger, target games
CONFIGS = [
    ("tap_ref", TAP, TAP, 1500), ("tap_ta", TAP, "ta", 1500), ("tap_hyb", TAP, HYB, 1000),
    ("tap_tapo", TAP, TAPO, 600), ("tap_hybo", TAP, HYBO, 600),
    ("hyb_ref", HYB, HYB, 600), ("hyb_tap", HYB, TAP, 600), ("hyb_ta", HYB, "ta", 600),
    ("hyb_hybo", HYB, HYBO, 600), ("hyb_tapo", HYB, TAPO, 600),
]


class Store:
    def __init__(self, tag, pool, chal, G):
        R = Rules().n_rounds
        self.tag, self.pool, self.chal, self.G = tag, pool, chal, G
        self.bids = np.zeros((G, R, 4), np.int8); self.tricks = np.zeros((G, R, 4), np.int8)
        self.scores = np.zeros((G, R, 4), np.float32); self.totals = np.zeros((G, 4)); self.wins = np.zeros((G, 4))
        self.secs = np.zeros(G); self.done = 0
        p = RES / f"q2_{tag}.npz"
        if p.exists():   # resume
            d = np.load(p)
            k = self.done = len(d["totals"])
            self.bids[:k], self.tricks[:k], self.scores[:k] = d["bids"], d["tricks"], d["scores"]
            self.totals[:k], self.wins[:k], self.secs[:k] = d["totals"], d["wins"], d["secs"]

    def save(self):
        d = self.done
        np.savez_compressed(RES / f"q2_{self.tag}.npz", bids=self.bids[:d], tricks=self.tricks[:d],
                            scores=self.scores[:d], totals=self.totals[:d], wins=self.wins[:d],
                            seat=np.arange(d) % 4, secs=self.secs[:d], seed_base=BASE,
                            pool=self.pool, chal=self.chal)


def play_one(st, g, rules):
    pols = [make_agent(st.pool, seed=int(np.random.SeedSequence([BASE, g, 7, p]).generate_state(1)[0]))
            for p in range(4)]
    pols[g % 4] = make_agent(st.chal, seed=int(np.random.SeedSequence([BASE, g, 5]).generate_state(1)[0]))
    t1 = time.time()
    res = play_game(rules, pols, seed=BASE + g, record_trace=False)
    st.secs[g] = time.time() - t1
    for r, rr in enumerate(res.rounds):
        st.bids[g, r] = rr.bids; st.tricks[g, r] = rr.tricks; st.scores[g, r] = rr.scores
    st.totals[g] = res.totals; st.wins[g] = res.wins
    st.done = g + 1


def main(deadline):
    rules = Rules()
    stores = [Store(*c) for c in CONFIGS]
    while time.time() < deadline and any(s.done < s.G for s in stores):
        for s in stores:
            if s.done >= s.G:
                continue
            t0 = time.time()
            for g in range(s.done, min(s.G, s.done + (CHUNK_CHEAP if s.tag in CHEAP else CHUNK))):
                if time.time() > deadline:
                    break
                play_one(s, g, rules)
            s.save()
            print(time.strftime("%H:%M:%S"), s.tag, s.done, f"{time.time()-t0:.0f}s", flush=True)
    print("Q2DONE", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]))

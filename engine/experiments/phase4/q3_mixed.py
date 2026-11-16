"""Q3: one table with TA, TA+, Hybrid, Hybrid+oracle (one seat each); game g uses seed
BASE+g and seat permutation g % 24.  python q3_mixed.py G DEADLINE -> results/q3_mixed.npz
python q3_mixed.py analyze -> results/q3_rating.md/.json (BT/Elo, TA+ = 0; game bootstrap)."""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "play"))
from common import make_agent  # noqa: E402
from judgement import Rules, play_game  # noqa: E402

RES = Path(__file__).parent / "results"
BASE = 450_000_000
POL = [("TA", "ta"), ("TA+", "ta_plus"), ("Hybrid", "hybrid:k=16,c=5"),
       ("Hybrid+oracle", "hyb_oracle:k=16,c=5,K=24,nmax=13")]
PERMS = list(itertools.permutations(range(4)))


def run(G, deadline):
    rules = Rules()
    tot = np.zeros((G, 4)); wins = np.zeros((G, 4)); secs = np.zeros(G)   # indexed by POLICY
    done = 0
    for g in range(G):
        if time.time() > deadline:
            break
        perm = PERMS[g % 24]            # seat s plays policy perm[s]
        pols = [make_agent(POL[perm[s]][1], seed=int(np.random.SeedSequence([BASE, g, s]).generate_state(1)[0]))
                for s in range(4)]
        t1 = time.time()
        res = play_game(rules, pols, seed=BASE + g, record_trace=False)
        secs[g] = time.time() - t1
        for s in range(4):
            tot[g, perm[s]] = res.totals[s]; wins[g, perm[s]] = res.wins[s]
        done = g + 1
        if done % 10 == 0:
            np.savez(RES / "q3_mixed.npz", tot=tot[:done], wins=wins[:done], secs=secs[:done])
    np.savez(RES / "q3_mixed.npz", tot=tot[:done], wins=wins[:done], secs=secs[:done])
    print("Q3DONE", done, flush=True)


def bt(tot):
    """Bradley-Terry strengths from in-game pairwise score comparisons (ties = 1/2)."""
    P = tot.shape[1]
    W = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            if i != j:
                W[i, j] = ((tot[:, i] > tot[:, j]) + 0.5 * (tot[:, i] == tot[:, j])).sum()
    p = np.ones(P)
    for _ in range(500):
        n = W + W.T
        p = W.sum(1) / np.array([sum(n[i, j] / (p[i] + p[j]) for j in range(P) if j != i) for i in range(P)])
        p /= np.exp(np.log(p).mean())
    return 400 * np.log10(p / p[1])     # Elo, TA+ = 0


def analyze():
    d = np.load(RES / "q3_mixed.npz")
    tot, wins = d["tot"], d["wins"]
    G = len(tot)
    rng = np.random.default_rng(0)
    elo = bt(tot)
    bs = np.array([bt(tot[rng.integers(0, G, G)]) for _ in range(300)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    rel = tot - tot[:, [1]]
    md = [f"Mixed table, n = {G} games (seat permutation g%24, seeds {BASE}+g); 95% CIs: Elo game-bootstrap (B=300), others 1.96 SE.", "",
          "| policy | Elo vs TA+ | score − TA+ (same game) | P(win) |", "|---|---|---|---|"]
    out = {}
    for i, (nm, _) in enumerate(POL):
        m, h = rel[:, i].mean(), 1.96 * rel[:, i].std(ddof=1) / np.sqrt(G)
        w, hw = wins[:, i].mean(), 1.96 * wins[:, i].std(ddof=1) / np.sqrt(G)
        out[nm] = dict(elo=float(elo[i]), elo_lo=float(lo[i]), elo_hi=float(hi[i]), d_score=float(m), d_ci=float(h),
                       pwin=float(w), pwin_ci=float(hw), n=G)
        md.append(f"| {nm} | {elo[i]:+.0f} [{lo[i]:+.0f}, {hi[i]:+.0f}] | {m:+.2f} ± {h:.2f} | {w:.3f} ± {hw:.3f} |")
    (RES / "q3_rating.md").write_text("\n".join(md) + "\n")
    (RES / "q3_rating.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        analyze()
    else:
        run(int(sys.argv[1]), float(sys.argv[2]))

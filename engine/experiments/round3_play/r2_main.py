"""R2: Phase-3 P2 rerun with the ties-as-win objective.

Identical design to experiments/phase3/p2_main.py (same seed 20261112 -> same deals, same
baseline games, same win-max RNG streams): one WinMax bidder (last k rounds) vs three
score-max (ModelBidder + TA) players, branch from the baseline at round 25-k, hero seat h=0..3.
Two treatment arms per (g, k, h):
  'tie' : WinMaxTieBidder - maximises P(total >= every other total)   (ties count as wins)
  'split': original WinMaxBidder - maximises P(win) with ties split 1/k  (Phase 3 P2 arm)
Both evaluated under ties-as-win (primary) and split (secondary).  Unit = game (mean of 4 seats).
-> results/r2_raw.npz, r2.md, r2.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1]))
sys.path.insert(0, str(H.parent / "phase3"))
from common3 import GAMES_TA, load_model  # noqa: E402
from judgement import Rules  # noqa: E402
from judgement.bidding.model import Agent, ModelBidder  # noqa: E402
from judgement.bidding2.winmax import ScoreBank, WinMaxBidder  # noqa: E402
from judgement.engine import deal, run_round  # noqa: E402
from judgement.policies.target_aware import TargetAwarePolicy  # noqa: E402
from judgement.state import RoundState  # noqa: E402

RES = H / "results"


class WinMaxTieBidder(WinMaxBidder):
    """WinMaxBidder with objective P(finish first, alone or tied)."""
    name = "winmax_tie"

    def pwin_by_bid(self, info, legal):
        totals, r = self.ctx
        N, me, bank = info.n_players, info.seat, self.bank
        g = self.rng.integers(bank.G, size=self.L)
        base = np.asarray(totals, float)[None, :] + bank.future[g, r, :]
        bid_seats = {p: b for p, b in info.bids}
        cur = np.zeros((self.L, N))
        for p in range(N):
            if p == me:
                continue
            if p in bid_seats:
                q = bank.hit_rate(r, (p - info.dealer - 1) % N, bid_seats[p])
                cur[:, p] = np.where(self.rng.random(self.L) < q, self.rules.score(bid_seats[p], bid_seats[p]),
                                     self.rules.score(bid_seats[p], -1))
            else:
                cur[:, p] = bank.scores[g, r, p]
        tot = base + cur
        top = np.delete(tot, me, axis=1).max(1)

        def W(x):
            return (tot[:, me] + x >= top).mean()
        ev, P = self._model_ev(info)
        miss = self.rules.score(1, 0)
        return {b: P[b, b] * W(self.rules.score(b, b)) + (1 - P[b, b]) * W(miss) for b in legal}, ev


def play_rounds(rules, agents, seed, g, r0, totals, winmax=None):
    totals = list(totals)
    rng0 = np.random.default_rng(0)
    snaps = {}
    for r, n in list(enumerate(rules.ladder()))[r0:]:
        snaps[r] = list(totals)
        rng = np.random.default_rng([seed, g, r])
        st = RoundState(rules, n, rules.trump_for_round(r), rules.dealer_for_round(r), deal(rules, n, rng))
        if winmax is not None:
            winmax.ctx = (tuple(totals), r)
        run_round(st, agents, [rng0] * 4, check=False)
        sc = st.scores()
        for p in range(4):
            totals[p] += sc[p]
    return totals, snaps


def w_tie(t, h):
    return float(t[h] >= max(t))


def w_split(t, h):
    m = max(t)
    return (1.0 / sum(1 for x in t if x == m)) if t[h] == m else 0.0


def ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def report(U, ks, G):
    """U[k]: (G, 4, 2 arms, 3: tie, split, score) treatment; B: baseline in arm slot via U[k][..., 2?]"""
    out, md = {}, ["| k | arm (objective) | n games | ΔP(win) ties-as-win | ΔP(win) split | Δscore/game | P(win) tie: base -> treat |",
                   "|---|---|---|---|---|---|---|"]
    for k in ks:
        A = U[k][:G]                      # (G, 4, 3 slots[base, tie, split], 3 metrics)
        for j, arm in ((1, "tie"), (2, "split")):
            d = (A[:, :, j, :] - A[:, :, 0, :]).mean(1)      # per game, mean over seats
            r = {m: ci(d[:, i]) for i, m in enumerate(("dp_tie", "dp_split", "dscore"))}
            r["n"] = G
            r["p_base"] = float(A[:, :, 0, 0].mean()); r["p_treat"] = float(A[:, :, j, 0].mean())
            out[f"k{k}_{arm}"] = r
            md.append(f"| {k} | {arm} | {G} | {r['dp_tie'][0]:+.4f} ± {r['dp_tie'][1]:.4f} | {r['dp_split'][0]:+.4f} ± {r['dp_split'][1]:.4f} "
                      f"| {r['dscore'][0]:+.2f} ± {r['dscore'][1]:.2f} | {r['p_base']:.4f} -> {r['p_treat']:.4f} |")
        d = (A[:, :, 1, :] - A[:, :, 2, :]).mean(1)
        r = {m: ci(d[:, i]) for i, m in enumerate(("dp_tie", "dp_split", "dscore"))}
        out[f"k{k}_tie_minus_split"] = r
        md.append(f"| {k} | tie − split (paired) | {G} | {r['dp_tie'][0]:+.4f} ± {r['dp_tie'][1]:.4f} | {r['dp_split'][0]:+.4f} ± {r['dp_split'][1]:.4f} "
                  f"| {r['dscore'][0]:+.2f} ± {r['dscore'][1]:.2f} | |")
    return out, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--ks", default="3,6")
    ap.add_argument("--L", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--deadline", type=float, default=float("inf"))
    a = ap.parse_args()
    rules, model, bank, ta = Rules(), load_model(), ScoreBank(GAMES_TA), TargetAwarePolicy()
    base_ag = Agent(ModelBidder(model), ta)
    ks = [int(x) for x in a.ks.split(",")]
    R = len(rules.ladder())
    U = {k: np.zeros((a.games, 4, 3, 3)) for k in ks}
    t0, G = time.time(), 0
    for g in range(a.games):
        if time.time() > a.deadline:
            break
        totals, snaps = play_rounds(rules, [base_ag] * 4, a.seed, g, 0, [0.0] * 4)
        for k in ks:
            S = snaps[R - k]
            for h in range(4):
                U[k][g, h, 0] = (w_tie(totals, h), w_split(totals, h), totals[h])
                for j, cls in ((1, WinMaxTieBidder), (2, WinMaxBidder)):
                    ss = np.random.SeedSequence([a.seed, g, h, k, 99])     # same stream as phase3 P2
                    wm = cls(model, rules, bank, k, L=a.L, seed=int(ss.generate_state(1)[0]))
                    ags = [base_ag] * 4
                    ags[h] = Agent(wm, ta)
                    tt, _ = play_rounds(rules, ags, a.seed, g, R - k, S, winmax=wm)
                    U[k][g, h, j] = (w_tie(tt, h), w_split(tt, h), tt[h])
        G = g + 1
        if G % 500 == 0 or G == a.games:
            out, md = report(U, ks, G)
            np.savez_compressed(RES / "r2_raw.npz", **{f"k{k}": U[k][:G] for k in ks}, seed=a.seed, L=a.L)
            (RES / "r2.md").write_text("\n".join(md) + "\n")
            (RES / "r2.json").write_text(json.dumps(out, indent=1))
            print(f"g={G} {time.time()-t0:.0f}s", flush=True)
    out, md = report(U, ks, G)
    np.savez_compressed(RES / "r2_raw.npz", **{f"k{k}": U[k][:G] for k in ks}, seed=a.seed, L=a.L)
    (RES / "r2.md").write_text("\n".join(md) + "\n")
    (RES / "r2.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md)); print("total", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()

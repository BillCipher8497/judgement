"""Q1: exact NashConv in toy Judgement.  python q1_toy.py GAME_TAG... -> results/q1_<tag>.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement.cfr.heuristics import profile, trick_dist_greedy  # noqa: E402
from judgement.cfr.solve import CFRPlus, cf_reach, nash_conv, reaches  # noqa: E402
from judgement.cfr.toy import NT, ToyGame, Tree  # noqa: E402

RES = Path(__file__).parent / "results"
GAMES = {
    "2p_diff": (ToyGame(N=2, S=2, R=4, n=2, trump=0, util="diff"), 1000),
    "2p_raw": (ToyGame(N=2, S=2, R=4, n=2, trump=0), 1000),
    "3p_t": (ToyGame(N=3, S=2, R=4, n=2, trump=0), 1000),
    "3p_nt": (ToyGame(N=3, S=2, R=4, n=2, trump=NT), 1000),
    "3p_3x3_t": (ToyGame(N=3, S=3, R=3, n=2, trump=0), 500),
    "4p_n1_t": (ToyGame(N=4, S=3, R=4, n=1, trump=0), 1000),
    "4p_n1_nt": (ToyGame(N=4, S=3, R=4, n=1, trump=NT), 1000),
    "4p_t": (ToyGame(N=4, S=2, R=4, n=2, trump=0), 500),
    "4p_nt": (ToyGame(N=4, S=2, R=4, n=2, trump=NT), 500),
    "3p_n3_t": (ToyGame(N=3, S=3, R=3, n=3, trump=0), 100),
}
CK = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000]


def gains3(t, s):
    nc, g, v = nash_conv(t, s)
    _, gb, _ = nash_conv(t, s, free=t.is_bid)
    _, gp, _ = nash_conv(t, s, free=~t.is_bid)
    return dict(nashconv=nc, gain=g.tolist(), gain_bid_only=gb.tolist(), gain_play_only=gp.tolist(),
                value=v.tolist(), nc_bid_only=float(gb.sum()), nc_play_only=float(gp.sum()))


def is_reach(t, s):
    """P(infoset reached) under profile s (chance x all players)."""
    r = reaches(t, s)
    x = t.node_c * r.prod(0)
    return np.bincount(t.node_is[t.node_pl >= 0], weights=x[t.node_pl >= 0], minlength=t.n_is)


def run(tag):
    g, T = GAMES[tag]
    t0 = time.time()
    t = Tree(g)
    out = dict(game=str(g), deals=t.n_deals, nodes=t.n_nodes, infosets=t.n_is,
               bid_infosets=int(t.is_bid.sum()), build_s=time.time() - t0, curve=[])
    c = CFRPlus(t)
    for k in [x for x in CK if x <= T]:
        c.iterate(k - c.T)
        nca, ga, _ = nash_conv(t, c.average())
        ncc, gc, _ = nash_conv(t, c.current())
        out["curve"].append(dict(T=k, nc_avg=nca, nc_cur=ncc, gain_avg=ga.tolist(), t=time.time() - t0))
        print(tag, k, f"{nca:.5f} {ncc:.5f}", f"{time.time()-t0:.0f}s", flush=True)
    avg = c.average()
    prof = {"cfr_avg": avg, "uniform": t.uniform(), "greedy": profile(t, "greedy"), "ta": profile(t, "ta")}
    # mixed: heuristic play + CFR bids, CFR play + myopic bids
    for h in ("greedy", "ta"):
        m = prof[h].copy(); m[t.is_bid] = avg[t.is_bid]; prof[f"{h}_play+cfr_bid"] = m
    m = avg.copy(); m[t.is_bid] = prof["ta"][t.is_bid]; prof["cfr_play+myopic_bid"] = m
    out["profiles"] = {k: gains3(t, s) for k, s in prof.items()}
    # CFR bid analysis: reach-weighted agreement of CFR modal bid with myopic bid, mixing
    rw = is_reach(t, avg)
    my = prof["ta"]
    bid_rows = []
    for p in range(g.N):
        m = t.is_bid & (t.is_pl == p)
        w = rw[m]
        agree = (avg[m].argmax(1) == my[m].argmax(1)).astype(float)
        mixed = (avg[m].max(1) < 0.99).astype(float)
        eb_c = (avg[m] * np.arange(g.A)).sum(1)
        eb_m = (my[m] * np.arange(g.A)).sum(1)
        bid_rows.append(dict(seat=p, agree=float(w @ agree / w.sum()), mixed=float(w @ mixed / w.sum()),
                             mean_bid_cfr=float(w @ eb_c / w.sum()), mean_bid_myopic=float(w @ eb_m / w.sum())))
    out["bids"] = bid_rows
    pm = ~t.is_bid
    out["play_mixed_frac"] = float(rw[pm] @ (avg[pm].max(1) < 0.99) / max(rw[pm].sum(), 1e-12))
    out["play_agree_ta"] = float(rw[pm] @ (avg[pm].argmax(1) == prof["ta"][pm].argmax(1)) / max(rw[pm].sum(), 1e-12))
    if g.n == 1:   # seat-0 threshold comparison
        rows = []
        for s, (p, hand, hist) in enumerate(t.is_key):
            if p == 0 and len(hist) == 0:
                pw = float(trick_dist_greedy(g, 0, hand)[1])
                rows.append(dict(card=hand[0], p_win=pw, threshold_bid=int(pw > 10 / 21), cfr_p_bid1=float(avg[s, 1])))
        out["seat0_threshold"] = sorted(rows, key=lambda r: r["p_win"])
    out["total_s"] = time.time() - t0
    (RES / f"q1_{tag}.json").write_text(json.dumps(out, indent=1))
    np.save(RES / f"q1_{tag}_cfr_avg.npy", avg)
    print(tag, "DONE", f"{out['total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        run(tag)

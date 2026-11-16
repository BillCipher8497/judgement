"""P1 likelihood validation.
(i) Self-play ModelBidder table (the table used throughout): does the vectorised
    likelihood's argmax reproduce the actual bid?  (bid is deterministic in the hand,
    so the exact likelihood is the indicator; MLE temperature -> 0.)
(ii) A non-model bidder: the uniform-world MC oracle bids b_U recorded by the previous
    attempt (experiments/round2_bid/results/p1_raw.json, 2160 situations; hands are
    regenerated from the recorded seeds and checked against the recorded prior bids).
    Fit tau in L(b|h) = softmax_b(EV_model(b|h)/tau) by MLE, report log-loss,
    top-1 accuracy and a reliability table (predicted P(bid) bins vs realised freq)."""
from __future__ import annotations

import json

import numpy as np

from common3 import ROOT, dump, load_model, status
from judgement import Rules
from judgement.bidding.model import ModelBidder
from judgement.bidding2.sampler import BidLikelihood
from judgement.cards import NT
from judgement.engine import deal
from judgement.state import RoundState

TR = [0, 1, 2, 3, NT]
B = lambda m: np.array([[(m >> c) & 1 for c in range(52)]], dtype=bool)  # noqa: E731


def main():
    rules = Rules()
    model = load_model()
    mb = ModelBidder(model)
    lik = BidLikelihood(model, rules, 0.0, 0.0)
    out = {"selfplay": {}}
    # (i)
    for n in (2, 3, 5, 8, 13):
        agree = tot = 0
        for i in range(300):
            rng = np.random.default_rng([515, n, i])
            st = RoundState(rules, n, TR[i % 5], 3, deal(rules, n, rng))
            for pos in range(4):
                info = st.info_state(pos)
                b = mb.bid(info, st.legal_bids(), rng)
                if pos < 3:
                    ev = lik.evs(B(st.hands[pos]), n, st.trump, pos, list(st.bids[:pos]))[0]
                    agree += int(ev.argmax() == b)
                    tot += 1
                st.apply_bid(b)
        out["selfplay"][n] = {"agree": agree / tot, "rows": tot}
    # (ii)
    d = json.load(open(ROOT / "experiments/round2_bid/results/p1_raw.json"))
    seed = d["args"]["seed"]
    EV, OBS, LEG, NS = [], [], [], []
    bad = 0
    for r in d["rows"]:
        n, pos, i = r["n"], r["pos"], r["i"]
        if n == 1:
            continue
        rng = np.random.default_rng([seed, n, pos, i])
        st = RoundState(rules, n, TR[i % 5], 3, deal(rules, n, rng))
        for _ in range(pos):
            st.apply_bid(mb.bid(st.info_state(st.to_act), st.legal_bids(), rng))
        if list(st.bids[:pos]) != r["prior"]:
            bad += 1
            continue
        ev = lik.evs(B(st.hands[pos]), n, st.trump, pos, r["prior"])[0]
        leg = np.array(st.legal_bids())
        EV.append(ev); OBS.append(r["bu"]); LEG.append(leg); NS.append(n)

    def nll(tau):
        s = 0.0
        for ev, o, lg in zip(EV, OBS, LEG):
            z = ev[lg] / tau
            z -= z.max()
            s += -(z[list(lg).index(o)] - np.log(np.exp(z).sum()))
        return s / len(EV)

    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 60))
    L = [nll(t) for t in grid]
    tau = float(grid[int(np.argmin(L))])
    f = np.exp(np.linspace(np.log(tau / 1.15), np.log(tau * 1.15), 15))
    Lf = [nll(t) for t in f]
    tau = float(f[int(np.argmin(Lf))])
    probs, hit, top1, by_n = [], [], [], {}
    for ev, o, lg, n in zip(EV, OBS, LEG, NS):
        z = ev[lg] / tau
        p = np.exp(z - z.max()); p /= p.sum()
        for k, b in enumerate(lg):
            probs.append(p[k]); hit.append(int(b == o))
        top1.append(int(lg[p.argmax()] == o))
        by_n.setdefault(n, []).append(int(lg[p.argmax()] == o))
    probs, hit = np.array(probs), np.array(hit)
    rel = []
    for lo in np.arange(0, 1, 0.1):
        m = (probs >= lo) & (probs < lo + 0.1 + (lo > 0.85))
        if m.sum():
            ph = hit[m].mean()
            rel.append([f"{lo:.1f}-{lo+0.1:.1f}", int(m.sum()), float(probs[m].mean()), float(ph),
                        float(1.96 * np.sqrt(ph * (1 - ph) / m.sum()))])
    out["oracle"] = {"rows": len(EV), "seed_mismatch": bad, "tau": tau, "nll_tau": float(min(Lf)),
                     "nll_uniform": float(np.mean([np.log(len(lg)) for lg in LEG])),
                     "nll_tau_inf": float(nll(1e6)), "top1": float(np.mean(top1)),
                     "top1_by_n": {k: float(np.mean(v)) for k, v in by_n.items()}, "reliability": rel}
    dump(out, "p1_calib.json")
    lines = ["P1 likelihood validation (p1_calib.py; results/p1_calib.json)",
             "(i) self-play ModelBidder table: vectorised-likelihood argmax == actual bid: " +
             ", ".join(f"n={k}: {v['agree']:.4f} ({v['rows']} bids)" for k, v in out["selfplay"].items()),
             f"(ii) MC-oracle (uniform worlds, K=64) bids, {len(EV)} rows: fitted tau={tau:.3f} pts; "
             f"log-loss {out['oracle']['nll_tau']:.3f} vs uniform-over-legal {out['oracle']['nll_uniform']:.3f} nats; "
             f"top-1 acc {out['oracle']['top1']:.3f}; seed mismatches {bad}",
             "reliability (bin, rows, mean pred, realised ± 95%): " +
             "; ".join(f"{a} {b} {c:.3f} {d_:.3f}±{e:.3f}" for a, b, c, d_, e in rel)]
    status("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

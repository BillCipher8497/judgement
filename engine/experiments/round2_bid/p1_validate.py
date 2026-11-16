"""Validation of the bid likelihood / informed sampler (no rollouts, cheap).

Situations as in P1 (dealer 3, earlier seats ModelBidder).  For every earlier
bidder j we compare the posterior-mean prediction of j's TRUE hand statistics
(trump length [NT: longest suit], HCP 4-3-2-1, aces) under each sampler against the
truth: mean abs error and 80% central-interval coverage.  Samplers: uniform,
hard (eps=0.02, the one used in P1), soft tau=1 and tau=3 points.
Also: acceptance rate and ESS, and a reliability check of the soft likelihood's
P(observed bid | true hand) (mean prob assigned to the realised bid)."""
from __future__ import annotations

import time

import numpy as np

from common2 import dump, fmtu, load_model  # noqa: E402
from judgement import Rules
from judgement.bidding.model import ModelBidder
from judgement.bidding2.sampler import BidLikelihood, BidPhaseSampler
from judgement.cards import NT
from judgement.engine import deal
from judgement.state import RoundState

TRUMPS = [0, 1, 2, 3, NT]


def stats(masks: np.ndarray, trump: int) -> np.ndarray:
    H = ((masks[:, None] >> np.arange(52, dtype=np.uint64)) & np.uint64(1)).astype(bool).reshape(-1, 4, 13)
    ln = H.sum(2)
    tl = ln.max(1) if trump == NT else ln[:, trump]
    hcp = (4 * H[:, :, 12] + 3 * H[:, :, 11] + 2 * H[:, :, 10] + H[:, :, 9]).sum(1)
    aces = H[:, :, 12].sum(1)
    return np.stack([tl, hcp, aces], 1).astype(float)


def main():
    rules = Rules()
    model = load_model()
    mb = ModelBidder(model)
    samplers = {"uniform": None, "hard": BidLikelihood(model, rules, 0.0, 0.02),
                "soft1": BidLikelihood(model, rules, 1.0), "soft3": BidLikelihood(model, rules, 3.0)}
    K = 400
    res = {}
    t0 = time.time()
    for n in (2, 3, 5, 8, 13):
        for pos in (1, 3):
            acc = {k: {"err": [], "cov": [], "ess": []} for k in samplers}
            ptrue = []
            for i in range(150):
                rng = np.random.default_rng([4242, n, pos, i])
                tr = TRUMPS[i % 5]
                st = RoundState(rules, n, tr, 3, deal(rules, n, rng))
                for _ in range(pos):
                    st.apply_bid(mb.bid(st.info_state(st.to_act), st.legal_bids(), rng))
                info = st.info_state(pos)
                truth = stats(np.array(st.hands[:pos], dtype=np.uint64), tr)       # (pos, 3)
                soft3 = samplers["soft3"]
                for j in range(pos):
                    H = ((np.array([st.hands[j]], dtype=np.uint64)[:, None] >> np.arange(52, dtype=np.uint64)) & np.uint64(1)).astype(bool)
                    ptrue.append(float(np.exp(soft3.log_lik(H, n, tr, j, list(st.bids[:j]), st.bids[j]))[0]))
                pool = BidPhaseSampler(info, None).propose(4000, rng)
                for name, lik in samplers.items():
                    s = BidPhaseSampler(info, lik)
                    w, stt = s.sample(K, rng, informed=lik is not None, first=pool)
                    arr = np.array([[ww[p] for p in range(pos)] for ww in w], dtype=np.uint64)
                    for j in range(pos):
                        sj = stats(arr[:, j], tr)
                        acc[name]["err"].append(np.abs(sj.mean(0) - truth[j]))
                        lo, hi = np.quantile(sj, 0.1, axis=0), np.quantile(sj, 0.9, axis=0)
                        acc[name]["cov"].append((truth[j] >= lo) & (truth[j] <= hi))
                    acc[name]["ess"].append(stt["ess"])
            res[f"{n}|{pos}"] = {k: {"mae": np.mean(v["err"], 0).tolist(),
                                     "mae_ci": (1.96 * np.std(v["err"], 0, ddof=1) / np.sqrt(len(v["err"]))).tolist(),
                                     "cov80": np.mean(v["cov"], 0).tolist(), "ess": float(np.median(v["ess"])),
                                     "rows": len(v["err"])} for k, v in acc.items()}
            res[f"{n}|{pos}"]["soft3_p_obs"] = float(np.mean(ptrue))
            print(n, pos, f"{time.time()-t0:.0f}s", flush=True)
    lines = ["## P1 validation: posterior predictive of earlier bidders' TRUE hands",
             "MAE of posterior mean (95% CI half-width) for [trump length (NT: longest suit), HCP, aces]; "
             "cov80 = coverage of the 10–90% posterior interval (target ≥0.80; discrete stats overcover). "
             "150 situations per row, K=400 posterior worlds from a shared 4000-proposal pool (+more if needed).", "",
             "| n | pos | sampler | MAE trump-len | MAE HCP | MAE aces | cov80 (tl, hcp, aces) | median ESS |",
             "|---|---|---|---|---|---|---|---|"]
    for key, v in res.items():
        n, pos = key.split("|")
        for name in samplers:
            r = v[name]
            m, c = r["mae"], r["mae_ci"]
            lines.append(f"| {n} | {pos} | {name} | {m[0]:.3f} ± {c[0]:.3f} | {m[1]:.3f} ± {c[1]:.3f} | {m[2]:.3f} ± {c[2]:.3f} | "
                         f"{r['cov80'][0]:.2f}, {r['cov80'][1]:.2f}, {r['cov80'][2]:.2f} | {r['ess']:.0f} |")
    dump({"res": res, "table": "\n".join(lines)}, "p1_validate.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

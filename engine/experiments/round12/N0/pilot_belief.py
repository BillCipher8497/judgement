"""N0 pilot (Claude, head of research): learned card-location beliefs vs rule-based posteriors.

Question: given a play-time information state of a seat in a TA table, how well do different belief
models predict WHERE each unseen card is (which opponent holds it)? Proper scoring rule: mean log-loss
of the true holder per unseen card (lower is better; no exact posterior needed). Methods:
  prop    holder prob ∝ remaining cards of opponent, zero if void in the suit (closed form)
  unif    marginals of the exact uniform consistent-deal sampler (HandSampler, K worlds)
  bi      marginals of the bid-informed SIR posterior used by Hybrid-BI (K worlds)
  net     numpy MLP residual on top of log(prop): per (card, opponent) features, softmax over legal holders
Pilot only (seed base 2_100_000_000, train/test split by round index). Not a validated result.

python -I pilot_belief.py gen  --split train --rounds 6000
python -I pilot_belief.py gen  --split test  --rounds 600 --K 200
python -I pilot_belief.py fit
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments" / "play"), str(ROOT / "math")]
from judgement import Rules, RoundState  # noqa: E402
from judgement.cards import NT  # noqa: E402
from judgement.engine import deal  # noqa: E402
from judgement.sampling import HandSampler  # noqa: E402
from judgement.bidding2.sampler import BidLikelihood  # noqa: E402
from judgement.policies.hybrid_bi import BidInformedWorlds  # noqa: E402
from common import M_TA  # noqa: E402
from m5_sim import make_agents  # noqa: E402

BASE = 2_100_000_000
RES = HERE / "results"
NS = (5, 8, 11)
EPS = 1e-3  # floor mixed into MC marginals (with prop) so log-loss is finite


def feats(info, c, j_rel, opp):
    """Feature vector for 'opponent opp (relative position j_rel in 1..3) holds unseen card c'."""
    n, tr = info.n_cards, info.trump
    s, r = c // 13, c % 13
    bids = dict(info.bids)
    b = bids.get(opp, 0)
    tw = info.tricks_won[opp]
    left = info.cards_left(opp)
    need = b - tw
    hist = info.history
    mine = [cc for p, cc in hist if p == opp]
    in_suit = [cc % 13 for cc in mine if cc // 13 == s]
    led = sum(1 for k in range(0, len(hist), 4) if hist[k][0] == opp and hist[k][1] // 13 == s)
    unseen_in_suit = 13 - sum(1 for _, cc in hist if cc // 13 == s) - sum(1 for k in range(13) if info.hand >> (13 * s + k) & 1)
    x = np.zeros(13 + 3 + 14, np.float32)
    x[r] = 1.0
    x[13 + j_rel - 1] = 1.0
    o = 16
    x[o:o + 14] = [b / n, tw / n, need / max(1, left), float(need > 0), float(need == 0), float(need < 0),
                   left / n, float(tr != NT and s == tr), float(tr == NT), len(in_suit) / 4,
                   (max(in_suit) / 12) if in_suit else -1.0, led / 4, unseen_in_suit / 13, r / 12 * (need > 0)]
    return x


def record(info, rng, K, lik):
    """Labels + baselines for one information state. Returns dict of arrays."""
    seat, N = info.seat, info.n_players
    opps = [(seat + k) % N for k in (1, 2, 3)]
    seen = info.hand | info.played_mask
    unseen = [c for c in range(52) if not (seen >> c & 1)]
    true = {p: info._s.hands[p] for p in opps}  # labels only
    holders = [p for p in range(N) if p != seat]
    in_play = [c for c in unseen if any(true[p] >> c & 1 for p in opps)]  # stock cards never held by anyone
    X, M, Y, LP = [], [], [], []
    left = np.array([info.cards_left(p) for p in opps], float)
    for c in in_play:
        s = c // 13
        legal = np.array([left[k] > 0 and not (info.voids[opps[k]] >> s & 1) for k in range(3)])
        lp = np.where(legal, np.log(np.maximum(left, 1e-9)), -np.inf)
        X.append(np.stack([feats(info, c, k + 1, opps[k]) for k in range(3)]))
        M.append(legal)
        Y.append([k for k in range(3) if true[opps[k]] >> c & 1][0])
        LP.append(lp)
    out = {"X": np.array(X), "M": np.array(M), "Y": np.array(Y), "LP": np.array(LP),
           "n": info.n_cards, "nt": int(info.trump == NT), "cards": np.array(in_play)}
    if K:
        hs = HandSampler(info)
        U = [hs.sample(rng) for _ in range(K)]
        B, _ = BidInformedWorlds(lik, batch=64, m_max=2048).sample(info, K, rng)
        for name, W in (("unif", U), ("bi", B)):
            P = np.zeros((len(in_play), 3))
            for w in W:
                for i, c in enumerate(in_play):
                    for k in range(3):
                        if w[opps[k]] >> c & 1:
                            P[i, k] += 1
            out[name] = P / len(W)
    return out


def gen(split, rounds, K):
    rules = Rules()
    agents = make_agents()["ta"]
    lik = BidLikelihood(M_TA, rules, eps=0.02)
    off = 0 if split == "train" else 10_000_000
    fn = RES / f"{split}.npz"
    rows = []
    t0 = time.time()
    for g in range(rounds):
        rng = np.random.default_rng([BASE, off + g])
        n = NS[g % 3]
        trump = NT if (g // 3) % 2 else 0
        hands = deal(rules, n, rng)
        st = RoundState(rules, n, trump, g % 4, hands)
        for _ in range(4):
            st.apply_bid(agents[st.current_player].bid(st.info_state(st.current_player), st.legal_bids(), rng))
        # one recorded decision per round: random seat, random trick in [1, n-2]
        hero = int(rng.integers(4))
        tgt = int(rng.integers(1, n - 1))
        rec = None
        while not st.is_terminal:
            p = st.current_player
            info = st.info_state(p)
            if rec is None and p == hero and len(st.history) // 4 >= tgt:
                rec = record(info, rng, K, lik)
            st.apply_card(agents[p].play(info, st.legal_cards(), rng))
        if rec is not None and len(rec["Y"]):
            rec["g"] = g
            rows.append(rec)
        if (g + 1) % 500 == 0:
            print(split, g + 1, f"{time.time() - t0:.0f}s", flush=True)
    keys = ["X", "M", "Y", "LP"] + (["unif", "bi"] if K else [])
    pack = {k: np.concatenate([r[k] for r in rows]) for k in keys}
    pack["dec"] = np.concatenate([np.full(len(r["Y"]), i) for i, r in enumerate(rows)])
    pack["n"] = np.concatenate([np.full(len(r["Y"]), r["n"]) for r in rows])
    pack["nt"] = np.concatenate([np.full(len(r["Y"]), r["nt"]) for r in rows])
    RES.mkdir(exist_ok=True)
    np.savez_compressed(fn, **pack)
    print("saved", fn, {k: v.shape for k, v in pack.items()})


# ---------- numpy residual MLP: logit = log prop + f(x) ----------
def init(d, h, rng):
    return {"W1": rng.normal(0, 1 / np.sqrt(d), (d, h)), "b1": np.zeros(h),
            "w2": np.zeros(h), "b2": np.zeros(1)}


def forward(P, X):
    H = np.maximum(0, X @ P["W1"] + P["b1"])
    return H, H @ P["w2"] + P["b2"][0]


def loss_grad(P, X, M, Y, LP, l2):
    N, J, D = X.shape
    Xf = X.reshape(-1, D)
    H, f = forward(P, Xf)
    z = np.where(M, LP + f.reshape(N, J), -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    L = -np.log(p[np.arange(N), Y] + 1e-300).mean()
    g = p.copy(); g[np.arange(N), Y] -= 1; g /= N; g = g.reshape(-1)
    G = {"w2": H.T @ g + l2 * P["w2"], "b2": np.array([g.sum()])}
    dH = np.outer(g, P["w2"]) * (H > 0)
    G["W1"] = Xf.T @ dH + l2 * P["W1"]; G["b1"] = dH.sum(0)
    return L, G


def predict(P, X, M, LP):
    N, J, D = X.shape
    _, f = forward(P, X.reshape(-1, D))
    z = np.where(M, LP + f.reshape(N, J), -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); return p / p.sum(1, keepdims=True)


def prop(M, LP):
    z = np.where(M, LP, -np.inf); z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def fit():
    tr, te = np.load(RES / "train.npz"), np.load(RES / "test.npz")
    rng = np.random.default_rng([BASE, 7])
    dec = tr["dec"]; ud = np.unique(dec); rng.shuffle(ud)
    val = np.isin(dec, ud[: len(ud) // 5]); trn = ~val
    D = tr["X"].shape[2]
    P = init(D, 64, rng)
    A = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in P.items()}
    lr, best, bestP = 3e-3, 1e9, None
    idx = np.flatnonzero(trn)
    for ep in range(40):
        rng.shuffle(idx)
        for s in range(0, len(idx), 512):
            b = idx[s:s + 512]
            _, G = loss_grad(P, tr["X"][b], tr["M"][b], tr["Y"][b], tr["LP"][b], 1e-4)
            for k in P:
                m, v = A[k]; m[:] = 0.9 * m + 0.1 * G[k]; v[:] = 0.999 * v + 0.001 * G[k] ** 2
                P[k] = P[k] - lr * m / (np.sqrt(v) + 1e-8)
        vl, _ = loss_grad(P, tr["X"][val], tr["M"][val], tr["Y"][val], tr["LP"][val], 0)
        if vl < best:
            best, bestP, bep = vl, {k: v.copy() for k, v in P.items()}, ep
    # evaluation on test: per-card log loss, aggregated per decision, paired across methods
    pr = prop(te["M"], te["LP"])
    preds = {"prop": pr, "net": predict(bestP, te["X"], te["M"], te["LP"])}
    for k in ("unif", "bi"):
        preds[k] = (1 - EPS) * te[k] + EPS * pr
    y = te["Y"]; dec = te["dec"]
    ll = {k: -np.log(v[np.arange(len(y)), y]) for k, v in preds.items()}
    ud = np.unique(dec)
    per = {k: np.array([v[dec == d].mean() for d in ud]) for k, v in ll.items()}
    def ci(x): return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))]
    res = {"n_decisions": int(len(ud)), "n_cards": int(len(y)), "best_epoch": int(bep), "val_loss": float(best),
           "logloss": {k: ci(v) for k, v in per.items()},
           "paired_vs_bi": {k: ci(per[k] - per["bi"]) for k in per if k != "bi"},
           "by_cell": {}}
    for n in NS:
        for nt in (0, 1):
            sel = np.array([(te["n"][dec == d][0] == n) and (te["nt"][dec == d][0] == nt) for d in ud])
            res["by_cell"][f"n{n}_{'NT' if nt else 'S'}"] = {k: ci(per[k][sel] - per["bi"][sel]) for k in ("prop", "unif", "net")} | {"n_dec": int(sel.sum())}
    (RES / "fit.json").write_text(json.dumps(res, indent=1))
    np.savez(RES / "net_params.npz", **bestP)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--split", default="train")
    ap.add_argument("--rounds", type=int, default=6000); ap.add_argument("--K", type=int, default=0)
    a = ap.parse_args()
    gen(a.split, a.rounds, a.K) if a.cmd == "gen" else fit()

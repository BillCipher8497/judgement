"""M10 (round 3, H2): lead-state models for the hero's trick count T.

Data: math/results/m10_raw_n{n}_{S|NT}.npz (m10_sim.py): per hand R rollouts, card-win mask and
trick-win mask W_1..W_n.  Hero = first bidder = first leader, so L_1 = 1 and L_{j+1} = W_j
(the trick winner leads next): the hero's lead state is a deterministic function of its own wins.

Models of P(T = t) per hand (all built from rollout statistics of that hand, like PB/SB):
  PB   independent cards (card marginals);
  TPB  control: independent TRICKS with trick-indexed marginals p_j (no lead state);
  LC0  homogeneous lead chain: P(W_1) and two numbers P(W | lead), P(W | no lead) (tricks 2..n);
  SB   suit-block (per-suit trick-count pmfs convolved; round 2);
  LC1  lead chain: time-inhomogeneous Markov chain, a_j = P(W_j=1 | L_j=1), b_j = P(W_j=1 | L_j=0);
       shrunk to the trick-j marginal p_j with kappa pseudo-rollouts;
  LC2  second order: P(W_j | W_{j-1}, W_{j-2}), shrunk to LC1;
  LCT  P(W_j | L_j, T_{<j} = t) (lead x tricks-so-far), shrunk to LC1 (with kappa large -> LC1,
       small -> the empirical counting process, which reproduces P(T) exactly in sample);
  LCU  control: P(W_j | T_{<j}) without the lead state ("good-world luck" accumulates in T_{<j});
  LCB  lead + budget (n+2 params): logit P(W_j) = alpha_j + beta L_j + gamma (T_{<j} - E T_{<j});
  LB   control: budget only (n+1 params);
  LCS  lead chain with finite on-lead supply: P(W | lead, k lead-wins so far), P(W | no lead, j) (~2n);
  LCSP LCS with on-lead logit alpha_k + theta*j/(n-1) (supply and time);
  LCX  lead x phase + lead-specific budget slopes (n+4 params);
  LSB  lead-mixed suit blocks: SB computed separately in the two halves of the rollouts split by
       the outcome of trick 1 (W_1 = 1 keeps the lead), mixed with weight P(W_1).
Metrics (per hand, mean +- 95% CI over hands): in-sample TV, KL, dispersion error delta_M - delta_emp,
bid match vs the empirical-optimal bid, EV loss; cross-fit (fit on rollouts 1..R/2, test vs empirical
R/2+1..R): xTV, bid match vs B's empirical bid (floor: empirical A vs empirical B), EV loss on B.
gap closed = (metric_PB - metric_M) / (metric_PB - target), target = 0 for TV/loss, 1 (in sample) or
the A/B floor (cross) for match.
python math/m10_analyze.py [--kappa 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5_analyze import H, bid, conv_all, hist, kl, pb_pmf, tv  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
MODELS = ("PB", "SB", "TPB", "LC0", "LC1", "LC2", "LCB", "LB", "LCX", "LCS", "LCSP", "LCT", "LCU", "LSB")


def logit_fit(F, y, l2=1e-2, iters=25):
    """Ridge logistic regression by Newton (no penalty on nothing special; tiny ridge for separability)."""
    w = np.zeros(F.shape[1])
    for _ in range(iters):
        z = np.clip(F @ w, -30, 30)
        q = 1 / (1 + np.exp(-z))
        g = F.T @ (q - y) + l2 * w
        Hm = (F * (q * (1 - q))[:, None]).T @ F + l2 * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def budget_chain(W, L, T_before, pj, n, lead=True):
    """LCB: logit P(W_j) = alpha_j + beta*L_j + gamma*(T_{<j} - sum_{i<j} p_i)   (n+2 parameters);
    LB (lead=False): same without beta.  The budget term lets a hand that has already cashed more
    tricks than expected win fewer later (finite force), which a pure lead chain cannot express."""
    R = len(W)
    cum = np.concatenate([[0.0], np.cumsum(pj)[:-1]])
    cols = [np.repeat(np.eye(n)[None], R, 0).reshape(R * n, n)]
    dev = (T_before - cum[None, :]).reshape(-1, 1)
    ph = np.repeat((np.arange(n) / max(n - 1, 1))[None], R, 0).reshape(-1, 1)
    if lead == "x":     # LCX: lead x phase, lead-specific budget slopes (n+4 params)
        Lf = L.reshape(-1, 1).astype(float)
        cols += [Lf, Lf * ph, dev * Lf, dev * (1 - Lf)]
    elif lead:
        cols.append(L.reshape(-1, 1).astype(float))
        cols.append(dev)
    else:
        cols.append(dev)
    F = np.hstack(cols)
    w = logit_fit(F, W.reshape(-1).astype(float))
    al = w[:n]

    def tr(j, s, t):
        l = s[-1] if s else 1
        d = t - cum[j]
        if lead == "x":
            z = al[j] + w[n] * l + w[n + 1] * l * j / max(n - 1, 1) + (w[n + 2] * l + w[n + 3] * (1 - l)) * d
        elif lead:
            z = al[j] + w[n] * l + w[n + 1] * d
        else:
            z = al[j] + w[n] * d
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    coef = (w[n], w[n + 1]) if lead is True else (np.nan, np.nan)
    return chain_pmf(tr, n), coef


def chain_pmf(trans, n):
    """trans(j, prev_state) -> P(W_j=1); forward over (t, state); state = tuple of last wins."""
    dist = {(0, ()): 1.0}
    for j in range(n):
        nd = {}
        for (t, s), pr in dist.items():
            q = trans(j, s, t)
            for w, pw in ((1, q), (0, 1 - q)):
                if pw <= 0:
                    continue
                ns = (s + (w,))[-2:]
                k = (t + w, ns)
                nd[k] = nd.get(k, 0.0) + pr * pw
        dist = nd
    P = np.zeros(n + 1)
    for (t, _), pr in dist.items():
        P[t] += pr
    return P


def fit_models(X, W, suit, n, kappa):
    """X (R, n) card wins, W (R, n) trick wins (trick order). Returns dict of pmfs."""
    R = len(W)
    out = {}
    p = X.mean(0)
    out["PB"] = pb_pmf(p)
    blocks = [hist(X[:, suit == s].sum(1).astype(int), int((suit == s).sum())) for s in np.unique(suit)]
    out["SB"] = conv_all(blocks, n)
    pj = W.mean(0)
    out["TPB"] = pb_pmf(pj)            # control: independent tricks, trick-indexed marginals, no lead state
    L = np.ones((R, n), int)
    L[:, 1:] = W[:, :-1]
    # LC0: homogeneous lead chain, 2 numbers (P(win | lead), P(win | no lead)) pooled over tricks 1..n-1,
    # trick 1 (always led) uses its own rate; last trick is included in the pooling
    a0 = [(W[:, 1:][L[:, 1:] == l].sum() + kappa * W.mean()) / ((L[:, 1:] == l).sum() + kappa) for l in (0, 1)]
    out["LC0"] = chain_pmf(lambda j, s, t: pj[0] if j == 0 else a0[s[-1]], n)
    T_before = np.zeros((R, n), int)
    T_before[:, 1:] = np.cumsum(W, 1)[:, :-1]
    a = np.zeros((n, 2))
    for j in range(n):
        for l in (0, 1):
            m = L[:, j] == l
            a[j, l] = (W[m, j].sum() + kappa * pj[j]) / (m.sum() + kappa) if m.sum() + kappa > 0 else pj[j]
    out["LC1"] = chain_pmf(lambda j, s, t: a[j, s[-1] if s else 1], n)
    # LC2
    c2 = {}
    for j in range(n):
        l1 = L[:, j]
        l2 = np.ones(R, int) if j < 2 else W[:, j - 2]
        if j < 1:
            l2 = np.ones(R, int)
        for x in (0, 1):
            for y in (0, 1):
                m = (l1 == x) & (l2 == y)
                c2[j, y, x] = (W[m, j].sum() + kappa * a[j, x]) / (m.sum() + kappa) if m.sum() + kappa > 0 else a[j, x]

    def tr2(j, s, t):
        x = s[-1] if s else 1
        y = s[-2] if len(s) >= 2 else 1
        return c2[j, y, x]
    out["LC2"] = chain_pmf(tr2, n)
    out["LCB"], out["_coef"] = budget_chain(W, L, T_before, pj, n, lead=True)
    out["LB"], _ = budget_chain(W, L, T_before, pj, n, lead=False)
    out["LCX"], _ = budget_chain(W, L, T_before, pj, n, lead="x")
    out["LCS"] = supply_chain(W, L, pj, n, kappa)
    out["LCSP"] = supply_chain(W, L, pj, n, kappa, phase=True)
    ct = {}
    for j in range(n):
        for l in (0, 1):
            for t in range(j + 1):
                m = (L[:, j] == l) & (T_before[:, j] == t)
                if m.any():
                    ct[j, l, t] = (W[m, j].sum() + kappa * a[j, l]) / (m.sum() + kappa)
    # LCU: "luck" control -- P(W_j | T_{<j} = t) WITHOUT the lead state (shrunk to p_j), same order of
    # flexibility as LC1 but conditioning on cumulative success instead of who holds the lead
    cu = {}
    for j in range(n):
        for t in range(j + 1):
            m = T_before[:, j] == t
            if m.any():
                cu[j, t] = (W[m, j].sum() + kappa * pj[j]) / (m.sum() + kappa)
    out["LCU"] = chain_pmf(lambda j, s, t: cu.get((j, t), pj[j]), n)
    out["LCT"] = chain_pmf(lambda j, s, t: ct.get((j, s[-1] if s else 1, t), a[j, s[-1] if s else 1]), n)
    # LSB: suit blocks conditional on trick-1 outcome
    w1 = W[:, 0] == 1
    mix = np.zeros(n + 1)
    for m, wt in ((w1, w1.mean()), (~w1, 1 - w1.mean())):
        if m.sum() == 0:
            continue
        Xm = X[m]
        bl = [hist(Xm[:, suit == s].sum(1).astype(int), int((suit == s).sum())) for s in np.unique(suit)]
        Pm = conv_all(bl, n)
        mix += wt * Pm
    out["LSB"] = mix
    return out


def supply_chain(W, L, pj, n, kappa, phase=False):
    """LCS: lead chain with a finite on-lead supply.  On lead, P(win) depends on k = tricks already won
    ON LEAD (the hand cashes its established winners, then the supply runs out); off lead, on the trick
    index j.  ~2n parameters, each shrunk (kappa) to the pooled on-lead / trick-j rates."""
    R = len(W)
    K = np.zeros((R, n), int)                      # lead-wins before trick j
    K[:, 1:] = np.cumsum(W * L, 1)[:, :-1]
    lw = W[L == 1].mean() if (L == 1).any() else 0.5
    A = {}
    for k in range(n):
        m = (L == 1) & (K == k)
        if m.any():
            A[k] = (W[m].sum() + kappa * lw) / (m.sum() + kappa)
    th = None
    if phase and (L == 1).sum() > 2:
        # LCSP: logit on-lead rate = alpha_k + theta * j/(n-1)  (supply AND time: cards also leave the hand
        # while following, so the remaining force falls with j even without lead wins)
        m = L == 1
        ks = np.unique(K[m])
        jj = np.repeat(np.arange(n)[None], R, 0)
        F = np.hstack([(K[m][:, None] == ks[None, :]).astype(float), (jj[m] / max(n - 1, 1))[:, None]])
        w = logit_fit(F, W[m].astype(float), l2=0.5)
        th = (dict(zip(ks.tolist(), w[:-1])), w[-1])
    Bq = np.zeros(n)
    for j in range(n):
        m = L[:, j] == 0
        Bq[j] = (W[m, j].sum() + kappa * pj[j]) / (m.sum() + kappa) if m.sum() + kappa > 0 else pj[j]
    dist = {(0, 1, 0): 1.0}                         # (t, lead, k)
    for j in range(n):
        nd = {}
        for (t, l, k), pr in dist.items():
            if l and th is not None:
                ak = th[0].get(k, min(th[0].values()) if k > max(th[0]) else max(th[0].values()))
                q = 1 / (1 + np.exp(-np.clip(ak + th[1] * j / max(n - 1, 1), -30, 30)))
            else:
                q = A.get(k, lw) if l else Bq[j]
            for w, pw in ((1, q), (0, 1 - q)):
                if pw > 0:
                    key = (t + w, w, k + (w and l))
                    nd[key] = nd.get(key, 0.0) + pr * pw
        dist = nd
    P = np.zeros(n + 1)
    for (t, _, _), pr in dist.items():
        P[t] += pr
    return P


def var_of(P):
    k = np.arange(len(P))
    mu = (k * P).sum()
    return float(((k - mu) ** 2 * P).sum())


def analyze_hand(cards, cmask, tmask, n, kappa):
    X = ((cmask[:, None] >> np.arange(n)) & 1).astype(float)
    W = ((tmask[:, None] >> np.arange(n)) & 1).astype(int)
    suit = np.asarray(cards) // 13
    T = W.sum(1)
    assert (T == X.sum(1)).all()
    R = len(T)
    P_emp = hist(T, n)
    vpb = float((X.mean(0) * (1 - X.mean(0))).sum())
    if vpb < 1e-9:
        return None
    full = fit_models(X, W, suit, n, kappa)
    A, B = slice(0, R // 2), slice(R // 2, R)
    fa = fit_models(X[A], W[A], suit, n, kappa)
    PA, PBe = hist(T[A], n), hist(T[B], n)
    ev = (H + np.arange(n + 1)) * P_emp
    evB = (H + np.arange(n + 1)) * PBe
    be, bB = bid(P_emp), bid(PBe)
    vemp = float(T.var())
    o = {"floor_xtv": tv(PBe, PA), "floor_match": bid(PA) == bB, "floor_loss": float(evB[bB] - evB[bid(PA)])}
    for k in MODELS:
        P = full[k]
        o[k] = {"tv": tv(P_emp, P), "kl": kl(P_emp, P), "derr": (var_of(P) - vemp) / vpb,
                "match": bid(P) == be, "loss": float(ev[be] - ev[bid(P)]),
                "xtv": tv(PBe, fa[k]), "xmatch": bid(fa[k]) == bB, "xloss": float(evB[bB] - evB[bid(fa[k])])}
    o["delta"] = vemp / vpb
    o["beta"], o["gamma"] = full["_coef"]
    Lh = np.ones_like(W)
    Lh[:, 1:] = W[:, :-1]
    w2, l2 = W[:, 1:], Lh[:, 1:]
    o["p_lead"] = float(w2[l2 == 1].mean()) if (l2 == 1).any() else np.nan
    o["p_nolead"] = float(w2[l2 == 0].mean()) if (l2 == 0).any() else np.nan
    # lag-k autocorrelation of trick wins, averaged over trick pairs (j-k, j) (within-j: no Simpson pooling).
    # First-order lead chain => decays ~geometrically with k; exchangeable "good-world" luck => flat in k.
    for k in (1, 2, 3):
        cs = []
        for j in range(k, n):
            x, y = W[:, j - k], W[:, j]
            if x.std() > 0 and y.std() > 0:
                cs.append(float(np.corrcoef(x, y)[0, 1]))
        o[f"acf{k}"] = float(np.mean(cs)) if cs else np.nan
    return o


def mci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--cfg", default="5:NT,8:NT,13:NT")
    ap.add_argument("--tag", default="")
    ap.add_argument("--maxh", type=int, default=10 ** 9)
    a = ap.parse_args()
    summ = {}
    L = [f"M10 lead-state models (hero = first bidder/leader, model bid), kappa={a.kappa}; mean ± 95% CI over hands",
         "in-sample: TV, KL, δ err (δ_M − δ_emp), match (= emp-optimal bid), loss (pts/round); "
         "cross: fit A (first half of rollouts) vs emp B",
         "| play | n | type | hands | model | TV | KL | δ err | match | loss | xTV | xmatch | xloss | "
         "gap closed TV | gap closed match | gap closed loss | gap closed xmatch |",
         "|" + "---|" * 17]
    for cfg in a.cfg.split(","):
        n, t = cfg.split(":")
        n = int(n)
        fn = RES / f"m10_raw_n{n}_{t}.npz"
        if not fn.exists():
            continue
        D = np.load(fn)
        for k, pol in enumerate(D["pols"]):
            hs = [analyze_hand(D["cards"][i], D["cmask"][k, i], D["tmask"][k, i], n, a.kappa)
                  for i in range(min(len(D["cards"]), a.maxh))]
            hs = [h for h in hs if h is not None]
            key = f"{pol}|n={n}|{t}"
            d = {"hands": len(hs), "delta": mci([h["delta"] for h in hs]),
                 "floor_xtv": mci([h["floor_xtv"] for h in hs]), "floor_match": mci([h["floor_match"] for h in hs]),
                 "floor_loss": mci([h["floor_loss"] for h in hs])}
            both = [h for h in hs if np.isfinite(h["p_lead"]) and np.isfinite(h["p_nolead"])]
            d["p_win_lead"] = mci([h["p_lead"] for h in both])
            d["p_win_nolead"] = mci([h["p_nolead"] for h in both])
            d["persistence"] = mci([h["p_lead"] - h["p_nolead"] for h in both])
            d["LCB_beta"] = mci([h["beta"] for h in hs])
            d["LCB_gamma"] = mci([h["gamma"] for h in hs])
            for k in (1, 2, 3):
                d[f"acf{k}"] = mci([h[f"acf{k}"] for h in hs if np.isfinite(h[f"acf{k}"])])
            for m in MODELS:
                d[m] = {q: mci([h[m][q] for h in hs]) for q in hs[0][m]}
            pb = d["PB"]
            for m in MODELS:
                dm = d[m]
                dm["gap_tv"] = (pb["tv"][0] - dm["tv"][0]) / pb["tv"][0]
                dm["gap_match"] = (dm["match"][0] - pb["match"][0]) / (1 - pb["match"][0])
                dm["gap_loss"] = (pb["loss"][0] - dm["loss"][0]) / pb["loss"][0] if pb["loss"][0] > 0 else float("nan")
                fl = d["floor_match"][0]
                dm["gap_xmatch"] = (dm["xmatch"][0] - pb["xmatch"][0]) / (fl - pb["xmatch"][0])
                L.append(f"| {pol} | {n} | {t} | {len(hs)} | {m} | {dm['tv'][0]:.3f} ± {dm['tv'][1]:.3f} | "
                         f"{dm['kl'][0]:.3f} | {dm['derr'][0]:+.3f} ± {dm['derr'][1]:.3f} | "
                         f"{dm['match'][0]:.3f} ± {dm['match'][1]:.3f} | {dm['loss'][0]:.2f} ± {dm['loss'][1]:.2f} | "
                         f"{dm['xtv'][0]:.3f} | {dm['xmatch'][0]:.3f} ± {dm['xmatch'][1]:.3f} | "
                         f"{dm['xloss'][0]:.2f} ± {dm['xloss'][1]:.2f} | {dm['gap_tv']:.2f} | {dm['gap_match']:.2f} | "
                         f"{dm['gap_loss']:.2f} | {dm['gap_xmatch']:.2f} |")
            L.append(f"| {pol} | {n} | {t} | floor (emp A vs emp B) | – | – | – | δ_emp {d['delta'][0]:.3f} | – | – | "
                     f"{d['floor_xtv'][0]:.3f} | {d['floor_match'][0]:.3f} ± {d['floor_match'][1]:.3f} | "
                     f"{d['floor_loss'][0]:.2f} | | | | |")
            L.append(f"| {pol} | {n} | {t} | lead persistence P(W|lead) − P(W|no lead), tricks 2..n: "
                     f"{d['p_win_lead'][0]:.3f} − {d['p_win_nolead'][0]:.3f} = {d['persistence'][0]:.3f} ± "
                     f"{d['persistence'][1]:.3f} (hands {d['persistence'][2]}); trick-win autocorrelation lag 1/2/3: "
                     + " / ".join(f"{d[f'acf{k}'][0]:+.3f} ± {d[f'acf{k}'][1]:.3f}" for k in (1, 2, 3))
                     + f"; LCB coef lead β {d['LCB_beta'][0]:+.2f} ± {d['LCB_beta'][1]:.2f}, budget γ "
                     f"{d['LCB_gamma'][0]:+.2f} ± {d['LCB_gamma'][1]:.2f} (logit units) |" + " |" * 12)
            summ[key] = d
            print(L[-1], flush=True)
    json.dump(summ, open(RES / f"m10_summary{a.tag}.json", "w"), indent=1)
    (RES / f"m10_tables{a.tag}.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

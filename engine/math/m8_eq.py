"""M8: 2-card round (trump S), fixed TA play, equilibrium bidding by iterated best responses.

Data: results/m8_raw_ta.npz (m8_sim.py): N deals x 81 bid vectors -> trick counts (TA play depends only on
each player's own bid, deterministically).  Seat 0 bids first and leads; seat 3 = dealer (hook: b3 != 2 - sum).
Information: own hand class (2 cards up to permutation of the side suits H,D,C) + prior bids.
Best responses are computed on deal half A by grouping deals by (class, history) (this is the exact MC
posterior: deals are uniform, and the history is a deterministic function of the earlier hands under the
current profile).  Gauss-Seidel sweeps dealer -> seat 0 until no change.  Seat values on held-out half B.
Also: myopic variant (later bidders do not react to my bid), ModelBidder (TA model) profile, hook off,
threshold/index tests.  Output: results/m8_eq.json, m8_tables.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "math" / "results"
H = 10


def card_list(h):
    return [c for c in range(52) if h >> c & 1]


def hand_key(h):
    a, b = card_list(int(h))
    sa, ra, sb, rb = a // 13, a % 13, b // 13, b % 13
    ta, tb = sa == 0, sb == 0
    if ta and tb:
        return ("TT", max(ra, rb), min(ra, rb))
    if ta or tb:
        return ("TX", ra if ta else rb, rb if ta else ra)
    if sa == sb:
        return ("XXs", max(ra, rb), min(ra, rb))
    return ("XXd", max(ra, rb), min(ra, rb))


def load():
    D = np.load(RES / "m8_raw_ta.npz")
    Hs, T = D["hands"], D["tricks"].astype(np.int16)
    t0, t1, t2 = T % 3, (T // 3) % 3, T // 9
    t3 = 2 - t0 - t1 - t2
    tr = np.stack([t0, t1, t2, t3], -1).astype(np.int8)  # (N, 81, 4)
    keys = {}
    cls = np.zeros(Hs.shape, np.int32)
    for i in range(Hs.shape[0]):
        for s in range(4):
            k = hand_key(Hs[i, s])
            cls[i, s] = keys.setdefault(k, len(keys))
    inv = [None] * len(keys)
    for k, v in keys.items():
        inv[v] = k
    return Hs, tr, cls, inv


class Game:
    def __init__(self, tr, cls, K, hook=True):
        self.tr, self.cls, self.K, self.hook = tr, cls, K, hook
        self.N = len(cls)
        self.ar = np.arange(self.N)

    def bids(self, S, fix=None):
        """S = [S0 (K), S1 (K,3), S2 (K,3,3), S3 (K,3,3,3)]; fix = (seat k, bid array, react) overrides seat k.
        If react is False, later seats see the PROFILE bid of seat k (myopic counterfactual)."""
        c = self.cls
        b = np.zeros((self.N, 4), np.int64)
        prof = np.zeros((self.N, 4), np.int64)
        for k in range(4):
            idx = (c[:, k],) + tuple(prof[:, j] for j in range(k))
            pk = S[k][idx]
            prof[:, k] = pk
            b[:, k] = pk
            if fix is not None and fix[0] == k:
                b[:, k] = fix[1]
                if fix[2]:
                    prof[:, k] = fix[1]
        return b

    def scores(self, b):
        idx = b[:, 0] * 27 + b[:, 1] * 9 + b[:, 2] * 3 + b[:, 3]
        t = self.tr[self.ar, idx]
        return np.where(t == b, H + b, 0).astype(float), t

    def br(self, S, k, react=True):
        """Best response of seat k; returns new S[k] array and per-infoset EV table."""
        c = self.cls
        prof = self.bids(S)
        hist = np.zeros(self.N, np.int64)
        for j in range(k):
            hist = hist * 3 + prof[:, j]
        nh = 3 ** k
        key = c[:, k] * nh + hist
        ev = np.full((self.K * nh, 3), np.nan)
        cnt = np.bincount(key, minlength=self.K * nh)
        for bb in range(3):
            b = self.bids(S, (k, np.full(self.N, bb), react))
            sc, _ = self.scores(b)
            ev[:, bb] = np.bincount(key, weights=sc[:, k], minlength=self.K * nh) / np.maximum(cnt, 1)
        if k == 3 and self.hook:
            hs = np.arange(nh)
            ssum = hs // 9 + (hs // 3) % 3 + hs % 3
            forb = 2 - ssum
            for h in range(nh):
                if 0 <= forb[h] <= 2:
                    ev[h::nh, forb[h]] = -np.inf
        new = S[k].reshape(-1).copy()
        ok = cnt > 0
        new[ok] = np.argmax(ev[ok], 1)
        return new.reshape(S[k].shape), ev, cnt

    def values(self, S):
        b = self.bids(S)
        sc, t = self.scores(b)
        return sc, b, t


def init_profile(K, hook):
    S = [np.zeros(K, np.int64), np.zeros((K, 3), np.int64), np.zeros((K, 3, 3), np.int64),
         np.zeros((K, 3, 3, 3), np.int64)]
    if hook:  # dealer: Σprior = 2 forbids 0 -> bid 1
        for b0 in range(3):
            for b1 in range(3):
                for b2 in range(3):
                    if b0 + b1 + b2 == 2:
                        S[3][:, b0, b1, b2] = 1
    return S


def solve(g, S, sweeps=25, react=True, log=None):
    hist = []
    for it in range(sweeps):
        changes = 0
        for k in (3, 2, 1, 0):
            new, _, cnt = g.br(S, k, react)
            # mass-weighted change count
            changes += int((new != S[k]).sum())
            S[k] = new
        hist.append(changes)
        if log is not None:
            print("  sweep", it, "changed infosets", changes, flush=True)
        if changes == 0:
            break
    return S, hist


def mci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def model_profile_bids(Hs, hook):
    """ModelBidder (TA model) bids on each deal, seats in order, dealer best legal."""
    from judgement import Rules
    from judgement.bidding.features import hand_features
    from judgement.bidding.model import BidModel, score_matrix
    rules = Rules()
    m = BidModel.load(ROOT / "experiments" / "phase1" / "results" / "model_ta_final.json").get(2, 0)
    Sm = score_matrix(rules, 2)
    out = np.zeros((len(Hs), 4), np.int64)
    for i in range(len(Hs)):
        prior = []
        for k in range(4):
            P = m.dist_all_bids(hand_features(int(Hs[i, k]), 0, 2, k, prior))
            ev = (P * Sm).sum(1)
            legal = [b for b in range(3) if not (hook and k == 3 and b == 2 - sum(prior))]
            b = max(legal, key=lambda x: (ev[x], -x))
            out[i, k] = b
            prior.append(b)
    return out


def lnds_mass(order_vals, bids, w):
    """max mass of a subsequence (in index order) with non-decreasing bids; DP over 3 bid levels."""
    best = np.zeros(3)
    for v, b, ww in sorted(zip(order_vals, bids, w)):
        best[b] = best[: b + 1].max() + ww
        for j in range(b + 1, 3):
            best[j] = max(best[j], best[b])
    return best.max()


def threshold_test(g, S, inv, k, static_index):
    """Per history h of seat k: is the bid monotone in an index of the class?
    dynamic index: E[t_k | class, h, b_k = 1]; static: history-free per-class index."""
    _, ev, cnt = g.br(S, k)
    nh = 3 ** k
    # dynamic index from tricks under b_k = 1
    prof = g.bids(S)
    hist = np.zeros(g.N, np.int64)
    for j in range(k):
        hist = hist * 3 + prof[:, j]
    key = g.cls[:, k] * nh + hist
    b1 = g.bids(S, (k, np.ones(g.N, np.int64), True))
    _, t = g.scores(b1)
    Et = np.bincount(key, weights=t[:, k], minlength=g.K * nh) / np.maximum(cnt, 1)
    res = {"dyn": [0.0, 0.0], "static": [0.0, 0.0]}
    Sk = S[k].reshape(-1)
    for h in range(nh):
        rows = np.arange(g.K) * nh + h
        w = cnt[rows].astype(float)
        if w.sum() < 50:
            continue
        m = w > 0
        bids = Sk[rows][m]
        for nm, idx in (("dyn", Et[rows][m]), ("static", static_index[m])):
            res[nm][0] += lnds_mass(idx, bids, w[m])
            res[nm][1] += w[m].sum()
    return {nm: v[0] / v[1] for nm, v in res.items()}


def main():
    Hs, tr, cls, inv = load()
    K = len(inv)
    N = len(cls)
    A = np.arange(N) < N // 2
    out = {"N_deals": int(N), "classes": K}
    L = []
    profiles = {}
    for hook in (True, False):
        gA, gB = Game(tr[A], cls[A], K, hook), Game(tr[~A], cls[~A], K, hook)
        for react in (True, False):
            nm = f"{'hook' if hook else 'nohook'}|{'eq' if react else 'myopic'}"
            print(nm, flush=True)
            S, hist = solve(gA, init_profile(K, hook), react=react, log=True)
            sc, b, t = gB.values(S)
            profiles[nm] = S
            out[nm] = {"sweeps_changes": hist, "seat_values_B": [mci(sc[:, k]) for k in range(4)],
                       "table_B": mci(sc.sum(1)), "hit_B": [mci(t[:, k] == b[:, k]) for k in range(4)],
                       "bid_dist": [np.bincount(b[:, k], minlength=3).tolist() for k in range(4)],
                       "P(sum bids = 2)": mci(b.sum(1) == 2)}
    # model-bidder profile (sanity vs Phase-1 E5 hook cost 2.84)
    print("model bids", flush=True)
    for hook in (True, False):
        mb = model_profile_bids(Hs[~A], hook)
        gB = Game(tr[~A], cls[~A], K, hook)
        sc, _ = gB.scores(mb)
        out[f"{'hook' if hook else 'nohook'}|model"] = {"seat_values_B": [mci(sc[:, k]) for k in range(4)],
                                                          "table_B": mci(sc.sum(1))}
        if hook:
            sc_mh = sc
        else:
            out["hook_cost_model_paired"] = mci(sc[:, 3] - sc_mh[:, 3])
    # equilibrium hook cost: paired on B deals
    gBh, gBn = Game(tr[~A], cls[~A], K, True), Game(tr[~A], cls[~A], K, False)
    sh, _, _ = gBh.values(profiles["hook|eq"])
    sn, _, _ = gBn.values(profiles["nohook|eq"])
    out["hook_cost_eq_paired"] = mci(sn[:, 3] - sh[:, 3])
    out["hook_cost_eq_others"] = [mci(sn[:, k] - sh[:, k]) for k in range(3)]
    # myopic vs eq: info-set disagreement mass and value gap (hook)
    gA = Game(tr[A], cls[A], K, True)
    Se, Sm = profiles["hook|eq"], profiles["hook|myopic"]
    be, bm = gA.bids(Se), gA.bids(Sm)
    out["myopic_vs_eq_bid_disagree_by_seat"] = [mci(be[:, k] != bm[:, k]) for k in range(4)]
    # unilateral: seat k plays myopic BR against eq profile of others (react=False BR to eq)
    uni = []
    gB = Game(tr[~A], cls[~A], K, True)
    base, _, _ = gB.values(Se)
    for k in range(4):
        new, _, _ = gA.br([s.copy() for s in Se], k, react=False)
        S2 = [s.copy() for s in Se]
        S2[k] = new
        sc2, _, _ = gB.values(S2)
        uni.append({"seat": k, "infosets_changed": int((new != Se[k]).sum()),
                    "value_change_B": mci(sc2[:, k] - base[:, k])})
    out["unilateral_myopic_deviation"] = uni
    # threshold / index tests (hook eq)
    # static index: E[t_0 | class] for seat 0 under b0 = 1 (history-free hand strength)
    g = gA
    b1 = g.bids(Se, (0, np.ones(g.N, np.int64), True))
    _, t = g.scores(b1)
    cnt0 = np.bincount(g.cls[:, 0], minlength=K)
    static = np.bincount(g.cls[:, 0], weights=t[:, 0], minlength=K) / np.maximum(cnt0, 1)
    out["threshold_consistency(mass frac, hook eq)"] = {k: threshold_test(g, Se, inv, k, static) for k in range(4)}
    # strategy description for seat 0 and seat 1
    desc = {}
    for k in (0, 1):
        Sk = Se[k]
        for h in range(3 ** k):
            row = Sk if k == 0 else Sk[:, h]
            groups = {}
            for c in range(K):
                groups.setdefault(int(row[c]), []).append(inv[c])
            desc[f"seat{k}|h={h}"] = {b: sorted(v)[:200] for b, v in groups.items()}
    out["eq_strategy_seat0_seat1"] = desc
    json.dump(out, open(RES / "m8_eq.json", "w"), indent=1, default=str)

    def f(m, d=2):
        return f"{m[0]:.{d}f} ± {m[1]:.{d}f}"
    L.append(f"M8 2-card round, trump S, TA play; {N} deals (BR on half A, values on held-out half B, {N - N // 2} deals); {K} hand classes")
    L.append("| profile | pos0 (leads) | pos1 | pos2 | dealer | table | P(Σb=2) | sweeps (infosets changed) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for nm in ("hook|eq", "hook|myopic", "hook|model", "nohook|eq", "nohook|myopic", "nohook|model"):
        d = out[nm]
        L.append(f"| {nm} | " + " | ".join(f(v) for v in d["seat_values_B"]) + f" | {f(d['table_B'])} | "
                 + (f"{d['P(sum bids = 2)'][0]:.3f}" if "P(sum bids = 2)" in d else "–") + " | "
                 + (str(d.get("sweeps_changes", "–"))) + " |")
    L.append(f"Hook cost (dealer, paired on B): equilibrium {f(out['hook_cost_eq_paired'])}; model-bidder {f(out['hook_cost_model_paired'])}; Phase-1 E5 measured 2.84 ± 0.08")
    L.append("Others' change (nohook − hook, eq): " + ", ".join(f(v) for v in out["hook_cost_eq_others"]))
    L.append("Myopic vs eq bid disagreement by seat (deal mass): " + ", ".join(f(v, 3) for v in out["myopic_vs_eq_bid_disagree_by_seat"]))
    L.append("Unilateral myopic deviation from eq (value change on B): " + ", ".join(
        f"seat{u['seat']} {f(u['value_change_B'], 3)} ({u['infosets_changed']} infosets)" for u in uni))
    L.append("Threshold consistency (mass fraction explained by a monotone rule; dyn = E[t|class,h,b=1], static = seat-0 E[t|class]): "
             + json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in out["threshold_consistency(mass frac, hook eq)"].items()}))
    open(RES / "m8_tables.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

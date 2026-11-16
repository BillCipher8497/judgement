"""S1: fit / evaluate the human bidding count on s1_train.pkl, report on s1_test.pkl.

Objective = mean counterfactual score cf[rule bid] (true-deal playouts, see s1_data.py).
Rule bids outside the stored candidates (n > 6: MB bid +-4) are clipped to the nearest candidate (rate reported).
"""
import json
import pickle
import sys

import numpy as np

from common import RES
from judgement.cards import NT
from judgement.policies.human_guide import GUIDE_BID, bid_counts

KEYS = ["tA", "tK", "tQ", "tlen", "sA", "sK2", "sK1", "sQ3", "voids", "singles", "nt_long", "sAlong", "low", "sQ", "sJ"]


def load(name):
    with open(RES / name, "rb") as fh:
        rows = pickle.load(fh)
    N = len(rows)
    D = {k: np.zeros(N, int) for k in KEYS}
    meta = {k: np.zeros(N, int) for k in ["n", "nt", "pos", "prior", "bm", "bo", "dealer_forb"]}
    cf = np.full((N, 14), np.nan)
    orc = np.full((N, 14), np.nan)
    for i, r in enumerate(rows):
        c = bid_counts(r["hand"], r["trump"], r["n"])
        for k in KEYS:
            D[k][i] = c[k]
        n = r["n"]
        meta["n"][i] = n; meta["nt"][i] = r["trump"] == NT; meta["pos"][i] = r["pos"]
        meta["prior"][i] = sum(r["prior"]); meta["bm"][i] = r["bm"]; meta["bo"][i] = r.get("bo", -1)
        meta["dealer_forb"][i] = (n - sum(r["prior"])) if r["pos"] == 3 else -1
        for b, v in r["cf"].items():
            cf[i, b] = v
        for b, v in r.get("orc", {}).items():
            orc[i, b] = v
    return D, meta, cf, orc


def rule_bids(w, D, meta):
    """Vectorised HumanBidder (checked equal to judgement.policies.human_guide.HumanBidder in tests)."""
    n, nt = meta["n"], meta["nt"].astype(bool)
    sm = n <= w["small_cut"]
    sideK = D["sK2"] + D["sK1"]
    pn = (w["n_A"] * D["sA"] + w["n_K2"] * D["sK2"] + w["n_K1"] * D["sK1"] + w["n_Q3"] * D["sQ3"] + w["n_long"] * D["nt_long"] - n // w["n_sub"])
    L = D["tlen"]
    g = w["t_guard"]
    hon = D["tA"] + ((D["tK"] == 1) & ((L >= 2) | (not g))) + ((D["tQ"] == 1) & ((L >= 3) | (not g)))
    pt = (w["t_each"] * L + w["t_hon"] * hon
          + np.where(sm, w["small_A"] * D["sA"] + w["small_K"] * sideK,
                     w["s_A"] * D["sA"] + w["s_K2"] * D["sK2"] + w["s_K1"] * D["sK1"])
          + w["s_Q3"] * D["sQ3"] + np.where(L >= 2, w["short"] * (2 * D["voids"] + D["singles"]), 0) - n // w["t_sub"])
    extra = w["low_w"] * D["low"] + w["s_Q"] * D["sQ"] + w["s_J"] * D["sJ"]
    p = np.maximum(0, np.where(nt, pn, pt) + extra)
    half = p % 2
    b = p // 2 + (half if w["round_up"] else 0)
    share = meta["pos"] * n / 4.0
    ex = meta["prior"] - share
    b = b - np.ceil(ex / w["prior_k"] - 0.5).astype(int)
    b = np.clip(b, 0, n)
    forb = meta["dealer_forb"]
    hooked = b == forb
    up = np.where(half == 1, 1 - w["round_up"], w["hook_up"]).astype(bool)
    alt = np.where(up & (b + 1 <= n), b + 1, np.where(b - 1 >= 0, b - 1, b + 1))
    return np.where(hooked, alt, b)


def value(b, cf):
    N = len(b)
    v = cf[np.arange(N), b]
    miss = np.isnan(v)
    if miss.any():
        for i in np.where(miss)[0]:
            cand = np.where(~np.isnan(cf[i]))[0]
            v[i] = cf[i, cand[np.argmin(np.abs(cand - b[i]))]]
    return v, miss.mean()


GRID = dict(t_each=[0, 1, 2, 3], t_hon=[0, 1, 2, 3], t_guard=[0, 1], s_A=[0, 1, 2, 3], s_K2=[0, 1, 2], s_K1=[0, 1, 2],
            s_Q3=[0, 1, 2], short=[0, 1, 2], t_sub=[1, 2, 3, 4, 5, 6, 8, 99],
            n_A=[1, 2, 3, 4], n_K2=[0, 1, 2, 3], n_K1=[0, 1, 2], n_Q3=[0, 1, 2], n_long=[0, 1, 2], n_sub=[2, 3, 4, 6, 8, 99],
            low_w=[-2, -1, 0], s_Q=[0, 1, 2], s_J=[0, 1],
            small_cut=[0, 1, 2, 3, 4, 5, 6], small_A=[0, 1, 2, 3], small_K=[0, 1, 2, 3], prior_k=[1, 2, 3, 4, 5, 6, 99], hook_up=[0, 1], round_up=[0, 1])


def fit(D, meta, cf, w0, keys, sweeps=3, log=print):
    w = dict(w0)
    best = value(rule_bids(w, D, meta), cf)[0].mean()
    for s in range(sweeps):
        changed = False
        for k in keys:
            for v in GRID[k]:
                if v == w[k]:
                    continue
                w2 = dict(w); w2[k] = v
                m = value(rule_bids(w2, D, meta), cf)[0].mean()
                if m > best + 1e-9:
                    best, w, changed = m, w2, True
        log(f"sweep {s} best {best:.4f} {w}")
        if not changed:
            break
    return w, best


if __name__ == "__main__":
    D, meta, cf, orc = load("s1_train.pkl")
    vm = value(meta["bm"], cf)[0].mean()
    print("train MB", vm)
    w = dict(GUIDE_BID)
    w, best = fit(D, meta, cf, w, list(GRID))
    json.dump(w, open(RES / "s1_fit_full.json", "w"))

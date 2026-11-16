"""Stretch: n=1 exact game.  Compare the Phase-1 bidder model (ModelBidder) with
(i) the Phase-2 exact iterated-best-response equilibrium (onecard.json) and
(ii) an exact unilateral best response to a table of ModelBidders.

All 52*51*50*49 ordered deals enumerated (positions 0..3, 0 leads, 3 = dealer with
hook).  Play is forced.  Strategy of position k: (own card, earlier-bid history) -> bid.
Exact expectation (no sampling) -> no CI needed; values are per round."""
from __future__ import annotations

import json

import numpy as np

from common2 import ROOT, dump, load_model  # noqa: E402
from judgement import Rules
from judgement.bidding.features import hand_features
from judgement.bidding.model import score_matrix
from judgement.cards import NT

RANKS = "23456789TJQKA"
SUITS = "SHDC"


def deals():
    c = np.arange(52)
    g = np.stack(np.meshgrid(c, c, c, c, indexing="ij"), -1).reshape(-1, 4)
    ok = (g[:, 0] != g[:, 1]) & (g[:, 0] != g[:, 2]) & (g[:, 0] != g[:, 3]) & (g[:, 1] != g[:, 2]) & \
         (g[:, 1] != g[:, 3]) & (g[:, 2] != g[:, 3])
    return g[ok].astype(np.int16)


def winners(D, trump):
    suit, rank = D // 13, D % 13
    led = suit[:, :1]
    key = np.where(suit == led, 50 + rank, 0)
    if trump != NT:
        key = np.where(suit == trump, 100 + rank, key)
    return key.argmax(1)


def model_table(model, rules, trump):
    """T[k][hist_int][card] = model bid (hist encoded as bits of earlier bids, in order)."""
    S = score_matrix(rules, 1)
    cl = model.get(1, trump)
    T = []
    for k in range(4):
        tk = np.zeros((2 ** k, 52), dtype=np.int8)
        for h in range(2 ** k):
            prior = [(h >> (k - 1 - j)) & 1 for j in range(k)]
            for c in range(52):
                x = hand_features(1 << c, trump, 1, k, prior)
                ev = (cl.dist_all_bids(x) * S).sum(1)
                legal = [0, 1]
                if k == 3:
                    legal = [b for b in legal if b != 1 - sum(prior)]
                tk[h, c] = max(legal, key=lambda b: (ev[b], -b))
        T.append(tk)
    return T


def eq_table(entry):
    T = []
    tab = entry["bid1_cards_by_pos_and_history"]
    for k in range(4):
        tk = np.zeros((2 ** k, 52), dtype=np.int8)
        for h in range(2 ** k):
            hs = "".join(str((h >> (k - 1 - j)) & 1) for j in range(k)) or "-"
            v = tab[str(k)].get(hs, "none")
            if v != "none":
                for s, ranks in v.items():
                    for r in ranks:
                        tk[h, SUITS.index(s) * 13 + RANKS.index(r)] = 1
        T.append(tk)
    return T


def play(D, W, T, override=None):
    """Bids under table T (override = (k, bids array for position k)) -> scores (M,4), hist."""
    M = len(D)
    h = np.zeros(M, dtype=np.int64)
    bids = np.zeros((M, 4), dtype=np.int8)
    for k in range(4):
        b = T[k][h, D[:, k]] if override is None or override[0] != k else override[1]
        bids[:, k] = b
        h = h * 2 + b
    won = (W[:, None] == np.arange(4)[None, :])
    sc = np.where(bids == won, 10 + bids, 0)
    return sc, bids


def best_response(D, W, T, k):
    """Exact BR of position k to the others' tables; returns new table row and value."""
    M = len(D)
    h = np.zeros(M, dtype=np.int64)
    for j in range(k):
        h = h * 2 + T[j][h, D[:, j]]
    vals = []
    for b in (0, 1):
        sc, _ = play(D, W, T, override=(k, np.full(M, b, dtype=np.int8)))
        vals.append(sc[:, k].astype(float))
    cell = h * 52 + D[:, k]
    ncell = 2 ** k * 52
    cnt = np.bincount(cell, minlength=ncell)
    v0 = np.bincount(cell, weights=vals[0], minlength=ncell)
    v1 = np.bincount(cell, weights=vals[1], minlength=ncell)
    legal1 = np.ones(ncell, bool)
    legal0 = np.ones(ncell, bool)
    if k == 3:
        hh = np.arange(ncell) // 52
        s = np.array([bin(x).count("1") for x in hh])
        legal1 &= s != 0          # forbidden bid = 1 - s
        legal0 &= s != 1
    choose1 = np.where(legal1 & legal0, v1 > v0, legal1)
    newk = choose1.reshape(2 ** k, 52).astype(np.int8)
    T2 = list(T)
    T2[k] = newk
    return T2


def main():
    rules = Rules()
    model = load_model()
    D = deals()
    eq = {e["trump"]: e for e in json.load(open(ROOT / "experiments/phase2/results/onecard.json"))}
    out = {}
    lines = ["## Stretch: n=1 exact game (all 6,497,400 ordered deals; exact expectations, no sampling error)", "",
             "| trump | profile | pos0 | pos1 | pos2 | dealer | table |", "|---|---|---|---|---|---|---|"]
    lines2 = ["", "| trump | pos | EV(model) at model table | EV(exact BR to model table) | EV lost by model | "
              "cells where BR≠model (reached) | model≠Phase-2 eq. bid (reached under eq.) |",
              "|---|---|---|---|---|---|---|"]
    for tname, trump in (("S", 0), ("NT", NT), ("H", 1)):
        W = winners(D, trump)
        Tm = model_table(model, rules, trump)
        scm, bm = play(D, W, Tm)
        vm = scm.mean(0)
        rec = {"model": vm.tolist()}
        lines.append(f"| {tname} | all ModelBidder | " + " | ".join(f"{x:.3f}" for x in vm) + f" | {vm.sum():.3f} |")
        Te = None
        if tname in eq:
            Te = eq_table(eq[tname])
            sce, be = play(D, W, Te)
            ve = sce.mean(0)
            rec["eq"] = ve.tolist()
            lines.append(f"| {tname} | Phase-2 IBR equilibrium | " + " | ".join(f"{x:.3f}" for x in ve) + f" | {ve.sum():.3f} |")
        for k in range(4):
            Tb = best_response(D, W, Tm, k)
            scb, bb = play(D, W, Tb)
            dif = (bb[:, k] != bm[:, k]).mean()
            mism = (be[:, k] != Tm[k][_hist(be, k), D[:, k]]).mean() if Te is not None else float("nan")
            rec[f"br{k}"] = float(scb[:, k].mean())
            lines2.append(f"| {tname} | {k} | {vm[k]:.3f} | {scb[:, k].mean():.3f} | {scb[:, k].mean() - vm[k]:.3f} | "
                          f"{dif:.3f} | {mism:.3f} |")
        out[tname] = rec
        print(tname, "done", flush=True)
    txt = "\n".join(lines + lines2)
    dump({"res": out, "table": txt}, "p4_onecard.json")
    print(txt)


def _hist(bids, k):
    h = np.zeros(len(bids), dtype=np.int64)
    for j in range(k):
        h = h * 2 + bids[:, j]
    return h


if __name__ == "__main__":
    main()

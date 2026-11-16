"""Build all Phase 1 tables from results/*.json|npz.  Prints markdown and writes
results/phase1_tables.md.  Every number: mean ± 1.96·SE (or Wilson 95% CI) and n."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from judgement import Rules  # noqa: E402

from common import RES, dist_stats, fmt, is_bimodal, mean_se, prop_ci  # noqa: E402


def dip_depth(p):
    p = np.asarray(p, float)
    if len(p) < 3:
        return 0.0
    left = np.maximum.accumulate(p)
    right = np.maximum.accumulate(p[::-1])[::-1]
    return float(max(0.0, max(min(left[k - 1], right[k + 1]) - p[k] for k in range(1, len(p) - 1))))

OUT = []
RULES = Rules()
POSNAME = {0: "pos0 (left of dealer, bids 1st, leads)", 1: "pos1", 2: "pos2", 3: "dealer (bids last)"}


def P(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)


def table(header, rows):
    P("| " + " | ".join(header) + " |")
    P("|" + "---|" * len(header))
    for r in rows:
        P("| " + " | ".join(str(x) for x in r) + " |")
    P("")


def load(name):
    p = RES / name
    if not p.exists():
        return None
    if name.endswith(".npz"):
        return dict(np.load(p))
    return json.load(open(p))


def rnd(x):
    return int(np.floor(x + 0.5))


# ---------------------------------------------------------------- fixed point
def fixed_point():
    for tag in ("ta", "greedy", "ta_neg_bid", "ta_neg_10", "ta_neg_diff"):
        L = load(f"fixed_point_{tag}.json")
        if not L:
            continue
        P(f"### Fixed point: {tag} (rounds/n/iter={L['rounds_per_n']}, explore={L['explore']}; eval = 400 fixed deals x 13 n x 4 seats = 20,800 bids)")
        rows = []
        for it in L["iters"]:
            c = it.get("change_vs_prev")
            t = it["table"]
            rows.append([it["iter"], "damped" if it.get("damped") else "plain",
                         f"{it['data_hit_rate_unexplored']:.3f}",
                         f"{c['frac_identical']:.3f}" if c else "-", f"{c['mean_abs_change']:.3f}" if c else "-",
                         f"{c['tv_bid_hist']:.3f}" if c else "-",
                         f"{t['13']['mean_bid']:.2f}", f"{t['13']['mean_sum_minus_n']:+.2f}",
                         f"{t['7']['mean_sum_minus_n']:+.2f}"])
        table(["iter", "update", "self-play hit rate", "bids identical vs prev", "mean abs dbid", "TV(bid hist)",
               "mean bid n=13", "sum-n n=13", "sum-n n=7"], rows)
        if "noise_floor_same_policy_two_datasets" in L:
            nf = L["noise_floor_same_policy_two_datasets"]
            f2 = L["final_vs_refit_on_fresh_pooled_data"]
            P(f"Noise floor (two independent {L['rounds_per_n']}-round/n datasets under the final model, fitted separately): identical={nf['frac_identical']:.3f}, mean|d|={nf['mean_abs_change']:.3f}, TV={nf['tv_bid_hist']:.3f}.  "
              f"Final model vs refit on fresh pooled data under it: identical={f2['frac_identical']:.3f}, mean|d|={f2['mean_abs_change']:.3f}, TV={f2['tv_bid_hist']:.3f}.")
            P("")


# ---------------------------------------------------------------- bidder
def gammas():
    p = RES / "model_ta_final.json"
    if not p.exists():
        return
    from judgement.bidding.model import GAMMA_NAMES, BidModel
    bm = BidModel.load(p)
    P("### TA bidder: fitted bid-pull coefficients γ (conditional logit; logit_t += Σ γ_j F_j(t,b))")
    rows = []
    for n in (2, 5, 8, 13):
        for nt in (False, True):
            if (n, nt) in bm.models:
                g = bm.models[(n, nt)].gamma
                rows.append([n, "NT" if nt else "trump"] + [f"{x:+.2f}" for x in g])
    table(["n", "type"] + GAMMA_NAMES, rows)


def bidder():
    gammas()
    for tag in ("ta", "greedy"):
        B = load(f"bidder_eval_{tag}.json")
        if not B:
            continue
        cal, o = B["calibration"], B["oracle"]
        P(f"### Bidder accuracy ({tag} play)")
        P(f"Held-out self-play rows: {cal['n_rows']} (1000 clean rounds per n, no exploration).  Log-loss of P(t): model {fmt(*cal['logloss_model'], 3)} vs per-n marginal {fmt(*cal['logloss_marginal_by_n'], 3)} nats.  "
          f"Mean predicted P(hit) {fmt(*cal['mean_pred_hit'], 3)} vs realized {fmt(*cal['mean_real_hit'], 3)}; mean predicted E[t] {cal['mean_pred_Et'][0]:.3f} vs realized {cal['mean_real_t'][0]:.3f}.")
        P("")
        table(["pred P(hit) bin", "rows", "mean pred", "realized ± 95%"],
              [[r["bin"], r["n"], f"{r['pred']:.3f}", fmt(r["real"], r["se"], 3)] for r in cal["reliability"]])
        table(["n", "pred hit", "real hit ± 95%", "pred E[t]", "real E[t]"],
              [[r["n"], f"{r['pred_hit']:.3f}", fmt(r["real_hit"], r["se"], 3), f"{r['pred_Et']:.2f}",
                f"{r['real_t']:.2f}"] for r in cal["by_n"]])
        P(f"Oracle comparison: {o['n_situations']} bidding situations (n=1..13 x trump S,H,D,C,NT x 4 positions, priors from model), "
          f"{B['samples']} MC worlds per situation (CRN across candidate bids; TA candidates = model bid ±3).")
        table(["metric", "value"], [
            ["model bid == oracle bid", fmt(*o["match_rate"], 3)],
            ["|model − oracle| ≤ 1", fmt(*o["within1"], 3)],
            ["EV loss vs oracle, in-sample (biased up)", fmt(*o["ev_loss_insample"], 3)],
            ["EV loss, split-sample (oracle picked on half A, scored on half B)", fmt(*o["ev_loss_heldout_split"], 3)],
            ["mean oracle EV / round", fmt(*o["mean_ev_oracle"], 3)],
            ["oracle at candidate-window edge", f"{o['frac_oracle_at_window_edge']:.3f}"]])
        recs = load(f"oracle_compare_{tag}.json")
        edge = 0
        for r in recs:
            ks = sorted(int(k) for k in r["dists"])
            if tag != "greedy" and ((r["b_oracle"] == ks[0] and ks[0] > 0 and r["b_model"] - 3 == ks[0]) or
                                    (r["b_oracle"] == ks[-1] and ks[-1] < r["n"] and r["b_model"] + 3 == ks[-1])):
                edge += 1
        P(f"Oracle bid at a truncated edge of the ±3 candidate window: {edge}/{len(recs)} (the summary's 'window edge' "
          f"field above also counts natural edges b=0/b=n and is superseded by this line).")
        P("")
        rows = []
        for n in range(1, 14):
            rr = [r for r in recs if r["n"] == n]
            if not rr:
                continue
            m = mean_se([r["b_model"] == r["b_oracle"] for r in rr])
            l = mean_se([r["ev_heldout_oracleA"] - r["ev_heldout_model"] for r in rr])
            rows.append([n, len(rr), fmt(*m, 2), fmt(*l, 2)])
        table(["n", "situations", "match", "EV loss (split)"], rows)


# ---------------------------------------------------------------- E1 / E2
def e1():
    P("### E1 — distribution shift, all-Greedy vs all-TA (hero seat 1 = first bidder, dealer 0, trump S)")
    P("Greedy: 400 MC worlds per hand (bid-free).  TA: 100 CRN worlds per candidate bid; TA distribution "
      "reported at the TA oracle bid b*_TA (the distribution the hero actually faces when bidding optimally).  "
      "± = 1.96·SE over hands (paired).")
    rows, rows2 = [], []
    allrecs = {}
    for n in (13, 8, 5):
        R = load(f"e1_n{n}.json")
        if not R:
            continue
        allrecs[n] = R
        sg = np.array([dist_stats(r["greedy"]) for r in R])
        st = np.array([dist_stats(r["ta"][str(r["b_ta"])]) for r in R])
        H = len(R)
        for j, nm in enumerate(["mean", "variance", "entropy (bits)"]):
            d = st[:, j] - sg[:, j]
            rows.append([n, H, nm, f"{sg[:, j].mean():.3f}", f"{st[:, j].mean():.3f}", fmt(*mean_se(d), 3)])
        mode_same = mean_se(sg[:, 3] == st[:, 3])
        rows.append([n, H, "mode equal", "", "", fmt(*mode_same, 3)])
        bg = np.array([r["b_greedy"] for r in R])
        bt = np.array([r["b_ta"] for r in R])
        ch = int((bg != bt).sum())
        lo, hi = prop_ci(ch, H)
        evg = np.array([r["ev_greedy"] for r in R])
        evt = np.array([r["ev_ta"] for r in R])
        ev_tg = np.array([r["ta_ev_at_greedy_bid"] for r in R])
        phit_g = np.array([r["greedy"][r["b_greedy"]] for r in R])
        phit_t = np.array([r["ta"][str(r["b_ta"])][r["b_ta"]] for r in R])
        rows2.append([n, H, f"{ch / H:.3f} [{lo:.3f},{hi:.3f}]", fmt(*mean_se(np.abs(bt - bg)), 3),
                      fmt(*mean_se(bt - bg), 3), f"{np.mean(bt > bg):.3f}/{np.mean(bt < bg):.3f}",
                      f"{phit_g.mean():.3f} → {phit_t.mean():.3f}", fmt(*mean_se(evg), 2), fmt(*mean_se(evt), 2),
                      fmt(*mean_se(evt - ev_tg), 3), f"{np.mean([r['edge'] for r in R]):.3f}"])
    table(["n", "hands", "stat", "Greedy", "TA (at b*)", "TA − Greedy (paired)"], rows)
    table(["n", "hands", "oracle bid changes [95% CI]", "mean |shift|", "mean signed shift (TA−G)",
           "frac TA higher/lower", "P(hit) at own oracle bid G→TA", "EV oracle Greedy", "EV oracle TA",
           "TA: EV(b*_TA) − EV(b*_G)", "b*_TA at window edge"], rows2)
    return allrecs


def e2(allrecs):
    P("### E2 — bimodality")
    P("Dip criterion: P is bimodal if ∃ i<k<j with P(i),P(j) ≥ 0.05, P(k) ≤ 0.5·min(P(i),P(j)) and min(P(i),P(j)) − P(k) ≥ 0.03 "
      "(using the max to the left/right of k).  Long-trump+void (LTV) = ≥6 trumps and ≥1 void.  "
      "Note: TA distributions use 100 worlds (noisier than Greedy's 400) — noise inflates false dips, so the TA rate is an upper bound.")
    rows = []
    for n, R in allrecs.items():
        for sub, f in (("all", lambda r: True), ("LTV", lambda r: r["tlen"] >= 6 and r["voids"] >= 1)):
            rr = [r for r in R if f(r)]
            if not rr:
                continue
            bg = sum(is_bimodal(r["greedy"]) for r in rr)
            bt = sum(is_bimodal(r["ta"][str(r["b_ta"])]) for r in rr)
            # greedy subsampled to 100 worlds equivalent: binomial noise proxy
            rng = np.random.default_rng(0)
            bgs = sum(is_bimodal(rng.multinomial(100, np.array(r["greedy"]) / sum(r["greedy"])) / 100) for r in rr)
            lo, hi = prop_ci(bg, len(rr))
            lo2, hi2 = prop_ci(bt, len(rr))
            rows.append([n, sub, len(rr), f"{bg / len(rr):.3f} [{lo:.3f},{hi:.3f}]", f"{bgs / len(rr):.3f}",
                         f"{bt / len(rr):.3f} [{lo2:.3f},{hi2:.3f}]"])
    table(["n", "subset", "hands", "Greedy bimodal (400 worlds)", "Greedy, resampled to 100 worlds", "TA at b* bimodal (100 worlds)"], rows)
    for n in (13, 8):
        D = load(f"e2_decomp_n{n}.json")
        if not D:
            continue
        rows = []
        for key in ("all_greedy", "hero_greedy_opp_ta", "hero_ta_opp_greedy", "all_ta"):
            k = sum(is_bimodal(d[key]) for d in D)
            lo, hi = prop_ci(k, len(D))
            # parity: P(odd) - P(even) alternation index = sum |P(t)-0.5(P(t-1)+P(t+1))|
            zig = [dip_depth(d[key]) for d in D]
            rows.append([key, len(D), f"{k / len(D):.3f} [{lo:.3f},{hi:.3f}]", fmt(*mean_se(zig), 3)])
        P(f"E2 decomposition, n={n}, LTV hands, 400 worlds each (TA hero bids b*_TA from E1):")
        table(["configuration", "hands", "bimodal [95% CI]", "mean dip depth ± 95%"], rows)
    L = load("e2_ltv.json")
    if L:
        D = L["recs"]
        keys = ("all_greedy", "hero_greedy_opp_ta", "hero_ta_opp_greedy", "all_ta")
        P(f"E2 targeted long-trump+void set (n=13, trump S, ≥6 trumps & ≥1 void; {len(D)} hands from {L['tried']} deals; "
          f"400 worlds per configuration; TA hero bids b*_TA). Dip depth = max_k [min(max_{{i<k}}P(i), max_{{j>k}}P(j)) − P(k)].")
        base = np.array([is_bimodal(d["all_greedy"]) for d in D])
        dd0 = np.array([dip_depth(d["all_greedy"]) for d in D])
        rows = []
        for key in keys:
            bm = np.array([is_bimodal(d[key]) for d in D])
            dd = np.array([dip_depth(d[key]) for d in D])
            lo, hi = prop_ci(int(bm.sum()), len(D))
            keep = f"{bm[base].mean():.2f}" if base.any() else "-"
            rows.append([key, f"{bm.mean():.3f} [{lo:.3f},{hi:.3f}]", fmt(*mean_se(dd), 3),
                         fmt(*mean_se(dd - dd0), 3), keep])
        table(["configuration", "bimodal [95% CI]", "mean dip depth", "Δ dip vs all-greedy (paired)",
               "still bimodal | greedy-bimodal"], rows)
    S = load("e2_specific.json")
    if S:
        for name, d in S.items():
            P(f"E2 specific hand {name} (greedy {d['samples_greedy']} worlds; TA {d['samples_ta']} per bid):")
            g = np.array(d["all_greedy"])
            hg = np.array(d["hero_greedy_opp_ta"])
            rows = [["all greedy"] + [f"{x:.3f}" for x in g], ["hero greedy, opp TA"] + [f"{x:.3f}" for x in hg]]
            for b, p in d["all_ta_by_bid"].items():
                rows.append([f"all TA, hero bids {b}"] + [f"{x:.3f}" for x in p])
            for b, p in d["hero_ta_opp_greedy_by_bid"].items():
                rows.append([f"hero TA bid {b}, opp greedy"] + [f"{x:.3f}" for x in p])
            table(["config"] + [str(t) for t in range(14)], rows)


# ---------------------------------------------------------------- E3
def e3(allrecs):
    P("### E3 — bid shading  s = b_oracle − round(E[t])")
    G = load("e3_greedy_grid.json")
    if G:
        sh, E, sd, nt, pos, n = [], [], [], [], [], []
        for r in G:
            m, v, _, _ = dist_stats(r["dist"])
            sh.append(r["b_oracle"] - rnd(m)); E.append(m); sd.append(np.sqrt(v)); nt.append(r["nt"])
            pos.append(r["pos"]); n.append(r["n"])
        sh, E, sd, nt, pos, n = map(np.array, (sh, E, sd, nt, pos, n))
        P(f"Greedy play grid: {len(G)} situations (40 per n x trump/NT x position), 300 worlds each.")
        rows = []
        for nn in range(1, 14):
            for ntv in (False, True):
                m = (n == nn) & (nt == ntv)
                rows.append([nn, "NT" if ntv else "trump", m.sum(), fmt(*mean_se(sh[m]), 2),
                             f"{np.mean(sh[m] < 0):.2f}", f"{np.mean(sh[m] > 0):.2f}", f"{E[m].mean():.2f}"])
        table(["n", "type", "sits", "mean shading ± 95%", "frac shade down", "frac shade up", "mean E[t]"], rows)
        rows = []
        for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 3), (3, 5), (5, 14)):
            m = (E >= lo) & (E < hi)
            if m.sum():
                rows.append([f"E[t] in [{lo},{hi})", m.sum(), fmt(*mean_se(sh[m]), 2), f"{np.mean(sh[m] < 0):.2f}",
                             f"{np.mean(sh[m] > 0):.2f}"])
        for lo, hi in ((0, .5), (.5, .8), (.8, 1.1), (1.1, 1.5), (1.5, 9)):
            m = (sd >= lo) & (sd < hi)
            if m.sum():
                rows.append([f"sd(t) in [{lo},{hi})", m.sum(), fmt(*mean_se(sh[m]), 2), f"{np.mean(sh[m] < 0):.2f}",
                             f"{np.mean(sh[m] > 0):.2f}"])
        for p in range(4):
            m = pos == p
            rows.append([POSNAME[p], m.sum(), fmt(*mean_se(sh[m]), 2), f"{np.mean(sh[m] < 0):.2f}", f"{np.mean(sh[m] > 0):.2f}"])
        # fractional part of E[t]
        frac = E - np.floor(E)
        for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
            m = (frac >= lo) & (frac < hi)
            rows.append([f"frac(E[t]) in [{lo},{hi})", m.sum(), fmt(*mean_se(sh[m]), 2), f"{np.mean(sh[m] < 0):.2f}",
                         f"{np.mean(sh[m] > 0):.2f}"])
        table(["slice (greedy)", "sits", "mean shading", "down", "up"], rows)
        # mode vs oracle
        modes = np.array([int(np.argmax(r["dist"])) for r in G])
        bo = np.array([r["b_oracle"] for r in G])
        P(f"Greedy: oracle bid == mode of P(t): {np.mean(bo == modes):.3f}; == round(E[t]): {np.mean(sh == 0):.3f}; "
          f"oracle below mode: {np.mean(bo < modes):.3f}, above: {np.mean(bo > modes):.3f} (n={len(G)}).")
        # linear fit
        X = np.column_stack([np.ones(len(sh)), E - np.floor(E) - 0.5, sd, nt, n / 13])
        beta, *_ = np.linalg.lstsq(X, sh, rcond=None)
        P(f"OLS shading ≈ {beta[0]:+.2f} {beta[1]:+.2f}·(frac(E)−.5) {beta[2]:+.2f}·sd {beta[3]:+.2f}·NT {beta[4]:+.2f}·n/13")
        P("")
    # TA shading
    O = load("oracle_compare_ta.json")
    if O:
        sh = np.array([r["b_oracle"] - rnd(r["Et_at_oracle"]) for r in O])
        P(f"TA play (oracle_compare_ta, {len(O)} situations): shading vs E[t | hero bids b*] = {fmt(*mean_se(sh), 3)}; "
          f"frac 0: {np.mean(sh == 0):.3f}.  (TA pulls E[t] toward the bid, so this is near 0 by construction.)")
    rows = []
    for n, R in allrecs.items():
        bt = np.array([r["b_ta"] for r in R])
        Eg = np.array([dist_stats(r["greedy"])[0] for r in R])
        bg = np.array([r["b_greedy"] for r in R])
        s_t = bt - np.floor(Eg + .5)
        s_g = bg - np.floor(Eg + .5)
        rows.append([n, len(R), fmt(*mean_se(s_g), 3), fmt(*mean_se(s_t), 3), f"{np.mean(s_t < 0):.3f}", f"{np.mean(s_t > 0):.3f}"])
    if rows:
        P("Shading relative to the bid-free greedy expectation, E1 hands (hero first bidder, trump S):")
        table(["n", "hands", "greedy b* − round(E_G[t])", "TA b* − round(E_G[t])", "TA down", "TA up"], rows)


# ---------------------------------------------------------------- games
def per_round(G):
    """Flatten to rows: game, r, n, trump_isnt, pos(rel. dealer) per seat."""
    R = RULES
    lad = np.array(R.ladder())
    nt = np.array([R.trump_for_round(r) == -1 for r in range(R.n_rounds)])
    dealer = np.array([R.dealer_for_round(r) for r in range(R.n_rounds)])
    return lad, nt, dealer


def e4(G, label="all-TA, model bids"):
    lad, nt, dealer = per_round(G)
    bids, tricks = G["bids"].astype(int), G["tricks"].astype(int)
    Ng = len(bids)
    over = bids.sum(2) - lad[None, :]                    # (G, R)
    hit = (bids == tricks)
    P(f"### E4 — table dynamics ({label}; {Ng} games x 25 rounds)")
    rows = []
    for n in range(1, 14):
        for t in (False, True):
            m = (lad == n) & (nt == t)
            if not m.any():
                continue
            o = over[:, m].ravel()
            rows.append([n, "NT" if t else "trump", len(o), fmt(*mean_se(o), 2), f"{np.mean(o < 0):.2f}",
                         f"{np.mean(o > 0):.2f}", f"{np.std(o):.2f}", fmt(*mean_se(hit[:, m].ravel()), 3)])
    table(["n", "type", "rounds", "mean Σbids−n", "frac under", "frac over", "sd", "hit rate"], rows)
    rows = []
    o = np.repeat(over[:, :, None], 4, 2).ravel()
    h = hit.ravel()
    for v in range(-5, 6):
        m = (o == v) if abs(v) < 5 else ((o <= -5) if v < 0 else (o >= 5))
        if m.sum() > 30:
            k = int(h[m].sum())
            lo, hi = prop_ci(k, int(m.sum()))
            rows.append([("≤" if v == -5 else "≥" if v == 5 else "") + f"{v:+d}", int(m.sum()) // 4,
                         f"{k / m.sum():.3f} [{lo:.3f},{hi:.3f}]"])
    table(["Σbids−n", "rounds", "per-player hit rate [95% CI]"], rows)
    # who misses: by position relative to dealer; split under/over tables
    pos = (np.arange(4)[None, :] - dealer[:, None] - 1) % 4      # (R, 4)
    posb = np.broadcast_to(pos[None], bids.shape)
    ob = np.repeat(over[:, :, None], 4, 2)
    rows = []
    for p in range(4):
        m = posb == p
        mu, mo = m & (ob < 0), m & (ob > 0)
        miss = ~hit
        over_t = (tricks > bids)
        rows.append([POSNAME[p], fmt(*mean_se(miss[m]), 3), fmt(*mean_se(miss[mu]), 3), fmt(*mean_se(miss[mo]), 3),
                     f"{np.mean(over_t[m & miss]):.2f}"])
    table(["position", "miss rate (all)", "miss in under-bid tables", "miss in over-bid tables", "frac of misses that are over-tricks"], rows)


def e5(G, Gnh):
    lad, nt, dealer = per_round(G)
    sc = G["scores"].astype(float)
    pos = (np.arange(4)[None, :] - dealer[:, None] - 1) % 4
    Ng = len(sc)
    P(f"### E5 — seat value ({Ng} games, all-TA, model bids)")
    rows = []
    for n in range(1, 14):
        m = lad == n
        row = [n, int(m.sum()) * Ng]
        for p in range(4):
            vals = np.concatenate([sc[:, r, pos[r] == p].ravel() for r in np.where(m)[0]])
            row.append(fmt(*mean_se(vals), 2))
        dl = np.concatenate([sc[:, r, pos[r] == 3].ravel() - sc[:, r, pos[r] != 3].mean(1) for r in np.where(m)[0]])
        row.append(fmt(*mean_se(dl), 2))
        rows.append(row)
    table(["n", "rounds", "pos0 (1st bidder, leads)", "pos1", "pos2", "dealer", "dealer − mean(others) (paired)"], rows)
    tot = G["totals"]
    wins = G["wins"]
    rows = [[f"seat {s}" + (" (deals round 1)" if s == 0 else ""), fmt(*mean_se(tot[:, s]), 1),
             fmt(*mean_se(tot[:, s] - tot.mean(1)), 2), fmt(*mean_se(wins[:, s]), 3)] for s in range(4)]
    table(["starting seat", "game score", "score − table mean", "P(win)"], rows)
    if Gnh is not None:
        sn = Gnh["scores"].astype(float)
        P(f"Hook cost: same {len(sn)} seeds with Rules(hook=False) (CRN; bidder unchanged, dealer's legal set widened).")
        rows = []
        for n in range(1, 14):
            m = np.where(lad == n)[0]
            d_dl = np.concatenate([sn[:, r, pos[r] == 3].ravel() - sc[:, r, pos[r] == 3].ravel() for r in m])
            d_ot = np.concatenate([(sn[:, r, pos[r] != 3] - sc[:, r, pos[r] != 3]).mean(1) for r in m])
            hk = np.concatenate([(sc[:, r, :].sum(1) * 0 + (G["bids"][:, r, :].sum(1) == n)) for r in m])
            bnh = Gnh["bids"].astype(int)
            fit_nh = np.concatenate([(bnh[:, r, :].sum(1) == n) for r in m])
            rows.append([n, len(d_dl), fmt(*mean_se(d_dl), 2), fmt(*mean_se(d_ot), 2), f"{fit_nh.mean():.3f}"])
        table(["n", "rounds", "dealer Δscore (no-hook − hook)", "others' mean Δ", "no-hook: frac Σbids==n"], rows)
        dt = sn.sum(1) - sc.sum(1)
        P(f"Full-game Δ (no-hook − hook) per seat: " + ", ".join(fmt(*mean_se(dt[:, s]), 2) for s in range(4)) +
          f"; table total Δ {fmt(*mean_se(dt.sum(1)), 2)}")
        P("")


def e6():
    P("### E6 — miss-penalty sweep (bidder refit per mode: 3 plain + 2 damped fixed-point iterations; 6,000 games each for the penalty modes, 10,000 for zero; all-TA)")
    rows = []
    for mode in ("zero", "neg_bid", "neg_10", "neg_diff"):
        G = load("games_ta.npz" if mode == "zero" else f"games_ta_{mode}.npz")
        if G is None:
            continue
        lad, nt, dealer = per_round(G)
        b, t, s = G["bids"].astype(int), G["tricks"].astype(int), G["scores"].astype(float)
        over = b.sum(2) - lad[None]
        hit = b == t
        zero_bids = (b == 0)
        pos_sc = np.where(hit, 10 + b, 0).sum((1, 2)) / 4    # hit points only, comparable across modes
        rows.append([mode, len(b), fmt(*mean_se(b.mean((1, 2))), 3), fmt(*mean_se(hit.mean((1, 2))), 4),
                     fmt(*mean_se(s.sum(1).mean(1)), 1), fmt(*mean_se(pos_sc), 1), fmt(*mean_se(over.mean(1)), 3),
                     f"{zero_bids.mean():.3f}", f"{np.mean(np.abs(b - t)[~hit]):.2f}"])
    table(["miss mode", "games", "mean bid", "hit rate", "game score/player (own mode)", "hit points/player (10+b on hits)",
           "mean Σbids−n", "frac zero bids", "mean |b−t| on misses"], rows)


def e7():
    T = load("games_ta.npz")
    S1 = load("games_spite1.npz")
    SA = load("games_spite_all.npz")
    if T is None or S1 is None:
        return
    Ng = min(len(T["totals"]), len(S1["totals"]))
    tt, st = T["totals"][:Ng], S1["totals"][:Ng]
    tw, sw = T["wins"][:Ng], S1["wins"][:Ng]
    k = np.arange(Ng) % 4
    idx = np.arange(Ng)
    own = st[idx, k] - tt[idx, k]
    mask = np.ones((Ng, 4), bool)
    mask[idx, k] = False
    opp = (st - tt)[mask].reshape(Ng, 3).mean(1)
    P(f"### E7 — value of spite ({Ng} full games per config, CRN; spite seat rotates g mod 4)")
    rows = [["(a) spite seat own score Δ", fmt(*mean_se(own), 2)],
            ["(a) each opponent's score Δ (mean of 3)", fmt(*mean_se(opp), 2)],
            ["(a) spite − opponents (relative)", fmt(*mean_se(own - opp), 2)],
            ["(a) P(win) spite seat (all-TA same seat)", f"{fmt(*mean_se(sw[idx, k]), 3)} ({fmt(*mean_se(tw[idx, k]), 3)})"],
            ["(a) P(win) Δ spite seat (paired)", fmt(*mean_se(sw[idx, k] - tw[idx, k]), 3)]]
    if SA is not None:
        sa = SA["totals"][:Ng]
        rows += [["(b) all-spite vs all-TA: table total Δ", fmt(*mean_se(sa.sum(1) - tt.sum(1)), 2)],
                 ["(b) per player Δ", fmt(*mean_se((sa - tt).mean(1)), 2)]]
    table(["quantity", "value ± 95%"], rows)
    # how often spite activates: hit rates
    for name, G in (("all-TA", T), ("one spite", S1), ("all-spite", SA)):
        if G is None:
            continue
        hit = (G["bids"] == G["tricks"])[:Ng]
        P(f"hit rate {name}: {fmt(*mean_se(hit.mean((1, 2))), 4)}")
    hs = (S1["bids"] == S1["tricks"])[:Ng]
    ht = (T["bids"] == T["tricks"])[:Ng]
    sp_hit = np.array([hs[g, :, k[g]].mean() - ht[g, :, k[g]].mean() for g in range(Ng)])
    op_hit = np.array([np.delete(hs[g], k[g], 1).mean() - np.delete(ht[g], k[g], 1).mean() for g in range(Ng)])
    P(f"(a) hit-rate Δ spite seat {fmt(*mean_se(sp_hit), 4)}, opponents {fmt(*mean_se(op_hit), 4)}")
    P("")


def bench():
    T, Gr, GT = load("games_ta.npz"), load("games_greedy.npz"), load("games_greedy_play_ta_bids.npz")
    if T is None or Gr is None:
        return
    P("### Benchmark — full games, model bids (each bidder fitted to its own play policy)")
    rows = []
    for name, G in (("all-TA (TA-fitted bidder)", T), ("all-Greedy (greedy-fitted bidder)", Gr),
                    ("all-Greedy play with TA-fitted bidder", GT)):
        if G is None:
            continue
        hit = G["bids"] == G["tricks"]
        rows.append([name, len(G["totals"]), fmt(*mean_se(G["totals"].mean(1)), 1), fmt(*mean_se(hit.mean((1, 2))), 4),
                     fmt(*mean_se(G["bids"].astype(float).mean((1, 2))), 3)])
    table(["config", "games", "game score / player", "hit rate", "mean bid"], rows)
    d = T["totals"].mean(1) - Gr["totals"].mean(1)
    P(f"Paired (same deals) TA − Greedy per-player game score: {fmt(*mean_se(d), 2)}")
    P("")


def main():
    fixed_point()
    bidder()
    allrecs = e1()
    e2(allrecs)
    e3(allrecs)
    T = load("games_ta.npz")
    if T is not None:
        e4(T)
        Gr = load("games_greedy.npz")
        if Gr is not None:
            e4(Gr, "all-Greedy, greedy-fitted bids")
        e5(T, load("games_ta_nohook.npz"))
    e6()
    e7()
    bench()
    (RES / "phase1_tables.md").write_text("\n".join(OUT))


if __name__ == "__main__":
    main()

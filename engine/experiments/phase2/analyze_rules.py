"""F6 theorem extraction: candidate play rules + shallow decision tree.

Dataset: results/main_dec.csv (every non-forced decision on the max^n path and
on the all-heuristic path, schemes R and O, n = 2..5, S and NT; exact own
max^n value of every legal card).  "Optimal" = own value equal to the best own
value (neutral max^n continuation).  For each rule: coverage (# decisions where
its condition holds and its action is defined), agreement (% where the rule's
card is optimal), mean loss over covered decisions and over disagreements.
Also checked on PIMC decisions (results/pimc_decisions.csv): agreement with the
PIMC choice set (cards with the best PIMC average) and with the true max^n values
of the real (hidden) deal.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from judgement.solver.features import ACTIONS, abstract_actions  # noqa: E402

RES = Path(__file__).parent / "results"

Z = lambda d: d.status == "zero"
P = lambda d: d.status == "pos"
F = lambda d: d.lead == 0

RULES = [
    # name, description, condition, action
    ("R0", "dead (cannot hit): any card (own-indifferent)", lambda d: d.status == "dead", "LOW"),
    ("R1", "need=0, following, can lose: highest surely-losing card", lambda d: Z(d) & F(d) & (d.can_lose == 1), "HIGHLOSE"),
    ("R1-alt", "need=0, following, can lose: LOWEST losing card", lambda d: Z(d) & F(d) & (d.can_lose == 1), "LOWLOSE"),
    ("R1|Σb<n", "R1 when Σbids<n", lambda d: Z(d) & F(d) & (d.can_lose == 1) & (d.sum_bids_minus_n < 0), "HIGHLOSE"),
    ("R1|Σb>n", "R1 when Σbids>n", lambda d: Z(d) & F(d) & (d.can_lose == 1) & (d.sum_bids_minus_n > 0), "HIGHLOSE"),
    ("R1|winner needs", "R1 when current winner still needs tricks", lambda d: Z(d) & F(d) & (d.can_lose == 1) & (d.winner_status == "pos"), "HIGHLOSE"),
    ("R1-last", "need=0, LAST to play, can lose: highest losing card", lambda d: Z(d) & (d.last == 1) & (d.can_lose == 1), "HIGHLOSE"),
    ("R2", "need=0, leading: lead lowest card", lambda d: Z(d) & (d.lead == 1), "LOW"),
    ("R2-alt", "need=0, leading: lead HIGHEST card", lambda d: Z(d) & (d.lead == 1), "HIGH"),
    ("R2|Σb<n", "R2 when Σbids<n", lambda d: Z(d) & (d.lead == 1) & (d.sum_bids_minus_n < 0), "LOW"),
    ("R2-alt|Σb<n", "need=0 leading, Σbids<n: lead highest (shed early)", lambda d: Z(d) & (d.lead == 1) & (d.sum_bids_minus_n < 0), "HIGH"),
    ("R3", "need=0, following, every card wins, last: play highest", lambda d: Z(d) & F(d) & (d.can_lose == 0) & (d.last == 1), "HIGH"),
    ("R3b", "need=0, following, every card wins, not last: play lowest", lambda d: Z(d) & F(d) & (d.can_lose == 0) & (d.last == 0), "LOW"),
    ("R4", "need>0, last, can win: cheapest winning card", lambda d: P(d) & (d.last == 1) & (d.can_win == 1), "CHEAPWIN"),
    ("R4|slack>0", "R4 when tricks left exceed total live need (surplus tricks)", lambda d: P(d) & (d.last == 1) & (d.can_win == 1) & (d.slack > 0), "CHEAPWIN"),
    ("R4|slack<=0", "R4 when tricks are scarce (live need >= tricks left)", lambda d: P(d) & (d.last == 1) & (d.can_win == 1) & (d.slack <= 0), "CHEAPWIN"),
    ("R5", "need>0, following not last, can win: cheapest winner", lambda d: P(d) & F(d) & (d.last == 0) & (d.can_win == 1), "CHEAPWIN"),
    ("R5-alt", "need>0, following not last, can win: strongest winner", lambda d: P(d) & F(d) & (d.last == 0) & (d.can_win == 1), "HIGHWIN"),
    ("R6", "need>0, following, cannot win: discard lowest", lambda d: P(d) & F(d) & (d.can_win == 0), "LOW"),
    ("R7", "need = tricks left, leading: lead highest", lambda d: P(d) & (d.need_eq_rem == 1) & (d.lead == 1), "HIGH"),
    ("R7-boss", "need = tricks left, leading, holds boss: lead highest boss", lambda d: P(d) & (d.need_eq_rem == 1) & (d.lead == 1) & (d.has_boss == 1), "HIGHBOSS"),
    ("R4|need=rem", "need = tricks left, last, can win: cheapest winner", lambda d: P(d) & (d.need_eq_rem == 1) & (d.last == 1) & (d.can_win == 1), "CHEAPWIN"),
    ("R7f", "need = tricks left, following, can win: strongest winner", lambda d: P(d) & (d.need_eq_rem == 1) & F(d) & (d.can_win == 1), "HIGHWIN"),
    ("R8", "need>0, leading, holds a boss card: lead lowest boss", lambda d: P(d) & (d.lead == 1) & (d.has_boss == 1), "LOWBOSS"),
    ("R8-alt", "need>0, leading, holds a boss: lead highest boss (trump first)", lambda d: P(d) & (d.lead == 1) & (d.has_boss == 1), "HIGHBOSS"),
    ("R9", "need>0, leading, no boss: lead lowest", lambda d: P(d) & (d.lead == 1) & (d.has_boss == 0), "LOW"),
    ("R9-alt", "need>0, leading, no boss: lead highest", lambda d: P(d) & (d.lead == 1) & (d.has_boss == 0), "HIGH"),
    ("HEUR", "whole target-aware heuristic", lambda d: d.status.notna(), "HEUR"),
]


def mci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else np.nan, np.nan, n)
    return (x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(n), n)


def eval_rules(d, vcol_prefix="v_", vmax="vmax", label=""):
    rows = []
    for name, desc, cond, act in RULES:
        col = vcol_prefix + act
        m = cond(d) & d[col].notna() & (d[col] != "")
        g = d[m]
        if len(g) == 0:
            rows.append({"rule": name, "desc": desc, "coverage": 0})
            continue
        v = g[col].astype(float)
        agree = (v == g[vmax]).astype(float)
        loss = g[vmax] - v
        nonind = g["indiff"] == 0
        rows.append({"rule": name, "desc": desc, "coverage": len(g), "cov%": round(100 * len(g) / len(d), 1),
                     "non-indiff": int(nonind.sum()),
                     "agree%": f"{100 * agree.mean():.2f} ± {196 * agree.std(ddof=1) / np.sqrt(len(g)):.2f}" if len(g) > 1 else f"{100*agree.mean():.1f}",
                     "agree% non-indiff": f"{100 * agree[nonind].mean():.2f}" if nonind.any() else "",
                     "mean loss": f"{loss.mean():.3f} ± {1.96 * loss.std(ddof=1) / np.sqrt(len(g)):.3f}" if len(g) > 1 else f"{loss.mean():.3f}",
                     "loss|disagree": f"{loss[agree == 0].mean():.2f}" if (agree == 0).any() else "-",
                     "always optimal": "YES" if agree.min() == 1 else "no"})
    return rows


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


FEATS = ["pos", "lead", "last", "nt", "rem", "need", "st_code", "need_eq_rem", "sum_bids_minus_n", "others_need",
         "others_zero", "others_dead", "slack", "win_code", "n_legal", "has_trump", "can_win", "can_lose", "has_boss"]
PRIO = ["LOW", "HIGHLOSE", "CHEAPWIN", "HIGH", "HIGHWIN", "LOWLOSE", "LOWBOSS", "HIGHBOSS"]


def prep(d):
    d = d.copy()
    d["st_code"] = d.status.map({"dead": -1, "zero": 0, "pos": 1})
    d["win_code"] = d.winner_status.fillna("").map({"": -2, "dead": -1, "zero": 0, "pos": 1})
    d["need"] = d["need"].clip(-1, 6)
    for a in ACTIONS + ("HEUR",):
        d[f"v_{a}"] = pd.to_numeric(d[f"v_{a}"], errors="coerce")
    return d


def label(d):
    lab = pd.Series("OTHER", index=d.index)
    done = pd.Series(False, index=d.index)
    for a in PRIO:
        ok = (~done) & (d[f"v_{a}"] == d["vmax"])
        lab[ok] = a
        done |= ok
    return lab


def tree_policy_value(d, pred):
    v = []
    for (_, r), a in zip(d.iterrows(), pred):
        x = r.get(f"v_{a}")
        if x is None or pd.isna(x):
            x = r["v_LOW"]
        v.append(x)
    return np.array(v, dtype=float)


def main():
    from sklearn.tree import DecisionTreeClassifier, export_text
    d = prep(pd.read_csv(RES / "main_dec.csv"))
    d = d[d.n <= 5].copy()          # F6 dataset: n <= 5 (n = 6 rows, if present, are excluded)
    out = {}
    md = []
    md.append(f"## F6 rules vs exact max^n decisions (perfect info).  Dataset: {len(d)} non-forced decisions, "
              f"{d.groupby(['n','trump','deal','scheme']).ngroups} (deal,scheme) rounds; {int((d.indiff==0).sum())} non-indifferent")
    cols = ["rule", "desc", "coverage", "cov%", "non-indiff", "agree%", "agree% non-indiff", "mean loss", "loss|disagree", "always optimal"]
    rows = eval_rules(d)
    out["rules_all"] = rows
    md.append(table(rows, cols))
    for nt in (0, 1):
        rows = eval_rules(d[d.nt == nt])
        out[f"rules_nt{nt}"] = rows
        md.append(f"\n### {'NT' if nt else 'trump (S)'} only")
        md.append(table(rows, ["rule", "coverage", "agree%", "agree% non-indiff", "mean loss", "always optimal"]))
    for sch in ("R", "O"):
        rows = eval_rules(d[d.scheme == sch])
        out[f"rules_{sch}"] = rows
        md.append(f"\n### bid scheme {sch} only")
        md.append(table(rows, ["rule", "coverage", "agree%", "agree% non-indiff", "mean loss", "always optimal"]))

    # ---- automatic search for always-optimal rules (theorem candidates) ----
    from itertools import combinations
    atoms = {
        "need=0": d.status == "zero", "need>0": d.status == "pos", "lead": d.lead == 1, "follow": d.lead == 0,
        "last": d.last == 1, "not last": (d.last == 0) & (d.lead == 0), "can win": d.can_win == 1,
        "cannot win": (d.can_win == 0) & (d.lead == 0), "can lose": d.can_lose == 1,
        "cannot lose": (d.can_lose == 0) & (d.lead == 0), "has boss": d.has_boss == 1,
        "need=tricks left": d.need_eq_rem == 1, "need=1": d.need == 1, "slack>0": d.slack > 0, "slack<=0": d.slack <= 0,
        "slack<0": d.slack < 0, "Σb<n": d.sum_bids_minus_n < 0, "Σb>n": d.sum_bids_minus_n > 0,
        "winner needs": d.winner_status == "pos", "winner at 0": d.winner_status == "zero",
        "winner dead": d.winner_status == "dead", "NT": d.nt == 1, "trump": d.nt == 0, "2 tricks left": d.rem == 2,
    }
    vm = d["vmax"].values
    found = []
    near = []
    for k in (1, 2, 3):
        for combo in combinations(atoms, k):
            m = np.logical_and.reduce([atoms[a].values for a in combo])
            if m.sum() < 200:
                continue
            for a in ACTIONS:
                v = d[f"v_{a}"].values
                mm = m & ~np.isnan(v)
                cov = int(mm.sum())
                if cov < 200:
                    continue
                ni = int((mm & (d["indiff"].values == 0)).sum())
                if ni < 50:
                    continue
                agr = float(np.mean(v[mm] == vm[mm]))
                if agr == 1.0:
                    if any(set(f[0]) <= set(combo) and f[1] == a for f in found):
                        continue
                    found.append((combo, a, cov, ni))
                elif agr >= 0.99 and ni >= 300:
                    if any(set(f[0]) <= set(combo) and f[1] == a for f in near):
                        continue
                    near.append((combo, a, cov, ni, agr, float(np.mean(vm[mm] - v[mm]))))
    found.sort(key=lambda x: -x[3])
    rows = [{"condition": " & ".join(c), "action": a, "coverage": cov, "non-indiff": ni} for c, a, cov, ni in found[:25]]
    out["always_optimal_search"] = rows
    md.append(f"\n## F6 automatic search: conjunctions of <=3 atoms where one abstract action is optimal in EVERY covered decision "
              f"(coverage >= 200, >= 50 non-indifferent; minimal conjunctions only; top 25 of {len(found)} by non-indifferent coverage)")
    if rows:
        md.append(table(rows, list(rows[0])))

    near.sort(key=lambda x: -x[3])
    rows = [{"condition": " & ".join(c), "action": a, "coverage": cov, "non-indiff": ni, "agree%": f"{100*g:.2f}",
             "mean loss": f"{l:.3f}"} for c, a, cov, ni, g, l in near[:25]]
    out["near_optimal_search"] = rows
    md.append(f"\n### near-theorems: >= 99% agreement, >= 300 non-indifferent decisions (top 25 of {len(near)})")
    if rows:
        md.append(table(rows, list(rows[0])))

    # ---- decision tree ----
    dn = d[d.indiff == 0].copy()
    rng = np.random.default_rng(0)
    keys = dn[["n", "trump", "deal", "scheme"]].astype(str).agg("|".join, axis=1)
    uk = keys.unique()
    test_keys = set(rng.choice(uk, size=len(uk) // 4, replace=False))
    te = keys.isin(test_keys)
    # cost-sensitive expansion: one row per optimal abstract action, weight 1/#optimal
    X, Y, Wt = [], [], []
    tr_rows = dn[~te]
    for a in PRIO:
        ok = tr_rows[f"v_{a}"] == tr_rows["vmax"]
        X.append(tr_rows.loc[ok, FEATS]); Y += [a] * int(ok.sum())
        Wt.append(ok[ok].astype(float))
    nopt = sum((tr_rows[f"v_{a}"] == tr_rows["vmax"]).astype(int) for a in PRIO)
    none = nopt == 0
    X.append(tr_rows.loc[none, FEATS]); Y += ["LOW"] * int(none.sum())
    Wt.append(none[none].astype(float))
    Xtr = pd.concat(X)
    wtr = np.concatenate([w.values for w in Wt])
    denom = np.concatenate([nopt[(tr_rows[f"v_{a}"] == tr_rows["vmax"])].values for a in PRIO] + [np.ones(int(none.sum()))])
    wtr = wtr / np.maximum(denom, 1)
    md.append(f"\n## F6 decision tree over abstract actions (non-indifferent decisions; cost-sensitive: each optimal "
              f"action is a positive example with weight 1/#optimal); train {int((~te).sum())} / test {int(te.sum())} "
              f"decisions, split by round.  Undefined predicted action -> LOW.")
    out["tree"] = []
    for depth in (2, 3, 4, 6):
        clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=0)
        clf.fit(Xtr, Y, sample_weight=wtr)
        pred = clf.predict(dn.loc[te, FEATS])
        vt = tree_policy_value(dn[te], pred)
        agree = (vt == dn.loc[te, "vmax"].values).astype(float)
        loss = dn.loc[te, "vmax"].values - vt
        heur = (dn.loc[te, "v_HEUR"].values == dn.loc[te, "vmax"].values).astype(float)
        r = {"depth": depth, "test agree%": f"{100*agree.mean():.2f} ± {196*agree.std(ddof=1)/np.sqrt(len(agree)):.2f}",
             "test mean loss": f"{loss.mean():.3f} ± {1.96*loss.std(ddof=1)/np.sqrt(len(loss)):.3f}",
             "heuristic agree% (same test set)": f"{100*heur.mean():.2f}", "n_test": len(agree)}
        out["tree"].append(r)
        if depth == 3:
            md.append("```\n" + export_text(clf, feature_names=FEATS) + "```")
    md.append(table(out["tree"], list(out["tree"][0])))

    # ---- PIMC decisions ----
    pf = RES / "pimc_decisions.csv"
    if pf.exists():
        p = pd.read_csv(pf)
        p = p[p.n <= 5].copy()
        rows_t = []
        for idx, r in p.iterrows():
            acts = abstract_actions(int(r.hand), int(r.legal), int(r.played), int(r.wc), int(r.trump_code))
            mvp = json.loads(r.mv)
            mvt = json.loads(r.mv_true)
            rec = {}
            for a, c in acts.items():
                rec[f"v_{a}"] = mvp[str(c)]
                rec[f"t_{a}"] = mvt[str(c)]
            rec["tvmax"] = max(mvt.values())
            rec["t_PIMC"] = mvt[str(int(r.chosen))]
            rows_t.append(rec)
        tv = pd.DataFrame(rows_t, index=p.index)
        for c in tv.columns:
            if c.startswith("v_"):
                p[c] = tv[c]
        p = prep(p)
        for c in tv.columns:
            if c.startswith("t_") or c == "tvmax":
                p[c] = tv[c]
        md.append(f"\n## F6 rules vs PIMC decisions (imperfect info, all-PIMC table, K samples). {len(p)} decisions")
        md.append("agree% = rule card is among PIMC's best-average cards; true-agree% = rule card optimal under the "
                  "true hidden deal (perfect-info max^n values)")
        rows = eval_rules(p)
        pt = p.copy()
        for a in ACTIONS:
            pt[f"v_{a}"] = p[f"t_{a}"]
        pt["v_HEUR"] = np.nan   # heuristic card's true value not logged
        pt["vmax"] = p["tvmax"]
        pt["indiff"] = 0
        rows2 = eval_rules(pt)
        for r1, r2, (name, desc, cond, act) in zip(rows, rows2, RULES):
            r1["true-agree%"] = r2.get("agree%", "")
            r1["true loss"] = r2.get("mean loss", "")
            col = "v_" + act
            m = cond(pt) & pt[col].notna()
            if m.any():
                r1["PIMC true-agree% (same cov.)"] = f"{100 * np.mean(p.loc[m, 't_PIMC'] == p.loc[m, 'tvmax']):.2f}"
        out["rules_pimc"] = rows
        md.append(table(rows, ["rule", "coverage", "agree%", "mean loss", "true-agree%", "true loss", "PIMC true-agree% (same cov.)"]))
        pa = (p["t_PIMC"] == p["tvmax"]).astype(float)
        md.append(f"\nPIMC chosen card optimal under the true deal: {100*pa.mean():.2f}% ± {196*pa.std(ddof=1)/np.sqrt(len(pa)):.2f} (n={len(pa)}); "
                  f"mean true 1-step loss {np.mean(p['tvmax'] - p['t_PIMC']):.3f}")
        out["pimc_true_agree"] = float(pa.mean())

    (RES / "summary_rules.json").write_text(json.dumps(out, indent=1, default=str))
    text = "\n".join(md)
    (RES / "summary_rules.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

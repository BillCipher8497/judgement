"""Round 7: exact welfare accounting and reanalysis of saved H1 experiments.

Run from engine: python math/m11_welfare.py
No new games or random draws. CIs use independent deals, or whole games.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "round7_paper" / "results"
SOURCE = ROOT / "experiments" / "surplus_hidden" / "results"


def ci(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("CI needs at least two finite independent observations")
    mean = float(x.mean())
    half = float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
    return {"mean": mean, "half95": half, "low95": mean-half,
            "high95": mean+half, "n": len(x)}


def ratio_ci(numerator, denominator):
    """Ratio of paired means; delta-method SE includes their covariance."""
    a, b = np.asarray(numerator, float), np.asarray(denominator, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or b.mean() <= 0:
        raise ValueError("ratio needs paired observations and a positive mean denominator")
    ratio = float(a.mean()/b.mean())
    influence = (a-ratio*b)/b.mean()
    out = ci(influence)
    half = out["half95"]
    return {"mean": ratio, "half95": half, "low95": ratio-half,
            "high95": ratio+half, "n": len(a)}


def information_comparison():
    """Descriptive comparison; policies differ, so this is not a causal effect.

    Match the mixture to equal spades/no-trump. Cluster PI seats within deal.
    Independent experiment seeds allow addition of the two influence variances.
    """
    path = ROOT/"experiments"/"phase2"/"results"/"main_seat.csv"
    groups = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["scheme"] != "R" or int(row["n"]) != 5:
                continue
            key = (row["trump"], int(row["deal"]))
            groups.setdefault(key, []).append((float(row["maxn_neutral"]), float(row["paranoid"])))
    assert len(groups) == 1000 and all(len(v) == 4 for v in groups.values())
    pi = np.array([np.mean(v, axis=0) for _, v in sorted(groups.items())])
    pi_share = ratio_ci(pi[:, 0]-pi[:, 1], pi[:, 0])
    with np.load(SOURCE/"rounds_main.npz", allow_pickle=False) as z:
        i = list(z["ns"]).index(5)
        count = int(z["done"][i])
        # H1 trump schedule d%5: S,H,D,C,NT. Equal S/NT sample.
        mask = np.isin(np.arange(count)%5, [0, 4])
        hidden = z["scores"][i, :count, :, 0][mask]
        hidden_share = ratio_ci(hidden[:, 0]-hidden[:, 3], hidden[:, 0])
    r, p = hidden_share["mean"], pi_share["mean"]
    value = r/p
    se2 = (hidden_share["half95"]/1.96/p)**2 + (r*pi_share["half95"]/1.96/p**2)**2
    half = float(1.96*np.sqrt(se2))
    out = {"hidden_share": hidden_share, "perfect_information_share": pi_share,
           "hidden_over_perfect_share": {"mean": value, "half95": half,
               "low95": value-half, "high95": value+half,
               "n_hidden_deals": len(hidden), "n_pi_deals": len(pi)},
           "mixture": "equal S/NT, five cards, seats averaged within PI deal",
           "qualification": "Descriptive ratio across different hero and bidding policies, not a causal information effect",
           "pi_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


def welfare_parts(bids, tricks, n_cards, scores=None, hit_base=10):
    """Validate completed 4-player hook rounds and return exact components.

    Leading dimensions are arbitrary; the last axis is seats. n_cards must
    broadcast to the leading dimensions. Integer outputs avoid MC tolerances.
    """
    b, t = np.asarray(bids), np.asarray(tricks)
    n = np.asarray(n_cards)
    if b.shape != t.shape or b.shape[-1] != 4:
        raise ValueError("expected matching arrays with four seats")
    if (not np.isfinite(b).all() or not np.isfinite(t).all()
            or np.any(b != np.floor(b)) or np.any(t != np.floor(t))
            or np.any(b < 0) or np.any(t < 0)
            or np.any(b > n[..., None]) or np.any(t > n[..., None])):
        raise ValueError("incomplete or invalid bids/tricks")
    if np.any(t.sum(-1) != n) or np.any(b.sum(-1) == n):
        raise ValueError("trick conservation or hook violated")
    hit = b == t
    calculated = np.where(hit, hit_base + b, 0)
    if scores is not None and not np.array_equal(calculated, scores):
        raise ValueError("saved scores disagree with exact-hit scoring")
    misses = (~hit).sum(-1)
    missed_tricks = np.where(hit, 0, t).sum(-1)
    welfare = calculated.sum(-1)
    if np.any(welfare != hit_base * (4-misses) + n - missed_tricks):
        raise AssertionError("welfare identity failed")
    if np.any(misses < 1) or np.any(welfare > 3*hit_base+n):
        raise AssertionError("hook welfare ceiling failed")
    return welfare, misses, missed_tricks


def exact_check(max_n=6):
    """All legal terminal count pairs, a superset of realizable card histories."""
    checks = []
    for n in range(1, max_n+1):
        bids = np.array(list(itertools.product(range(n+1), repeat=4)))
        bids = bids[bids.sum(1) != n]
        count, ceiling = 0, 0
        for t in itertools.product(range(n+1), repeat=4):
            if sum(t) != n:
                continue
            tricks = np.broadcast_to(t, bids.shape)
            w, _, _ = welfare_parts(bids, tricks, n)
            ceiling = max(ceiling, int(w.max()))
            count += len(bids)
        assert ceiling == 30+n
        checks.append({"n_cards": n, "legal_count_pairs": count,
                       "max_welfare": ceiling, "identity_violations": 0})
    return checks


def comparisons(scores, misses, missed_tricks, hero, conds):
    """Inputs have independent-unit x condition x seat (or no seat) shapes."""
    count = len(scores)
    hero = np.broadcast_to(hero, (count,)).astype(int)
    ix = np.arange(count)
    ht = scores[ix, :, hero]
    welfare = scores.sum(-1)
    out, raw = {}, {}
    for k, name in enumerate(conds):
        if name == "self":
            continue
        h_loss = ht[:, 0]-ht[:, k]
        all_loss = welfare[:, 0]-welfare[:, k]
        opp_loss = (all_loss-h_loss)/3
        hit_component = 10*(misses[:, k]-misses[:, 0])
        trick_component = missed_tricks[:, k]-missed_tricks[:, 0]
        np.testing.assert_array_equal(all_loss, hit_component+trick_component)
        values = {"hero_loss": h_loss, "opponent_mean_loss": opp_loss,
                  "table_loss": all_loss, "extra_misses_points": hit_component,
                  "missed_tricks_points": trick_component}
        # Recompute ties-as-win from totals; legacy H1 wins stores split ties.
        wins = scores.max(-1) == ht
        values["hero_win_self"] = wins[:, 0]
        values["hero_win_condition"] = wins[:, k]
        values["hero_win_difference"] = wins[:, k].astype(int)-wins[:, 0].astype(int)
        out[name] = {key: ci(val) for key, val in values.items()}
        raw.update({name+"_"+key: val for key, val in values.items()})
    return out, raw


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    status = RES.parent / "STATUS.txt"
    def checkpoint(message):
        with status.open("a", encoding="utf-8") as f:
            f.write(message+"\n")
    summary = {"design": "Reanalysis of H1 saved CRN outcomes; no new sampling",
               "ci": "mean +/- 1.96 SE; unit = deal for rounds, game for ladder",
               "exact_enumeration": exact_check(),
               "information_comparison": information_comparison(), "sources": {}}
    save_json(RES/"welfare.json", summary)
    checkpoint("M11 exact count enumeration complete; welfare.json checkpoint saved.")
    for name in ("rounds_main.npz", "games_main.npz"):
        path = SOURCE/name
        summary["sources"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with np.load(path, allow_pickle=False) as z:
            conds = list(z["conds"])
            assert conds == ["self", "s1", "s2", "s3"]
            summary["sources"][name]["base_seed"] = int(z["base"])
            if name.startswith("rounds"):
                summary["rounds"] = {}
                for i, n in enumerate(z["ns"]):
                    done = int(z["done"][i])
                    s = z["scores"][i, :done]
                    _, m, u = welfare_parts(z["bids"][i, :done],
                                           z["tricks"][i, :done], int(n), s)
                    out, raw = comparisons(s, m, u, 0, conds)
                    summary["rounds"][str(n)] = out
                    np.savez_compressed(RES/f"paired_n{n}.npz", **raw)
                    save_json(RES/"welfare.json", summary)
                checkpoint("M11 all H1 round arrays validated; deal-level paired components saved.")
            else:
                ladder = np.array(list(range(13, 0, -1))+list(range(2, 14)))
                s = z["scores"]
                _, m, u = welfare_parts(z["bids"], z["tricks"], ladder, s)
                out, raw = comparisons(s.sum(2), m.sum(2), u.sum(2), z["hero"], conds)
                summary["games"] = out
                np.savez_compressed(RES/"paired_games.npz", **raw)
                save_json(RES/"welfare.json", summary)
                checkpoint("M11 H1 full games validated; CIs clustered by game, ties-as-win recomputed.")
    lines = ["# Welfare lost under hidden information spoilers", "",
             "Reanalysis of saved H1 outcomes. Differences are self-interested minus spoiler condition.",
             "Each interval is mean +/- 1.96 SE over paired deals or whole games.", "",
             "| sample | condition | n | hero loss | mean opponent loss | table loss | extra misses points | missed tricks points |",
             "|---|---|---|---|---|---|---|---|"]
    for label, rows in [("5 cards", summary["rounds"]["5"]), ("full ladder", summary["games"])]:
        for cond, vals in rows.items():
            keys = ["hero_loss", "opponent_mean_loss", "table_loss", "extra_misses_points", "missed_tricks_points"]
            formatted = [f"{vals[k]['mean']:.3f} +/- {vals[k]['half95']:.3f}" for k in keys]
            lines.append(f"| {label} | {cond} | {vals['table_loss']['n']} | "+" | ".join(formatted)+" |")
    lines += ["", "For s1/s2, mean opponent loss includes bystanders. For s3, all opponents are spoilers.",
              "Table loss = hero loss + three times mean opponent loss exactly for every paired unit.",
              "Table loss = extra misses points + missed tricks points exactly for every paired unit.",
              "These are policy interventions with fixed bidding policies, not equilibrium welfare or optimal coalition values.",
              "Round win indicators in the JSON describe round score leadership; only games describe game wins.", ""]
    (RES/"welfare.md").write_text("\n".join(lines), encoding="utf-8")
    checkpoint("M11 welfare summary complete; source hashes and all paired units preserved.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

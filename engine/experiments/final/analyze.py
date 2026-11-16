"""Tables for experiments/final -> results/final.json, results/final.md.
All Δ are paired on the same game (same deals, same rng streams, same seat); 95% CI = 1.96 sd/sqrt(n)."""
import json
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from judgement import Rules
from judgement.cards import NT

HERE = Path(__file__).parent
RES = HERE / "results"
PLAY = HERE.parent / "play" / "results"
P4 = HERE.parent / "phase4" / "results"
rules = Rules()
NT_R = np.array([rules.trump_for_round(r) == NT for r in range(rules.n_rounds)])
SMALL_R = np.array([n <= 5 for n in rules.ladder()])


def load(*parts):
    """Concatenate runs (path, first game index); returns dict keyed by game index arrays."""
    out = {k: [] for k in ("game", "totals", "wins", "scores", "secs")}
    for p, g0 in parts:
        if not Path(p).exists():
            continue
        d = np.load(p)
        n = len(d["totals"])
        out["game"].append(d["game"] if "game" in d.files else np.arange(g0, g0 + n))
        for k in ("totals", "wins", "scores", "secs"):
            out[k].append(d[k])
    if not out["game"]:
        return None
    o = {k: np.concatenate(v) for k, v in out.items()}
    _, idx = np.unique(o["game"], return_index=True)
    return {k: v[idx] for k, v in o.items()}


def ci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan"), int(len(x))]


def paired(a, b):
    """a - b at the challenger seat (seat = game % 4) over common games."""
    common, ia, ib = np.intersect1d(a["game"], b["game"], return_indices=True)
    s = common % 4
    r = np.arange(len(common))
    d_tot = a["totals"][ia, s] - b["totals"][ib, s]
    d_win = a["wins"][ia, s] - b["wins"][ib, s]
    ds = a["scores"][ia][r, :, s] - b["scores"][ib][r, :, s]
    return {"d_game": ci(d_tot), "d_pwin": ci(d_win), "d_round_trump": ci(ds[:, ~NT_R].mean(1)),
            "d_round_nt": ci(ds[:, NT_R].mean(1)), "d_round_n<=5": ci(ds[:, SMALL_R].mean(1)),
            "pwin_a": ci(a["wins"][ia, s]), "s_per_game_a": float(a["secs"][ia].mean())}


def main():
    out = {}
    TA = load((RES / "f_ta_ta.npz", 0), (PLAY / "1v3_x_ta.npz", 0))
    TAP = load((RES / "f_ta_tap.npz", 0))
    TAP3 = load((RES / "f_ta_tap3.npz", 0))
    TAP3S = load((RES / "f_ta_tap3s.npz", 0))
    HYB = load((PLAY / "1v3_x_hyb_c5.npz", 0), (RES / "f_ta_hyb_ext.npz", 1000))
    HYB3 = load((RES / "f_ta_hyb3.npz", 0))
    HYBO = load((PLAY / "1v3_x_hyb_orc.npz", 0), (RES / "f_ta_hybo_ext.npz", 500))
    K32 = load((PLAY / "1v3_x_hyb_k32_c5.npz", 0), (RES / "f_ta_k32_ext.npz", 400))
    HYB3O = load((RES / "f_ta_hyb3o.npz", 0))
    F1 = {}
    for nm, x in [("TA+ (v2)", TAP), ("TA+3 (L any suit)", TAP3), ("TA+3 (L same suit)", TAP3S), ("Hybrid c5 K16", HYB),
                  ("Hybrid3 c5 K16", HYB3)]:
        if x is not None:
            F1[nm] = {"vs_TA": paired(x, TA)}
    if TAP3 is not None:
        F1["TA+3 (L any suit)"]["vs_TA+"] = paired(TAP3, TAP)
    if TAP3S is not None:
        F1["TA+3 (L same suit)"]["vs_TA+"] = paired(TAP3S, TAP)
    if HYB3 is not None:
        F1["Hybrid3 c5 K16"]["vs_Hybrid"] = paired(HYB3, HYB)
    out["F1"] = F1
    F2 = {}
    for nm, x in [("Hybrid c5 K16", HYB), ("Hybrid + oracle bidder", HYBO), ("Hybrid c5 K32", K32),
                  ("Hybrid3 c5 K16", HYB3), ("Hybrid3 + oracle bidder", HYB3O)]:
        if x is not None:
            F2[nm] = {"vs_TA": paired(x, TA)}
            if nm != "Hybrid c5 K16":
                F2[nm]["vs_Hybrid"] = paired(x, HYB)
    out["F2"] = F2
    REF = load((P4 / "q2_hyb_ref.npz", 0))
    F3 = {}
    for nm, p in [("Hybrid + oracle bidder (phase4)", P4 / "q2_hyb_hybo.npz"), ("TA+ (phase4)", P4 / "q2_hyb_tap.npz"),
                  ("Hybrid3 + oracle bidder", RES / "f_hyb_hyb3o.npz"), ("Hybrid3", RES / "f_hyb_hyb3.npz"),
                  ("TA+3", RES / "f_hyb_tap3.npz")]:
        x = load((p, 0))
        if x is not None:
            F3[nm] = {"vs_3xHybrid_ref": paired(x, REF)}
    out["F3"] = F3
    (RES / "final.json").write_text(json.dumps(out, indent=1))
    f = lambda v, k=2: f"{v[0]:+.{k}f} ± {v[1]:.{k}f}"  # noqa: E731
    L = ["# experiments/final results", "", "Paired Δ at the challenger seat (seat g%4), same game/deal/rng; ± = 95% CI; n = games.", ""]
    for sec, d in out.items():
        L += [f"## {sec}", "", "| challenger | baseline | n | Δ/game | ΔP(win) | P(win) | Δ/round trump | Δ/round NT | Δ/round n≤5 | s/game |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for nm, v in d.items():
            for bl, p in v.items():
                L.append(f"| {nm} | {bl} | {p['d_game'][2]} | {f(p['d_game'])} | {f(p['d_pwin'], 3)} | {f(p['pwin_a'], 3)} | "
                         f"{f(p['d_round_trump'], 3)} | {f(p['d_round_nt'], 3)} | {f(p['d_round_n<=5'], 3)} | {p['s_per_game_a']:.2f} |")
        L.append("")
    (RES / "final.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

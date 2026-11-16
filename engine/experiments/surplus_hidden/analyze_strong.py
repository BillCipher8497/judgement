"""H1 comparison table: heuristic spoiler (main) vs PIMC-paranoid spoiler vs omniscient (cheating,
perfect-information paranoid) spoiler, all against the same TA+ hero on the same CRN deals, plus the
Phase-2 perfect-information max^n - paranoid share.  -> results/summary_strong.md / .json
"""
import json
from pathlib import Path

import numpy as np

from analyze import PI_SHARE, PI_SURPLUS, f, mci, ratio_ci

RES = Path(__file__).parent / "results"


def load(tags):
    out = {}
    for tag in tags:
        fn = RES / f"rounds_{tag}.npz"
        if not fn.exists():
            continue
        z = np.load(fn)
        conds = list(z["conds"])
        for i, n in enumerate(z["ns"]):
            D = int(z["done"][i])
            if D < 2:
                continue
            s = z["scores"][i, :D]
            out[int(n)] = {"self": s[:, conds.index("self"), 0], "s3": s[:, conds.index("s3"), 0], "D": D,
                           "spo_self": s[:, conds.index("self"), 1:].mean(1), "spo_s3": s[:, conds.index("s3"), 1:].mean(1),
                           "hit_s3": (z["bids"][i, :D, conds.index("s3"), 0] == z["tricks"][i, :D, conds.index("s3"), 0])}
    return out


def main():
    m = np.load(RES / "rounds_main.npz")
    main_self = {int(n): m["scores"][i, :, 0, 0] for i, n in enumerate(m["ns"])}
    main_s3 = {int(n): m["scores"][i, :, 3, 0] for i, n in enumerate(m["ns"])}
    kinds = {"pimc": load(["pimc_2_5", "pimc_6", "pimc_7"]), "omni": load(["omni_2_6", "omni_7", "omni_8"]),
             "mc": load(["mc_3_5", "mc_2_5"])}
    summ = {}
    L = ["## H1 spoiler strength vs information (same TA+ hero, same CRN deals; share = surplus / self score)",
         "heuristic = SpoilerPolicy (rows restricted to the same deals as each comparison); PIMC = PIMC-paranoid "
         "K=16 (hidden info); omni = cheating PI paranoid solver on the true hands; PI = Phase-2 max^n − paranoid "
         "(PI hero AND PI opponents, scheme R)",
         "MC = hidden-info Monte-Carlo spoiler that KNOWS the hero's policy (rollouts: hero TA+, mates heuristic spoilers)",
         "| n | deals | self | heuristic s3 | PIMC s3 | MC s3 | omni s3 | share heur | share PIMC | share MC | share omni | PI share | "
         "PIMC−heur (pts) | omni−PIMC (pts) | hit s3 heur/PIMC/MC/omni | spoiler cost PIMC | spoiler cost omni | CRN self check |",
         "|" + "---|" * 18]
    for n in sorted(set(kinds["pimc"]) | set(kinds["omni"])):
        Ds = [kinds[k][n]["D"] for k in kinds if n in kinds[k]]
        D = min(Ds)
        base = main_self[n][:D]
        heur = main_s3[n][:D]
        row = {"deals": D, "self": mci(base), "heur_s3": mci(heur), "share_heur": ratio_ci(base - heur, base)}
        crn = all(np.array_equal(kinds[k][n]["self"][:D], base) for k in kinds if n in kinds[k])
        row["crn_ok"] = bool(crn)
        hits = [float(np.mean(m["bids"][list(m["ns"]).index(n), :D, 3, 0] == m["tricks"][list(m["ns"]).index(n), :D, 3, 0]))]
        for k in ("pimc", "mc", "omni"):
            if n in kinds[k]:
                s3 = kinds[k][n]["s3"][:D]
                row[f"{k}_s3"] = mci(s3)
                row[f"share_{k}"] = ratio_ci(base - s3, base)
                hits.append(float(kinds[k][n]["hit_s3"][:D].mean()))
                row[f"{k}_spoiler_cost"] = mci(kinds[k][n]["spo_self"][:D] - kinds[k][n]["spo_s3"][:D])
            else:
                hits.append(float("nan"))
        if "pimc_s3" in row:
            row["pimc_minus_heur"] = mci(kinds["pimc"][n]["s3"][:D] - heur)
        if "pimc_s3" in row and "omni_s3" in row:
            row["omni_minus_pimc"] = mci(kinds["omni"][n]["s3"][:D] - kinds["pimc"][n]["s3"][:D])
        summ[n] = row
        g = lambda k, d=2: f(row[k], d) if k in row else "–"  # noqa: E731
        L.append(f"| {n} | {D} | {g('self')} | {g('heur_s3')} | {g('pimc_s3')} | {g('mc_s3')} | {g('omni_s3')} | "
                 f"{g('share_heur', 3)} | {g('share_pimc', 3)} | {g('share_mc', 3)} | {g('share_omni', 3)} | "
                 f"{PI_SHARE.get(n, float('nan')):.3f} | {g('pimc_minus_heur')} | {g('omni_minus_pimc')} | "
                 + "/".join(f"{h:.3f}" for h in hits) + f" | {g('pimc_spoiler_cost')} | {g('omni_spoiler_cost')} | {crn} |")
    (RES / "summary_strong.json").write_text(json.dumps(summ, indent=1))
    (RES / "summary_strong.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

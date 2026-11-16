"""Rule values from the S3 ablation jobs: value = guide - guide_with_rule_changed (paired per game, 95% CI).
-> results/s3_ablation.txt / .json"""
import glob
import json
import os

from common import RES
from pair import compare

lines = ["Rule ablations: full guide agent (guide bid + guide play), 1 vs 3 TA, 6000 games, BASE 71M, paired per game;",
         "value = guide(default) - guide(rule changed); positive = the default rule is better; P(win) ties as wins"]
out = {}
for f in sorted(glob.glob(str(RES / "g_f_ta__guide_*.npz"))):
    t = os.path.basename(f)[2:-4]
    r = compare("f_ta__guide", t)
    lines.append(f"{t[12:]:55s} value {r['d']:+6.2f} +- {r['dci']:.2f}/game   dP(win) {r['dpw']:+.3f} +- {r['dpwci']:.3f}")
    out[t] = dict(n=r["n"], value=[r["d"], r["dci"]], dpwin=[r["dpw"], r["dpwci"]])
r = compare("f_tap3__guide", "f_tap3__guide_b.prior_k-99")
lines.append(f"{'(pool TA+3) b.prior_k=99':55s} value {r['d']:+6.2f} +- {r['dci']:.2f}/game   dP(win) {r['dpw']:+.3f} +- {r['dpwci']:.3f}")
out["tap3_prior"] = dict(n=r["n"], value=[r["d"], r["dci"]], dpwin=[r["dpw"], r["dpwci"]])
lines.append("")
lines.append("Where the strength is lost (paired, per game):")
for pool, P in (("ta", "TA"), ("tap3", "TA+3")):
    for a, b, lab in [("tap3", "gbid", "bidding cost, TA+3 play   (MB bid -> guide bid)"),
                      ("gplay", "guide", "bidding cost, guide play  (MB bid -> guide bid)"),
                      ("tap3", "gplay", "play cost, MB bidding      (TA+3 play -> guide play)"),
                      ("gbid", "guide", "play cost, guide bidding   (TA+3 play -> guide play)"),
                      ("tap3", "guide", "total                      (TA+3 -> guide)")]:
        r = compare(f"f_{pool}__{a}", f"f_{pool}__{b}")
        lines.append(f"pool {P:5s} {lab}: {r['d']:+6.2f} +- {r['dci']:.2f}/game  dP(win) {r['dpw']:+.3f} +- {r['dpwci']:.3f}")
        out[f"{pool}|{a}-{b}"] = dict(n=r["n"], value=[r["d"], r["dci"]], dpwin=[r["dpw"], r["dpwci"]])
open(RES / "s3_ablation.txt", "w").write("\n".join(lines) + "\n")
json.dump(out, open(RES / "s3_ablation.json", "w"), indent=1)
print("\n".join(lines))

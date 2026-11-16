"""S1 evaluation table on the test set: match vs U-oracle and vs ModelBidder, EV loss vs MB, by n and trump/NT."""
import json
import sys

import numpy as np

from common import RES
from s1_fit import load, rule_bids, value


def ci(x):
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def table(w, name="s1_test.pkl", out=None):
    D, m, cf, orc = load(name)
    b = rule_bids(w, D, m)
    vr, clip = value(b, cf)
    vm, _ = value(m["bm"], cf)
    has_o = m["bo"] >= 0
    lines = [f"S1 table ({name}, {len(b)} situations; oracle subset {has_o.sum()}; rule bids clipped {clip:.4f})",
             "group | N | match MB | N_orc | match oracle (rule) | match oracle (MB) | EV/round MB | EV loss rule vs MB (95% CI) | mean |bid-MB|"]
    groups = [("all", np.ones(len(b), bool)), ("trump", m["nt"] == 0), ("NT", m["nt"] == 1)]
    for n in sorted(set(m["n"])):
        groups.append((f"n={n}", m["n"] == n))
    for n in [1, 2, 3, 5, 8, 13]:
        for nt in (0, 1):
            groups.append((f"n={n} {'NT' if nt else 'trump'}", (m["n"] == n) & (m["nt"] == nt)))
    for pos in range(4):
        groups.append((f"pos={pos}" + (" (dealer)" if pos == 3 else ""), m["pos"] == pos))
    res = {}
    for g, sel in groups:
        so = sel & has_o
        mm = (b[sel] == m["bm"][sel]).mean()
        mo = (b[so] == m["bo"][so]).mean() if so.any() else np.nan
        mo_m = (m["bm"][so] == m["bo"][so]).mean() if so.any() else np.nan
        loss, lci = ci(vm[sel] - vr[sel])
        res[g] = dict(N=int(sel.sum()), match_mb=mm, N_orc=int(so.sum()), match_orc=mo, match_orc_mb=mo_m,
                      ev_mb=vm[sel].mean(), loss=loss, loss_ci=lci, absdiff=np.abs(b[sel] - m["bm"][sel]).mean())
        r = res[g]
        lines.append(f"{g} | {r['N']} | {mm:.3f} | {r['N_orc']} | {mo:.3f} | {mo_m:.3f} | {r['ev_mb']:.3f} | "
                     f"{loss:+.3f} +- {lci:.3f} | {r['absdiff']:.2f}")
    txt = "\n".join(lines)
    if out:
        open(RES / out, "w").write(txt + "\n")
        json.dump(res, open(RES / out.replace(".txt", ".json"), "w"), indent=1, default=float)
    return txt, res


if __name__ == "__main__":
    w = json.load(open(RES / sys.argv[1]))
    w = w.get("final", w)
    print(table(w, sys.argv[2] if len(sys.argv) > 2 else "s1_test.pkl")[0])

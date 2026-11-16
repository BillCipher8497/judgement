"""Analyse m1_ta_oracle_raw.json: under TA, is the bid chosen by the independent-card
model (applied bid-by-bid) the empirical-oracle bid?  Writes m1_ta_oracle.json / .txt"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pb import mean_var, pb_pmf  # noqa: E402

R = HERE / "results"


def mci(x):
    x = np.asarray(x, float)
    return [float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))), int(len(x))]


def main():
    recs = json.load(open(R / "m1_ta_oracle_raw.json"))
    rows = []
    for r in recs:
        n = r["n"]
        bs = sorted(int(b) for b in r["counts"])
        ev_e, ev_p, disp, hitcurve = {}, {}, {}, {}
        for b in bs:
            c = np.array(r["counts"][str(b)], float)
            P = c / c.sum()
            p = np.array(r["wins"][str(b)], float) / c.sum()
            Q = pb_pmf(p)
            ev_e[b] = (10 + b) * P[b]
            ev_p[b] = (10 + b) * Q[b]
            ve, vp = mean_var(P)[1], mean_var(Q)[1]
            disp[b] = ve / vp if vp > 0.02 else np.nan
        be = max(bs, key=lambda b: (ev_e[b], -b))
        bp = max(bs, key=lambda b: (ev_p[b], -b))
        bm = r["b_model"]
        rows.append({"n": n, "nt": r["trump"] < 0, "match_pb": be == bp, "match_model": be == bm,
                     "loss_pb": ev_e[be] - ev_e[bp], "loss_model": ev_e[be] - ev_e[bm],
                     "disp_at_oracle": disp[be], "edge": be in (bs[0], bs[-1]) and be not in (0, n),
                     "pb_overconf": ev_p[bp] - ev_e[bp]})
    out = {}
    L = ["TA bid-conditional oracle (hero first bidder; candidates model±2; R rollouts per bid, CRN worlds)",
         "group | hands | PB-oracle == emp-oracle | model == emp-oracle | EV loss PB | EV loss model | disp at oracle bid | window edge"]
    for name, f in (("all", lambda x: True), ("trump", lambda x: not x["nt"]), ("NT", lambda x: x["nt"]),
                    *[(f"n={n}|{t}", (lambda n, t: lambda x: x["n"] == n and x["nt"] == (t == "NT"))(n, t))
                      for n in (5, 8, 13) for t in ("S", "NT")]):
        g = [x for x in rows if f(x)]
        if not g:
            continue
        d = {k: mci([x[k] for x in g]) for k in ("match_pb", "match_model", "loss_pb", "loss_model", "edge")}
        dd = np.array([x["disp_at_oracle"] for x in g], float)
        d["disp_at_oracle"] = mci(dd[np.isfinite(dd)])
        out[name] = d
        L.append(f"{name} | {len(g)} | {d['match_pb'][0]:.3f}±{d['match_pb'][1]:.3f} | {d['match_model'][0]:.3f}±{d['match_model'][1]:.3f} | "
                 f"{d['loss_pb'][0]:.2f}±{d['loss_pb'][1]:.2f} | {d['loss_model'][0]:.2f}±{d['loss_model'][1]:.2f} | "
                 f"{d['disp_at_oracle'][0]:.3f}±{d['disp_at_oracle'][1]:.3f} | {d['edge'][0]:.3f}")
    json.dump(out, open(R / "m1_ta_oracle.json", "w"), indent=1)
    open(R / "m1_ta_oracle.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""Q1 tables -> results/q1_summary.md (all numbers exact; pts per round)."""
import json
from pathlib import Path

RES = Path(__file__).parent / "results"
TAGS = ["2p_diff", "2p_raw", "3p_t", "3p_nt", "3p_3x3_t", "3p_n3_t", "4p_n1_t", "4p_n1_nt", "4p_t", "4p_nt"]
D = {t: json.loads((RES / f"q1_{t}.json").read_text()) for t in TAGS if (RES / f"q1_{t}.json").exists()}
f = lambda x: f"{x:.2g}" if abs(x) < 0.01 else f"{x:.3f}"  # noqa: E731
L = ["## A. Games and CFR+ NashConv (sum over seats of exact BR gain, pts/round)", "",
     "| game | deals | nodes | infosets | NC avg T=10 | T=100 | final (T) | current iterate final |", "|---|---|---|---|---|---|---|---|"]
for t, d in D.items():
    c = {r["T"]: r for r in d["curve"]}
    last = d["curve"][-1]
    L.append(f"| {t} | {d['deals']} | {d['nodes']:,} | {d['infosets']:,} | {f(c[10]['nc_avg'])} | {f(c[100]['nc_avg'])} | "
             f"{f(last['nc_avg'])} ({last['T']}) | {f(max(last['nc_cur'], 0))} |")
L += ["", "## B. NashConv of profiles (full / bid-only deviations / play-only deviations)", "",
      "| game | uniform | greedy+myopic | TA+myopic | TA play + CFR bids | CFR play + myopic bids | CFR avg |", "|---|---|---|---|---|---|---|"]
for t, d in D.items():
    P = d["profiles"]
    cell = lambda k: f"{P[k]['nashconv']:.2f} / {P[k]['nc_bid_only']:.2f} / {P[k]['nc_play_only']:.2f}"  # noqa: E731
    L.append(f"| {t} | {P['uniform']['nashconv']:.2f} | {cell('greedy')} | {cell('ta')} | {cell('ta_play+cfr_bid')} | "
             f"{cell('cfr_play+myopic_bid')} | {f(P['cfr_avg']['nashconv'])} |")
L += ["", "## C. Per-seat BR gain against TA+myopic (seat 0 bids first, last seat = dealer) and seat values", "",
      "| game | BR gain per seat | TA value per seat | CFR-eq value per seat |", "|---|---|---|---|"]
for t, d in D.items():
    P = d["profiles"]
    j = lambda xs: " / ".join(f"{x:.2f}" for x in xs)  # noqa: E731
    L.append(f"| {t} | {j(P['ta']['gain'])} | {j(P['ta']['value'])} | {j(P['cfr_avg']['value'])} |")
L += ["", "## D. CFR bids vs myopic bidder (reach-weighted under the CFR profile)", "",
      "| game | modal-bid agreement per seat | P(CFR mixes bid) per seat | mean bid CFR vs myopic per seat | CFR play mixed / agrees with TA |",
      "|---|---|---|---|---|"]
for t, d in D.items():
    b = d["bids"]
    L.append(f"| {t} | {' / '.join(f'{x['agree']:.2f}' for x in b)} | {' / '.join(f'{x['mixed']:.2f}' for x in b)} | "
             f"{' / '.join(f'{x['mean_bid_cfr']:.2f} vs {x['mean_bid_myopic']:.2f}' for x in b)} | "
             f"{d['play_mixed_frac']:.2f} / {d['play_agree_ta']:.2f} |")
for t in ("4p_n1_t", "4p_n1_nt"):
    if t in D:
        L += ["", f"Seat 0 in {t}: (card, P(win) uniform, threshold bid, CFR P(bid 1)): " +
              ", ".join(f"({r['card']}, {r['p_win']:.3f}, {r['threshold_bid']}, {r['cfr_p_bid1']:.2f})" for r in D[t]["seat0_threshold"])]
p = RES / "q1_eqsel.json"
if p.exists():
    E = json.loads(p.read_text())
    L += ["", "## E. Equilibrium selection: CFR+ from different warm starts (NC, seat values)", "",
          "| game | start | T | NashConv | seat values | modal action differs from uniform-start CFR (reach-wt) |", "|---|---|---|---|---|---|"]
    for t, rows in E.items():
        u = D[t]["profiles"]["cfr_avg"]
        L.append(f"| {t} | uniform | {D[t]['curve'][-1]['T']} | {f(u['nashconv'])} | {' / '.join(f'{x:.3f}' for x in u['value'])} | – |")
        for k, r in rows.items():
            L.append(f"| {t} | {k} | {r['T']} | {f(r['nashconv'])} | {' / '.join(f'{x:.3f}' for x in r['value'])} | "
                     f"{r['modal_action_diff_reach_ref']:.2f} |")
(RES / "q1_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

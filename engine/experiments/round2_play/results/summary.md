# Round 2 play — results

## 1-vs-3 (challenger in seat g%4 vs three TA; seeds 61,000,000+g (final; fresh); paired vs the all-TA table, same seat)

| challenger | games | Δscore vs TA /game | P(win game) | hit rate | overshoot share of misses | over / under rate per round | opponents' Δscore /game (each) | TA seat (same games): hit / overshoot share |
|---|---|---|---|---|---|---|---|---|
| TA (reference, all-TA table) | 3000 | +0.00 ± 0.00 | 0.258 ± 0.016 | 0.5771 ± 0.0036 | 0.656 [0.651,0.661] | 0.2774 / 0.1455 | +0.00 ± 0.00 | 0.5771 / 0.656 [0.651,0.661] |
| Greedy (greedy bidder) | 3000 | -43.94 ± 1.18 | 0.019 ± 0.005 | 0.4163 ± 0.0034 | 0.663 [0.659,0.667] | 0.3871 / 0.1966 | +3.82 ± 0.57 | 0.5771 / 0.656 [0.651,0.661] |
| TA-spite | 3000 | +0.00 ± 0.00 | 0.261 ± 0.016 | 0.5771 ± 0.0036 | 0.656 [0.651,0.661] | 0.2774 / 0.1455 | -0.21 ± 0.10 | 0.5771 / 0.656 [0.651,0.661] |
| **TA+** | 3000 | +5.21 ± 0.72 | 0.313 ± 0.017 | 0.5937 ± 0.0035 | 0.622 [0.616,0.627] | 0.2527 / 0.1536 | -1.08 ± 0.32 | 0.5771 / 0.656 [0.651,0.661] |
| TA+ ablation: shed only | 3000 | +1.95 ± 0.36 | 0.272 ± 0.016 | 0.5835 ± 0.0036 | 0.625 [0.620,0.630] | 0.2602 / 0.1564 | -0.65 ± 0.15 | 0.5771 / 0.656 [0.651,0.661] |
| TA+ ablation: lead-high (trump) only | 3000 | +3.30 ± 0.65 | 0.290 ± 0.016 | 0.5875 ± 0.0035 | 0.651 [0.646,0.656] | 0.2686 / 0.1440 | -0.27 ± 0.29 | 0.5771 / 0.656 [0.651,0.661] |
| TA+ with lead-high in NT too | 3000 | +4.72 ± 0.76 | 0.304 ± 0.016 | 0.5923 ± 0.0035 | 0.616 [0.610,0.621] | 0.2511 / 0.1566 | -1.20 ± 0.33 | 0.5771 / 0.656 [0.651,0.661] |
| TA+ without need=r theorems | 3000 | +5.05 ± 0.72 | 0.309 ± 0.016 | 0.5933 ± 0.0035 | 0.621 [0.616,0.626] | 0.2527 / 0.1540 | -0.86 ± 0.32 | 0.5771 / 0.656 [0.651,0.661] |
| TA+ + surplus-duck | 3000 | +5.24 ± 0.72 | 0.311 ± 0.016 | 0.5938 ± 0.0035 | 0.620 [0.615,0.626] | 0.2520 / 0.1542 | -1.09 ± 0.32 | 0.5771 / 0.656 [0.651,0.661] |
| **Hybrid** (PIMC K=16, ≤5 cards) | 800 | +18.02 ± 1.77 | 0.439 ± 0.034 | 0.6305 ± 0.0062 | 0.575 [0.565,0.586] | 0.2124 / 0.1570 | -2.42 ± 0.74 | 0.5748 / 0.653 [0.644,0.663] |

Per-round Δscore vs TA split by round type (same games):

| challenger | trump rounds Δ/round | NT rounds Δ/round | rounds with n≤5 Δ/round | rounds n≥6 Δ/round |
|---|---|---|---|---|
| Greedy (greedy bidder) | -1.558 ± 0.053 | -2.558 ± 0.108 | -1.074 ± 0.058 | -2.142 ± 0.066 |
| TA-spite | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 |
| **TA+** | +0.233 ± 0.034 | +0.111 ± 0.040 | +0.078 ± 0.025 | +0.282 ± 0.043 |
| TA+ ablation: shed only | +0.070 ± 0.015 | +0.112 ± 0.040 | +0.005 ± 0.014 | +0.119 ± 0.021 |
| TA+ ablation: lead-high (trump) only | +0.165 ± 0.032 | +0.000 ± 0.000 | +0.061 ± 0.022 | +0.172 ± 0.039 |
| TA+ with lead-high in NT too | +0.233 ± 0.034 | +0.013 ± 0.062 | +0.079 ± 0.026 | +0.250 ± 0.045 |
| TA+ without need=r theorems | +0.225 ± 0.034 | +0.112 ± 0.040 | +0.066 ± 0.026 | +0.278 ± 0.043 |
| TA+ + surplus-duck | +0.238 ± 0.034 | +0.095 ± 0.041 | +0.078 ± 0.025 | +0.284 ± 0.043 |
| **Hybrid** (PIMC K=16, ≤5 cards) | +0.856 ± 0.081 | +0.179 ± 0.136 | +0.376 ± 0.081 | +0.915 ± 0.100 |

## 1-vs-3 (challenger in seat g%4 vs three TA; seeds 41,000,000+g (validation; TA+ here = lead-high in all rounds, before the trump-only change); paired vs the all-TA table, same seat)

| challenger | games | Δscore vs TA /game | P(win game) | hit rate | overshoot share of misses | over / under rate per round | opponents' Δscore /game (each) | TA seat (same games): hit / overshoot share |
|---|---|---|---|---|---|---|---|---|
| TA | 3000 | +0.00 ± 0.00 | 0.263 ± 0.016 | 0.5781 ± 0.0035 | 0.658 [0.653,0.663] | 0.2777 / 0.1441 | +0.00 ± 0.00 | 0.5781 / 0.658 [0.653,0.663] |
| Greedy | 3000 | -43.63 ± 1.20 | 0.018 ± 0.005 | 0.4182 ± 0.0033 | 0.663 [0.658,0.667] | 0.3855 / 0.1963 | +4.02 ± 0.56 | 0.5781 / 0.658 [0.653,0.663] |
| TA-spite | 3000 | +0.00 ± 0.00 | 0.263 ± 0.016 | 0.5781 ± 0.0035 | 0.658 [0.654,0.663] | 0.2777 / 0.1441 | -0.09 ± 0.10 | 0.5781 / 0.658 [0.653,0.663] |
| TA+ (lead-high all rounds) | 3000 | +5.09 ± 0.76 | 0.321 ± 0.017 | 0.5946 ± 0.0035 | 0.618 [0.612,0.623] | 0.2505 / 0.1549 | -1.20 ± 0.34 | 0.5781 / 0.658 [0.653,0.663] |
| shed only | 3000 | +2.21 ± 0.36 | 0.287 ± 0.016 | 0.5853 ± 0.0035 | 0.626 [0.621,0.631] | 0.2596 / 0.1551 | -0.90 ± 0.16 | 0.5781 / 0.658 [0.653,0.663] |
| lead-high (all rounds) only | 3000 | +2.78 ± 0.71 | 0.289 ± 0.016 | 0.5871 ± 0.0034 | 0.646 [0.640,0.650] | 0.2666 / 0.1463 | -0.21 ± 0.32 | 0.5781 / 0.658 [0.653,0.664] |
| TA+(lh all) + surplus-duck | 3000 | +5.10 ± 0.77 | 0.319 ± 0.017 | 0.5946 ± 0.0035 | 0.617 [0.611,0.621] | 0.2500 / 0.1555 | -1.24 ± 0.34 | 0.5781 / 0.658 [0.653,0.663] |

Per-round Δscore vs TA split by round type (same games):

| challenger | trump rounds Δ/round | NT rounds Δ/round | rounds with n≤5 Δ/round | rounds n≥6 Δ/round |
|---|---|---|---|---|
| Greedy | -1.537 ± 0.054 | -2.576 ± 0.110 | -1.087 ± 0.058 | -2.115 ± 0.068 |
| TA-spite | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 |
| TA+ (lead-high all rounds) | +0.251 ± 0.035 | +0.015 ± 0.063 | +0.072 ± 0.026 | +0.278 ± 0.045 |
| shed only | +0.086 ± 0.015 | +0.098 ± 0.042 | -0.014 ± 0.014 | +0.146 ± 0.021 |
| lead-high (all rounds) only | +0.160 ± 0.033 | -0.084 ± 0.054 | +0.070 ± 0.023 | +0.134 ± 0.043 |
| TA+(lh all) + surplus-duck | +0.255 ± 0.035 | -0.001 ± 0.064 | +0.072 ± 0.026 | +0.278 ± 0.046 |

## E1 robustness (n=13, trump S, hero seat 1; first 200 Phase-1 E1 hands, identical Greedy distributions; 100 worlds/candidate)

| quantity | TA play (Phase 1, same hands) | TA+ play | paired Δ (TA+ − TA) |
|---|---|---|---|
| P(oracle bid ≠ greedy-oracle bid) | 0.695 [0.628,0.755] | 0.635 [0.566,0.699] | -0.060 ± 0.063 |
| mean |shift| | 0.890 ± 0.103 | 0.815 ± 0.102 | -0.075 ± 0.079 |
| mean signed shift | -0.080 ± 0.160 | -0.175 ± 0.151 | |
| EV at own oracle bid | 7.72 ± 0.27 | 7.79 ± 0.25 | +0.07 ± 0.15 |
| EV(b*) − EV(greedy bid) under that play | 1.677 ± 0.270 | 1.638 ± 0.299 | |
| TA+ oracle bid = TA oracle bid | | 0.660 [0.592,0.722] | |

## Perfect-information agreement with max^n (n=5, Phase-2 F4b protocol: scheme-O bids, decisions on the max^n path)

| subset | decisions | TA optimal | TA+ optimal | Δ (paired) | TA 1-step loss | TA+ 1-step loss | Δ loss | same card |
|---|---|---|---|---|---|---|---|---|
| all | 11918 | 0.923 ± 0.005 | 0.920 ± 0.005 | -0.003 ± 0.003 | 0.879 ± 0.055 | 0.913 ± 0.056 | +0.035 ± 0.034 | 0.868 |
| non-indifferent | 3535 | 0.741 ± 0.014 | 0.730 ± 0.015 | -0.010 ± 0.010 | 2.963 ± 0.166 | 3.079 ± 0.168 | +0.117 ± 0.115 | 0.837 |
| non-indiff, trump S | 1677 | 0.699 ± 0.022 | 0.704 ± 0.022 | +0.005 ± 0.015 | 3.475 ± 0.255 | 3.417 ± 0.253 | -0.058 ± 0.172 | 0.831 |
| non-indiff, NT | 1858 | 0.779 ± 0.019 | 0.755 ± 0.020 | -0.024 ± 0.014 | 2.500 ± 0.214 | 2.774 ± 0.222 | +0.274 ± 0.153 | 0.843 |

## Mixed 4-policy tables (5 subsets × 4 seat rotations per block, all 20 tables of a block share seed 51,000,000+b; 1 blocks = 20 games)

| policy | mean score (games=16/block) | Δ vs TA (paired by block) | P(win game) | P(win) Δ vs TA | hit rate | overshoot share | BT strength | Elo vs TA [95% CI] |
|---|---|---|---|---|---|---|---|---|
| greedy | 136.25 ± nan | -13.31 ± nan | 0.000 ± nan | -0.031 ± nan | 0.4825 ± nan | 0.643 [0.643,0.643] | 0.261 | -115 [-115,-115] |
| ta | 149.56 ± nan | +0.00 ± nan | 0.031 ± nan | +0.000 ± nan | 0.5350 ± nan | 0.801 [0.801,0.801] | 0.505 | +0 [+0,+0] |
| ta_spite | 165.38 ± nan | +15.81 ± nan | 0.312 ± nan | +0.281 ± nan | 0.5900 ± nan | 0.768 [0.768,0.768] | 1.193 | +149 [+149,+149] |
| ta_plus | 176.31 ± nan | +26.75 ± nan | 0.500 ± nan | +0.469 ± nan | 0.6300 ± nan | 0.716 [0.716,0.716] | 2.209 | +256 [+256,+256] |
| hybrid | 192.19 ± nan | +42.62 ± nan | 0.406 ± nan | +0.375 ± nan | 0.6825 ± nan | 0.583 [0.583,0.583] | 2.884 | +303 [+303,+303] |

Pairwise finishing order (row finishes above column; ties = ½), fraction over all shared tables:

| | greedy | ta | ta_spite | ta_plus | hybrid |
|---|---|---|---|---|---|
| greedy | - | 0.250 | 0.250 | 0.208 | 0.000 |
| ta | 0.750 | - | 0.167 | 0.250 | 0.125 |
| ta_spite | 0.750 | 0.833 | - | 0.250 | 0.333 |
| ta_plus | 0.792 | 0.750 | 0.750 | - | 0.500 |
| hybrid | 1.000 | 0.875 | 0.667 | 0.500 | - |

(pairs per cell: 12; per-game wall time: tables with Hybrid 0.52 s, without 0.012 s)


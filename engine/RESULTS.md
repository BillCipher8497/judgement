# Phase 0 Results

- Tests: 82 passed (~4 s): unit, hypothesis invariants, determinism/replay, sampler uniformity (exact brute-force check), differential vs naive reference, OpenSpiel cross-val.
- Benchmarks (single core, py3.13): 9,533 random 13-card 4p rounds/s (target ≥5,000); greedy 9,359/s; 5,684/s via play_round with per-round seeding; full 25-round games 552/s random, 402/s greedy. Oracle 1,000-sample 13-card distribution: 0.14 s (target <2 s).
- Cross-validation: 0 mismatches. See crossval/REPORT.md.

## Greedy baseline (4p, 13 cards, trump S, hero seat 1, all seats greedy, 10k samples/hand)
Oracle bid = argmax_b Σ_t P(t)·score(b,t). Bimodal hand 5: mean-tricks bid 8 lands in the trough (P=.057) and scores 1.03; oracle bid 7 scores 6.94.

4p, 13 cards, trump S, hero seat 1, all-greedy play, 10000 samples/hand

| # | hero hand | source | E[t] (sd) | mean-tricks bid -> E[score] | oracle bid -> E[score] |
|---|---|---|---|---|---|
| 0 | `S:8543 H:J9865 D:- C:Q863` | seed 0 | 1.50 (1.22) | 1 -> 4.04 | 1 -> 4.04 |
| 1 | `S:J92 H:A986 D:K4 C:KQJ8` | seed 1 | 3.28 (1.20) | 3 -> 3.30 | 4 -> 4.86 **(diverges)** |
| 2 | `S:AQJ652 H:AQ3 D:432 C:8` | seed 2 | 6.10 (0.70) | 6 -> 9.67 | 6 -> 9.67 |
| 3 | `S:KQT632 H:985 D:J876 C:-` | seed 1316, first long-trump+void | 4.38 (0.86) | 4 -> 4.53 | 5 -> 7.62 **(diverges)** |
| 4 | `S:KQJ9843 H:7 D:- C:AK986` | seed 2706, first bimodal long-trump+void | 9.56 (0.92) | 10 -> 14.63 | 10 -> 14.63 |
| 5 | `S:QT86532 H:Q D:- C:AKQJT` | seed 3653, first bimodal with mean in trough (extra) | 7.80 (1.38) | 8 -> 1.03 | 7 -> 6.94 **(diverges)** |

Trick distributions P(k):

| # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.216 | 0.367 | 0.203 | 0.143 | 0.055 | 0.011 | 0.002 | . | . | . | . | . | . | . |
| 1 | 0.001 | 0.091 | 0.173 | 0.254 | 0.347 | 0.116 | 0.018 | 0.001 | . | . | . | . | . | . |
| 2 | . | . | . | . | 0.021 | 0.122 | 0.604 | 0.242 | 0.011 | . | . | . | . | . |
| 3 | . | 0.003 | 0.028 | 0.117 | 0.324 | 0.508 | 0.016 | 0.005 | 0.001 | . | . | . | . | . |
| 4 | . | . | . | . | . | . | 0.002 | 0.022 | 0.191 | 0.019 | 0.731 | 0.035 | . | . |
| 5 | . | . | . | . | . | 0.007 | 0.154 | 0.408 | 0.057 | 0.214 | 0.160 | . | . | . |

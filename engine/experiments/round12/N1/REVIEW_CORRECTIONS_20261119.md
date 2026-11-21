# N1 interpretation and diagnostic corrections — Codex, 2026-11-19

Claude external numerical verdict stands. Independent Nov19 checks reaggregated all saved high-budget diagnostic arrays and replayed14stored16384-proposal loss/ESS records. These are verification replays, not a new experiment. Auditor artifacts are preserved in ../CLAUDE_EXTERNAL_20261118 with SHA256 manifest.

## What the comparison establishes

The predeclared confirmation primary remains net-full minus the **K256 resampled SIR implementation, batch64, maximum2048uniform proposals, ESS target256**, not exact or converged bid-informed inference. Primary−0.048316 [−0.053126,−0.043506] nats/decision,n3520independent rounds,95%paired176stratum CI conditional on the frozen ensemble. Smoothing sensitivities address zero counts; they do not eliminate finite-proposal or resampling variance.

The auditor high-budget method instead uses **16384uniform proposals and weighted marginals**, with the same.001prop floor and no final256-world resampling. Both proposal count and estimator changed. 16384/256=64 compares proposals with output worlds; it is **not measured compute**. Relative to the original proposal ceiling, the count ratio is8, with larger ratios when the adaptive original bank was smaller. No timing benchmark or convergence certificate was supplied for these diagnostics.

|Post-hoc diagnostic|Contrast (nats/decision)|Mean[95%CI]|n rounds|
|---|---|---|---|
|initial_100_n_ge8|BI_K256-minus-weighted16384|+0.057397 [+0.027638, +0.087156]|100|
|initial_100_n_ge8|full-minus-weighted16384|-0.028313 [-0.044023, -0.012602]|100|
|initial_100_n_ge8|bids-minus-weighted16384|+0.023674 [+0.009874, +0.037474]|100|
|n8_13|full-minus-highA|-0.000816 [-0.020896, +0.019263]|60|
|n8_13|full-minus-highB|-0.002248 [-0.022278, +0.017782]|60|
|n8_13|bids-minus-highA|+0.046964 [+0.028120, +0.065808]|60|
|n8_13|bids-minus-highB|+0.045532 [+0.026339, +0.064725]|60|
|n8_13|highA-minus-highB|-0.001432 [-0.009884, +0.007020]|60|
|n5_7|full-minus-highA|-0.005486 [-0.017155, +0.006182]|60|
|n5_7|full-minus-highB|-0.001551 [-0.012679, +0.009578]|60|
|n5_7|bids-minus-highA|+0.016471 [+0.006498, +0.026443]|60|
|n5_7|bids-minus-highB|+0.020406 [+0.010296, +0.030517]|60|
|n5_7|highA-minus-highB|+0.003936 [-0.000372, +0.008244]|60|

The A/B columns within each range use **the same60contexts**, with different Monte Carlo seeds. They are not two independent chance replications; no pooling as120independent contexts. The initial100and later60subsets also overlap. Preserve both the initial favourable subset and subsequent near-zero comparisons. All are post-hoc auditor diagnostics, with paired round-normal CIs, not prospective equivalence tests. Describe the net as showing comparable measured log loss on these subsets; do not claim equivalence to exact/converged BI, general superiority to BI, or a demonstrated64×runtime saving.

## Original sampler degeneracy

|Original round size|n rounds|Reached2048before ESS256, count/fraction[Wilson95%]|ESS<32, count/fraction[Wilson95%]|
|---|---:|---|---|
|3|320|56; 0.1750 [0.1373,0.2204]|4; 0.0125 [0.0049,0.0317]|
|4|320|136; 0.4250 [0.3720,0.4797]|9; 0.0281 [0.0149,0.0526]|
|5|320|180; 0.5625 [0.5077,0.6158]|22; 0.0688 [0.0458,0.1019]|
|6|320|193; 0.6031 [0.5486,0.6552]|27; 0.0844 [0.0586,0.1200]|
|7|320|213; 0.6656 [0.6122,0.7151]|38; 0.1187 [0.0878,0.1588]|
|8|320|226; 0.7063 [0.6541,0.7535]|59; 0.1844 [0.1457,0.2305]|
|9|320|239; 0.7469 [0.6965,0.7914]|62; 0.1938 [0.1542,0.2406]|
|10|320|226; 0.7063 [0.6541,0.7535]|53; 0.1656 [0.1289,0.2103]|
|11|320|244; 0.7625 [0.7129,0.8058]|80; 0.2500 [0.2057,0.3002]|
|12|320|236; 0.7375 [0.6867,0.7827]|60; 0.1875 [0.1485,0.2339]|
|13|320|218; 0.6813 [0.6283,0.7299]|63; 0.1969 [0.1570,0.2439]|

## Uniform zero count and calibration

Observed raw uniform true-holder zero counts are **1/112696scored cards in1/3520rounds**. These are observed totals; cards within a round are dependent. For the affected-round proportion, the Wilson95%CI is [0.00005015,0.00160755]. Do not use the old negative normal bound or silently change a round-weighted mean card fraction into an iid-card estimate. Original diagnostic JSON remains preserved.

ECE is descriptive fixed10-bin top-label calibration with round-equal weights, argmax tie convention, upward absolute-error bias and sparse-bin limitations. It is not full-joint calibration or a posterior certificate. Bootstrap draws are paired across methods; overlapping individual CIs are not a test of their difference.

All10fits selected epoch50from{10,25,50}; alpha1was the blend boundary. The validation curve was still falling, so this does not bracket an optimum. Detailed secondary diagnostics were specified **after pilot labels but before confirmation**; they were prospective to confirmation, not before all data. Frozen scientific code/data/seeds remain unchanged. No human, arbitrary-pool or game-score claim follows.

# Matched controls — Claude external validated with corrections

Primary R-L supported on both fresh stages for these fixed ensembles. R=tanhMLP+explicit canonical offset; L=linear+offset; N=sameMLP without explicit offset but retaining canonical predictions as inputs. All share frozenM4training/selection data,5paired2060Mfit seeds,50epochs and epoch/blend selection. Selection reused; no universal architecture claim.

|stage|contrast|points/game +/-95%CI halfwidth|ties-as-win +/-95%CI halfwidth|n|
|---|---|---:|---:|---:|
|screen|R-L|0.908000 +/-0.333384|0.006000 +/-0.009603|2000|
|screen|R-N|-0.232000 +/-0.267681|-0.002500 +/-0.008486|2000|
|screen|R-canonical|1.078500 +/-0.392777|0.012000 +/-0.011077|2000|
|screen|L-canonical|0.170500 +/-0.270738|0.006000 +/-0.008307|2000|
|screen|N-canonical|1.310500 +/-0.427539|0.014500 +/-0.012112|2000|
|confirmation|R-L|0.567500 +/-0.340144|0.008500 +/-0.010778|2000|
|confirmation|R-N|-0.325000 +/-0.282475|-0.002000 +/-0.008767|2000|
|confirmation|R-canonical|0.686000 +/-0.392865|0.013500 +/-0.011377|2000|
|confirmation|L-canonical|0.118500 +/-0.272733|0.005000 +/-0.008434|2000|
|confirmation|N-canonical|1.011000 +/-0.425342|0.015500 +/-0.012725|2000|

Intervals are marginal equal-seat paired normal95%CIs across games, conditional on fixed5-fit ensembles; no across-training-population uncertainty claimed. PrimaryR-Lrequiresbothstagespositive. R-N is secondary and does not show an explicit-offset benefit; N-Rpositiveonlyconfirmation is not a replicated advantage. Linear-canonical includeszero in bothstages; no gain/equality claim. All nulls retained.

Separate verify.py recomputed45candidatevalidation totals/15selectedmodelchoices/threeensembleblendchoices,4000rawquadruplets/score-hook-trick totals, all pairedCIs and160freshfull-game replays. Frozen source/data hashes match. Seeds2040M/2050M; hero g%4. Source freeze was created before these stages, but code was not committed before this legacyrun: futureprotocols require Claude-reviewed pre-runcommits and selectionNIgate. No originalM4results changed. Claude external verdict dated 2026-11-17; corrections appended before scoped local commit.

## Claude external review corrections (2026-11-17)

Validated with required documentation changes; applied below. The score gain R-L replicated; the ties-as-win gain did not (both primary win intervals include zero). R-L is about 0.3-0.5% of a canonical hero mean near 167; this rounded scale is descriptive, not a separately estimated ratio. Exact mean/CI scale diagnostics follow below.
R-L compares whole selected pipelines: different selected epochs and alpha (L=0.5, R=1). It does not isolate nonlinearity. N retains canonical predictions as inputs but removes the explicit logit offset. N-L and R-N are important descriptive controls, not hidden comparisons.

| family | selected epochs (five fits) | alpha | selection score total | n selection contexts |
|---|---|---:|---:|---:|
|L|[50, 50, 50, 50, 25]|0.5|79059|12000|
|R|[10, 10, 25, 10, 10]|1.0|80665|12000|
|N|[50, 25, 50, 25, 50]|1.0|80569|12000|
L selected epoch50 in4/5fits while validation score was still rising; R selected epoch10 in4/5. R/N alpha1 is an endpoint. The grids did not bracket the optimum; validation was reused for epoch and blend selection. Totals above are exact sums over the fixed selection contexts, not independent effect estimates.
Provenance: run.py was modified at 2026-11-17T18:45Z (13:45 ET), after M4 outcomes, and was not committed before execution. Preserve source/data freezes as executed; this will be a post-run commit. Frozen source hashes, not a pre-run Git commit, establish in-run consistency.
Seed collision search command (repeated diagnostic only): `git grep -n -E "2040000000|2050000000|2060000000|2040M|2050M|2060M" HEAD -- engine`. An empty output means no matching tracked HEAD entries, not proof about every untracked/external use. Current source/metadata registrations are documented separately. No fresh experimental draws.

### Raw-derived N-L and magnitude diagnostic

All rows n=2000 games/stage, paired/equal-seat normal95%CIs; ratios use a paired delta method. These are post-hoc descriptive scales.

|stage|canonical mean [95%CI]|N-L [95%CI]|R-canonical / canonical mean [95%CI]|
|---|---:|---:|---:|
|screen|166.796500 [165.567524, 168.025476]|1.140000 [0.754542, 1.525458]|0.006466 [0.004101, 0.008831]|
|confirmation|166.262500 [165.036406, 167.488594]|0.892500 [0.493240, 1.291760]|0.004126 [0.001758, 0.006494]|

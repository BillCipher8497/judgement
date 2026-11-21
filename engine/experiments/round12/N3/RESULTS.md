> External verdict, Claude18Nov; independently confirmed Codex19Nov: **reproducible but uninformative**. 38/40fits selected alpha0; the other two are suppressed by the five-model ensemble. Fixed-coefficient activation bounds confirm every frozen ensemble retains the baseline argmax. Game zeros are identities, not evidence that activated hard/set neural players perform equally. No retuning or reuse of consumed2130M/2140M seeds.

# N3 — late-play teacher distillation

Status: externally reviewed(Claude,2026-11-18), reproducible but uninformative; independently rechecked(Codex,2026-11-19). Local result packet only. Fixed-source two-card intervention, not an all-stage neural player.

## Predeclared primary full-game comparisons

All game contrasts below are deterministic identities, not sampling CIs. All eight deployed students are action-identical to TA+v2. The fixed count was n=2000matched nine-arm blocks per stage. Each hero rotates g%4; all seats use the canonical ModelBidder, with three TA opponents. No inferential game interval is assigned to these structural zeros.

|Primary|Screen2130M|Fresh-deal confirmation2140M|
|---|---|---|
|PRIMARY_set-minus-hard|0 (identity)|0 (identity)|
|PRIMARY_best-minus-baseline|0 (identity)|0 (identity)|

The name `hard` won an exact eight-way validation tie by list order. There was no activated learned intervention in either primary game contrast. Two co-primary contrasts and all exploratory controls are retained, without a familywise significance claim.

## All game controls and table consequences

|Contrast|Stage|Hero score|Ties-as-win change|Table total|Mean opponent score|
|---|---|---|---|---|---|
|PRIMARY_set-minus-hard|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|PRIMARY_best-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|hard-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|uniform-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|q-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|direct-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|linear-hard-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|linear-set-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|no-offset-set-minus-baseline|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|uniform-minus-hard|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-uniform|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-linear-set|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|hard-minus-linear-hard|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-no-offset-set|screen|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|PRIMARY_set-minus-hard|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|PRIMARY_best-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|hard-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|uniform-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|q-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|direct-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|linear-hard-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|linear-set-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|no-offset-set-minus-baseline|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|uniform-minus-hard|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-uniform|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-linear-set|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|hard-minus-linear-hard|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|
|set-minus-no-offset-set|confirmation|0 (identity)|0 (identity)|0 (identity)|0 (identity)|

All four game metrics above are zero by policy identity on the n2000paired blocks/stage; no inferential CI or learned-policy equivalence test is claimed. Table/opponent effects are secondary general-sum consequences, not proof of optimal cooperation. Baseline absolute means and CIs, all totals, round scores and partial-arm recovery files are retained in raw JSON.

|Exploratory selected-pipeline interaction|Screen|Confirmation|
|---|---|---|
|(MLPset−MLPhard)−(linearset−linearhard)|0 (identity)|0 (identity)|

This interaction is not an isolated nonlinearity effect: capacity, selected epochs and blends differ. The no-offset arm still receives baseline indicators; it tests explicit offset structure, not absence of baseline information.

## Held finite-world teacher diagnostic

n=2400independent conditional contexts,96fixed strata, K16shared worlds/context. CIs are paired/context-stratified95%normal intervals, not16independent samples per observation. This diagnostic uses a structural-uniform, PI-MaxN-continuation teacher surrogate; it is not actual TA continuation, a true posterior or a hidden-value certificate.

|Decoder/student|Surrogate regret, mean[95%CI]|
|---|---|
|hard|0.428307 [0.388103, 0.468511]|
|uniform|0.000000 [0.000000, 0.000000]|
|set|0.000000 [0.000000, 0.000000]|
|student-hard|0.261719 [0.220456, 0.302981]|
|student-uniform|0.261719 [0.220456, 0.302981]|
|student-set|0.261719 [0.220456, 0.302981]|
|student-q|0.261719 [0.220456, 0.302981]|
|student-direct|0.261719 [0.220456, 0.302981]|
|student-linear-hard|0.261719 [0.220456, 0.302981]|
|student-linear-set|0.261719 [0.220456, 0.302981]|
|student-no-offset-set|0.261719 [0.220456, 0.302981]|
|set-minus-hard|-0.428307 [-0.468511, -0.388103]|
|set-minus-uniform|0.000000 [0.000000, 0.000000]|
|student-set-minus-student-hard|0.000000 [0.000000, 0.000000]|
|baseline|0.261719 [0.220456, 0.302981]|

Uniform-optimal and membership label-only ordering coincide with the two-action/public-hit-payoff finite-world optimum by the restricted M15 identity. Any zero diagnostic is that surrogate identity, not an empirical claim of zero true-game regret. Hard labels choose the lowest card in each tied world. The reported teacher-label gap is separate from finite neural optimization and full-game outcomes.

## Selection and reproducibility

|Arm|Epochs, five fits|Blends, five fits|
|---|---|---|
|hard|5, 5, 5, 5, 5|0.0, 0.0, 0.0, 0.0, 0.0|
|uniform|5, 5, 5, 5, 5|0.0, 0.0, 0.0, 0.0, 0.0|
|set|5, 5, 5, 5, 5|0.0, 0.0, 0.0, 0.0, 0.0|
|q|5, 5, 5, 5, 5|0.0, 0.0, 0.0, 0.0, 0.0|
|direct|5, 5, 10, 5, 5|0.0, 0.0, 0.25, 0.0, 0.0|
|linear-hard|5, 5, 5, 5, 5|0.0, 0.0, 0.0, 0.0, 0.0|
|linear-set|5, 5, 5, 5, 5|0.0, 0.0, 0.0, 0.0, 0.0|
|no-offset-set|25, 5, 5, 5, 5|0.5, 0.0, 0.0, 0.0, 0.0|

Epochs5/10/25/50and blends0/.25/.5/1were selected on reused2400validation contexts, then ensembles ranked on that same finite-K surrogate. Candidate CIs are descriptive selection diagnostics. Search endpoints do not certify optima. Train9600contexts, SHA-original-own-hand split, held-label2400contexts; pilot192excluded. Equal accepted96strata upweight rare active round types, while utility tests are unfiltered fresh games including all hand partitions. No held-context tuning or sample extension.

Pre-run source commit: `55971df8371c3e38cf6a4d77de62d323af72d986`; source fingerprint `1b36d09d470c2c5f3b3b0500bf42c7f207e258645c8caf1fc8403c90a4cc4361`. Five paired initialization/batch schedules use2120M/domain0x4E33, with actual attempt roots and child-state collision checks in the journal. Runtime bounds, original raw worlds/prefixes, Adam checkpoints, selected models, stage totals and every uncertainty calculation are retained.

Separate checks: every saved context/teacher world uses fresh manual deal/rules/solver replay (sampler and fixed policies reused as definitions), all40model choices and held-context CIs checked, every game score/hook/tie/paired CI recomputed, and144full games independently replayed with a manual engine and numerical forward pass. This is same-investigator cross-checking, not external validation.

The conditional dataset, summarized observations, finiteK, teacher/opponent mismatch and narrow opponent pool limit scope. Neither game improvement nor a null identifies pure loss-function dynamics, a general hidden-information optimum, human benefit or novelty. No publication release is implied.


## Claude external audit addendum (2026-11-18)
See root `CLAUDE_AUDIT_2026-11-18.md`, section N3, for the verdict, the post-hoc diagnostics and the requested changes. That section governs the interpretation of this packet: where it conflicts with the text above, the audit wording applies.

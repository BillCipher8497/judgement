# N12 — belief KL versus exact private decision regret

**Externally validated by Claude, 2026-11-20**, with one reporting change applied: the decision-relevant stratification below. See CLAUDE_AUDIT_2026-11-20.md.

PRIMARY largeNT/summaryMLP/80 joint-minus-marginal regret: 0.00000000 [0.00000000, 0.00000000] hero points, n5 paired frozen optimizer seeds, t95% df4 ONLY. All five observed differences are exactly0 on this fixed subset. The arithmetic optimizer interval[0,0] is degenerate; it does not establish population equivalence, universal identity or a guarantee for new seeds/hands. N9 improved KL on its larger test benchmark, but that alone does not establish a decision gain here.

All1375 eligible held keys were selected prospectively by an observable two-remaining-card/two-legal-action predicate. Each key uses the COMPLETE structural allocation set and exact posterior subset from frozen N9/N2. Force either current action, then fixed private TA opponents; the hero last card is forced. There is no future free hero choice or strategy fusion. These are one-step private-continuation diagnostics, not solved game values.

All240 frozen recipes were decoded without refitting or selecting a replacement. Numerical model expectations within1e-10 score points choose the lower card; exact integer control ties choose the lower card. The posterior-weighted deficit is an exact rational sum of integer payoffs conditional on that numerical decoding. No world/hand sampling CI applies to those fractions. Cross-seed intervals below cover only the five optimizer seeds on known, previously inspected hand splits; all secondary cells descriptive/unadjusted.

| condition | selected keys | exact reach | uniform-F action regret | exact-R oracle regret |
|---|---:|---:|---:|---:|
| S2R4n2_t0 | 388 | 1289 | 0 | 0 |
| S2R4n2_t-1 | 419 | 1203 | 100/1203 | 0 |
| S3R4n3_t0 | 294 | 806 | 10/403 | 0 |
| S3R4n3_t-1 | 274 | 648 | 11/648 | 0 |

## Decision-relevant stratification (added after Claude external audit, 2026-11-20)

The pooled regret above is divided by total posterior reach, which includes keys
where the two legal actions have identical exact values. Those keys contribute a
structural zero whatever any model predicts, so the pooled denominator alone does
not tell an informative zero from an uninformative one.

| condition | keys | reach | keys with differing exact action values |
|---|---:|---:|---:|
| S2R4n2_t0 | 388 | 1289 | 54 |
| S2R4n2_t-1 | 419 | 1203 | 57 |
| S3R4n3_t0 | 294 | 806 | 53 |
| S3R4n3_t-1 | 274 | 648 | 60 |
| total | 1375 | 3946 | 224 |

1151 of 1375 keys (84%) are exactly action-indifferent. In the primary cell
(S3R4n3_t-1 / summary_mlp / 80) the 60 decision-relevant keys are all decided
optimally by both arms in all five seeds, so the primary zero is not produced by
indifference. No number above changes and nothing was re-run; this is a
reporting addition. The systematic version across all conditions and all 240
recipes is N17.

Exact finite-support controls above have no sampling intervals. Oracle zero is by construction. results/evaluation.json records each recipe's exact regret numerator/fraction, KL on THIS SAME restricted subset, near-tie count, and the individual arm intervals. Do not compare this subset KL mean directly to N9's larger test-set mean.

## Joint minus marginal regret

| fixed cell | paired mean [95% CI], n5 optimizer seeds only |
|---|---|
| S2R4n2_t0/summary_linear/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/summary_linear/80 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/summary_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/summary_mlp/80 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/rich_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/rich_mlp/80 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_linear/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_linear/80 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_mlp/80 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/rich_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/rich_mlp/80 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/summary_linear/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/summary_linear/80 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/summary_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/summary_mlp/80 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/rich_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/rich_mlp/80 | 0.01637717 [-0.02909315, 0.06184749] |
| S3R4n3_t-1/summary_linear/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/summary_linear/80 | -0.01697531 [-0.01697531, -0.01697531] |
| S3R4n3_t-1/summary_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/summary_mlp/80 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/rich_mlp/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/rich_mlp/80 | 0.00000000 [0.00000000, 0.00000000] |

## Long80 minus short20 selection-pipeline regret

| fixed cell | paired mean [95% CI], n5 optimizer seeds only |
|---|---|
| S2R4n2_t0/summary_linear/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/summary_linear/joint | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/summary_mlp/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/summary_mlp/joint | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/rich_mlp/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/rich_mlp/joint | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_linear/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_linear/joint | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_mlp/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/summary_mlp/joint | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/rich_mlp/marginal | -0.01662510 [-0.01662510, -0.01662510] |
| S2R4n2_t-1/rich_mlp/joint | -0.01662510 [-0.01662510, -0.01662510] |
| S3R4n3_t0/summary_linear/marginal | -0.02481390 [-0.02481390, -0.02481390] |
| S3R4n3_t0/summary_linear/joint | -0.02481390 [-0.02481390, -0.02481390] |
| S3R4n3_t0/summary_mlp/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/summary_mlp/joint | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/rich_mlp/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/rich_mlp/joint | 0.01637717 [-0.02909315, 0.06184749] |
| S3R4n3_t-1/summary_linear/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/summary_linear/joint | -0.01697531 [-0.01697531, -0.01697531] |
| S3R4n3_t-1/summary_mlp/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/summary_mlp/joint | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/rich_mlp/marginal | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/rich_mlp/joint | 0.00000000 [0.00000000, 0.00000000] |

## Rich minus summary MLP regret

| fixed cell | paired mean [95% CI], n5 optimizer seeds only |
|---|---|
| S2R4n2_t0/marginal/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/marginal/80 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/joint/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t0/joint/80 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/marginal/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/marginal/80 | -0.01662510 [-0.01662510, -0.01662510] |
| S2R4n2_t-1/joint/20 | 0.00000000 [0.00000000, 0.00000000] |
| S2R4n2_t-1/joint/80 | -0.01662510 [-0.01662510, -0.01662510] |
| S3R4n3_t0/marginal/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/marginal/80 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/joint/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t0/joint/80 | 0.01637717 [-0.02909315, 0.06184749] |
| S3R4n3_t-1/marginal/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/marginal/80 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/joint/20 | 0.00000000 [0.00000000, 0.00000000] |
| S3R4n3_t-1/joint/80 | 0.00000000 [0.00000000, 0.00000000] |

## Separate checks and limitations

The independent checker reproduced all49,126 integer world/action outcomes with separate public legal/winner/score transitions; reconstructed complete structural F sets; independently forwarded all240 selected networks, normalizers, decisions and rational deficits; recomputed all160 arm/contrast/KL intervals. R membership relies on unchanged previously audited N2/N9 supports; this is not a new complete posterior reimplementation. All source/model/data hashes checked. Same investigator, not external approval.

The first output attempt stopped after all integer tables at JSON serialization of a NumPy int64 tie count. Original source/freeze/error retained; api_resume.py converts ONLY NumPy scalars to equal native JSON values, with fixture and hash binding. No arithmetic, payoff, decoder, source, model, count or seed was changed. There are NO new RNG draws or seed bases: this deliberately reuses frozen2170M/domain0x4e39 models as a derived known-hand test, not a fresh replication.

Files: PROTOCOL.md, append-onlySTATUS, lib12.py/run.py, source-preservingapi_resume.py, manualverify.py, report.py, all per-key hands/action tables and per-model exact decisions/raw summaries. Commands(cd engine): bounded_job.py N12/run.py(failedserializer); bounded_job.py N12/api_resume.py; separately bounded_job.py N12/verify.py; report.py. Fresh absolute stops/checkpoints inSTATUS and launch records. No core/existing policy/frozen prior sources changed; no new proof/52-card/optimal-belief/novelty claim; no commit/push awaiting Claude.

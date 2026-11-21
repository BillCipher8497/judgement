# N10 — private-counterfactual late-play students

**Externally validated by Claude, 2026-11-20, as a scope failure** (all 10 fits confirmed alpha=0, zero game data present, and 171/240 rejected candidates confirmed to activate without improving; see CLAUDE_AUDIT_2026-11-20.md). Model weights and bulk result trees remain local.

The prespecified activation gate failed. All ten fits selected alpha=0 at epoch5, so both deployed ensembles are exactly the TA+ baseline. The residual difference bound is exactly0, the baseline margin1/4 and strict slack1/4. These are arithmetic identities, not empirical equivalence claims. ZERO scientific full-game blocks were run; no game effect or confidence interval was estimated. Reserved2200M/2210M addresses were registered as metadata only, not drawn.

The 768 label-free activation fixtures showed0 changes for each arm (diagnostic rate0, conservative95% Hoeffding interval[0,0.04900627], n768). This is conditional active-context frequency, not a natural game activation rate. The exact alpha=0 identity is the stronger result. Validation contrast is exactly0 by policy identity, without a sampling CI.

## Protocol and provenance

Full52-card ladder, fixed canonical ModelBidder, TA opponents and TA+v2 baseline. Eligible contexts have two legal own cards and need in{0,1,2}; there can be only one intervention per round and no future free hero decision. This tests private opponent continuation, NOT strategy-fusion repair. Own initial hands are split by a fixed hash; all10080 contexts and disjoint train/validation/held hands were independently replayed. Counts:96pilot,6144train,1536validation,1536held and768label-free fixtures, with96seat/round strata.

Before any N10 chance draw, the proxy-primary draft was archived and the protocol revised: PRIMARY labels evaluate both actions on the actual simulated hidden deal, followed by strictly private fixed-policy continuation. Predictor inputs remain683 own/public features. K16 structural-uniform worlds are auxiliary diagnostics ONLY, never used for training, selection or gates. Actual/proxy hidden hands and reward labels are offline supervision, not prediction inputs. This gives counterfactual reward samples for this conditional generator; it does not solve the posterior/privateBR or guarantee full-game improvement. Both privileged per-world label decoders have zero hindsight regret by construction, not a private Bayes-label-risk result.

Fresh namespaces:2180M/domain0x4e40data,2190Mpairedoptimizer fits,2220Mlabel-free fixtures.2200M/2210M game targets were conditional on both gates, which failed. Counts,50epochs, five paired seeds/loss, candidate checkpoints/alphas, margin.25 and gates were fixed before data. Two losses hardCE/setBCE; NumPy MLP683-64-32-2. The synthetic guard caught a zero-padding seed alias before rollout; a separate synthetic namespace was registered and all32 immutable-fixture private-transition pairs plus gradients passed. Originalfailed checker source retained. No scientific seed/data/model was revised after outcomes.

## Held conditional diagnostics

Actual-deal hindsight regret includes hidden-current-state uncertainty. It is not private Bayes regret. CIs here use1536 independent conditional draws in96strata, conditional on the fixed selected ensemble; no optimizer/population extrapolation. The identical student/baseline values below are the same policy evaluated on the same rewards.

| fixed policy/diagnostic | actual hindsight regret, points/context |
|---|---|
| baseline | 0.401693 [0.290386, 0.512999] (n=1536; 95% CI) |
| hard | 0.401693 [0.290386, 0.512999] (n=1536; 95% CI) |
| set | 0.401693 [0.290386, 0.512999] (n=1536; 95% CI) |
| proxy_hard_actual_hindsight_regret | 0.641276 [0.505380, 0.777172] (n=1536; 95% CI) |
| proxy_set_actual_hindsight_regret | 0.236328 [0.154767, 0.317889] (n=1536; 95% CI) |

Privileged label-hard and label-set hindsight regret are exactly0 by target construction (sampling CI not applicable). The frozen generic raw helper also stores[0,0] arithmetic intervals for those two all-zero arrays; this report treats them as identities and makes no population precision claim. Set-hard selected-policy contrast is exactly0 by alpha=0 identity.

| auxiliary proxy decoder | actual score minus TA+ |
|---|---|
| proxy_hard | -0.239583 [-0.384206, -0.094961] (n=1536; 95% CI) |
| proxy_set | 0.165365 [0.055295, 0.275434] (n=1536; 95% CI) |

These privileged proxy decoders see K16 simulated counterfactual labels and are not the deployable students. Their uniform-world proxy is not the exact posterior; do not infer posterior calibration, a sampler effect or a general rollout gain.

## All rejected candidates retained

results/candidate_diagnostics.json contains all240 checkpoint/alpha candidates, validation action counts/Hoeffding intervals, paired score intervals and coefficient identity bounds. This is a posthoc diagnostic on reused selection data, not a new held-out comparison or a replacement selection. No rejected candidate was deployed after the gate failed.

## Separate checks and files

verify.py independently regenerates ALL10080 prefixes, including earliest accepted rejection attempts; checks strict public observations/features/hand partitions; manually replays both actual-deal actions and all32 auxiliary proxy actions per labeled context; recomputes all ten model selections, independent forwards, gates, held metrics and95%CIs. No full-game replay applies because zero blocks ran. Same investigator, not Claude external validation.

Files: PROTOCOL.md, append-onlySTATUS.txt, n10lib.py/run.py/fit.py/evaluate.py, reference.py, self_check.py, verify.py, report.py; results/raw64-row chunks/public histories/worlds/rewards, immutable optimizer checkpoints/models/freezes, gate/held/candidate reports. Exact commands(cd engine): bounded_job.py N10/run.py --preflight; bounded_job.py N10/self_check.py; bounded_job.py N10/run.py; separately bounded_job.py N10/verify.py; report.py. Absolute launch cuts/epochs inSTATUS and launch.json/verify_launch.json; every64-row/epoch checkpoint retained. All science source hashes verified unchanged. No core/existing policy/frozen prior source/N0/Torch edit; no commit or push.

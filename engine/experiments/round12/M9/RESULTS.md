# M9 private sequential deployment — Claude external validated, corrections applied


**Every deployed root-trained student is worse than the existing feasible private policy in both large games.** The primary descriptive contrast is large-NT set BCE minus hard CE. It avoids some label-induced cost but does not deliver a competitive sequential policy. No causal separation of root mis-decoding and later-key behavior is supported.

All chance deals are evaluated exactly per fitted profile:2520 per small condition,369600 per large condition. Per-fit exact means/bounds have no sampling CIs. Summaries below use five paired optimization seeds with t95%CIs (df4); these intervals do not quantify future-hand or chance-sampling uncertainty.

| condition | exact deals/profile | PI | J1 | feasible upper | hard deployed mean +/- seed95%CI | set deployed mean +/- seed95%CI |
|---|---:|---:|---:|---:|---:|---:|
|S2R4n2_t0|2520|1257/140|1257/140|1257/140|8.978571 +/- 0.000000|9.308492 +/- 0.468742|
|S2R4n2_t-1|2520|946/105|946/105|946/105|9.371429 +/- 0.000000|9.371429 +/- 0.000000|
|S3R4n3_t0|369600|367921/61600|446669/73920|574247/92400|7.815024 +/- 0.031443|7.533880 +/- 0.059002|
|S3R4n3_t-1|369600|106419/15400|213259/30800|109709/15400|8.235500 +/- 0.033086|7.808720 +/- 0.059056|

| condition | hard deployed-minus-upper +/- seed95%CI | set deployed-minus-upper +/- seed95%CI | paired set-hard [seed95%CI] |
|---|---:|---:|---:|
|S2R4n2_t0|0.000000 +/- 0.000000|0.329921 +/- 0.468742|0.329921 [-0.138822, 0.798663]|
|S2R4n2_t-1|0.361905 +/- 0.000000|0.361905 +/- 0.000000|0.000000 [0.000000, 0.000000]|
|S3R4n3_t0|1.600230 +/- 0.031443|1.319086 +/- 0.059002|-0.281144 [-0.338864, -0.223424]|
|S3R4n3_t-1|1.111539 +/- 0.033086|0.684759 +/- 0.059056|-0.426780 [-0.473004, -0.380556]|
All rows have n=5 fits per arm, paired by seed. For large NT, exact per-fit hard-minus-upper ranges +1.06 to+1.13 and set-minus-upper +0.63 to+0.74 hero points/deal; large trump +1.58 to+1.64 and +1.24 to+1.37. These are ranges of five exact enumerations, not sampling intervals.
Small trump: hard CE attains the exact optimum in5/5fits, while set BCE is worse in3/5. Small NT: all10profiles are identical. Both small conditions have only one reached held-out own hand per split; the deployed students largely memorize training hands. All teammate hand partitions mix in this deployment stress test; it is not held-out team generalization.
The neural profile acts at every matching private key, with no hidden root-stage flag or omniscient continuation. Root keys can recur later:9164/41712(trump),16192/42440(NT), exact counts. M3 label alignment does not establish sequential policy quality. This failed transfer motivates sequential distillation N3 and strategy-fusion N6.
Verification: Codex fresh128manual deals/profile (5120 checks total) plus cached full exact evaluation; Claude independently reimplemented full-deal replay for8of40profiles (all2520or369600deals/profile), matching per deal. PI<=J1<=feasible upper<=deployed holds for all40. No improved feasible upper.
Provenance correction: PROTOCOL.md was last modified 2026-11-17T05:29:46Z, after M3 outcomes at05:26:02Z. Its hash was first recorded at05:55:50Z. Thus this is a post-outcome deployment diagnostic, not a fully preregistered independent experiment. No original protocol/hash/raw files changed; first Git commit is post-run.


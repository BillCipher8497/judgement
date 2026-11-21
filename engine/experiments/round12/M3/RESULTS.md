# Root label-misalignment audit

Externally validated (Claude,2026-11-17). Set supervision avoids a label-induced cost; hard-CE gap is almost entirely label misalignment, not a learning-dynamics advantage. Label and decoder ties choose lowest original-hand index. Exact population costs have no sampling CI. Learned intervals below are paired seed t95%CIs only.

Primary hypothesis supported.

discovery: set-BCE minus hard-CE -0.33342967 +/-0.02564339; n=5paired seeds,95%tCI; 38753worlds/4967keys/20distinct reached hands.

confirmation: set-BCE minus hard-CE -0.53784717 +/-0.04233016; n=5paired seeds,95%tCI; 24340worlds/2549keys/16distinct reached hands.

Discovery/confirmation spread shows hand-sampling uncertainty dominates; seed CIs do not cover it. Same fits, two fixed populations, not independent training replications.

|condition/split|actual hands|worlds|information floor|hard excess|uniform excess|set excess|

|---|---:|---:|---:|---:|---:|---:|

|S2R4n2_t-1/discovery|1|162|0.00000000|0.61111111|0.00000000|0.00000000|

|S2R4n2_t-1/confirmation|1|162|0.00000000|0.00000000|0.00000000|0.00000000|

|S2R4n2_t0/discovery|1|162|0.00000000|0.00000000|0.00000000|0.00000000|

|S2R4n2_t0/confirmation|1|162|0.00000000|0.00000000|0.00000000|0.00000000|

|S3R4n3_t-1/discovery|20|38753|0.00882512|0.35339200|0.00278688|0.00000000|

|S3R4n3_t-1/confirmation|16|24340|0.03134758|0.56581758|0.00098603|0.00000000|

|S3R4n3_t0/discovery|20|38071|0.06821465|0.20713929|0.00824775|0.00000000|

|S3R4n3_t0/confirmation|16|24655|0.09856013|0.37769215|0.01415534|0.00000000|

|MLP condition/split|hard CE|uniform CE|set BCE|Q regression|direct|

|---|---:|---:|---:|---:|---:|

|S2R4n2_t0/mlp/discovery|0.00000 +/-0.00000|0.52593 +/-0.63088|1.05185 +/-1.78838|0.52593 +/-1.46020|0.29630 +/-0.50377|

|S2R4n2_t0/mlp/confirmation|0.00000 +/-0.00000|0.35556 +/-0.24680|0.26667 +/-0.30226|0.00000 +/-0.00000|0.08889 +/-0.24680|

|S2R4n2_t-1/mlp/discovery|0.61111 +/-0.00000|1.14691 +/-0.91856|0.61111 +/-0.00000|0.61111 +/-0.00000|0.45062 +/-0.25351|

|S2R4n2_t-1/mlp/confirmation|0.00000 +/-0.00000|0.29630 +/-0.25188|0.00000 +/-0.00000|0.01481 +/-0.04113|0.02963 +/-0.05038|

|S3R4n3_t0/mlp/discovery|0.21162 +/-0.01125|0.03738 +/-0.01590|0.05651 +/-0.02720|0.06790 +/-0.02312|0.05507 +/-0.01498|

|S3R4n3_t0/mlp/confirmation|0.35757 +/-0.01539|0.03295 +/-0.01100|0.04861 +/-0.02169|0.07007 +/-0.03405|0.07595 +/-0.00517|

|S3R4n3_t-1/mlp/discovery|0.35195 +/-0.01608|0.03446 +/-0.00337|0.01852 +/-0.01157|0.05853 +/-0.02026|0.03911 +/-0.00766|

|S3R4n3_t-1/mlp/confirmation|0.59730 +/-0.04562|0.07495 +/-0.01712|0.05946 +/-0.03312|0.07117 +/-0.00988|0.08621 +/-0.04436|

Every learner cell: n5seeds,95%tCI, fixed exact population. Secondary comparisons descriptive; small controls do not uniformly favor BCE. Targets retain different label information, no label-efficiency claim.

Numerical stack: results/numerical_stack.json. Audit minimum top-two logit gap1.6e-7; exact integer totals require the same numerical stack. No full-team/52-card/general learning-advantage or novelty claim.

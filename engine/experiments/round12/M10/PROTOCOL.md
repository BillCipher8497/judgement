# Matched residual-model ablations — beforeM4held-out score summaries

Motivation: a neural residual gain against the old bidder alone would not isolate
architecture or the explicit offset. OldR4 used different features/data/fitting.
Compare matched new families on the **same frozenM4train/selection contexts**,
then evaluate on untouched2040M/2050Mfullgames. M4test outcomes must not select
this family or its hyperparameters; this protocol is written before their CI
summary/audit is executed. No originalM4model, result, source or seed is changed.

Three arms: (L)linearF→1canonical-logit residual, (R)MLPF→64→32→1canonical-logit
residual, (N)the sameMLP with the additive canonical offset disabled. All receive
the same namedM4observable features including the canonical logit/EV columns;
Ntests explicit offset insertion, not complete removal of baseline information.
All use BCE only, same40000train/12000selectioncontexts, training mean/SD/fixed
public-payoff normalization,50epochs/Adam/L2/clipping/batch128 asM4. Linearzero
initialization; bothMLPs share Xavierhidden/zerofinal initialization exactly.
Five paired fresh optimizer seeds2060M (collisionaudit before consumption), same
batchorder for all arms; no extra grid, restarts or pretraining.

Epoch10/25/50chosen by realized selection score, earlier ties. Ensembles average
five corrected probabilities; alpha[0,.25,.5,1]blend against canonical selected
by same selection score, loweralpha ties. OriginalM4source/features/rawhashes
must match before fit. Save each epoch/candidate/model/freeze. Reused selection
numbers are descriptive and optimistic; do not relabel this as a fresh-label
training replication or compare original2030Mfits as paired optimizer seeds.

Screen2040M/confirmation2050M,2000complete CRN **quadruplets** perstage:
L,R,N,canonical. SameModelBidderopponents/TA+v2play, hero g%4, base+g, rotating
executionorder, allnonNTcanonical. PrimaryR−Lhero total points/game, paired
equal-seat95%CIs; positive lowerbound in BOTH stages required to support this
fixed nonlinear residual family over its matched linear counterpart. Other
contrasts(R−N,each−canonical,ties-as-win)descriptive withn/95%CIs. No global
neural superiority or isolation beyond these exact family/initialization rules.

Resource/hard stop03:30ET17Nov2026 within07:00ET scientific cap; oneworker,
BLASthreads1. Fifteen50-epoch fits and4000quadruplets fixed; do not extendn,
changefeatures/alpha/epochs from screenresults or replace a failedoriginalM4
primary. If incomplete atstop, preservecompletequadruplets/checkpoints and
report scope. Separate raw/source/selection/CI/fresh-game replay checks before
self-validation; then fullfreshprocesspytestgate. Newproofsexternalgate stays.

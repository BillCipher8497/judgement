# Frozen root-learning comparison, before root-data inspection

Scope: first free spoiler action with fixed myopic bids/TA hero and PI
continuation. Four exact games: S2R4n2/S3R4n3 × trump0/NT. Learners never see
hidden hands, world IDs, PI costs at test time, or test labels during fitting.
Root observations comprise acting seat, sorted original own hand, full public
bid/card history, known game constants and the legal-action mask. Actions are
indices0..2 of that sorted hand; unused/illegal slots are masked.

Split all possible own hands by ascending SHA256 of the ASCII JSON tuple
[2000000000,S,R,n,hand], compact separators, shared across actor seats and trump
conditions. Cumulative cutpoints floor(.6H),floor(.8H),floor(.9H) give
train/validation/discovery/confirmation. All worlds/public histories sharing
an own hand stay together. No random per-world split. Models are selected on
validation only and frozen before either disjoint held-out hand set is read.

Input basis: own-card12binary;3sorted own-card slots ×13onehot including pad12;
4public bids ×5onehot;12public played-card slots ×13onehot including pad12;
seat4onehot;trump4onehot(NTslot3);n/3,S/3,R/4;legal3binary. Total241features.
Only training groups determine any normalization. No suit-symmetry label
assumption: the frozen teacher's tie-breaking may depend on card identities.

Architecture arms: linear241→3 and tanh MLP241→64→32→3. Xavier hidden weights,
final weights0.01normal, zero biases. Five paired learner seeds from2010M;
within condition/architecture, objectives share initialization and batch order.
Adam lr1e-3,betas(.9,.999),eps1e-8;L2=1e-4 on weights only;global gradient
norm clipping10;batch256groups,100epochs. Epochs10/25/50/100 are selected by
lowest exact validation hero cost; ties favor earlier epoch. No extra grid,
restarts, changed features or tuning after either held-out result.

Five objectives, chance-group count weights and public payoff scale c=10+b0:

1. Hard-first-argmin teacher categorical CE.
2. Uniform-over-argmin teacher categorical CE.
3. Independent per-action optimal-set membership BCE (no categorical label
   normalization), averaged across legal actions.
4. Q/c regression, half squared loss averaged across legal actions.
5. Direct expected Q/c under masked softmax policy.

Group targets are exact training-world averages. Weights count×c/13 are
normalized by their full-training mean, then minibatch averages use the
fixed normalization (not a varying weighted-ratio denominator). L2 convention
is .5λsum(W²). CE/BCE/direct heads choose highest legal logit; Q regression
chooses lowest predicted cost. Ties choose lowest index. All methods have the
same number of parameters, exposure, optimization budget and validation rule.

Primary: **set-BCE MLP minus hard-CE MLP avoidable expected hero cost** in the
large NT condition. Negative favors set-BCE. Require negative paired-seed95%
t-CIs on BOTH disjoint held-out hand sets to claim a learner benefit there.
Five independent optimization seeds are the uncertainty units (df4); exact
held-out deal counts/private-key counts are also reported. A given fitted
policy's value over the entire fixed held-out population is exact, not an IID
world-sampling estimate. Other methods/conditions, agreement accuracy,
representation diagnostics and linear comparisons are descriptive with their
seed-CIs; no selected-condition, global-ML or full-game team claim.

Always report information floor, learned avoidable regret and teacher-label
agreement separately. Later omniscient continuation makes this a relaxation,
not a feasible full private team or a comparison with optimal52-card play.
Save every epoch candidate/model/validation choice, per-seed exact integer
score totals and source/data fingerprints. Hard task stop03:30ET17Nov2026,
checkpoint each epoch/model. One worker; nulls/numerical/resource failures
reported rather than changing the fixed family/counts.

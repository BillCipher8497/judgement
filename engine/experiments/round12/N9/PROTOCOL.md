# N9 — matched structured-joint training, new local preregistration

19November2026, no N9fits/outcomes yet. New recipe on the KNOWN N2toy
benchmark, not fresh-hand replication. Original N2fits/sources/data/seeds
unchanged. One NumPy worker; no PyTorch, core or existing-policy edits.

## Fixed population and grid

Four original all-private-TA/myopic S2R4n2/S3R4n3trump0/NT reach datasets.
Keep the N2own-hand60/20/20split. Eligible keys have at least2unseen cards
(observable), including forced current actions. For each split, order keys
within each own hand by SHA256([2170000000,0x4e39,S,n,trump,split,keyhex]); round-robin
across hands in mask order, take2048train/512validation/1024test or all if less.
This is a deterministic hand-balanced KEY subset, not a probability sample.
Record actual hands/keys/reach weights, all key IDs and support hashes before
fits. Use complete exact R(I)and structurally feasible F(I)for EVERY selected
key. No resampling/approximate normalizers. Test keys are previously inspected
benchmark keys; models do not access their labels during fitting/selection.

Three configurations: summary-linear, summary-MLP, rich-MLP. Every input has
245columns. Summary uses unchanged N2's35observable per(card,holder)features
and zero padding. Rich appends original ownhand12bits, bids4x4onehot,
publiccards12x13onehot(pad12), actor4,trump4,n/Sscalars2, cardidentity12 and
absolute candidateholder4:210additional columns. No hidden hand, future score,
reach/target count, split or key ID in predictor inputs. Structural decoder
uses the same public capacities/void support in all configurations.

Linear245→1; MLP245→32ReLU→1sharedholderhead. Nominal identical MLPparameters
and same initialization for summary/rich; summary's padded weights receive no
data signal. Rich-input effects are an input intervention, NOT a certified
representation floor or pure parameter-capacity mechanism. Factorized scores
conditioned on F remain a restricted joint family even with full context.

Two matched objectives: raw per-card marginal CE versus exact conditional-joint
CE(logpartition over F minus exact posterior mean log factor). Both divide each
key's data loss by unseen-card count and use identical fixed reach/meanreach
weights, minibatches and L2. Logits=log(capacityprop)+network. No N2mixture/alpha
selection; this changes the old recipe but BOTH new arms share the change.
Strict positive finite-log support; stable logsumexp, no probability clipping
in the loss. Objective contrast is within this new matched recipe only.

Adam1e-3,betas.9/.999,epsilon1e-8,clip10,L2=1e-4,batch128keys. Five paired
seeds,80epochs per trajectory; save candidate5/10/20/40/80. Select short20from
5/10/20and long80from allfive by lowest chance-weighted validation JOINT KL,
earlier ties, common selection criterion for both objectives.120trajectories,
240selected fits. Short/long share the exact prefix; no reranking/extension
after test outcomes. Immutable epoch optimizer states, atomic pointer/receipts.

## Primary, controls and uncertainty

PRIMARY: largeNT summary-MLP long80 joint-minus-marginal held-subset joint KL,
paired n5optimizer seeds,t95%df4. Secondary: allsame objective contrasts,
long-minus-short and rich-minus-summary at fixed loss/budget; linear control.
Allcells/nulls/adverse results retained, no selected winning cell, no claim of
independent four-condition replications. Individual and paired n5seed95%CIs
measure optimization variability ONLY. Exact finite subset reach-weighted KL
has no chance-samplingCI; report actual test hands and per-hand seed-averaged
contrasts descriptively. OldN2/bidbaseline comparisons are secondary and do
not isolate training duration or diagnose a unique failure cause. No game-score,
fusion, equilibrium,52card or novelty claim.

Base2170M/domain0x4e39. Initialization[base,domain,condition,arch,seed,0], arch0
linear/1MLP(sharedsummary/rich); epoch order[base,domain,condition,arch,seed,1,e].
Objectives share roots; preflight synthetic root[base,domain,99]. Generate
and check against known N0–N5root/child states+journals before consuming.
Metadata grep before this source found no2170M–2220Mbase uses. Unknown external
uses/undocumented descendants cannot be certified absent.

Fresh absolute <=3hcut and checkpoint cadence recorded in STATUS before job.
Maximum two3h windows are permitted as an unchanged-count resource-only
continuation, each with a newly logged absolute cutoff; not outcome-based
extension. No new architectures/epochs/keys/seeds between windows.
Resource gate: first condition data build/one epoch timing, retain allprefix
data; if projected remaining target exceeds window, report scope failure or
resource-only unchanged-count continuation with a new logged cap, never alter
n/models/loss/selection based on outcomes. Separate finite-difference/invariance
checks BEFOREfits; after allfitfreezes a new forward/directworld normalizer/
selection/split/raw/CI audit. All new scope stays local pending Claude external.

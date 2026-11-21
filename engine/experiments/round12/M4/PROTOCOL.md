# Frozen neural NT residual experiment — before fresh labels

Question: can a baseline-preserving nonlinear probability correction improve
ModelBidder on actual52-card ladder NT bidding? This is separate from the root
teacher benchmark and asserts no transfer theorem. The previous R4 candidates
lost; all old sources/data remain frozen and are not tuning/validation data here.

Game: standard25-round ladder, NT sizes9,4,3,8,13. Canonical model is the fixed
phase1/results/model_ta_final.json; hero and all opponents use TA+v2play, canonical
ModelBidder bidding except the specified NT intervention. Fullgame hero g%4,
same base+g for all three arms, rotating execution order, no nonNT change.

Fresh resource-only pilot:200contexts1940M. Train40000contexts1900M and
selection12000contexts1910M. In each stage i, cell=i%20, g=i//20,
n=[9,4,3,8,13][cell//4], hero bidding position=cell%4, dealer=(-pos-1)%4,
SeedSequence([base,n,pos,g]). Hero seat0 for context labels. Deal once; generate
earlier canonical bids; for each legal hero bid clone state and RNG tapes, finish
round with canonical remaining bids/TA+v2play. Save all legal-bid binary hit/score
outcomes, all hands for replay, named observable context, seeds and fingerprints
incrementally. No full hands/outcomes supplied to predictions.

Resource gate, no performance inspection: pilot<=120s and projected52000context
time plus7200s reserved fitting/evaluation <=remaining time before06:30ET.
If rejected, scope failure; do not shrink counts from outcomes. Fit/evaluate hard
stop06:30ET17Nov2026, within07:00ET session stop. One worker, BLASthreads1;
checkpoint each context/epoch/frozen model/fullgame triplet.

Observable features reuse the frozen R4 variantB design (suit-canonical hand
shape/honors, public bid position/prior/hook and bid interactions), dropping its
constant column. Append canonical logit(hit probability) and expected score/23
for that legal bid. Canonical probability is P(tricks=b | same hand, own bid=b),
from the original conditional model, clipped[1e-6,1-1e-6] only for the logit.
Raw canonical probability is retained for zero blend. Standardize with training
context weights1/(number legal bids), training mean/SD only;SD<1e-10→1.
Network F→64→32→1 tanh; Xavier hidden weights, final weights/bias zero, so at
initialization corrected probabilities exactly match the clipped canonical law.

Five paired seeds2030M (reserve/collision-audit before consumption), each arm
shares initialization and context minibatch order. Adam1e-3,beta(.9,.999),eps1e-8,
weightL2=.5*1e-4sumW²,gradnormclip10;batch128contexts,50epochs. Fixed epoch
candidates10,25,50; choose each seed's epoch by mean realized selection round
score under greedy expected-score decisions, ties earlier. No extra grids.

Two objectives, same parameters/exposure:

* **BCE primary**: corrected Bernoulli hit likelihood, average each context
  across legal bids with public weight(10+b)/23. A context minibatch mean and
  fixed training normalization, not variable random denominators.
* **Direct secondary**: negative realized score/23 under softmax over corrected
  predicted expected scores (temperature1hero point). Derivative through the
  canonical-offset sigmoid. This is a fixed differentiable decision surrogate,
  not a general RL or posterior-consistency claim.

Ensemble is arithmetic mean of five corrected hit probabilities. Select blend
alpha in[0,.25,.5,1] against raw canonical probability by selection realized
round score, tie smalleralpha;alpha0exactly recovers canonical NT decisions.
Freeze both ensembles/epochs/alpha and all sources before any fullgame screen.
Selection numbers descriptive, not evidence of held-out gain.

Screen1920M and confirmation1930M, **2000complete triplets each**:
BCEensemble,Directensemble,canonical. Primary paired BCE−canonical hero total
points/game; secondary ties-as-win and Direct−canonical. Report all contrasts
with n and seat-stratified95%CIs; normal paired mean +/-1.96*SE with equal seat
weights, per-seat sample variances. Require positive score lowerCI in BOTH stages
to claim primary gain. No n extension, changedseeds, tuning or selected-size claim.
Ifalpha0gives identity, report exact paired0 with n2000,CI[0,0], not improvement
or noninferiority evidence. Opponent bids can respond after hero; this estimates
the whole deployable bidder intervention, not an isolated play/lead mechanism.

Save scores/tricks/bids per round and totals. Assert hook/score arithmetic,
nonNT identity, earlier opponent bids agree, hero rotation and complete arms.
Separate replay/CI/leakage/source audit before self-validation. New result and
artifact stay local pending the project publication review/reproduction gate.

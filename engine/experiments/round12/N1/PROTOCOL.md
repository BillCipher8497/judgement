# N1 source preflight — learned four-holder beliefs

Prepared after reading Claude's N0/RESULTS.md and updated ML addendum. N0 is
read-only. This is pilot-informed design, with untouched N1 confirmation.
No claim that a residual architecture guarantees out-of-sample improvement.

Primary: fixed five-fit full-history residual ensemble minus BI mean
unconditional log-loss(nats) per decision, negative better; paired equal-stratum
normal95%CI across3520independent recorded rounds, not cards or optimizer seeds.
Require upperCI<0. Secondary: bids+structural-legality ensemble, net-prop,
conditional-on-opponent-held-card losses, reliability/ECE,44n/trump/table cells.
All44cells use n3..13, S/NT, TA/TA+; four observer seats balanced, hero=g%4.
One observation per independent round, after a seeded trick in1..n-2, at hero's
next turn. Dealer=rep%4. Both tables use canonical ModelBidder, play differs.

Exact fixed counts: train30800rounds, validation4400, confirmation3520,
resource-only pilot176. Cell=(g//4)%44, rep=g//176.44cells×4seats; train175,
validation25,confirmation20replicates per joint cell. No extension on a null.
Train/validation/pilot use base2100M with explicit N1namespace0x4E31 and distinct
stagecodes1/2/0; confirmation2110M stage3. Data entropy addresses
[base,0x4E31,stage,g,3], observation selector[...,4], sampler[...,5]. Nonzero
namespace/stage-tail separates N0's two-integer entropy addresses; audit source
registrations and generated states before consumption. Optimization addresses
[2100M,0x4E31,20,seed_index]. Calibration bootstrap uses2110M namespace/stage90.

All52 minus own/current publicly played cards are scored; never filter by true
opponent holdings. Classes: three relative opponents plus undealt stock.
Stockcapacity52-4n; opponents' remaining capacities/public voids determine mask.
Closed form prop is masked capacity normalized across four holders. Features
receive only a strict named observable dictionary. Hidden current hands produce
Y only; full initial hands/observations are retained separately for replay.

The shared per(card,holder)35features are documented in belief.py. Bids-only
zeros behavioral-history fields but retains structural capacities/void support
and public card removal. Full adds tricks won, need, seen same-suit cards/leads.
Known controlled table ID is available to both nets; no unknown human type claim.
Each is35→64ReLU→1 per holder, logits=log(prop)+residual, finalzero at start.
Five paired initialization/batch-order seeds; Adam1e-3,betas(.9,.999),eps1e-8,
L2=.5*1e-4sumW²,batch32decisions,gradclip10,50epochs. Targets weight each
round equally, then each unseen card equally within that round. Floored proper
loss uses .999softmax+.001prop; analytic gradient must pass finite differences.
Epoch10/25/50selected by exact validation mean decision log-loss, earlier ties.
Five-fit probability mean, alpha[0,.25,.5,1]blend with prop selected on validation,
smaller alpha ties. Source/model freeze before confirmation.

Selection resource/non-inferiority gate: full ensemble alpha>0 and paired
validation95%CI upper bound for net-prop<=0 before confirmation is generated.
This is a selection filter, not independent population non-inferiority evidence.
Failure documents unsupported scope/identity and leaves2110Mconfirmation unused.

Uniform and BI marginals each useK256; BI batch64,m_max2048,ess_target1,
eps=.02,tau0. Both get identical addressed sampler RNG starts. Stock assignment
is complement of sampled opponent masks. All reported probabilities mix .001
prop, including nets, so finite-MC zeros do not make incompatible scoring laws.
Record ESS/M/runtime. Conditional diagnostic renormalizes every method over
opponents identically, and filters active cards only in scoring, never inputs.

Reliability:10fixed equal-width top-confidence bins, round-equal card weights.
ECE95%CIs use2000paired stratified round bootstrap draws; marginal intervals,
not simultaneous cell guarantees. Log-loss comparisons use176balanced strata.
No exact posterior or universal robustness claim. Calibration is descriptive.

Resource gate: pilot176with bothK256baselines; projected training/validation
generation plus confirmation plus7200sfit/audit reserve must fit remaining
window. Count-only scope failure, no outcome-based shrinking or seed changes.
Fresh proposed compute hardstop2026-11-18T03:00Z(22:00ET17Nov); must be
re-recorded before actual launch if source review takes longer. Chunk64round
atomic saves and per-epoch optimizer checkpoint. One worker, BLASthreads1.

Pre-run gate: Claude externally validates this source preflight, then code is
committed locally before any experimental case. The runner verifies the commit
contains identical pipeline/model/protocol bytes and records its SHA. No code
or label run until that gate. Full task remains running until results are
ready-for-review; only Claude external may validate completed N1. Numpy only.

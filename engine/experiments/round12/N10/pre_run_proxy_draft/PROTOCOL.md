# N10 — activated late-play students with private rollout continuation

Pre-run local protocol19November2026, no N10data/seeds/outcomes consumed.
Runs AFTER N9completion/separate checks. New independent recipe; original
N3frozen sources/data/models/2120M–2140Mgames unchanged. NumPyonly, one worker,
new absolute hardstop each long stage, all results local pending Claude.

## Teacher and conditional context population

Original52card ladder/rules, hero g%4, canonical ModelBidder for all seats,
TAopponentplay and TA+v2hero baseline. Act only with exactly2own cards, both
legal, need in{0,1,2}; otherwise existing baseline.96strata:4hero seats times
24ladder rounds excludingn1, stratum g%96. Within each stratum independently
deal/roll baseline to first eligible hero decision, bounded1000attempts/row.
Condition on original-hand split:SHA256([2180000000,0x4e40,'hand',handmask]),
first8bytes modulo10:0–5train,6–7validation,8–9held. Pilot uses TRAIN partition
but never enters fitting. Fixtures unrestricted and receive NO labels.
This upweights conditional active contexts, not natural fullgame frequency.

Pilot96,train6144,validation1536,held1536; independent label-free activation
fixtures768(8/stratum). Every row records source/seed/attempt/original hand,
strict own/public observation, legalcards, baselinechoice, featurevector.

For each labeled context K16COMMON structural-uniform HandSampler worlds,
not an exact/bid/play-conditioned posterior. Two current actions compared on
the SAME worlds. After the forced hero card, EVERY future player receives a
detached own/public snapshot only; hero uses fixed TA+v2, others fixed TA.
No PI/MaxN, hidden-state argument or world ID enters a future policy. This is
a one-step policy-improvement rollout teacher, NOT an exact privateBR or
true-game value. All world maps and both terminal score arrays saved per row.
Scores0orpublic10+hero bid; hard labels choose first per-world maximum,
membership labels retain allper-world optimal actions before averaging.

This changes teacher, fit recipe and activation safeguard relative to N3;
no isolated causal attribution to private continuation or offset alone.
Pending-audit M17motivates this control but is not taken as externally proved.
The two-card predicate allows at most ONE intervention perround; after it the
hero's last card is forced. Thus hero-future strategy fusion is structurally
absent here, and this task does NOT test a fusion repair. The new teacher holds
opponents' private policies fixed instead of substituting MaxNopponents.
Within-round future hero continuation matches the deployed student's inactive
last-card behavior; belief/Kerror and finite model/summary limits still remain.

## Predictors/selection and prespecified gates

Use unchanged N3observable683encoder/private snapshot definitions through a
new source-hashed adapter; no hidden arrays/labels in prediction. Two arms:
hardCE and membershipBCE, five paired MLP683→64→32→2tanh fits per arm.
Training logit=.25baseline-indicator+f; both include baselinebits. Finalhead
zero. Adam1e-3,L2=1e-4,clip10,batch128,50epochs. Public hit-score weights
normalized by fixed training mean. Init/shuffle roots paired across losses.
Checkpoints5/10/25/50, alpha0/.25/.5/1/2/4; choose highest mean validation
finiteKteacher score, then earlier epoch, then smaller alpha. Ensemble means
.25baseline+alpha*f overfive selected fits. Allchoices freeze before fixture,
held or fullgame evaluation. No picking a replacement arm/alpha after gates.

Both arms must pass BEFORE ANY scientific fullgame blocks:
(A) label-free768fixture counterexamples: >=16changed actions(>=2% nominal),
atleast8distinct original hands, everyhero seat represented among changes,
each changed argmax margin>=1e-6. Save ALL scores/actions and examples; this
certifies nonidentity on those fixtures, not a natural gameplay activation rate.
If a rate is reported, give n and conservative95%Hoeffding bound for the
independent bounded context draws, not an equivalence/performance interval.
(B) ensemble validation rollout-score minus baseline lower paired96stratum
normal95%CI >=−0.05points/active context. This reuses selection validation,
optimistic conditional gate, NOT an independent noninferiority guarantee.
Held labels never alter gates/models. Report all held finiteKregrets/differences
with n1536,96stratum95%CIs conditional on the fixed ensemble.
If the existing bounded-tanh coefficient argument proves a selected ensemble
globally baseline-identical, record its exact rational margin bound and zero
validation contrast as an IDENTITY(ci95notapplicable), not a samplingCI. Its
activationgate necessarily fails. New bound numbers remain pending external
check; this is not a new theorem or an equivalence test.

If either arm fails AorB: complete the task as a documented activation/selection
scope failure, report held/gate results and all nulls; run ZERO scientific games.
Do not turn zero unrun games into a zero effect/CI, weaken thresholds, increase
alpha/epochs/n or retune. Such followups need a new task/protocol/seeds.

## Fresh namespaces and fullgame target (only if gates pass)

Data2180M/domain0x4e40, stages0pilot/1train/2validation/3held. Prefix
[base,domain,stage,g,attempt,3] engine deal/policy roots; world RNG same ending5.
Matched private-rollout roots[base,domain,stage,g,attempt,w,6],same across
candidateactions, per-seat tapes independent of sample-world draw seed.
Fixtures2220M/domain0x4e40/stage4,g,attempt,3; no world/label draws.
Fit2190M/domain0x4e40/20,seed; shuffle[2190M,domain,21,seed,epoch].
Synthetic preflight2190M/domain0x4e40/99; any rollout fixture descendants
are individually registered before use and do not enter training/inference.
Register allroots+5children against known previous addresses BEFOREdraws,
including N9roots. No2170Mreuse; metadata grep prior to reservation found
no2180M–2220Muses. Unknown external roots/descendants remain a stated limit.

PRIMARY set-ensemble minus hard-ensemble hero score/game, separately on
2200Mscreen and2210Mconfirmation, exactly1000complete three-armCRNblocks
each (baseline/hard/set), hero g%4, seed=base+g. Secondary set/hard minusbaseline
and ties-as-win recomputed from totals; paired equal-seat normal95%CIs,
n1000/stage,250perseat. No post-screen choice/extra games/significance chasing.
Save each complete arm and block incrementally, rotate execution order.
Null/negative completion is a pass. No weak-pool human/general52card/neural
architecture/fusion/equilibrium guarantee from this intervention.

## Resources and separate review

At most two3hwindows for unchanged scientific targets, each with newly logged
absolute UTCcut, per64data rows/perfit-epoch/pergamearm checkpoints. Pilot96
timing plus conservative old-game timing forecast and fixed900sfit reserve
must fit total6h; if not, document resource scope failure. Allpilot/raw/errors
retained. Future windows are count-only/source-matching, never outcome-tuned.
Before fits: synthetic gradient, detached-feature, rollout-private-capability,
seed-collision and zero/active fixture checks. After allfreezes: independent
prefix/rollout transition replay, forward/selection/gate/held-CI audit; ifgames
run, allrawseed/seat/totals/CIs plus fixed balanced game replays. Literal full
pytest once at closure; scientific standalone receipts separate. No core,
existing policy/N0/frozen science edit, commit or push awaiting Claude review.

# N3 — bounded late-play privileged-teacher distillation

Drafted 2026-11-18T02:34Z before N3 labels, pilot draws, or game outcomes.
Implementation and source review must be complete and source-committed before
any reserved stream is consumed. New results remain local pending external audit.

## Question and tractable scope

Does optimal-set supervision outperform arbitrary first-argmax teacher labels
when distilled into a private-observation card player? The intervention acts
only when the hero has exactly two cards, both legal, and remaining need is
0, 1 or 2. At all other decisions it plays existing TA+v2. All other seats play
TA; every seat bids with the frozen canonical ModelBidder. The original
52-card, 25-round ladder is used in evaluation, including forced/irrelevant
decisions and rounds. This is a late-play intervention, not an all-stage neural
player, a hidden-information optimum, or a full-game information-cost bound.

At the active hero decision every opponent has at most two cards. Use K=16
shared structural-uniform HandSampler worlds and exact neutral MaxN move
values on each reconstructed world. This teacher ignores bid/play likelihood,
assumes future omniscience and PI opponents, and differs from the actual TA
continuation. Its empirical mean is a finite-K surrogate, not a posterior or
actual game-value certificate. No oracle filtering with actual hidden hands.

## Data and seeds

2120M, namespace0x4E33. Data stages0pilot,1train,2validation,3held-label-test;
address [2120000000,0x4E33,stage,g,attempt,3] uses engine spawn deal/policy
streams; the paired world RNG uses the same address ending5. Check every
new address and its five spawned roots against all prior experiment roots
before drawing. Attempts are recorded; at most1000attempts per accepted row.

Pilot192 contexts (two per stratum), train9600, validation2400, held2400.
Time16nine-arm full-game blocks on2120M/domain0x4E33/90/g using zero-head
fixture ensembles (all five registered initializations) before fitting. These
are resource-only pilot games, excluded from every effect estimate. The
resource forecast uses pilot data time per context for all14400nonpilot
contexts and pilot game time per block for4000blocks, multiplied by a fixed
1.5timing/overhead margin, plus a fixed900second
fitting/selection reserve. A forecast fitting six hours is a scheduling gate,
not a guarantee; any actual timeout preserves all checkpoints and is reported.
Each set balances96strata: hero=g%4, round index from the24ladder rounds
excluding the one-card round, indexed by(g//4)%24. Dealer/trump/n use that
actual scheduled round. Each independent attempt plays baseline policies to
the hero's first active decision in the round. Activity is measurable from
own/public information. Reject rounds without such a decision. This is a
conditional active-context population; it is not the natural round frequency.
There is no inverse acceptance-probability weighting across strata: rare active
round types are upweighted by this conditional design. Held context regret
uses that same declared mixture; full-game utility uses unfiltered deals.

Split original own hand by SHA256(b'N3-hand-v1:'+decimal-mask), first8bytes
big-endian modulo10:0..5train,6..7validation,8..9held. Pilot uses the train
partition and is timing-only, excluded from fitting/selection/inference.
All histories/actors sharing an original hand use the same partition. The
actual independent samples remain independently generated conditional draws;
latent opponent cards may recur across splits. Record distinct hand counts.

## Models and selection

Two sorted legal-card outputs. Observable683-vector: ownhand52,
public played-by-seat208, bids4x14, tricks-won4x14, current-trick3x53
(empty sentinel), voids16, cards-left4/13, actor4, dealer4, trump5,
originaln13, two card identities104, baseline-choice2. No hidden arrays,
teacher scores, actual future outcomes or truth-dependent active mask enter.
This summary is not claimed injective in perfect-recall history.

Five objectives: hard CE (each world's first maximum, averaged), uniform CE
(each world's uniform optimal set, averaged), optimal-set membership BCE,
mean-cost regression and direct expected cost. All use the same worlds,
features, fit seeds and batch orders. Cost=(max-world score minus move score)
averaged and divided by the public hit score10+bid; this differs from1-Q
by an action-constant and has the same decision optimum. Weight each row by
10+bid. Hard/uniform/membership targets are computed per world before averaging.

MLP683->64->32->2,tanh; linear683->2 hard and set controls; an MLP set arm
without the explicit offset, eight arms total. Offset arms score=baseline
logits(2for chosen,0otherwise)+alpha*residual; no-offset arm score=
(1-alpha)*baseline+alpha*network. Output heads initialize zero; input includes
baseline indicators in every arm. Thus this is not a baseline-information
ablation. Regression minimizes squared loss on negative scores as costs;
all deployed decoders maximize scores, ties choose lower card index.

Five paired fit seeds[2120M,0x4E33,20,seed], fixed50epochs, Adam1e-3,
L2=1e-4, clip10, batch128. Batch RNG[2120M,0x4E33,21,seed,epoch].
Evaluate checkpoints5,10,25,50 and alpha0,.25,.5,1 on the same validation
contexts; choose highest mean finite-K teacher score, then earlier epoch,
then smaller alpha. Freeze each fit; deploy the mean of five score vectors.
Select best student among eight ensembles on validation surrogate, ties by
listed order (hard,uniform,set,q,direct,linear-hard,linear-set,no-offset-set).
No screen/confirmation or held-label choice. Validation reuse and bounded
epoch/blend searches are selection limitations; no isolated nonlinearity claim.

## Primary contrasts, uncertainty and fixed game counts

Before launching: **MLP set minus MLP hard score per full game**, and
**validation-selected best ensemble minus TA+v2 score per full game**.
Screen2130M and independent confirmation2140M, exactly2000complete matched
nine-arm blocks each; hero=g%4, seed=base+g. Same engine deal stream across
all arms; save totals and round summaries after each complete block.
Full-game deals are unfiltered and include all SHA hand partitions: these are
fresh-seed utility tests, not a second unseen-original-hand generalization
benchmark. The held label-context split alone enforces that hand exclusion.
A partial block is retained separately and completed unchanged on resume. Stop at fixed
counts regardless of significance. No seed swap, retuning or sample extension.
Report both primary contrasts and all secondary pairwise baseline/control
contrasts, paired seat-stratified95%normal CIs, n2000/stage. Ties count as wins
from totals, not GameResult.wins. Replication means same direction in both
fixed stages; multiplicity and secondary exploratory comparisons disclosed.
Secondary controls include uniform-minus-hard, set-minus-uniform, matched
MLP-minus-linear contrasts, and explicit-offset-minus-no-offset. Predeclare
an exploratory label-gain interaction (MLPset-MLPhard)-(linearset-linearhard),
which compares selected pipelines rather than isolating nonlinearity.
For each contrast also retain paired table-total and mean-opponent score CIs;
hero improvement need not improve table welfare in this general-sum game.

Held label diagnostic: empirical label-only decode cost against the argmax of
the same K-world mean move score, hard/uniform/set, plus deployed student
surrogate regret. Paired96-stratum round CIs,n2400; do not treat16worlds as
independent decisions. Training-seed uncertainty is separate from these
fixed-ensemble context CIs. For two legal actions and world payoffs0or10+bid,
membership differences equal normalized mean-score differences, and the
uniform-optimal decoder also has the same ordering; audit this
algebra on saved labels, not a new general-game claim. It does not certify
actual TA continuation or true hidden value. Log nulls, ties and all arms.

## Resource and reproducibility gates

One scientific worker. Before every long launch record a fresh absolute UTC
hard stop, max3h, checkpoints per64data rows, epoch, and complete game block.
At most two3h windows are allocated to this fixed experiment before outcome;
unfinished counts are a scope failure, never a partial-population headline.
Pilot forecasts remaining data+training+game time; if it cannot fit the
allocated resource budget, preserve the pilot and report scope failure.
Source commit/hash, model JSON, dependencies and reserved addresses frozen.
Separate manual prefix/label replay, numerical forward/gradient/selection,
raw paired CIs, seat rotation and all scientific hashes must pass before
ready-for-review. Full regression suite remains a separate gate; N0/core/
existing policies and old freezes remain unchanged. No unreviewed push.

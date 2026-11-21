# N2 protocol proposal — exact play-history posteriors

Draft for Claude (head) before implementation/compute. N1 remains the only
compute worker. No N2 trajectories, labels, fits or outcomes exist.

## Conditioning and fixed policies

Use the four M1 game definitions:4players, S2R4n2/S3R4n3, trump0/NT, uniform
chance over all2520/369600ordered deals. Keep M1datasets/trees untouched.
Bids are the existing deterministic myopic bidder with the dealer hook;
**all four players use the existing private/public TA toy policy**. This is
a declared fixed private profile, not M13's omniscient minimizing team.
Record every pre-play actor infoset, including forced moves, along each full
trajectory; do not prune hero-impossible branches. Infoset I=(acting seat,
original own hand, complete public bids/cards). No first-free-stage flag,
pooled team hand, root-only weighting or hidden prior-action information.

For every I, enumerate every chance deal whose fixed-policy trajectory reaches
I. Fixed deterministic policies and uniform chance make the exact posterior
uniform over this complete reach set. Save explicit deal-index support and
integer count; check no deal contributes twice to one I. Grouping is across
all deals, never the M1root-selected subset. Exact per-I marginals and entropy
come from these same supports. Include all actors symmetrically; this is an
exact observational posterior audit, not a hero/challenger game comparison.
Later full-game interventions retain hero=g%4.

Proposed exporter: checkpoint every1024deals in compressed NPZ with explicit
fixed-width public/private key, deal index and holder bits. A deterministic
merge groups keys, saves CSR support/deal indices and integer holder counts.
All raw chunks remain. Full histories/keys specify action order via the public
trick winner. Independently replay selected supports and exhaustive small
populations with a fresh rules path; compare reach membership by re-running
policy prefixes, without trusting M1root labels. Large full-policy traversal
may fail the resource gate; a partial exact checkpoint is not a complete
posterior result.

## Joint laws (not card marginals mislabeled as densities)

Let F(I) be all hidden initial deals consistent with own original hand, observed
card ownership, remaining capacities, and public follow-suit/void constraints,
without requiring fixed-policy choices or bids to have likelihood1. Enumerate
this legal support (at most1680initial allocations at n3); no invalid-world
mass is silently dropped differently across methods.

* Uniform: q_U(w|I)=1/|F(I)|.
* Prop-induced joint: product of masked remaining-capacity card marginals,
  **renormalized over F(I)**. On these all-dealt toys, every feasible allocation
  has the same holder capacities; its product is constant. Thus this defined
  prop joint equals uniform exactly. Report the identity/null openly. It is
  not a statement that marginal calibration implies a correct joint.
* BI-style: q_B proportional to product over opponents of the noisy myopic-bid
  likelihood, evaluated on reconstructed original hands. For public hook-legal
  bid b, L=(1-.02)*1[b==myopic_argmax]+.02/number_legal_bids. Own bid is
  conditioned on own hand and contributes a common factor. No play-policy
  likelihood is included. Strictly positive L ensures support on F(I).
* Net-induced joint: q_N proportional to product of strictly positive learned
  per-card holder probabilities, renormalized over the **same F(I)**. The net
  uses bids/history once; do not multiply another bid likelihood. This is a
  factorized learned joint model under constraints, not an exact posterior.

Evaluate KL(P_exact||q) over complete supports. Uniform KL is
log(|F(I)|/|reach(I)|). Product models use an exact subset-allocation dynamic
program for the normalizer; exact posterior integer card counts give expected
log joint factors. This avoids a separate approximate world sampler. BI's
factor is per original opponent hand, not a per-card marginal. Store exact
integer normalizer counts where available, float/log arithmetic precision and
cross-check direct enumeration on all small and fixed large contexts. KL is
an exactly enumerated population computation up to floating arithmetic, not
a rational bound or Monte Carlo estimate; no chance-sampling CI.

The prop/uniform identity and finite deterministic-reach posterior reasoning
are elementary but remain **pending external audit**, not a new theorem or
policy-optimality certificate.

## Proposed finite net and evaluation

New N2encoder only: same observable35feature vocabulary as N1, adapted to the
known toy deck/rank range and zero-stock mask. No modification of N1's frozen
encoder, sources or data. Only own/public features; full deal is a label.
Residual logits on prop,35->32ReLU->1 per candidate holder, finalzero. Proper
mixture .999softmax+.001prop. Five paired fits per condition,20epochs,
Adam1e-3/clipping10/L2 1e-4, batch128private keys. Train with exact reach-count
weight, then equal unseen-card weight per recorded decision; terminal/no-unseen
cases contribute no marginal target and are documented rather than filtered
with hidden truth. Epoch5/10/20 chosen on held-out marginal logloss; no KL-based
retuning. Alpha0/.25/.5/1 selected on same validation loss, smaller ties.
A declared validation net-versus-prop noninferiority filter is diagnostic only;
all exact held-out joint KLs are reported even if the learned arm loses.

Own initial hands hashed into60/20/20train/validation/test, shared across seats
and S/NT within a deck. Every reached history for an own hand stays together.
Net joint results on the complete reached test-hand population are primary;
all-key results are explicitly descriptive/in-sample mixed. Every method is
compared on the same test keys with identical reach-count weights. Primary
contrast: net-induced-joint KL minus BI-style KL, each condition separately,
negative better; no selection of a winning toy. Five-fit paired t95%CIs describe
optimization variation only. Fixed-fit KL/population calculations have no
chance CI. Report actual worlds/keys/hands, card-marginal logloss separately,
and all nulls/negative results. No claim that these curves measure52-card play.

Proposed namespace2100M,domain0x4E32 (N2), optimizer addresses
[2100000000,0x4E32,20,seed], batch[...,21,seed,epoch]; hand split uses the same
explicit N2namespace in a SHA256 key, not N1draws. Collision-check all roots
against known N0/N1streams before consumption; please confirm this reservation
since the board lists no separate N2seed base. Never consume2120M–2160M.

## Resource and source gate

No N2compute until N1 releases the one worker, Claude approves the concrete
source/protocol and a matching pre-run local commit exists. Record actual UTC
primary predeclaration and a fresh absolute hard stop at launch. Proposed
maximum3elapsedhours and absolute latest2026-11-18T06:00Z; use the earlier
bound. A fixed initial256deal timing pilot projects the full enumeration and
fixed fits; fail transparently rather than change n/profile/seeds after seeing
outcomes. Count-only checkpoints may resume unchanged under a new recorded
bound. The optional larger toy belongs N5 and is not silently added here.

Questions for head's protocol review: all-TA fixed private profile; prop joint
identity; proposed namespaced2100Mseed reservation; scope of fixed-profile
posteriors for later N6 unilateral decision probe. No operational approval or
new mathematical validation is asserted by this proposal.

## Conditioning details for implementation review

Zero-unseen-card infosets remain in the posterior artifact and joint-KL
population: all opponent hands are then publicly determined, F/reach are
singletons and joint KL is exactly zero. They supply no marginal labels.
Define their per-decision marginal loss as zero, retaining their reach weight
in the full recorded-decision objective rather than conditioning on a hidden
active-card event. Training/selection normalizers must state this convention.

Raw capacity marginals generally differ from marginals of the capacity-
conditioned product joint (which is uniform on F). Likewise raw neural outputs
can differ from marginals of the normalized net joint. Joint KL is the primary
N2measure; any raw marginal-card logloss must be labeled as such and not
presented as the same distribution after structural conditioning. The prop-
joint/uniform identity does not imply identity of their raw marginal reports.

N6 later needs a fixed objective and deviation model: a unilateral private
best response with other players' policies fixed is distinct from joint team
optimization. PIMC with an exact posterior may still assume privileged future
actions. A future exact private continuation must preserve actor infosets and
update support under the deviating action and subsequent public observations.
N2 alone establishes neither that continuation nor a strategy-fusion bound.

The proposed N2net's35features are history summaries, as in N1, not an injective
encoding of the full private/public key. Its excess must not be called pure
belief-estimation error with zero representation loss. Representation,
approximation and fitting can all contribute. The exact posterior, structural
joint support and fixed trained-network joint KL are nevertheless separately
well-defined. N5 would need its own explicit encoder audit for a certified
representation curve.

## Proposed finite-fitting resource cap (pre-outcome)

The exact posterior artifact and joint-KL evaluation still cover every recorded
play infoset, with complete test-hand populations. To keep the finite learner
within the declared3hbudget, propose a fixed cap of30000training keys and
4000validation keys per condition, chosen by lowest SHA256 priorities in the
N2namespace after the own-hand split. Use all keys if fewer exist. This is
key subsampling for fitting/selection only, never a Monte Carlo approximation
to a posterior or a subset of test outcomes. Train/selection losses keep exact
reach-count weighting over those selected keys. Report available/selected
keys, reached hands and effective reach weights explicitly; no claim of a
full-training-population optimum. Please accept or change this cap before
source freeze or any N2trajectory generation. No N2data exists.

## Bid-model scope for the head's review

The proposed data profile uses exactly the same deterministic myopic bidder
as the BI-style likelihood's argmax. Therefore it has no out-of-library bidder-
type mismatch. BIstill softens that deterministic law with epsilon.02 and
omits the fixed TAplay-action likelihood. Its KL is not a pure measure of
only missing play information. Please approve a further **exact** diagnostic:
condition uniform F on every opponent's deterministic myopic bid matching the
observed bid (hard bid-consistent subset). Its KL is log(|F_bid|/|reach|).
Then compare the declared soft-BI KL to that hard-bid law to identify the
softening/modeling choice explicitly, and the hard-bid law to full reached
posterior to identify omitted fixed-play-history consistency. No sampled
worlds or extra seed stage is needed. This is a separate defined comparison,
not an additive five-way information-cost theorem. All claims remain local
pending external audit; no N2data exists.

## Generalization unit

The split isolates original own hands and private/public keys. A complete chance
deal contributes observations for multiple actors and can therefore contribute
a training observation for one actor and a held-hand observation for another.
This is **not a disjoint latent-deal split**. The exact posterior target always
uses the full declared chance prior and fixed profile; do not redefine the
posterior by conditioning on a private train/test split marker. Report this as
hand/key generalization on one completely enumerated game population, with
seed-only fitting intervals. No independent-deal or arbitrary-future-hand
uncertainty claim. No hidden deal or another actor's private hand enters the
learner input.

Blend selection is **per fitted seed** after selecting that fit's epoch; no
five-fit ensemble is a primaryN2arm. The five exact held-population model
values give optimization-seed intervals relative to each fixed baseline.
Training's grouped soft-target loss uses reach_count/mean_training_reach and
1/unseen_card_count with uniform minibatches of keys; zero-card keys retain
the objective denominator and have zero data gradient. Fixed L2 is separate.

## Exact baseline arithmetic

For epsilon1/50 and public legal-bid counts k_j, each opponent hand's likelihood
has a common denominator50*k_j and integer numerator49*k_j*1[bid match]+1.
The denominator is constant across F(I) and cancels. A nonnegative-integer
subset DP computes Z_integer and W_max=product_j(49*k_j+1). The soft-BI KL is
log(Z_integer/(reach_count*W_max)); persist both integers per key. Hard-bid
and uniform KL similarly retain exact counts. Check the ordering through
integer inequalities before floating logs. No claim that neural-network KL
is rational or that these belief comparisons bound a game score. This
algebra, like the proposed posterior/prop identities, remains pending external
review; no N2 data or numerical result exists yet.

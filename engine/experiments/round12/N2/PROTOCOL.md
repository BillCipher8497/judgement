# N2 — Exact play-history posteriors: final pre-run protocol

No N2 experiment data exist at finalization. The draft was reviewed by Claude
on2026-11-17: all-TA profile,2100M/domain0x4E32,30k/4kkeycaps, hard-bid
baseline and the M16conditional identities approved. C1–C4implementation
checks are required below. The owner subsequently authorized independent
Codex continuation. Source-only review/commits may be self-reviewed(Codex),
pending external audit, with no forged Claude token or new external verdict.
N1source/data remain unchanged. No PyTorch, core or existing-policy edits.

## Target population and observations

Four exact conditions: N4,S2,R4,n2,trump0/NT and N4,S3,R4,n3,trump0/NT.
All2520/369600ordered deals per condition, uniform chance; all cards dealt.
Fixed dealer3, seat0bids/leads first, hook, exact hit10+bid/miss0. All four
players use existing deterministic myopic bidding and private/public toyTAplay;
source-preserving bounded memoization only. This is not a team optimum.

Record every pre-play actor infoset, including forced decisions and
hero-impossible branches, along every full fixed-profile trajectory. I is
acting seat, sorted original own hand and full public bids/cards, with fixed
rules known. No hidden first-free-stage marker, pooled team hand or root-only
M1weight. One chance world reaches a particular full-history key at most once.
The exact posterior is uniform on its complete reached-deal support R(I).
Explicit CSR deal indices and integer card-holder counts are saved.
Zero-unseen-card keys remain and have exact zero joint KL; their marginal
loss is zero while their chance weight stays in the denominator.

Each record has a20byte key (seat,history length,uint16ownmask,history bytes,
zero padding), lexicographic deal index, holder bits and chosen card.
The first256deals/condition are resource timing prefixes, retained as raw
chunks; later chunks contain1024deals. No raw chunk is discarded on a null.
Hashes/atomic receipts and source checks support unchanged count-only resume.

## Common joint support and M16

F(I) contains all original hidden deals consistent with own hand, observed
card ownership, capacities and public follow-suit/void constraints. It does
not require bid/play choices to follow the fixed policies. Define:

- Uniform joint:1/|F|.
- Prop-induced joint: product of masked capacity/card factors, normalized on
  F. Every feasible assignment has fixed capacities, so this joint equals
  uniform exactly. Raw capacity marginals need not equal that joint's marginals.
- Hard-bid joint: uniform on F_bid, where every opponent's hook-legal myopic
  argmax equals the observed bid, evaluated on reconstructed original hands.
- Soft-BI-style joint: product of L_j=(1-1/50)*1[bid matches]+1/(50*m_j),
  normalized on F, m_j=public legal-bid count. Own-bid factor is constant.
  Integer factors49*m_j*1[match]+1give exact Z and W_max. Store integers
  |F|,|F_bid|,|R|,Z,W_max before floating logs. No TAplay-action likelihood.
- Net-induced joint: product of strictly positive learned card-holder factors,
  normalized over the same F. Do not multiply another bid likelihood.

M16is proved(externally checked by Claude2026-11-17), conditional on the
uniform-chance/deterministic-profile/matched-hook-legal-bid premises. It gives
KL(P||U_bid)=log(|F_bid|/|R|), KL(P||BI)=log(Z/(|R|W_max)),
KL(P||Uniform)=log(|F|/|R|) and their ordering. This is nats-valued posterior
model mathematics, not a game-score or five-way cost decomposition. Net KL
is not ordered by it. Neural factors/normalizers use recorded floating
arithmetic, with raw unclipped KL and tolerance1e-10, not rational certificates.

## Required implementation checks

C1: synthetic checks for no RNG use in the fixed policies, repeated cold/warm
choices, explicit lower-bid/card tie rules. In-memory test patches restored.
C2: data bidder and model use the same myopic function and dealer hook-legal
argmax. An independent manual hook-argmax check runs for every deal and an
assertion at every exported record establishes R subset F_bid. The separate
verifier independently enumerates every deal and checks every raw record's
matched bids, true holder bits, own hand, legal action and prefix chaining.
C3: report actual test own-hand counts; small conditions are descriptive.
C4: integer baseline ratios retained. The separate direct enumerator checks
actual policy-conditioned reach sets and all net-joint normalizers on every
small-game key and64fixed evenly spaced sorted keys/large condition, without
using the primary DP/rules transitions. Every raw grouping/model selection/
summary is checked; no independent full-large-posterior replay claim.

## Finite network and split

New strict observable encoder only:35per(card,holder)features from own/public
history summaries; known toy deck/rank ranges, no nonexistent cards, stock
capacity0and fourth class masked. Same vocabulary as N1adapted to rank4and
n<=3. The summaries are not injective; structural support F is also part of
the effective input/decoder. Do not claim zero representation loss or pure
estimation error. Full hidden hands produce labels only.

Five paired seeds per condition;35->32ReLU->1sharedholderhead; finalzero.
Logits=log(prop)+f_theta. Training uses proper .999softmax+.001prop mixture
cross entropy against exact per-key marginal holder-count targets. Adam1e-3,
betas.9/.999,epsilon1e-8,globalgradientclip10; L2=.5e-4sumW²,batch128keys,
20epochs. Exact reach_count/mean_selected_training_reach weights and equal
unseen-card weighting per key give the selected-key chance-weighted loss.
Zero-card keys retain the denominator and have zero data gradient.

Own initial hands are SHA256sorted into cutpoints floor(.6H),floor(.8H),
train/validation/test, shared across seats and S/NT within a deck. Hash input
[2100000000,0x4E32,'hands',S,R,n,hand], canonical JSON separators. Train cap
30000keys,validation cap4000keys, use all if fewer. Larger sets use lowest
SHA256priorities of canonical prefix[base,domain,'fit-key-v1',S,R,n,trump]
plus fixed-width20byte key. No test-key subsampling. All histories for an
own hand stay together, but a latent deal can appear through different
actors on different splits. This is hand/key generalization, not independent-
deal sampling. Posterior targets always use the full chance prior, not a
private train/test marker.

Each fit selects epoch5/10/20by exact selected-validation marginal loss,
earlier ties; then alpha0/.25/.5/1against prop, lower ties, on that same
validation set. Selection reuse/boundaries are reported. Validation alpha0
is a fallback, not a held-out noninferiority theorem; every test outcome is
reported even if the net loses. Selection is per fit, not a primary ensemble.

Optimization roots[2100000000,0x4E32,20,seed], batch order
[2100000000,0x4E32,21,seed,epoch]. Reuse across the four conditions is intended
CRN, not20independent optimizer seeds. Check105new roots against123545known
N0/N1roots including N1secondary91before consumption; no2120M–2160Muse.

## Contrasts and reporting

Primary per condition: net-joint KL minus soft-BI joint KL, negative better,
weighted by exact reach counts on the complete reached test-hand population.
Five-fit t95%CIs(df4) measure optimizer variability only. A fixed model's
population KL is completely enumerated (up to floating logs), no chance CI.
No winning-toy selection or arbitrary-hand extrapolation; all/train/validation
are descriptive, small held-hand populations explicitly limited.

Report side by side uniform/prop,hard-bid,soft-BI and all five net-joint values.
Secondary: descriptive share of mean uniform KL removed=(KL_U-KL_m)/KL_U;
retain negatives, undefined if denominator0, no additive/game-score meaning.
Net shares get five-fit seed CIs; fixed baseline calculations have no sampling
interval. Also report raw capacity/net marginal-card logloss separately, not
as marginals of the constrained joints. Every empirical estimate keeps its n
and95%CI; exact chance populations/counts are labeled exact, never given a
sampling interval. Nulls and failures are retained.

## Resource/source gates

One scientific worker after N1releases it. Before launch append UTCprimary,
exact source fingerprint/pre-run local commit, absolute cutoff and cadence.
Each primary launch has max3elapsedhours; proposed initial latest06:00UTC
18Nov, use earlier actual launch+3h. A resource-only gate uses each condition's
fixed256deal timing prefix, projects full traversal and adds5400sfit/joint/
audit reserve. If it does not fit, record scope failure; do not change the
profile/count/seeds based on outcomes. A hard cut preserves complete chunks,
optimizer epochs and1024key joint-KL checkpoints. Any count-only continuation
uses a newly recorded bound with identical target/source/seeds; no retuning.
Separate verification gets its own fresh bound and never overlaps a worker.

Exact scientific source hashes and NumPy/BLAS stack are checked at chunk/epoch
boundaries. Before any experimental trajectory, source must pass checks and
have a matching source-only local Git commit. Under the owner's independent
continuation, a separately recorded Codex source-only review may authorize
execution; it remains pending external implementation/result audit. Never
manufacture Claude approval. Existing actual Claude protocol/M16verdicts are
retained only within scope. No unchecked publication or private-master push.

The primary was drafted before N1confirmation outcomes and before all N2data.
Final implementation/pre-run self-review occurs after observing N1's result;
this is not an independent confirmation of N1 or a novelty claim. Original
draft/proof/review history is retained. No N2outcomes exist at finalization.

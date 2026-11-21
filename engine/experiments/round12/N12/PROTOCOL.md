# N12 — does better belief KL improve a fixed private decision?

Prospective deterministic follow-up, 20 November 2026. Written after N9
results, before inspecting these decision payoffs. N9 remains a known,
previously inspected toy-hand benchmark; this is not a fresh replication.
Run AFTER N10 completion and separate checking. One worker, a freshly logged
absolute stop and per-key/per-model checkpoints. All scope local pending
Claude external review. No fitting, retuning or modification of prior files.

Use ALL N9 held keys in each of its four conditions having exactly two
remaining hero cards and two legal actions. No further outcome-based filter,
cap, extra keys or RNG. There are NO new chance or optimizer seeds: reuse the
frozen 2170M/domain0x4e39 model outputs, explicitly as a derived benchmark.
Predeclare selected key IDs before their payoff calculation.

At each key enumerate the entire N9 structural allocation set F, retaining
its ordering and exact posterior subset R. Force each current hero action;
all opponents thereafter use the fixed private TA definition from N2. The
hero's last card is forced. Thus there is no future free hero decision or
strategy fusion in this task. This isolates a belief-to-decision intervention
conditional on this fixed continuation, not optimal full-game card play.
Actions ordered by card ID; exact ties choose the first. Save complete integer
world/action payoff tables and reconstruct original hands from own/public
cards and allocation owners. Exact R sums yield each action's private value.

For ALL 240 frozen N9 selected recipes, normalize their saved factors over F
and choose maximum expected score, then evaluate under uniform exact R.
For floating model expectations, values within1e-10 score points of the
maximum are numerical ties and choose the first action; count all such ties.
Exact integer oracle/uniform ties choose the first without a tolerance.
Uniform-F and oracle-R action choices are exact integer controls. The oracle
action regret is zero by construction; it is not a sampling result.

PRIMARY: largeNT, summaryMLP,80-budget, joint-minus-marginal reach-weighted
one-step private decision regret, paired across the five frozen optimizer
seeds, t95% df4. Negative favors joint training. Other conditions, linear/rich
configs,20-budget and duration/feature contrasts are descriptive, unadjusted;
keep ALL cells including zeros/adverse outcomes. Arm and contrast CIs cover
optimizer variation only, not unseen hands, contexts or worlds. Each recipe's
regret is an exact rational sum of integer score deficits over the complete
selected R reach, conditional on numerical model decoding. No sampling CI
on that rational or on uniform/oracle exact controls. Belief KL is computed
on this SAME restricted set; don't compare its mean to N9's larger test set.
No monotonic KL-to-score claim, additive gap stack, population confidence,
new-hand replication, 52-card benefit or novelty claim.

Before completion, independently replay ALL saved F/action integer outcomes
with separate public legal/winner/score transitions, reconstruct normalized
factor probabilities and all decisions from independent selected-model
forwards, and recompute all exact regret numerators and paired intervals.
Use fresh read-only audit stop. Same-investigator check is not Claude review.

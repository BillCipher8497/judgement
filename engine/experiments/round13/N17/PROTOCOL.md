# N17 — Decision-relevant misspecification

Predeclared 2026-11-21T01:55Z, before any stratified regret number for the
primary contrast was computed. Owner-authorized independent continuation.

## Question

N12 found exactly zero one-step private decision regret in its primary cell,
and N15/M19 exhibits a context with strictly positive product-family belief
error and zero decision regret. The obvious reviewer attack is that both are
selection rather than structure: if the legal actions have equal true value,
regret is zero by construction and no belief can be punished.

N17 tests the claim where it can actually fail. Stratify the exact contexts by
the **spread of true action values**, restrict to contexts where the legal
actions genuinely differ in value, and measure regret there.

## Stratification (declared before any primary regret is computed)

For each context (key) k with legal actions a0 < a1, let R(k) be the set of
structural worlds that reach k under the frozen fixed policies, and let
Q[w, a] be the exact integer hero score of taking action a in world w followed
by the fixed private continuation. Define the exact true action values

    V(a) = (1/|R(k)|) * sum over w in R(k) of Q[w, a]

as exact rationals, and the **spread** s(k) = |V(a1) - V(a0)|.

Strata, fixed now:

- **S0**: s(k) = 0 — action-indifferent. Regret is structurally zero for every
  possible belief. Reported for completeness, never pooled into a headline.
- **S+**: s(k) > 0 — **decision-relevant. This is the primary stratum.**
- Within S+, three fixed magnitude bands for a secondary breakdown:
  **small** 0 < s <= 5, **medium** 5 < s <= 10, **large** s > 10 hero points.

The spread depends only on the frozen game tables and the frozen posterior. It
contains no model output and no regret, so the stratification cannot be tuned
to an outcome.

## Primary contrast

The **exact conditional-product projection**, with no learning in the loop.
For each context, take the true reaching posterior P (uniform on R(k)) and
compute Q*, the maximum-entropy distribution on the structural support whose
card-holder marginals equal those of P. Q* is the I-projection of P onto the
conditional-product belief family: the best any product-factored belief can do,
with optimizer error, finite features and sample noise all removed.

Q* is computed through the exponential-family dual, reusing N13's machinery
rather than a new solver: minimize `log Z(theta) - theta.mu - log|R|` over the
card-holder indicator features on the exact structural support with L-BFGS-B,
requiring gradient (max marginal deviation) L-infinity below 1e-8. Any key that
fails that tolerance is reported as a failure, not dropped. This is the same
family and the same objective N13 optimizes, so the projection here and the
belief error there refer to one object.

Because the projection decision depends only on the sign of
`sum_w Q*(w) (Q[w,a1] - Q[w,a0])`, floating-point error can only matter near
zero. The absolute value of that quantity is recorded as the decision margin
for every key. Any key with margin below 1e-6 is declared **numerically
ambiguous** and reported separately, with its regret evaluated under both
choices so the stratum total is bracketed rather than guessed.

The projection decision rule picks

    argmax over a of sum over w of Q*(w) * Q[w, a]

breaking ties by the lower card, the same frozen convention N12 uses. Its exact
one-step regret on key k is V(best) - V(chosen), an exact rational.

**Primary statistic:** reach-weighted mean projection regret over the S+
stratum, per condition, reported as an exact fraction and a decimal. This is an
exact enumeration over a complete finite set of contexts. It has **no sampling
confidence interval** and must never be given one.

## Secondary contrasts

1. The 240 frozen N12 recipes' regret restricted to S+, per cell, so the
   learned-model result can be read against the projection bound. Optimizer
   seed intervals (t 95%, df 4, n=5 paired seeds) apply to these and only to
   these, and only across optimizer seeds on a known hand benchmark.
2. The S+ magnitude-band breakdown.
3. The uniform-belief control already present in N12 (uniform over the
   structural support rather than the projection), restricted to S+.

## Acceptance criteria (declared now)

- If projection regret is **exactly 0 on every S+ context in all four
  conditions**, the paper's claim is structural, not selection: even the best
  product-factored belief, on contexts where actions genuinely differ, loses
  nothing. T3's framing stands and strengthens.
- If projection regret is **positive anywhere**, that is reported plainly, with
  the per-key witnesses, and T3's framing is revised to "misspecification costs
  decisions only in this identified subset", with the subset characterised.
  This outcome is equally publishable and no number is re-run to avoid it.
- A partial run is reported as partial.

Either way the S0 count is reported alongside, so no reader can mistake an
indifference zero for a decision result.

## Data and determinism

- Contexts: the complete frozen N12 key set, 1375 keys over four conditions
  (S2R4n2 and S3R4n3, trump 0 and NT), read from
  `engine/experiments/round12/N12/results/<tag>/<key>.npz`.
- **No RNG, no new seeds, no fits, no games, no seed namespace reserved**, so
  no collision check is required; this is exact deterministic arithmetic over
  already-frozen arrays. Reported explicitly rather than left implicit.
- Exact rational arithmetic (`fractions.Fraction`) for all values, spreads and
  regrets. Floating point is used only inside the IPF iteration, whose output
  only selects an action; the selected action's regret is then evaluated
  exactly. A declared near-tie margin of 1e-12 in the projected expectation
  triggers the lower-card rule.
- No core file, existing policy, N0, or frozen round12 source, dataset or model
  is modified. N17 writes only under `engine/experiments/round13/N17/`.

## Disclosure

During the N12 audit earlier the same night I computed the spread
stratification itself and observed that the N12 **primary cell**
(S3R4n3_t-1 / summary_mlp / 80) has zero regret on its 60 S+ keys. That single
observation is already in CLAUDE_AUDIT_2026-11-20.md and is what motivated
N17. The primary N17 contrast — the exact conditional-product projection — had
not been computed in any form when this protocol was written, nor had the S+
regret of any other N12 cell.

## Resource bound

Exact enumeration over 1375 keys with at most a few hundred structural worlds
each. No long run, no checkpoint risk. Absolute stop
**2026-11-21T04:30:00Z**; if unfinished, partial per-condition results are
saved and labelled partial. Runs alongside the N14 window-2 fit worker, so
N17 carries no timing or equal-compute claim.

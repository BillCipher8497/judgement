# N19 — The card-holder marginals identify the decision

Status: complete, separately verified by an independently written reader. Exact
and deterministic: no RNG, no seeds, no fits, no games, no namespace reserved.
Protocol predeclared in PROTOCOL.md before any bound was computed.

## Headline

**No belief that matches the true posterior's card-holder marginals — of any
family, product or not — strictly prefers the inferior action at any of the 224
decision-relevant contexts.** The worst case over the entire marginal-consistent
polytope is exact indifference, reached at 9 contexts, and strict preference for
the inferior action is reachable at none.

| condition | S+ contexts | identified by marginals | reach exact indifference | strictly wrong reachable | worst-case strict regret |
|---|---:|---:|---:|---:|---|
| S2R4n2_t0 | 54 | 53 | 1 | **0** | 0 |
| S2R4n2_t-1 | 57 | 51 | 6 | **0** | 0 |
| S3R4n3_t0 | 53 | 52 | 1 | **0** | 0 |
| S3R4n3_t-1 | 60 | 59 | 1 | **0** | 0 |
| total | 224 | 215 | 9 | **0** | 0 |

Exact LP bounds with exact rational vertex certificates over a complete finite
context set. **No sampling confidence interval applies and none is given.**

## Why this matters

N17 showed the exact conditional-product projection has zero decision regret on
all 224 decision-relevant contexts. That left two explanations open: either the
product family happens to contain a member that decides correctly, or the
first-order marginals already determine the decision so that *every*
marginal-consistent belief decides correctly.

N19 settles it as the second. For each context we bounded
`q . d`, where `d(w) = Q[w,a1] - Q[w,a0]` is the integer action-value difference
per world, over the whole polytope `{q >= 0 : A q = b}` of beliefs on the
structural support whose card-holder marginals equal the true posterior's — with
no family restriction at all. At 215 contexts both endpoints carry the same
strict sign as the true posterior, so the decision is identified by the
marginals. At the remaining 9 the adversarial endpoint is **exactly 0**: a
marginal-consistent belief can be made indifferent between the two actions, but
none can be made to strictly prefer the worse one.

So the paper's central null has a structural reason rather than a lucky one. On
these contexts, getting the first-order holder marginals right is sufficient for
the one-step decision, and the product family's provable error as a
*distribution* — strictly positive KL, with an exact infimum of
log 3 - (4/3) log 2 at one context (M19) — lives entirely in the directions the
decision does not read. A belief model that matches the marginals cannot be
punished here however badly it models the correlations.

This also explains why N17's uniform-belief control *does* lose points: the
uniform distribution over the structural support is not required to match any
marginal, so it lies outside this polytope. N17 therefore tests a strictly
larger set of beliefs than N19 does, and the two results are consistent: wrong
marginals cost points, wrong correlations do not.

## The one real caveat: tie sensitivity

At the 9 contexts where the polytope reaches exact indifference, a
marginal-consistent belief can leave the decision entirely to the tie rule, and
a tie rule that happens to pick the inferior card would then pay the full
spread. That adversarial-tie bound is:

| condition | contexts | tie-sensitive worst-case regret |
|---|---:|---|
| S2R4n2_t0 | 1 (key 2305) | 48/77 = 0.6234 |
| S2R4n2_t-1 | 6 (keys 29, 107, 158, 161, 199, 203) | 910/317 = 2.8707 |
| S3R4n3_t0 | 1 (key 1667487) | 12/47 = 0.2553 |
| S3R4n3_t-1 | 1 (key 1068474) | 66/205 = 0.3220 |

This is the same tie sensitivity M17 flags in the fusion setting, and it is the
honest limit of the identification claim: marginals identify a *weak* preference
everywhere and a *strict* preference at 215 of 224 contexts. The exact
interval endpoints for every context, including the nine, are in
`results/<tag>/contexts.json`.

## Separate verification

`verify19.py`, written independently of `run19.py`, rebuilds the constraint
system from the raw structural arrays rather than reusing it.
`results/verification.json`:

- **A.** The true posterior itself satisfies the rebuilt exact constraints at
  all 224 contexts, so the polytope genuinely contains the object it should.
- **B.** All 448 endpoint certificates (min and max per context) re-verified in
  exact rational arithmetic: non-negativity, exact equality `A q = b` against
  the full system, and the exact objective matching the stored value. No
  floating LP status was accepted as a feasibility or sign certificate anywhere.
- **C.** Zero contexts admit a marginal-consistent belief that strictly prefers
  the inferior action.
- **D.** The tie-sensitive bound above.
- **E.** Cross-check against N17: the projection N17 uses is itself
  marginal-consistent, so its objective must lie inside this interval. It does,
  at all 224 contexts. Two tasks built from different machinery agree.

The posterior's own value `P . d` was also confirmed to lie within
`[min, max]` at every context, which it must.

## Scope

- Two toy conditions at two trump settings; one-step private continuation with
  fixed private opponents; hero side only. Nothing extrapolates to the 52-card
  ladder.
- Identification is of the **one-step** decision at these contexts, not of a
  policy, a game value, or anything multi-step.
- The constraint set is the first-order card-holder marginals. A belief model
  that does not match those marginals is outside the polytope and is not covered
  — N17's uniform control is exactly such a model, and it does lose points.
- Strict identification holds at 215 of 224 contexts; at the other 9 only weak
  identification holds, and the tie rule decides.
- These are the same frozen known-hand contexts as N12 and N17, so this is a
  derived exact benchmark, not a fresh replication on unseen hands.
- No novelty claim. This is an exact finite computation, not a general theorem
  about when moments identify decisions.

## Commands

```
cd engine
JUDGEMENT_JOB_DEADLINE=<epoch> python experiments/round13/N19/run19.py
JUDGEMENT_JOB_DEADLINE=<epoch> python experiments/round13/N19/verify19.py
```

Ran inside the declared absolute stop 2026-11-21T05:30:00Z.

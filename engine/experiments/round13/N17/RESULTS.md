# N17 — Decision-relevant misspecification

Status: complete, separately verified by a second independently written reader.
Exact deterministic enumeration; **no RNG, no seeds, no fits, no games**.
Protocol predeclared in PROTOCOL.md before the primary contrast was computed.

## Headline

On the contexts where the two legal actions genuinely differ in exact value, the
**best possible conditional-product belief incurs exactly zero one-step decision
regret** — in all four conditions, on every one of the 224 such contexts.

| condition | keys | S0 (action-indifferent) | S+ (decision-relevant) | S+ reach | projection regret on S+ | uniform-belief control on S+ |
|---|---:|---:|---:|---:|---|---|
| S2R4n2_t0 | 388 | 334 | 54 | 231 | **0** | 0 (no power, see below) |
| S2R4n2_t-1 | 419 | 362 | 57 | 317 | **0** | 100/317 = 0.3155 |
| S3R4n3_t0 | 294 | 241 | 53 | 235 | **0** | 4/47 = 0.0851 |
| S3R4n3_t-1 | 274 | 214 | 60 | 205 | **0** | 11/205 = 0.0537 |

Exact enumeration over the complete frozen key set. **No sampling confidence
interval applies to any number in this table** and none is given.

The S+ magnitude bands are populated, so this is not a near-tie artifact: 110 of
the 224 S+ contexts have a spread above 10 hero points, 86 between 5 and 10, and
28 at or below 5. Projection regret is 0 in every band.

## Why this answers the reviewer attack

N15/M19 exhibits a context with strictly positive product-family belief error
and zero decision regret, but that context has both actions scoring 10 in every
world, so no belief could have been punished. The obvious reading is selection.

N17 removes the selection. It stratifies on the **spread of true action values**,
a property of the frozen game tables and the frozen posterior with no model
output and no regret in it, then restricts to the stratum where regret is
possible. The answer does not change: the exact I-projection of the true
posterior onto the conditional-product family picks an optimal action on all 224
decision-relevant contexts.

So the zero is **structural, not selection**. Belief-family misspecification is
real and strictly positive in KL (N13/M18 certifies positive witnesses; N15/M19
proves an exact positive infimum for one context), and it still buys nothing at
the decision.

## The measurement is not vacuous — and where it has no power

A zero is only worth reading if something could have been nonzero. Two controls:

- **Uniform structural belief.** Replacing the projection with the uniform
  distribution over the structural support, on the *same* S+ keys, gives
  **positive** regret in three of the four conditions: 100/317, 4/47 and 11/205
  hero points per unit reach. So the metric does punish a wrong belief on these
  very contexts, and the projection's zero is informative rather than an
  arithmetic impossibility.
- **Point-mass beliefs.** In the same three conditions there are S+ keys
  (4, 7 and 4 respectively) where some single-world belief would choose the
  wrong action.

**Limitation, stated plainly: S2R4n2_t0 has no power.** On its 54 S+ keys no
wrong belief of either control form errs, so that condition's zero carries no
information and should not be counted as evidence. The load-bearing evidence is
the **170 S+ contexts across the other three conditions**, where the metric
demonstrably discriminates and the projection still scores zero.

## Secondary: the 240 frozen N12 recipes restricted to S+

`results/n12_recipes_Splus.json`. Every N12 recipe's regret numerator restricted
to S+ equals its full numerator, which is an identity rather than a finding:
S0 keys contribute a structural zero by construction, so no pooled N12 number
was ever carrying indifference mass in its numerator — only in its denominator.
That is what the audit's reporting change to N12/RESULTS.md fixes. Optimizer
seed intervals, where quoted for these cells, remain n=5 paired optimizer seeds,
t 95% df 4, on a known hand benchmark, and apply to nothing else here.

## Separate verification

`verify17.py` was written independently of `run17.py` and re-derives rather than
re-runs. `results/verification.json`:

- **A.** All 1375 keys revalidated: exact action values, spread and stratum
  recomputed from the raw npz alone, all matching.
- **B.** All 224 S+ keys re-optimized with a far tighter budget (20000
  iterations, gtol and ftol both 0). Identical action choice everywhere, regret
  still exactly 0 everywhere, and the decision margin moved by at most
  1.7e-9.
- **C.** Path invariance. Scaling theta by 0.1, 0.25, 0.5, 1, 2, 4 and 10 — four
  orders of magnitude along the path toward the limiting distribution — leaves
  the sign of the projected action difference unchanged on **all 224** keys,
  with zero flips. The conclusion therefore does not depend on where the dual
  optimizer stopped, which matters because for several keys the projection
  infimum is approached rather than attained (the same non-attainment M19
  proves).
- **D.** Non-vacuity as above.

The largest gradient (max marginal deviation) after the tight refit is 5.6e-8,
against a smallest S+ decision margin of 0.618 — about seven orders of magnitude
of headroom, so no S+ decision is numerically close. No key was numerically
ambiguous under the predeclared 1e-6 margin rule.

One protocol note: `run17.py` flags keys with gradient above 1e-8 as
non-converged, and 247 of 1375 keys are so flagged. Their gradients lie between
1.5e-12 and 5.6e-8, i.e. the 1e-8 threshold was simply tight rather than the
optimization having failed, and check C shows the decisions are invariant well
beyond that range. The flags are kept in the raw records rather than relaxed
after the fact.

## Scope

- Two toy conditions at two trump settings, hero-side one-step private
  continuation with fixed private opponents. Nothing here extrapolates to the
  52-card ladder.
- One-step regret at a single decision. Not a full-game policy result, not a
  solved game value, not a statement about strategy fusion (that is N11/M17).
- The conditional-product family only. A correlation-aware family is N16.
- These are the same frozen known-hand keys N12 uses, so this is a derived exact
  benchmark, not a fresh replication on unseen hands.
- No novelty claim.

## Commands

```
cd engine
JUDGEMENT_JOB_DEADLINE=<epoch> python experiments/round13/N17/run17.py
JUDGEMENT_JOB_DEADLINE=<epoch> python experiments/round13/N17/verify17.py
```

Ran inside the declared absolute stop 2026-11-21T04:30:00Z, alongside the N14
window-2 fit worker, so N17 carries no timing or equal-compute claim.

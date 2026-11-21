# Claude external audit — 2026-11-20

Scope: N9, N10, N12, N13/M18, N15/M19, N11/M17 (the "ready-for-review" packets
listed in `engine/experiments/round12/HANDOFF_CLAUDE_20261120.md`), audited in
the order that handoff proposes.

Method: independent recomputation. For each numerical claim I wrote my own
reader over the raw arrays (`results/**/*.npz`, per-key JSON, model
checkpoints) rather than re-running the task's own analyzer, and recomputed the
reported statistic from scratch. For the two new proofs I re-derived the
argument by hand, including the full enumeration behind M19's partition
polynomial. The audit readers are throwaway scripts run from a scratch
directory; no task file, source, seed, dataset or model was modified.

Engine gate at audit time: `cd engine && python -m pytest -q -p no:cacheprovider`
→ **283 passed, 1 skipped in 35.09s**, exit 0. Matches the baseline inventory.

| item | verdict |
|---|---|
| N9 | **PASS** |
| N10 | **PASS** (scope failure, correctly labelled) |
| N12 | **PASS** on the number; **CHANGES REQUESTED** on reporting |
| N13 / M18 | **PASS** — proof validated, pending label removed |
| N15 / M19 | **PASS** — proof validated, pending label removed |
| N11 / M17 | **PASS** on the certificate algebra; posterior support inherited |

---

## N9 — structured-joint versus marginal belief objective

Reproduced the primary cell from the raw held-out arrays. For
`S3R4n3_t-1 / summary_mlp / budget 80` I loaded each of the ten
`models/summary_mlp_{joint,marginal}_s{0..4}/held_80.npz` KL vectors, took the
reach-count weights from `results/S3R4n3_t-1/data.npz` (`ids` to `count`), and
formed the count-weighted mean per seed.

Joint 1.08393597 nats, marginal 1.10098322 nats, paired difference
**-0.01704725 [-0.02449983, -0.00959466]** nats, n=5 paired optimizer seeds,
t 95% df 4. Identical to the reported value in all eight decimals. Per-seed
differences -0.02420084, -0.01504765, -0.01566367, -0.02148287, -0.00884120 —
all five negative, so the interval is not carried by one seed.

The budget-20 cell also reproduces (-0.03126846 [-0.05514175, -0.00739518]).

Scope as reported stands: this is an optimizer-seed interval on a known,
previously inspected hand benchmark, not a fresh-hand replication, and a KL
improvement is not a decision improvement. N12 is the task that tests the
latter, and it finds no decision gain.

## N10 — activated residual late-play students

The specific question put to this audit was whether the failed activation gate
is reported as a scope failure rather than a measured zero effect. It is, in
three places: `results/scope_failure.json` (`"scientific_games": 0`, `"scope":
"Not a zero effect/CI"`), the first paragraph of `RESULTS.md`, and the board
row. Confirmed.

Independent checks:

- All ten fits selected **alpha = 0**. I read
  `results/models/{hard,set}_s{0..4}/frozen.json` directly; every one records
  `alpha: 0.0`. A residual with alpha 0 is the TA+ baseline exactly, so both
  deployed ensembles are the baseline by identity, not by measurement.
- **Zero scientific games.** `results/data/` contains `pilot`, `train`,
  `validation`, `held` and `fixtures` only. No game blocks exist, and no file
  references the reserved 2200M/2210M game namespaces.
- The gate failure is **not** an implementation artifact. Of the 240 rejected
  candidates in `results/candidate_diagnostics.json`, **171 do activate**
  (`n_changed > 0`), spread across every nonzero alpha level (all 40 at
  alpha 4.0, 36 at 2.0, 35 at 1.0, 30 each at 0.5 and 0.25). So the machinery
  can change actions; it simply never improves. **None** of the 171 activating
  candidates has a positive mean score against the baseline on the reused
  validation set. That is a stronger and more interesting negative than "the
  net never fired", and RESULTS.md states it.

One wording note, not a change request: `RESULTS.md` and the board say "many
rejected candidates activate", which is right, but the field that carries this
is `n_changed`, while the separate `changed_fixture_actions` field is 0 for all
240. A reader checking the wrong field would conclude nothing activates. Worth
one clarifying clause if the packet is ever written up.

## N12 — belief KL to exact one-step decision regret

This is the load-bearing zero, so I rebuilt it from the integer game tables
rather than trusting any aggregate.

For each of the 1375 selected two-card keys I loaded `results/<tag>/<key>.npz`,
took `Q` (structural worlds by 2 actions, integer hero scores) and `posterior`
(indices of the worlds that reach the key), and formed the exact
posterior-weighted action values myself. Then for the primary cell
(`S3R4n3_t-1 / summary_mlp / 80`) I replayed each stored per-key decision and
recomputed its deficit as `max_a V(a) - V(choice)`.

Result: my reach is 648 over 274 keys, matching the report. Across all ten
model/seed combinations (joint and marginal, seeds 0-4) my recomputed regret
numerator is **0**, with **zero mismatches** against the 2740 stored per-key
deficits. The primary paired difference is exactly 0 in all five pairs. The
reported degenerate interval reproduces.

**Is the [0,0] a structural identity or an aggregation artifact?** It is
neither, and the distinction matters:

- It is not an aggregation artifact. 76 of the 240 recipes have a **nonzero**
  regret numerator (60 rows at 20, 15 rows at 11, 1 row at 66). The pipeline is
  capable of recording positive regret and does so for other cells. The primary
  cell's zero is a property of those models, not of the measurement.
- It is not a universal identity either. Two cells in the report are negative
  (-0.01697531, -0.01662510) and one is positive with a non-degenerate interval
  (0.01637717 [-0.02909315, 0.06184749]).
- Most importantly, in the primary cell the zero is **not** produced by
  indifference. I stratified the 274 keys by exact action-value spread: **60 of
  them have genuinely different true action values**, and in all five seeds both
  arms choose the optimal action on **all 60**. So the primary zero says
  something real: on this benchmark the models get every decision-relevant
  context right, and the joint objective's KL gain buys nothing because there
  was nothing left to buy.

RESULTS.md never calls the zero an equivalence; the hedging is correct.

**CHANGES REQUESTED (reporting only).** The report does not say how many keys
are decision-relevant. Across the four conditions, **1151 of 1375 keys (84%)
have exactly equal true action values**, so they contribute a structural zero
to the numerator no matter what any model predicts:

| condition | keys | reach | keys with differing exact action values |
|---|---:|---:|---:|
| S2R4n2_t0 | 388 | 1289 | 54 |
| S2R4n2_t-1 | 419 | 1203 | 57 |
| S3R4n3_t0 | 294 | 806 | 53 |
| S3R4n3_t-1 | 274 | 648 | 60 |
| **total** | **1375** | **3946** | **224** |

Without that denominator a reader cannot tell the informative zero (all 60
decision-relevant keys correct) from the uninformative one (nothing to get
wrong). The fix is to report the stratified count alongside the pooled regret.
The number itself does not change and nothing needs re-running. This table has
been added to N12/RESULTS.md as part of the audit; N17 turns the stratification
into a predeclared experiment across all four conditions and all 240 recipes.

Also noted, and already stated in the report: `n_numeric_nearties` (308/388
etc.) counts near-ties in the *model's* expectation, which is a different and
much larger set than the exact value ties above. The two should not be confused.

## N13 / M18 — finite conditional-product entropy lower bound

**Proof checked by hand and correct.** With `Q_theta(w) = exp(theta.f(w))/Z(theta)`
on finite F, P uniform on R within F, mu = E_P f:

    KL(P || Q_theta) = log Z(theta) - theta.mu - H(P)

and for any q on F with E_q f = mu, `KL(q || Q_theta) >= 0` expands to
`log Z(theta) >= theta.mu + H(q)`. Substituting gives
`KL(P || Q_theta) >= H(q) - H(P)`, uniformly in theta, so the bound survives the
infimum over the family and its closure. Every step checks out; no optimization
of q is needed, as claimed.

Two observations, neither an error:

1. Since P is uniform on R, `H(P) = log|R|` is the *maximum* entropy on R, so a
   useful witness must place mass on F minus R. That is exactly what the code's
   mixtures do, and it is why the bound is non-vacuous here rather than
   trivially zero.
2. The bound is actually **tight**: the supremum of H(q) over the
   moment-matching set is attained by the max-entropy member of that set, which
   is itself in the exponential family, so
   `inf_theta KL(P || Q_theta) = H(q*) - H(P)` by the standard I-projection
   Pythagorean identity. BOUND.md states only the inequality direction, which is
   conservative rather than wrong. N15/M19 is precisely the case where the
   matching upper limit is exhibited, confirming tightness there.

Numerical claims reproduce. I recomputed every per-key certified lower bound as
`H(q) - log|R|` from the stored exact rational mixture weights, independently of
the task's enclosure arithmetic, and reach-weighted them:

| condition | keys | reach | positive witnesses | reach-weighted certified lower (nats) |
|---|---:|---:|---:|---|
| S3R4n3_t-1 | 64 | 170 | 6 | 0.0660799372 |
| S3R4n3_t0 | 64 | 236 | 7 | 0.0881641563 |
| S2R4n2_t-1 | 64 | 135 | 1 | 0.0032926298 |
| S2R4n2_t0 | 64 | 123 | 1 | 0.0081165408 |

The largeNT primary 0.0660799372 matches the report exactly. 15 positive
witnesses in total and 3 optimizer status failures, both as reported. All 256
LP proposals are exactly feasible, and in every key the certified rational
lower bound lies at or below the numerical free-potential value — the ordering
the two must satisfy to be consistent, and it holds without exception.

**The "pending external audit" label is removed from M18.** The lemma is
validated as a correct, correctly scoped, standard finite entropy duality
argument. No novelty claim attaches to it.

## N15 / M19 — exact single-context product-family KL infimum

**Proof checked by hand, end to end, and correct.** Each link verified
independently:

- **Moment face.** The four worlds with positive moment-compatible mass are
  structural indices 0, 5, 10, 6 with owner vectors (0,0,1,2), (0,2,1,0),
  (1,2,0,0), (1,0,0,2). I recomputed the exact card-holder marginals of the
  uniform 3-world posterior and confirmed every other structural world touches a
  zero-marginal (card, owner) pair and is therefore excluded. Exactly 4 remain.
- **Dimension.** The marginal constraints reduce to `p0+p5 = 2/3` and
  `p0+p6 = 1/3` (each appearing twice) plus normalization, giving the
  one-parameter family `(1/3-t, 1/3+t, 1/3-t, t)`, `0 <= t <= 1/3`. Matches the
  stated RREF result.
- **Maximizer.** `dH/dt = 2 log(1/3-t) - log(1/3+t) - log t = 0` gives
  `(1/3-t)^2 = t(1/3+t)`, i.e. `t = 1/9`, so `q* = (2/9, 4/9, 2/9, 1/9)`.
  Matches.
- **Entropy gain.** `H(q*) = 2 log3 - (4/3) log2`, so
  `H(q*) - H(P) = log3 - (4/3) log2`. Verified symbolically; equals
  0.174416047921516.
- **Matching upper limit.** This is the step worth checking properly, so I
  enumerated all 12 structural allocations myself (two cards to owner 0, one
  each to owners 1 and 2, over unseen cards 5,7,8,9; C(4,2) times 2 = 12) and
  applied the stated potentials — factor 2 on (card 5, owner 0) and
  (card 7, owner 2), 1 on every other positive-marginal factor, epsilon on
  zero-marginal factors. World weights come out: constants 2 + 4 + 1 + 2 = 9,
  linear 2e + 4e + e + e = 8e, quadratic 2e^2 + 2e^2 + e^2 + e^2 = 6e^2. So
  **Z(e) = 9 + 8e + 6e^2**, exactly as claimed, and the three posterior worlds
  carry weights 2, 4, 2 independent of e. Then
  `KL(P || Q_e) = log Z(e) - log3 - (4/3) log2 -> log9 - log3 - (4/3) log2
  = log3 - (4/3) log2`.

Upper limit meets lower bound, so the product-family KL infimum for this one
context is exactly **log3 - (4/3) log2, approximately 0.174416047921516 nats**.
The non-attainment caveat is correct: the limit puts zero mass off the moment
face, so no strictly positive potential achieves it.

Cross-artifact consistency, which is a good independent test of all three
packets at once: N13's certified lower bound for this same key (105013) is
0.166389086604960, N15's exact infimum is 0.174416047921516, and N13's
numerical free-potential optimizer landed at 0.174416047921740. Lower bound <
infimum < numerical upper, with the optimizer converging to the proved infimum
from above to eleven decimals. Three artifacts built by different code paths
agree.

**The "pending external audit" label is removed from M19.**

The decision part — both legal actions scoring 10 in every posterior world, so
zero current regret despite strictly positive belief-family error — is correct
as arithmetic and honestly labelled as a posthoc single-context illustration.
It is also the paper's weakest load-bearing claim, because the context was
selected as N13's smallest witness and happens to be action-indifferent. That
is exactly the reviewer attack N17 is built to answer, and N12's stratified
result above already gives it support beyond this one context.

## N11 / M17 — a PI-optimal action need not be private-optimal

I recomputed the headline certificate
(`results/certificates/S3R4n3_t0_0507870.json`) from its own tables, without the
recursive solver. For each of the three legal cards I summed the terminal
constants, then for each future private infoset took the sum of per-world row
maxima (PI, which may choose a different card per world) and the maximum over
column sums (private BR, one common card across the infoset), and checked the
column sums against the stored ones.

| card | my PI sum | reported | my BR sum | reported |
|---|---:|---:|---:|---:|
| 4 | 140 | 140 | 100 | 100 |
| 6 | 140 | 140 | 140 | 140 |
| 10 | 140 | 140 | 140 | 140 |

All 14 worlds are accounted for in each branch. PI-optimal set {4, 6, 10}
strictly contains private-optimal {6, 10}; the predeclared lower-card tie rule
picks 4 and loses (140-100)/14 = **20/7** hero points conditionally. Reproduced
exactly. The structural count also checks: C(8,2) times C(6,3) = 28 times 20 =
560 assignments before bid/lead filtering.

The tie-sensitivity caveat is correct and important — best-PI-tie loss is 0, so
this witness shows that *a* PI-optimal decoder can lose, not that every one
must. PROOF_PENDING_EXTERNAL.md says exactly that.

**Audit scope limit.** I verified the certificate's algebra and internal
consistency completely. I did **not** re-implement the posterior enumeration
that reduces 560 structural assignments to the 14-world support; that step
rests on the previously audited N2 reachability machinery and the fixed TA
replay, as the Codex review already notes. The claim "the exact posterior here
is these 14 deals" is therefore inherited, not independently re-derived. Since
the certificate is self-contained once the support is granted, the algebra is
cleared and the support dependency is recorded rather than leaving the whole
item pending.

M17 is cleared as a correct finite computational witness of a known
strategy-fusion phenomenon, with that dependency noted.

---

## What this audit did not cover

- N14: mid-run at audit time, handled separately (see N14/STATUS.txt).
- N16: code only, no data, no seeds — nothing to audit.
- N4's missing equal-wallclock and SIR reset/ESS diagnostics remain open; N18
  is the task that closes them.
- The 52-card ladder. Every item above is a toy-condition or known-hand
  benchmark result. Nothing here extrapolates to the full game.
- Novelty. The literature map is bounded and no novelty claim is made or
  cleared.

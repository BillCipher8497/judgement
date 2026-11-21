# N19 — Does matching the card-holder marginals already pin the decision?

Predeclared 2026-11-21T02:55Z, before computing any bound. Exact, deterministic,
no RNG, no seeds, no fits, no games, no namespace reserved.

## Question

N17 established that the exact conditional-product projection incurs zero
one-step decision regret on all 224 decision-relevant contexts. That is a fact
about one particular belief — the best member of one family. It does not explain
why.

Two very different explanations fit N17 equally well:

- **(a) Lucky family.** The product family happens to contain a member that
  decides correctly, and other misspecified families might not.
- **(b) Moment identification.** The card-holder marginals of the true posterior
  already determine the decision, so *every* belief consistent with those
  marginals decides correctly, whatever family it comes from.

If (b) holds, the zero regret is not about the product family at all. It is a
statement about what the first-order marginals identify, and it explains N12,
N15/M19 and N17 at once: a belief model that gets the marginals right cannot go
wrong at these decisions even though it is provably wrong as a distribution.

N19 separates (a) from (b) exactly.

## Primary contrast

For each decision-relevant (S+) context, let F be the structural allocation
support, R the reaching worlds, P uniform on R, and
`d(w) = Q[w, a1] - Q[w, a0]` the integer action-value difference per world.
Let A q = b be the card-holder marginal constraints of P, i.e. for every unseen
card c and owner o, `sum_{w : owner_c(w) = o} q(w) = P(owner_c = o)`, together
with normalization.

Define the **moment-matching decision interval**

    L = min  q . d        U = max  q . d
        s.t. A q = b, q >= 0         s.t. A q = b, q >= 0

over the whole polytope of moment-matching beliefs on F, with no family
restriction. Then:

- If **L and U have the same strict sign** as the true posterior's
  `P . d`, every moment-matching belief — product, correlation-aware, or
  arbitrary — picks the same action the true posterior picks. The decision is
  **identified by the marginals**, and explanation (b) holds for that context.
- If the interval **straddles zero**, some moment-matching belief flips the
  decision, so the product family's success at that context is a property of the
  family or of the projection, not of the marginals. Explanation (a) holds there,
  and the flipping belief is a witness we report.

**Primary statistic:** the number of S+ contexts, per condition, whose
moment-matching decision interval excludes zero on the true-optimal side, out of
the S+ total. Exact enumeration over a complete finite context set; **no
sampling confidence interval applies and none is given.**

## Secondary

1. The worst-case regret over the whole moment-matching polytope, per condition:
   the reach-weighted regret a maximally adversarial marginal-consistent belief
   would incur. This upper-bounds every marginal-consistent belief model,
   learned or not.
2. For contexts whose interval straddles zero, the explicit flipping belief and
   its regret.
3. Whether the N17 uniform-belief control's positive-regret keys are explained
   by the uniform belief lying outside this polytope (it is not required to match
   any marginal), which would make N17's control a test of a strictly larger set
   of beliefs than N19's.

## Exactness

The two linear programs are solved in floating point with HiGHS, and then
**every reported sign is certified exactly**:

- The LP's optimal basis gives a candidate vertex; the vertex is reconstructed
  in exact rational arithmetic by solving the active equality system with
  `fractions.Fraction`, and feasibility (`A q = b` exactly, `q >= 0`) and the
  objective `q . d` are recomputed exactly. A floating LP status is never
  accepted as a feasibility or sign certificate.
- For the "identified" conclusion the exact objective at the minimizing vertex
  is what must have the correct strict sign, since the minimum of a linear
  function over a polytope is attained at a vertex. If the exact rational
  reconstruction fails for a context, that context is reported as
  **uncertified** and is not counted as identified.
- A context whose interval straddles zero needs an exact feasible flipping
  belief, reconstructed and checked the same way, or it is reported as
  uncertified rather than as a counterexample.

## Acceptance criteria (declared now)

- If **every** S+ context is certified identified, explanation (b) holds
  throughout and the paper gains a structural reason for its central null. The
  claim becomes: on these contexts, first-order holder marginals identify the
  one-step decision, so belief misspecification beyond the marginals is
  decision-irrelevant by construction.
- If some contexts straddle zero, that is reported with counts and witnesses,
  and the claim is narrowed to "identified on this subset", with the
  non-identified subset characterised. N17's zero is unaffected either way; only
  its explanation changes.
- Mixed outcomes are reported as mixed. No context is reclassified after seeing
  its interval.

## Data

The complete frozen N17/N12 S+ context set (224 contexts over four conditions),
read from `round12/N12/results/<tag>/<key>.npz` for action tables and
`round12/N9/results/<tag>/data.npz` for the structural support, the same inputs
N17 uses. Nothing in round12 is modified. N19 writes only under
`engine/experiments/round13/N19/`.

## Resource bound

224 small LPs plus exact rational reconstruction. Absolute stop
**2026-11-21T05:30:00Z**; partial per-condition results are saved and labelled
partial if it is reached. No compute worker is running for this task beyond the
single process.

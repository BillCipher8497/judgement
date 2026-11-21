# M15 — Privileged root supervision and decomposed private decision risk

**Status: externally validated (Claude,2026-11-17), documentation corrections applied; see root audit.** Elementary finite-decision observations,
not a claimed novel theorem. Exact game references are M13/M12. We condition
on the first free team decision; later team decisions remain omniscient.

Let w be a chance world that reaches a free root decision, I(w) its acting
player's own/public history, A(I) its legal original-hand action indices, and
Q(w,a) the hero score under action a followed by perfect-information minimization
against the fixed hero. Terminal roots contribute a common constant T. Use
unconditional world probabilities, so all following values include T.

PI = T + sum_w p(w) min_a Q(w,a).
J1 = T + sum_I min_a sum_{w:I(w)=I} p(w)Q(w,a).

The minimum of a sum is at least the sum of minima, giving PI<=J1.
For phi=phi(I), a function of I alone, with the same legal index set in
each representation class (actions are original-hand indices), define J_phi by replacing I with phi(I). Every
phi-policy is an I-policy, hence J1<=J_phi<=J_student. Algebra gives

J_student-PI = (J1-PI)+(J_phi-J1)+(J_student-J_phi).

Every term is nonnegative. This requires a deterministic action decoder using
only phi; any supplied legal mask is part of phi. It cannot be ignored while
claiming representation loss. Randomized policies obey the same bound by
linearity. This does not solve the full decentralized team or establish an
equilibrium. Our241-feature input is injective in I over the benchmark; its
representation term is therefore zero, to be checked by exhaustive key mapping.

## Binary public-payoff alignment

In the present fixed-policy deterministic game each Q(w,a) is0 or c(I)=10+b0,
and b0 is public. Define y(w,a)=1 when a minimizes Q(w,.), allowing all ties.
If at least one zero action exists, y=1-Q/c. If every action costs c, y=1
for every legal action. Consequently

E[y(w,a)|I] = 1 - E[Q(w,a)/c|I] + Pr(all actions cost c|I).

The last term is common to all legal actions. Thus maximizing the conditional
unnormalized optimal-membership probability minimizes conditional expected
hero score. Proper independent Bernoulli loss at its unrestricted population
optimum equals those conditional probabilities at each observed private key; normalized Q regression and direct
expected cost also align. Finite function classes, regularization, optimization,
generalization and the particular global weights can all prevent finite fitted
models from attaining this optimum. No universal finite-network guarantee.

If c varied with unobserved w, membership alone would not generally suffice.
Neither categorical normalization of optimal sets nor arbitrary hard tie-breaking
has this identity. Two exact counterexamples, with common c=1 and one private I:

* Hard first-argmin: world masses(1/5,3/10,1/2), costs(0,1),(1,0),(0,0).
  Hard labels chooseA with probability7/10, but risksA=3/10,B=1/5.
* Uniform optimal labels: masses(3/5,2/5), costs(0,0,1),(1,1,0).
  Categorical probabilities(3/10,3/10,2/5) chooseC, while risksA=B=2/5,C=3/5.

These are supervision-objective artifacts, distinct from the information floor.
Their exact arithmetic needs no sampling CI. They are not evidence that a
particular finite learner will improve or that teacher ties explain every gap.

J_student is explicitly a root-relaxation score: a private root action followed by omniscient continuation. It is not a feasible full-team score and must not be compared as an achieved policy to the M13 feasible uppers. J_phi depends on the original-hand-index action encoding; public_mask excess is not an intrinsic information measure. No guarantee applies to held-out own-hand keys.

Load-bearing assumptions: single-round hit/miss scoring without penalty; deterministic fixed hero; public hero bid; hero score as sole team objective; perfect-information continuation. Label and decoder ties use lowest original-hand index in the reported convention. Decoder tie sensitivity is in results/audit_diagnostics.json.

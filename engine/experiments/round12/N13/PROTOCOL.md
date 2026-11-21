# N13 — conditional-product oracle and exact entropy witnesses

Prospective deterministic diagnostic,20November2026; run after N12 separate
checks. This responds to N9's objective effect and does not use N13 outcomes
to select its keys, budget or certificate procedure. All pending Claude.

For each of four N9 conditions, sort its held keys by SHA256 of UTF8 JSON
["N13",tag,keyhex20bytes] with compact separators, take first64. Save IDs
before optimization. No RNG/new chance/optimizer seeds, no fitting/deployment
on fresh hands. This known-hand oracle uses exact R and cannot be deployed.

For each complete structural allocation F, use unrestricted card-holder
potentials (not a shared neural feature encoder) and minimize
logsumexp_F(theta.f)-theta.mu-log|R| with L-BFGS-B, start all0,
maxiter1000,gtol1e-10,ftol1e-14. Save finite potentials, objective, gradients,
status and iterations. This is a NUMERICAL feasible-potential value/upper
estimate for the optimal KL, not a rigorous optimum or certified interval.
No optimizer convergence or tiny residual establishes a zero factor floor.

Independently solve an LP maximizing mass outside R subject to exact card
holder marginals mu and total1, q>=0 on F. SciPy/HiGHS uses one thread.
Reconstruct its nonzero weights as Fractions with max denominator10^9.
Accept ONLY if normalization/nonnegativity/ALL moment equalities pass EXACT
rational arithmetic. Otherwise preserve failed proposal and use P itself
as a zero witness, with explicit failure; no tolerance called feasibility.

For feasible q, consider the fixed grid q_e=(1-e)P+e*q, e=0 and2^-1..2^-40.
Pick maximum FLOAT entropy on this deterministic grid (a witness search,
not sampling/selection inference). Independently certify entropy gain with
exact rational log enclosures: scale x to y in[1,2], use
log(y)=2*sum_{j=0..31}z^(2j+1)/(2j+1), z=(y-1)/(y+1), with remaining tail
<=2*z^65/(65*(1-z^2)). Use the same enclosure for log2 and scale powers;
round endpoints outward to dyadics denominator2^128. Arithmetic is rational.

For any such exact-moment q_e, the variational entropy inequality gives
KL(P||Q_theta)>=H(q_e)-H(P) for EVERY conditional product distribution with
finite potentials, and by limits for its closure. Save explicit rational
weights and the rational nats enclosure. A positive certified lower endpoint
is a strict family-floor witness, conditional on this finite model/proof.
New proof note is PENDING EXTERNAL AUDIT, no novelty claim. The unqualified
label "solved floor" is forbidden even if numerical upper and lower agree.

PRIMARY: largeNT's exact reach-weighted sum of nonnegative certified lower
endpoints on these64keys. Report all four conditions, ALL per-key outcomes,
zero/failed witnesses, numeric optimizer status and reach/counts. Exact
deterministic sums/bounds have NO sampling CI; neither does numerical
optimization. No estimate of unseen-key prevalence/52-card effect, no additive
gap stack or transfer to deployed NN performance.

One worker, new absolute scientific stop and per-key checkpoint. Separate
read-only check: independently verify all q marginals, alternate40-term
rational log enclosures, entropy-bound containment and positivity, finite
potential objectives/gradients and all aggregate fractions. No original
N2/N9/data/model/core/policy edit; no commit/push awaiting Claude validation.

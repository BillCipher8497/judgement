# One exact product-belief limitation with no current action cost

**Status: externally validated by Claude, 2026-11-20** (moment face, RREF
dimension, entropy maximizer and the full 12-allocation partition polynomial
all re-derived by hand; see CLAUDE_AUDIT_2026-11-20.md). Separate Codex
reach/linear-algebra/partition/log checks passed first. This is an
illustrative application of established entropy duality, not a new general
variational theorem or a claim of novelty.

In the all-private TA/myopic,12-card no-trump toy, hero0 has original hand
{2,4,6} and observes bids(0,0,2,0), public plays(2,0,3,1,11,10). The complete
reaching posterior is uniform on three deals. The unseen cards are(5,7,8,9);
relative opponent capacities are(2,1,1). There are12 structural allocations.
Complete world arrays/indices and private-reach checks are saved in results/.

In the immutable structural ordering, posterior indices are0,5,10. Exact
holder marginals exclude every positive-mass moment-compatible world except
indices0,5,10,6, whose owner vectors are respectively

    (0,0,1,2), (0,2,1,0), (1,2,0,0), (1,0,0,2).

Here owner0/1/2 means seats1/2/3. Rational row reduction gives every
moment-compatible distribution on this face as

    (1/3-t, 1/3+t, 1/3-t, t),  0 <= t <= 1/3.

The rational distribution q*=(2/9,4/9,2/9,1/9), corresponding to t=1/9,
matches the exact posterior's card-holder marginals. Consequently the N13
entropy inequality lower-bounds KL(P||Q) for every conditional product Q by

    H(q*) - H(P) = log(3) - (4/3) log(2).

For a matching limit, assign card-holder factor2 to the first unseen card in
owner0 and second unseen card in owner2; assign1 to every other factor having
positive posterior marginal, and epsilon to factors with zero marginal.
For every epsilon>0 these are positive finite potentials on the structural
support. Exact enumeration gives the partition polynomial

    Z(epsilon) = 9 + 8 epsilon + 6 epsilon^2.

The three posterior worlds have unnormalized weights2,4,2, independent of
epsilon. Thus

    KL(P||Q_epsilon) = log Z(epsilon) - log3 - (4/3)log2
                   -> log3 - (4/3)log2.

The upper limit matches the entropy lower bound. The product-family KL
**infimum for this one context** is therefore exactly
log3-(4/3)log2, approximately0.174416047921516nats. Positive-potential
attainment is not asserted: the limiting distribution has zero mass outside
the moment face. Saved rational log enclosures certify positivity; these
are exact finite arithmetic bounds, not sampling confidence intervals.

The same preselected context has two legal hero actions4and6. Under the true
three-world posterior and fixed private TA future opponents, each action
scores10 in every world; hero's last card is forced. A separate manual
transition check verifies all six integer scores. Thus either current action
has zero one-step private regret even though the product belief family's KL
infimum is strictly positive. This is a single-context illustration that
belief fit and decision cost differ, not a universal claim that beliefs do
not matter or a result about a full-game policy.

The case was chosen as N13's smallest positive witness before these action
values were inspected; it is posthoc illustration, not a replication,
prevalence estimate, fresh test set or general factor-family optimum. The
complete toy posterior/game premise and both proofs require external audit.

# Exact objective mismatch and representation checks

**Externally validated (Claude,2026-11-17), requested documentation corrections applied.** Values are exact complete
finite-population root-relaxation means, not sampling estimates. Each condition
has4players, fixed myopic bids/TAhero, and omniscient continuation after the
first free team action. Costs are hero points, so smaller is better.

| condition | n exact deals | private groups | information floor J1−PI | hard-first-label Bayes regret | uniform-optimal CE Bayes regret | unnormalized optimal-set Bayes regret |
|---|---:|---:|---:|---:|---:|---:|
| S2R4n2 trump0 |2520|702|0|0|0|0|
| S2R4n2 NT |2520|767|0|19/105 (best decoder1/42)|0|0|
| S3R4n3 trump0 |369600|41712|25819/369600|101621/369600|257/52800|0|
| S3R4n3 NT |369600|42440|421/30800|7819/17600|19/30800|0|

Here a Bayes teacher-label decoder has exact conditional label frequencies
at every full private key, not a fitted/generalizing network. Hard and uniform
labels refer to precisely specified tie conventions. Their regret is excess
over J1, separate from irreducible J1−PI. The structural alignment identity and
feature injectivity passed in all four full populations. No neural comparison
has yet been evaluated on the held-out hand populations at this checkpoint.

The uniform-label mismatch is much smaller than the hard-first mismatch here;
this is not evidence that every categorical distillation method is poor. A
different teacher tie rule or student family can behave differently. Full
241features are injective in private observation; two explicit coarsenings are
reported in results/diagnostics.json with the supplied legal mask retained.

Large data extraction inherits the audited tree, checked by exact PI/J1
rationals and fresh64selected complete-deal recursions perregime. Small controls
were independently traversed on all2520deals/regime. This is not a second
independent full-population implementation for the large dataset. No52-card
extrapolation, hidden-team feasibility, equilibrium, novelty or finite-training
consistency claim. Sources, raw integer arrays and contexts retained locally.

Claude external audit2026-11-17 correction: hard labels and hard/uniform decoders use lowest original-hand index. SmallNT hard Bayes excess19/105 must accompany best decoder case1/42. Fixed-label decoder best/worst risks and tied-group counts are in round12/M2/results/audit_diagnostics.json; these ranges differ from changing the teacher priority. Exact complete populations2520/369600deals, no sampling CI. M3 supports avoiding label-induced cost, not a learning-dynamics advantage.

Fixed-label decoder tie convention: lowest original-hand index for labels/decoder. SmallNT hard best/worst[1/42,19/105]; see results/audit_diagnostics.json for all exact best/worst risks and tied-group counts, distinct from changing teacher priorities. Membership excess0under every tie-break.

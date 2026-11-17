# Pure team policies against a fixed hero

**Status: proved and externally validated (Claude, 2026-11-16).** Elementary model clarification; no novelty claim.

Consider a finite extensive game in which the team minimizes expected hero score. All non-team policies, including the hero's responses, are fixed. Every member's information contains its full observation/action history, and team randomization is independent of the deal/chance and non-team random tapes. Internal team coins influence non-team decisions only through legal observed histories: an extra coin observed directly by the hero is not silently added to the game. Public signaling through legal actions remains permitted; private-card pooling remains outside the admissible information.

Fix the team's full collection of random tapes, including shared and private randomness. Each member now uses a deterministic action plan on its own information histories. The resulting tuple is a feasible pure team policy. Because the game is finite, there are finitely many distinct such pure tuples, even when tapes are continuous. A randomized team policy therefore induces a distribution λ over feasible pure tuples p. With non-team behavior fixed and exogenous tapes independent,

    J(randomized team) = sum_p λ(p) J(p) >= min_p J(p).

Pure tuples are themselves admissible, so the minimum over all randomized private team policies equals the minimum over pure tuples and is attained. Shared randomness independent of the deal cannot improve that global minimum.

The statement concerns the unrestricted finite private-policy class. It does not show that a small parameterized bot family contains the best pure plan, that a particular seed finds it, or that CFR/local best responses reach it. The optimal plan can be computationally difficult to find. Arbitrary rowwise argmax rounding can worsen a saved mixed profile; every proposed feasible upper policy still needs replay.

If the hero strategically best-responds to the team distribution instead of following a fixed policy, the displayed fixed-value mixture argument does not establish pure-team sufficiency. Correlation with the deal changes the information model. Restricted-memory/absent-minded policy classes are excluded by the full-history assumption. Do not present this as a theorem for team-maxmin equilibria or general multiplayer CFR.

An exact scope example is a hidden-bit matching game, not Judgement: the hero knows the team's announced bit distribution p and optimally guesses the bit, scoring1 for a match. The team cost is max(p,1−p), minimized at1/2; every pure bit gives cost1. This is precisely a strategic-response setting outside the fixed-hero premise. These are algebraic values, not sampled estimates.

`check.ps1` is a small deterministic arithmetic check, not a new Judgement game experiment. It enumerates a weighted four-world table, all16 pure private plans and81 profiles with action probabilities0,1/2,1. The manual minimum by the second member's four plans is1,8/5,9/10,19/10; the overall pure minimum is9/10. The uniform profile is43/20. The continuous-policy conclusion follows from the proof, not from the half-grid check. No sampling CI or seed applies.

For direct checking, each member observes its own type bit. In the table, costs are ordered by joint actions00,01,10,11:

| Types | Chance weight | Costs |
|---|---|---|
| 00 | 1/10 | 0,4,4,2 |
| 01 | 2/10 | 3,0,2,5 |
| 10 | 3/10 | 0,3,4,1 |
| 11 | 4/10 | 1,4,0,3 |

Fixing the second member's type-contingent action pair to00,01,10,11, and minimizing the first member's two separate type decisions, gives weighted cost numerators10,16,9,19 over10. A minimizing pure tuple has the first member always choose1, while the second chooses1 at type0 and0 at type1; its four costs are2,2,1,0, hence9/10. The all-half profile has numerator86 over40, hence43/20. This hand calculation is independent of the enumeration's base3 profile encoding.

# First-decision private-information lower bound

**Status: proved and externally validated (Claude, 2026-11-16). See the root CLAUDE_AUDIT_2026-11-16.md. Feasible upper-policy independent replay remains task E5.**

The compact tree already fixes all bids and hero moves, and skips forced spoiler moves. Its root for a given deal is therefore the first spoiler choice, or a terminal; no earlier free spoiler action affects the probability of reaching that root.

Let Q(w,a) be the integer PI-minimum hero score after root action a in deal w. Group nonterminal roots by the acting spoiler's private information set I (seat, own hand, complete public history). Every feasible hidden team must use the same behavioral distribution at all roots in I. Grant it perfect information after this decision. Its expected hero cost is at least

    J1 = (sum of terminal-root scores + sum_I min_a sum_{w rooted in I} Q(w,a)) / number of deals.

This is a relaxation: future play is omniscient, and any coupling between these root choices and decisions that are non-root in other worlds is removed. Every actual hidden team still satisfies the retained root constraints. Minimizing the linear root objective picks a pure action separately for each I. Thus PI <= J1 <= optimal hidden-team hero score <= achieved feasible score. Integer sums give an exact rational J1 over the complete chance space. A strict J1 > PI would certify some information cost in this toy; equality would leave this bound inconclusive about a strict separation.

Root selection must be measurable before any free team action: fixed bids, deterministic hero moves and forced team moves satisfy this condition here. The full public sequence includes every collapsed move. Conditioning on a root information set therefore retains the correct chance weights; no team strategy can alter which deal reaches its first choice. The same information set may be non-root in another world; dropping that cross-world coupling is explicitly part of the relaxation.

Independent shared team randomness does not evade the bound. Condition on its value to get a feasible root rule and apply the inequality, then average. A mediator that observes pooled private cards is outside the allowed policy class. This argument supplies no mixing, CFR convergence or equilibrium guarantee.

Implementation is in root_bound.py. This is a mathematical strengthening within D3, with no change to its preregistered primary PI/achieved interval. The full-tree values were computed in the original timebox and saved in the per-regime JSON; later consolidation only replays saved policies and recomputes checks. The old draft sentence saying the bound was not computed is superseded by these saved results. Separate review recomputation is in ../reviews/audit_d3.py. Exact enumeration covers all 369,600 ordered deals per regime, so no sampling CI applies: trump PI=367921/61600, J1=446669/73920, strict gap=25819/369600; NT PI=106419/15400, J1=213259/30800, strict gap=421/30800. The lower bounds and proof are externally validated; no 52-card extrapolation applies.

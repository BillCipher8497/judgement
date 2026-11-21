# Round 13 — post-audit ML continuation (2026-11-20/21 ET)

Opened after the Claude external audit of 2026-11-20 cleared N9, N10, N12,
N13/M18, N15/M19 and N11/M17 (see CLAUDE_AUDIT_2026-11-20.md at the repo root).
Round 12 stays frozen: its sources, seeds, datasets and models are not edited
here, and round13 writes only under its own task folders.

Same protocol as round12: one folder per task, PROTOCOL.md predeclared before
compute, append-only STATUS.txt, results/, then a separately written
verification. One Judgement compute worker per machine. Fresh absolute stops
recorded before any long run; expired deadlines never carry over. Every
empirical number keeps its n and 95% CI, and exact enumerations, optimizer-seed
intervals and paired-game intervals stay strictly distinct.

## Current

**2026-11-21T03:20Z** — N17 and N19 complete and separately verified. N19
explains N17's zero structurally: the card-holder marginals identify the
one-step decision. N18 predeclared, with no data or seeds. N14 (round12)
finished in its final window: primary gate failed, zero games, a scope failure
rather than a zero effect.

| id | task | status | notes |
|---|---|---|---|
| N17 | Decision-relevant misspecification: stratify exact contexts by true action-value spread, then measure regret of the exact conditional-product projection where actions genuinely differ. | **complete, verified** | Projection regret exactly 0 on all 224 S+ contexts, all four conditions, all three spread bands. Uniform-belief control positive on the same keys in three conditions (100/317, 4/47, 11/205), so the measurement is not vacuous; S2R4n2_t0 has no power and its zero is reported as uninformative. Second independent reader revalidates all 1375 keys; choice and zero regret stable under tighter refit and theta rescaled 0.1x-10x. Exact enumeration, no sampling CI, no RNG/seeds/games. Answers the selection objection to the paper's thesis. |
| N18 | Equal-wallclock belief-in-search, closing the hole Claude's audit left open in N4: fixed-K only, with equal-wallclock and SIR reset/ESS diagnostics missing. | predeclared, no data | Needs fresh collision-checked seeds, a resource pilot and a fresh absolute stop before any draw. N4's 2150M raw and its fixed-K null stay unchanged; this is a new protocol, not a rerun. |

| N19 | Does matching the card-holder marginals already pin the decision? Bound the action-value difference over the whole marginal-consistent polytope, with no family restriction, and certify the sign exactly. | **complete, verified** | **Zero of 224 decision-relevant contexts admit a marginal-consistent belief that strictly prefers the inferior action**; 215 are strictly identified and 9 reach exact indifference, where the tie rule decides (bounded at 48/77, 910/317, 12/47, 66/205). Exact LP bounds with exact rational vertex certificates; no floating LP status accepted as a certificate. Independently written verifier rebuilds the constraints, re-checks all 448 endpoint certificates, and confirms N17's projection lies inside every interval. Explains N17 and M19: the product family's KL error lives in directions the decision does not read. |

## Seed namespaces

- N17 and N19 reserve **none**: it is exact deterministic arithmetic over already-frozen
  round12 arrays, with no RNG anywhere, so no collision check is required.
- N18 will reserve fresh bases and must pass a collision check against every
  prior consumed namespace before it draws anything. Nothing is reserved yet.

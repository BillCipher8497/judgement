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

**2026-11-21T02:45Z** — N17 complete and separately verified. N18 predeclared,
with no data or seeds. N14 (round12) finished in its final window: primary gate
failed, zero games, a scope failure rather than a zero effect.

| id | task | status | notes |
|---|---|---|---|
| N17 | Decision-relevant misspecification: stratify exact contexts by true action-value spread, then measure regret of the exact conditional-product projection where actions genuinely differ. | **complete, verified** | Projection regret exactly 0 on all 224 S+ contexts, all four conditions, all three spread bands. Uniform-belief control positive on the same keys in three conditions (100/317, 4/47, 11/205), so the measurement is not vacuous; S2R4n2_t0 has no power and its zero is reported as uninformative. Second independent reader revalidates all 1375 keys; choice and zero regret stable under tighter refit and theta rescaled 0.1x-10x. Exact enumeration, no sampling CI, no RNG/seeds/games. Answers the selection objection to the paper's thesis. |
| N18 | Equal-wallclock belief-in-search, closing the hole Claude's audit left open in N4: fixed-K only, with equal-wallclock and SIR reset/ESS diagnostics missing. | predeclared, no data | Needs fresh collision-checked seeds, a resource pilot and a fresh absolute stop before any draw. N4's 2150M raw and its fixed-K null stay unchanged; this is a new protocol, not a rerun. |

## Seed namespaces

- N17 reserves **none**: it is exact deterministic arithmetic over already-frozen
  round12 arrays, with no RNG anywhere, so no collision check is required.
- N18 will reserve fresh bases and must pass a collision check against every
  prior consumed namespace before it draws anything. Nothing is reserved yet.

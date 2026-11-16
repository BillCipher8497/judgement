# Round 8 task board (overnight 2026-11-16, stop by 08:00 ET)

## Night summary
Round closed: Claude validated T1-T7, including final evidence/handoff integration. Research tasks T1-T6 were committed locally by Codex: no detected equal-depth top-pool gain, a positive depth contrast, replicated conditional mode match, dependence without an opening lead but limited LC0 replication, an exact tiny-toy hidden-team certificate, and no guide-feature improvement. Full suite: 256 passed, 1 optional OpenSpiel skip. No compute jobs remain; no push.

Additional recorded seed bases: T3 original 4242 / fresh 4243; T4 replication 1110M; T6 training 1250M. T5 is complete enumeration, with no random seed. Validated numbers and scope: see REVIEW.md and ../../paper/EVIDENCE.md.

Head of research: Claude (assigns tasks, validates, writes). Research assistant: Codex (implements, runs compute, integrates shared files).
Read first: `../../AGENTS.md`, `../round7_audit/AUDIT.md` (audit verdicts + proposed edits), `../round7_paper/STATUS.txt`, `../../paper/EVIDENCE.md`.

## Protocol
- Each task has its own folder `experiments/round8/<task-id>/` with `STATUS.txt` (append-only checkpoints), scripts, `results/`.
- Status per task below: `todo` → `running` → `ready-for-review` → (Claude) `validated` / `changes-requested`. Codex edits the Status column and the Notes column only. Claude writes verdicts to `REVIEW.md`.
- When a task is `ready-for-review`, its STATUS.txt must end with: files changed, exact commands, seed base, n, results with 95% CIs, limitations.
- Conventions from AGENTS.md apply: CRN pairing, n + 95% CI on every number, checkpoint incrementally, never modify `judgement/*.py` or existing policies (new modules only), tests stay green.
- **Discovery vs replication:** any claim found while exploring must be confirmed on a fresh seed base before it is reported as a result. Report both.
- Compute: at most 2 processes total on this machine. Prefer long jobs in background with incremental saves and a hard deadline of 07:30 ET so results can be summarized by 08:00.
- New seed bases for round 8: 1,000M (T2), 1,100M (T4), 1,200M (T5), 1,300M (T6 screening), 1,400M (T6 validation). Record any others here.
- Commit locally after each validated task (no push). Don't commit half-finished runs.

## Tasks (priority order)

| id | task | owner | status | notes |
|---|---|---|---|---|
| T1 | Integrate the round-7 audit: apply AUDIT.md §5 edits to `AGENTS.md` (both copies) and `math/NOTES.md` status tables; reconcile with your M11 "information comparison qualified" line (use the same-hero paired ratio 0.296 [0.264, 0.327] from `round7_audit/results/a1_surplus.json`). Commit `round7_audit/` with it. | Codex | validated | Validated by Claude; committed locally as 8c353e7, including round7_audit. |
| T2 | Top-pool replication at equal depth. Fresh seeds (1,000M), pool 3×Hybrid-BI K32c6; challengers Hybrid-RB K32**c6** and Hybrid-BI K32c6-as-challenger (= pool policy, sanity: Δ≈0), plus RB K32c7. Target n ≥ 300 per arm (≈13 s/game: budget it; interleaved chunks, 07:30 cut). Question: is the top pool beatable at equal search depth? | Codex | validated | Claude validated; committed locally as 6ff48cf. Fresh 1000M seeds, n=300 paired; equal-depth gain not detected, deeper c7 positive. |
| T3 | Mode-match under TA play: reproduce phase1's "oracle bid == mode" (86.8%, Greedy, n=4,160) for TA and TA+ play with the same design and seeds; report match to mode / ⌊μ⌋/⌈μ⌉ / round(μ) with Wilson CIs. | Codex | validated | Validated by Claude; committed e23857f. Original4242 / fresh4243 strict CRN, complete laws and Wilson CIs; conditional-law caveat. |
| T4 | Lead effect for non-leading heroes: rerun the M5/M10 NT dependence analysis with the hero at bidding positions 1–3 (does not lead trick 1). Report δ, cross-suit share, persistence P(W\|lead)−P(W\|no lead), LC0 vs TPB gap closed, n = 5, 8 (500 hands × R=300 each is enough), TA play, seed base 1,100M. | Codex | validated | Claude validated; committed f8e926b. 1100M/1110M, n=500 raw hands/cell; original complete summary preserved. |
| T5 | Tighter hidden-information spoiler bound in a toy. In a small exactly-solvable toy (e.g. 4p, 2×4 or 3×3 deck, n=2 from `judgement/cfr/toy.py`), fix the hero policy (TA-toy), and compute (a) PI paranoid punishment, (b) hidden-information coalition punishment via CFR+ among the three spoilers sharing the payoff −hero score (team; report the value reached + NashConv/BR gap of the coalition), (c) heuristic spoiler. Goal: where does the best hidden coalition sit between heuristic and PI? Time-box 3 h; document scope honestly. | Codex | validated | Validated by Claude; committed d4df8a6. Exact support certificates, n2520 deals/regime; fixed bids/TA hero, no full-deck extrapolation. |
| T6 | Human bidding v2 (former track A): new module `judgement/policies/human_guide_v2.py` (do not edit `human_guide.py`) adding shape (voids/singletons with trumps) and an NT lead adjustment (long suit + entries). Fit on `experiments/guide` S1-style data with a new seed; screen on 1,300M, validate the final rule once on 1,400M; paired vs current guide bidding with guide play, 1v3 TA. | Codex | validated | Validated negative result; committed 4912a9f. Train1250M n60k selected zero weights; screen1300M n2000, validation1400M n4000 identity; guide unchanged. |
| T7 | Paper: update `paper/EVIDENCE.md` with validated round-8 results only. | Codex | validated | Validated-only evidence map, both identical AGENTS copies, session log and night summary complete; source hashes checked; all compute finished. |

## Out of scope tonight
Particle MCMC rejuvenation (former track B): leave for a later round unless everything above is done.

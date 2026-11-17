# AGENTS.md — Judgement (Oh Hell) research project

Read this file first. It is the hand-off point for any agent (Claude Code, Cowork, or other) picking up this project. **Keep it current: at the end of every work session, update "Current status", "Results so far" and "Next steps", and add a row to the session log.**

Last updated: 2026-11-16 (validated publication snapshot)

## What this project is
Game-theory research on the 4-player trick-taking card game **Judgement** (Oh Hell / Kachufool). Owner: Arvin.

Thesis: Judgement is a general-sum game that looks zero-sum. The work quantifies that (cooperation surplus, hook tax, under-bidding), backs it with proofs and exact solves, then builds bots and a human strategy guide.

Living report: https://claude.ai/artifact/Gv6NtDpvDr3QUE7SgjkJ76 (Claude Doc "Judgement Research — Interim Report"). The claude.ai project "Judgement Research" also holds `claude/research-log.md`, a running log.

## Rules (confirmed by Arvin, 2026-11-12; engine verified to match)
- 4 players (engine supports 3–7). Standard 52-card deck.
- Ladder: 13, 12, …, 1, …, 13 cards per player (25 rounds).
- Trump rotates by round: Spades, Hearts, Diamonds, Clubs, No-Trump, repeat.
- The deal passes to the left. The player left of the dealer bids first, and the dealer bids last.
- **Hook rule:** the dealer may not bid a number that makes total bids equal the cards in hand.
- The player left of the dealer leads the first trick, and each trick's winner leads the next.
- Must follow suit. A player who is void may play any card; there is no obligation to trump.
- Scoring: exact hit scores 10 + bid; a miss scores 0. The engine also supports penalty variants.
- After 25 rounds the highest total wins. **A tie for first counts as a win.** Note that the engine's `GameResult.wins` still splits ties 1/k, so recompute ties-as-win from the totals.

## Layout (inside `engine/`)
| Path | What |
|---|---|
| `judgement/` | Core package: cards, rules, state, engine, oracle, sampling, trace |
| `judgement/policies/` | Players: `greedy`, `target_aware` (TA), `ta_plus` (`ta_plus_v2()` = TA+), `ta_plus3` (TA+3, best heuristic), `hybrid` (TA+ plus PIMC late-trick search), `hybrid_bi` (bid-weighted worlds), `hybrid_rb` / `hybrid_rb2` (learns each opponent's bidding online; current best bot), `spoiler`, `human_guide`; experimental `human_guide_v2` (zero fitted adjustments; no improvement) |
| `judgement/bidding/` | Fitted bidder model (`ModelBidder`, model JSONs in `experiments/phase1/results/`) |
| `judgement/bidding2/` | Bid-informed sampler and likelihood |
| `judgement/solver/` | Exact perfect-information solver (max^n, paranoid, social), PIMC |
| `judgement/cfr/` | Toy-game CFR+ and exact best response / NashConv |
| `math/` | `NOTES.md` (proofs, claims and their status), `STATUS.txt`, scripts `m1_*` to `m10_*`, `results/` |
| `experiments/<round>/` | One folder per round, each with `STATUS.txt` (append-only checkpoints) and `results/` |
| `experiments/guide/GUIDE.txt` | Human strategy guide (bidding count + play checklist) |
| `tests/` | pytest + hypothesis (256 passed, 1 optional OpenSpiel skip on 2026-11-16) |
| `crossval/`, `bench/` | OpenSpiel cross-validation, benchmarks |

Experiment folders, in order: `phase1`, `phase2`, `phase3`, `phase4`, `play`, `final`, `round2_*` (superseded), `round3_play`, `surplus_hidden`, `round4`, `round5_bots`, `guide`, `round6`, `round7_paper`, `round7_audit`, `round8`.

## How to run
```bash
cd engine
pip install -r requirements.txt            # numpy, pytest, hypothesis; open_spiel optional
python -m pytest -q -p no:cacheprovider    # must stay green
```
Full-game 1-vs-3 comparisons use the runner in the latest round folder, e.g. `experiments/round6/run.py`. Job queues are `queue*.sh` with `jobs*.txt` (claims in `results/claims/`).

## Working conventions (keep these)
1. **Common random numbers.** Compare policies on the same deals and the same seeds, with the challenger in seat g%4, and report paired differences. Record the seed bases. In use so far: 71M (vs TA), 400M (vs Hybrid), 500M (round 4 pools), 600M (vs Hybrid-BI), 700M (mixed), 900M (guide-bidding pool).
2. **Every number gets n and a 95% CI.** Don't report a point estimate alone.
3. **Checkpoint as you go.** Append to the round's `STATUS.txt` after each task and save raw results (`.npz`/`.json`) incrementally. Past runs were cut off by usage limits, and the checkpoints saved them.
4. **Don't modify core files** (`judgement/*.py` and existing policies). Add new modules instead and keep all tests passing.
5. Use at most 2 CPU processes (the cloud sandbox has 2 cores).
6. Proofs and claims go in `math/NOTES.md` with a status: proved, verified numerically, or refuted.

## Results so far (headline numbers; details in the report and STATUS files)
- **Round 7 welfare (new):** proved W = 40+n−10M−U, where M is the number who miss and U their tricks; the hook gives the sharp global bound W≤30+n. A fixed-deal, fixed-bid legal example also changes total score through play alone.
- **Punishment destroys score (new):** in 4,000 paired H1 games, three spoilers reduce hero score by 51.661 ± 1.105 and each spoiler's mean score by 40.015 ± 0.605, for table loss 171.708 ± 1.985 points/game (95% CIs). Additional missed bonuses account for 154.425 ± 1.628; tricks held by missers account for 17.282 ± 0.440.
- **Historical M11 cross-policy comparison (descriptive only):** at five cards with equal S/NT mixture, hidden surplus share is 0.22984 ± 0.02751 (4,000 deals), versus PI 0.81095 ± 0.01465 (1,000 deal clusters); ratio 0.28342 ± 0.03431. Policies still differ, so this is descriptive, not a causal information estimate. Prefer the audit’s same-hero paired ratio 0.296 [0.264, 0.327] (4,000 deals) for the information contrast; neither establishes optimal hidden punishment.
- **H1 wins corrected (new):** legacy H1 summaries used split ties. Ties-as-win from totals gives baseline 0.24600 ± 0.01335, s3 0.16975 ± 0.01164, paired change −0.07625 ± 0.01697 (4,000 games). Score-surplus findings are unchanged.
- **Bidding:** Greedy oracle = mode: 86.8% [85.8, 87.8] (n=4,160; Wilson 95% CI), reproduced distribution-by-distribution. TA 94.6% [93.9, 95.3] and TA+ 95.6% [95.0, 96.2], each n=4,160; fresh replication TA 94.7% [94.0, 95.3], TA+ 95.6% [95.0, 96.2] (same n). For TA/TA+, mode means the law at the oracle-selected bid: a conditional diagnostic, not a bid-free recommendation. Under independent-card PB laws the unconstrained optimum is the mode or one above and lies in {floor μ, ceil μ}; hook and dependent laws require separate treatment.
- **Lead dependence:** TA, opening leader: LC1 closes 0.98 [0.94, 1.02] of the PB-to-floor bid gap at n=5 (1,000 hands), LCX 0.82 [0.75, 0.89] at n=8 (1,000), 0.30 [0.23, 0.36] at n=13 (996). Per-hand rollout diagnostics, not usable bidders; flexible no-lead LCU also reaches the floor. At n=5 parsimonious LC0 closes 0.58 [0.49, 0.65] versus TPB 0.04 [0.02, 0.07] (1,000 hands).
- **Hook cost:** PI rational-bid dealer costs at n=1..6: 10.17 ± 0.02, 9.64 ± 0.11, 8.92 ± 0.16, 8.21 ± 0.24, 7.82 ± 0.37, 6.96 ± 0.99 points/round (2,000/2,000/2,000/1,200/600/90 deals; last interval see phase2 F3). Hidden TA: 5.41 ± 0.11 at n=1 (10,000 rounds), 0.27 ± 0.06 at n=13 (20,000). PI non-dealers also pay; policy and information differ across regimes. Exact one-card threshold is 10/21 when both bids are legal.
- **Cooperation surplus:** Same TA+ hero, play-only punishment: heuristic-hidden/omniscient surplus ratio 0.296 [0.264, 0.327] at n=5 (4,000 paired deals); PIMC ratio 0.356 [0.313, 0.398] (2,000 deals). Whether a stronger hidden-information coalition can do better remains open. Full-game heuristic surplus share 0.311 [0.306, 0.317] (4,000 games).
- **Play rules:** M9 winner/loser exchange proofs require same-suit pairs and open-loop opponents (Assumption O); empirical checks used TA opponents. Any-suit TA+3 extension is heuristic.
- **Bots vs 3×TA:** TA+3 +7.98 ± 0.43 (10,000 games), Hybrid c5 +17.07 ± 1.32 (1,500), Hybrid-BI +20.74 ± 1.62 (1,000). BI is not separated from Hybrid+oracle or Hybrid K32 on matched TA games; its advantage over plain Hybrid in the Hybrid pool is +5.31 ± 1.59 (1,000).
- **Robustness:** RB−Hybrid: model +6.51 ± 1.81 (800 games), noisy +2.45 ± 2.28 (500), honour +2.72 ± 2.14 (500), mixed +3.58 ± 1.83 (800). These pools mostly overlap its type library. Out-of-library guide +2.37 ± 2.28 (500). Top pool (3×Hybrid-BI K32c6), fresh 1,000M seeds, n=300 paired: RB c6 −0.74 ± 2.96 points/game (no detected equal-depth gain); RB c7 +4.64 ± 3.23; c7−c6 +5.38 ± 3.12. Depth contrast does not establish opponent-modelling benefit or a win-rate gain.
- **Toy-game equilibria:** Exact BR checks verify small NashConv in the finite toys (not a multiplayer CFR convergence guarantee). Material payoff non-uniqueness appears in 3p/4p NT; see phase4 exact tables.
- **Human guide:** Retains 0.808 [0.794, 0.822] of TA+3’s edge over greedy (6,000 games, TA pool), but scores −2.01 ± 0.78 points/game versus TA. Bidding is the larger loss. Final defaults were re-evaluated on selection seeds; human playtesting is outstanding. Round 8: none of 143 nonzero shape/NT-lead weightings beat the current rule on 60,000 training situations (best selected difference −0.002 ± 0.006 points/round; descriptive, selection-biased). Zero weights were frozen; screen n=2,000 and fresh validation n=4,000 give paired Δ=0 ± 0 by identity. No improvement promoted; other feature encodings remain open.

- **Non-leading NT heroes (round 8):** dependence persists; replication n=5 pos1 δ=1.575 ± 0.030 and lead persistence 0.237 ± 0.013 (n=500 hands). LC0 gap closed: discovery pos1 0.314 [0.153, 0.455], pos2 0.295 [0.117, 0.442] (n=499 each), replication 0.426 [0.271, 0.561] and 0.286 [0.064, 0.458] (n=500 each). Dealer effect fails replication; no n=8 LC0 interval excludes zero. Per-hand rollout diagnostics; comparison with the opening leader is descriptive across designs.
- **Hidden-team toy certificate (round 8):** an 8-card, 4-player, n=2 toy with fixed TA hero and myopic bids has feasible private-infoset spoiler strategies attaining PI punishment exactly on every deal. Mean hero scores are 1257/140 (trump) and 946/105 (NT), joint gap exactly zero; complete enumeration of 2,520 deals/regime, no sampling CI. Proved support-certificate lemma in M12. This toy does not bound the 52-card information gap.

## Current status
- Public repository: curated validated snapshot through round8 plus separately self-validated round9 D1,D2,D4 and document/protocol artifacts from round10. Original full research history remains local.
- D1/D2 raw outcomes and fresh audit scripts are included; D4 includes a visually checked one-page PDF. Self-review is not independent external validation; strong D1 causal paper claims remain gated.
- Working manuscript and bounded primary-source literature draft are included. No novelty claim or fitted new NT model is made. Future protocol seed reservations are unused.
- Unreleased work is retained in a local review logbook. Do not assume an omitted experiment or theorem has been validated.
- Latest integrated local suite before publication selection:262 passed,1 optional OpenSpiel skip. This snapshot contains a subset of those tests; no separate full publication-suite rerun is claimed.

## Next steps
1. Reproduce the included audits and broaden independent validation and literature review.
2. Extend the local manuscript while distinguishing implemented policies from optimal coalition values; no strong D1 causal claims before external review.
3. Review the unexecuted NT-bidder protocol before implementation or seed use. Keep core/existing policies unchanged and use paired game-level inference.
4. Consult the local review logbook before releasing withheld work. Push only validated snapshots, never the full private research branch without review.

## Session log
| Date (ET) | Session | What happened |
|---|---|---|
| 2026-11-10 | Phase 0–2 | Engine (cross-validated against OpenSpiel), target-aware play, exact solver, cooperation surplus |
| 2026-11-10 | Phase 3 | Math layer round 1, strategic bidding, win-maximizer |
| 2026-11-11 | Overnight | TA+/Hybrid tournament, math round 2, Phase 4 CFR/exploitability, interim report |
| 2026-11-14 | Overnight | Rules confirmed; hidden-info surplus; Hybrid-BI; lead-state model; fixed bid model shown fragile |
| 2026-11-15 | Overnight | Hybrid-RB (learns opponents' bidding), human strategy guide, RB2, top-of-ladder update; AGENTS.md created |
| 2026-11-15 | Round 7 paper and welfare | Restored missing snapshot files and initialized local Git; paper outline/evidence map; M11 welfare identity, sharp hook ceiling and paired table-loss decomposition; standardized PI/hidden comparison and corrected H1 ties-as-win; coordination prompt for external agents. |
| 2026-11-16 | Round 8 overnight (Claude head, Codex compute/integration) | Integrated audit; equal-depth top-pool null and positive depth contrast; mode-match replication with conditional-law caveat; non-leading NT dependence and limited LC0 replication; exact fixed-bid toy hidden-team certificate; negative guide-feature grid. All research results Claude-validated; 256 tests passed, 1 optional skip; raw data and local task commits saved. |

| 2026-11-16 | Validated publication | Curated owner-authorized GitHub snapshot; withheld unreviewed proof/experiment content and private history; source-selection checks pass. |

## External-audit release,2026-11-16
D1,D2 and M13/M14 proofs/lower bounds externally validated by Claude; full report is CLAUDE_AUDIT_2026-11-16.md. Fresh full local suite262passed/1optional skip, exit0,49.07s pytest,53.91s startup-inclusive output wall interval. D3/M13 and round10/R3 are now released. Feasible-upper independent replay remains E5; D5 remains withheld. M13 gaps are about0.070 and0.014 hero points: strict information cost only, no material-magnitude or52-card implication. The public manuscript/ledger reconciliation is tracked as E4.

# Judgement paper evidence map

This map connects the outline to saved evidence. Empirical ± values are 95% intervals; counts refer to independent units used in the interval. Exact algebra and complete enumerations do not require sampling intervals.

Local manuscript artifacts: `MANUSCRIPT_DRAFT.md` has an11-group quantitative source ledger under `experiments/round10/R0/`; `RELATED_WORK_DRAFT.md` has a bounded primary-source map under R1. These remain working drafts with broader literature/novelty review outstanding. Claude externally validated D1/D2 and M13/M14 on2026-11-16; M14 is a fixed-hero pure-team clarification, not a new observed card-game value. R4 execution and any resource shortfall are tracked in the local round11/E3 packet; no unexecuted or unvalidated stage supplies performance evidence. E0 freshly passed262tests/1optional OpenSpiel skip (49.07s pytest,53.910273s startup-inclusive output interval).

| Claim | Evidence and uncertainty | Source | Scope |
|---|---|---|---|
| Score is general-sum even with fixed bids | Exact two-card engine witness: same deal and bids produce table scores 30 and 31 under different legal discards | `tests/test_round7_welfare.py::test_fixed_bids_play_changes_total_score`; M11 in `math/NOTES.md` | Standard exact-hit score; distinguish final victory utility |
| Welfare identity and hook ceiling | W = 40 + n − 10M − U; W ≤ 30 + n under hook; sharp legal construction | `math/m11_welfare.py`; M11 proof; engine witnesses in tests | Zero miss score, hit base ten, four players |
| Punishment destroys table score | Three spoilers: 171.708 ± 1.985 points/game lost; hero 51.661 ± 1.105, each opponent mean 40.015 ± 0.605; n = 4,000 games | `experiments/round7_paper/results/welfare.json`, `paired_games.npz` | H1 TA+ hero and fixed fitted bidder; hostile policies specified |
| Decomposition of table loss | Missed-bonus component 154.425 ± 1.628, missed-trick component 17.282 ± 0.440 points/game; n = 4,000 | Same M11 files | Exact paired accounting; full games are CI units |
| Hidden/perfect information surplus contrast | Equal S/NT at five cards: hidden share 0.22984 ± 0.02751 (n = 4,000), PI share 0.81095 ± 0.01465 (n = 1,000), ratio 0.28342 ± 0.03431 | `welfare.json` field `information_comparison`; H1 round NPZ and phase2 seat CSV | Descriptive ratio, different hero and bidding regimes; PI seats clustered within deal |
| Information constrains punishments | Common five-card subset: hidden PIMC spoiler hero score minus heuristic −0.28 ± 0.32; omniscient minus PIMC −3.44 ± 0.32 points/round; n = 1,000 deals | `experiments/surplus_hidden/results/summary_strong.md`, raw strong-spoiler NPZs | Fixed hidden-information hero; computational methods differ; no proof of hidden coalition optimality |
| Exact hidden-information obstruction | Same opponent information set requires incompatible spoiling discards in two deals | `tests/test_round3_math_h3.py`; H3 in `math/NOTES.md` | Counterexample, not a population-average information effect |
| Hidden hook tax declines with hand size | One card 5.41 ± 0.11 (n = 10,000 rounds), thirteen cards 0.27 ± 0.06 (n = 20,000 rounds) | `experiments/phase1/results/phase1_tables.md`, hook table | Same seeded hook/no-hook experiment; fitted bidder |
| PI dealer hook cost remains large | One card 10.17 ± 0.02 (n = 2,000 deals), five cards 7.82 ± 0.37 (n = 600 deals) | `experiments/phase2/results/summary_main.md`, F3 by position | Rational PI bids and max^n; different regime from hidden hook experiment |
| Lead dependence explains short NT diagnostic gap | Five-card TA cross-fit bid match: PB 0.739 ± 0.027, LC1 0.978 ± 0.009, split-half reference 0.982 ± 0.008; n = 1,000 hands | `math/results/m10_tables.txt`, `m10_summary.json`, `m10_raw_n5_NT.npz` | Per-hand rollout fit; first bidder and leader; not a deployable hand-feature model |
| Robust opponent bidding inference helps score | RB−Hybrid: model +6.51 ± 1.81 (n = 800), noisy +2.45 ± 2.28 (n = 500), honour +2.72 ± 2.14 (n = 500), mixed +3.58 ± 1.83 (n = 800), points/game | `experiments/round5_bots/results/r5.md`, raw pool NPZs | Named pools, paired seeds; no universal superiority claim |
| RB2 score advantage is unresolved | RB2−RB: guide +0.37 ± 2.20 (n = 500), mixed +0.29 ± 1.66 (n = 800), model −0.49 ± 1.66 (n = 800), points/game | `experiments/round6/results/r6.md` | Better predictive likelihood does not establish better score |
| Human guide retains much of heuristic edge | 0.808 [0.794, 0.822], n = 6,000 games against TA pool | `experiments/guide/results/s3_summary.txt`, final defaults in GUIDE.txt | Ratio relative to greedy; bot evaluation, no human playtest |

## Reproduction

From `engine/`, run `python math/m11_welfare.py`. It validates the saved H1 outcomes, enumerates all hook-legal bid/trick count pairs through six cards, and writes paired components, summary intervals and input hashes to `experiments/round7_paper/results/`.

Run `python -m pytest -q -p no:cacheprovider` for engine and research checks. OpenSpiel cross-validation tests skip when its optional dependency is absent; record that skip explicitly.

M11 uses no new random seeds. Existing H1 round seeds are `SeedSequence([131000000, n, d])`; games are `133000000 + g`. The PI comparison uses scheme R rows at five cards with seeds `SeedSequence([20261110, 5, trump_code, deal])`, as documented in `experiments/phase2/exp_main.py`. The S/NT standardization retains H1 indices d mod 5 in {0, 4}; each trump state contributes equally.

## Corrections and unresolved evidence

The legacy H1 game summary reads split-tie wins from the saved `wins` array. M11 recomputes the requested ties-as-win metric from totals: baseline 0.24600 ± 0.01335, three-spoiler 0.16975 ± 0.01164, paired difference −0.07625 ± 0.01697, n = 4,000 games. Keep the historical result files unchanged and use these corrected values in new prose.

The original PI/hidden comparison mixed all five trump states on the hidden side against only S/NT on the PI side. The standardized M11 ratio corrects that mixture difference. It cannot correct the hero/bid-policy difference without another experiment.

The living Claude report was not accessible during this session. The local snapshot and experiment logs provide the evidence above; reconciliation with that report and literature review remain manuscript tasks.

## Round 9 — separate self-review and completed external-audit updates

Codex is head of research. The original separate reviews in `experiments/round9/REVIEW.md` were by the same investigator; Claude subsequently independently recomputed D1/D2 and rebuilt the complete D3 lower-bound enumeration, and externally validated M13/M14 (root CLAUDE_AUDIT_2026-11-16.md). D1 wording is confined to its audited intervention. D4 retains its original self-review label. Fresh feasible-upper forward replay has now passed separately in E5, self-validated (Codex). D5 remains outside this evidence map/publication scope.

### D1: implemented information-intervention diagnostic

Fixed TA+ hero, fixed ModelBidder bids, five cards, equal S/NT mixture, dealer-balanced, hero seat0 as in the a1 design. All arms use the same seeded deal and per-seat policy RNG streams. The preregistered primary is revealed-hero minus own/public-information surplus share. Original paired-deal bootstrap95% intervals:

| Stage | Seed base | Paired deals | Primary share contrast |
|---|---|---|---|
| Discovery | 1,600M | 1,000 | 0.369315 [0.315244,0.424218] |
| Fresh replication | 1,610M | 1,000 | 0.341204 [0.289874,0.394787] |

Sources: `round9/D1/results/summary.json`, all2,000 raw five-arm files; independent legality/scoring/deal audit and bootstrap in `round9/reviews/audit_d1.py`, `results/D1_raw_audit.json`. All-information PIMC and omniscient reference trajectories match on every deal. Own/hero/all differ only in the world sampler, with K16 per active decision, identical solver/fallback/tie-breaking. Active-decision counts and total compute are endogenous. Own information includes public bids/history/voids; sampling omits bid/play likelihoods. The omniscient paranoid reference internally assumes a maximizing hero and is not a certified optimum against the actual fixed TA+ hero. These are policy-specific round diagnostics, not optimal hidden punishment or a full-game/full-deck theorem. Claude independently recomputed and externally validated the intervention. Causal wording is limited to revealing the hero hand to this particular K16 sampler at n5 with fixed TA+v2 hero/bids; it does not generalize to other heroes, optimal coalitions, the full 25-round ladder or humans. K is fixed per active decision, not total wall time.

### D2: replicated guide bidder score decomposition

Guide play is held fixed in both arms against3×ModelBidder/TA, challenger seatg%4, full25-round games paired by seed. ModelBidder−guide bidding gives **5.29425 ±0.82665 points/game** at1,500M discovery and **5.05950 ±0.81963** at1,510M fresh replication, each n=4,000 paired games,95% CIs. Ties-as-win differences from totals are **0.04975 ±0.01510** and **0.03375 ±0.01533**, same n/coverage. Intervals use full-game clusters within rotated-seat strata; repeated rounds stay together.

The discovery-frozen cells are n2/trump/position2, n7/trump/position3(dealer), n10/trump/position1. Their replication loss contributions per full game are respectively **0.23875 ±0.07760**, **0.15175 ±0.11261**, **0.13625 ±0.12954**, each n=4,000 game clusters (2,000 contributing rounds), nominal95% CIs. Selected discovery intervals are descriptive. A post-review Bonferroni familywise95% sensitivity retains the first two positive cells but includes zero for n10; do not claim three familywise-confirmed cells or unchanged top-three ranking. No new bidding feature is validated.

Sources: `round9/D2/results/raw_{1500000000,1510000000}.jsonl`, `summary.json`, `frozen_top3.json`; independent all-cell/context/deal audit `round9/reviews/audit_d2.py`, `results/D2_raw_audit.json`, `D2_multiplicity.json`. ModelBidder is a comparator, not ground truth; guide−model bid difference is not oracle error. Whole-bidder replacement includes later bid and target-aware play responses. The historical overall guide deficit versus TA must not be called its isolated bidding cost. No human playtest or feature retuning occurred.

### D3: larger fixed toy, exact optimization bounds

Four players,3 suits×4 ranks, three cards each, no stock; fixed myopic bids and original toy TA hero on every counterfactual branch. All369,600 ordered deals per regime are enumerated. Spoilers retain separate own-hand/public-history information sets and optimize play only. Complete-population values have no sampling95% CI.

| Regime | Baseline hero | Composed independent BR | PI lower | J1 private-root lower | Exact feasible upper |
|---|---|---|---|---|---|
| Trump | 8.7291991342 | 7.5562472944 | 367921/61600 | 446669/73920 | 574247/92400 |
| NT | 10.0370725108 | 7.8805194805 | 106419/15400 | 213259/30800 | 109709/15400 |

Each row has n=369,600 full-population deals. The exact hidden-team optimum lies in **[446669/73920,574247/92400]** (trump) and **[213259/30800,109709/15400]** (NT), with M13's proof/lower bounds externally validated by Claude; the feasible uppers are separately confirmed by E5 fresh forward replay, self-validated (Codex). These are optimization bounds, not confidence intervals. Integer J1−PI gaps are25819/369600 ≈ 0.070 and421/30800 ≈ 0.014 hero points. They certify a strict information cost and nothing about its magnitude; they neither establish a material effect nor estimate the actual optimal hidden-team cost minus PI. M13 constrains the first free spoiler action by private information, then grants omniscience; no earlier free action changes root reach. The proof is **proved and externally validated (Claude,2026-11-16)**. No CFR convergence or solved hidden optimum is claimed. Previously evaluated one-hot checkpoint50 policies supply integer-replayed feasible uppers; later review performed no extra training. The original3-hour box includes infrastructure failures/pauses; the extra trump run completed120 updates with best selected at50; main NT final completed count is unknown.

Sources: `round9/D3/results/summary_final.json`, saved profiles and `review_pure_trump_*.npz`, `D3/ROOT_BOUND.md`, M13; independent grouped recurrences/BR/manual-rule audit `round9/reviews/audit_d3.py`, integer upper audit `exact_policy_d3.py` and their result packets. Review reused source-fingerprinted derived construction caches, checked16 manual game trees, and did not independently rebuild the whole population. Claude subsequently independently reconstructed all369600deals/regime and reproduced exact PI/J1 values; Claude checked only the feasible uppers' ordering, not their replay. E5 subsequently completed that distinct numerical check: all369600deals/regime, exact integer score vectors reproduce574247/92400 and109709/15400, with portable private policies. This is same-investigator self-validation, not a new Claude upper-replay verdict. Baseline support completion and private BR caveats are in D3/README.md. Neither this toy nor its positive bound quantifies the52-card information gap.

### D4: source-faithful printable card

`paper/guide_card.md` condenses GUIDE.txt into bidding count/hook advice and eight ordered play rules. The supplied `round9/D4/results/guide_card.pdf` is one A4 page, with exact source-word-order and rendered visual checks recorded in `render_audit.json`. No strategy/performance result or human playtest is added; inherited source edge cases are not claimed to be resolved. Arbitrary Markdown renderers may paginate differently.

## Round 8 — Claude-validated additions (2026-11-16)

Only results validated in `experiments/round8/REVIEW.md` enter this section. The raw files, exact commands, seed design and limitations are in each task's append-only STATUS.txt. All ± and bracketed intervals below are 95%; exact complete enumerations have no sampling CI.

### Audit correction: prefer the same-hero information comparison

At five cards, the heuristic-hidden/omniscient surplus ratio is **0.296 [0.264, 0.327]**, n=4,000 paired deals with the same TA+ hero and fixed bids. The hidden PIMC/omniscient ratio is **0.356 [0.313, 0.398]**, n=2,000 paired deals. Source: `experiments/round7_audit/results/a1_surplus.json`. These supersede the cross-policy M11 ratio as the preferred information contrast. Neither computes the optimum hidden coalition; keep the old M11 ratio above as descriptive history only.

### T2: equal-depth top-pool replication

Pool: 3×Hybrid-BI K32c6. Fresh seed base 1,000,000,000; challenger seat g%4; n=300 paired games per comparison. Game-level paired normal intervals:

| Challenger / contrast | Score difference, points/game | Ties-as-win probability difference |
|---|---|---|
| BI K32c6 sanity − pool | 0.00 ± 0.00 | 0.0000 ± 0.0000 |
| RB K32c6 − pool | −0.74 ± 2.96 | +0.0167 ± 0.0566 |
| RB K32c7 − pool | +4.64 ± 3.23 | +0.0333 ± 0.0569 |
| RB c7 − RB c6 | +5.38 ± 3.12 | +0.0167 ± 0.0589 |

Source: `experiments/round8/T2/results/summary.json` and 1,200 per-game JSON records. Equal-depth score improvement is not detected; the positive depth contrast does not establish an opponent-modelling benefit or a victory-probability improvement. The sanity arm is a deterministic plumbing check: all total vectors match; it does not calibrate independent Monte Carlo noise.

### T3: mode match under target-aware play

Both grids use 40 situations × 13 sizes × 2 trump types × 4 bid positions = **4,160 situations**, R=300 hidden worlds per legal bid. Every table entry is a percentage with a Wilson 95% CI, n=4,160 situations. Earlier bids use the original Greedy context design; later bidders use the policy's fitted model. The 4,242 grid reproduces every original Greedy distribution exactly. Its policy MC streams are independent; confirmatory 4,243 uses strictly common hidden worlds across Greedy, TA, TA+ and all candidate bids.

| Grid / play | Oracle = mode | = floor(mean) | = ceil(mean) | = round(mean) | = floor or ceil |
|---|---|---|---|---|---|
| 4242 / Greedy | 86.83 [85.77, 87.82] | 59.01 [57.51, 60.50] | 34.52 [33.09, 35.98] | 63.73 [62.25, 65.17] | 92.81 [91.99, 93.56] |
| 4242 / TA | 94.64 [93.91, 95.28] | 78.87 [77.60, 80.08] | 19.16 [17.99, 20.38] | 74.88 [73.54, 76.17] | 97.21 [96.67, 97.67] |
| 4242 / TA+ | 95.65 [94.99, 96.23] | 76.37 [75.06, 77.64] | 21.32 [20.10, 22.59] | 77.40 [76.11, 78.65] | 96.90 [96.33, 97.38] |
| 4243 / Greedy | 87.02 [85.96, 88.01] | 58.87 [57.37, 60.36] | 34.78 [33.35, 36.24] | 63.75 [62.28, 65.20] | 93.00 [92.19, 93.74] |
| 4243 / TA | 94.69 [93.96, 95.33] | 78.77 [77.51, 79.99] | 19.54 [18.37, 20.78] | 75.60 [74.27, 76.88] | 97.52 [97.01, 97.95] |
| 4243 / TA+ | 95.65 [94.99, 96.23] | 77.45 [76.16, 78.70] | 20.65 [19.45, 21.91] | 77.67 [76.38, 78.91] | 97.19 [96.64, 97.65] |

Source: `experiments/round8/T3/results/summary.json`, complete raw laws in `raw_4242.jsonl` and `raw_4243.jsonl`; diagnostic-only `unpaired_prefix_4243.jsonl` excluded. For TA/TA+, each bid changes play: the mode and mean are evaluated in **the law at the oracle-selected bid**, not a single exogenous trick law. Thus the high match is a conditional diagnostic and may be self-targeting; it is not an implementable rule to bid an unknown distribution's mode. Finite-rollout selection and argmax tie handling remain. These dependent laws do not inherit the independent-card PB theorem. Discovery has no stored hand/prior metadata; the fresh replication records and fully checks both.

### T4: dependence without the opening lead

NT, TA play, fixed prior/own bids and hook-legal diagnostic bids. Discovery base 1,100M; independent replication 1,110M. Each cell has 500 raw hands × R=300 rollouts, split 150/150 for cross-fitting; degenerate hands are excluded from the affected estimates. δ = Var(total tricks)/PB variance; the cross-suit term is covariance contribution divided by PB variance, not a fraction of total excess dispersion. Persistence excludes the first trick. Parentheses give each estimate's actual hand count; gap CIs use 4,000 paired-hand bootstrap replicates.

| Base / cards / position | δ | Cross-suit dispersion | Lead persistence | LC0 gap closed | TPB gap closed | Gap n |
|---|---|---|---|---|---|---|
| 1100M / n5 / pos1 | 1.581 ± 0.032 (n=499) | 0.378 ± 0.021 (n=499) | 0.233 ± 0.014 (n=498) | 0.314 [0.153, 0.455] | 0.048 [-0.032, 0.126] | 499 |
| 1100M / n5 / pos2 | 1.579 ± 0.031 (n=499) | 0.369 ± 0.021 (n=499) | 0.231 ± 0.014 (n=498) | 0.295 [0.117, 0.442] | 0.054 [-0.033, 0.142] | 499 |
| 1100M / n5 / pos3 | 1.503 ± 0.029 (n=498) | 0.327 ± 0.019 (n=498) | 0.211 ± 0.013 (n=498) | 0.440 [0.132, 0.706] | 0.060 [-0.093, 0.209] | 498 |
| 1100M / n8 / pos1 | 1.569 ± 0.033 (n=500) | 0.362 ± 0.024 (n=500) | 0.297 ± 0.012 (n=500) | 0.119 [-0.024, 0.252] | -0.007 [-0.080, 0.061] | 500 |
| 1100M / n8 / pos2 | 1.495 ± 0.034 (n=500) | 0.300 ± 0.025 (n=500) | 0.275 ± 0.013 (n=500) | 0.163 [-0.037, 0.345] | -0.031 [-0.136, 0.066] | 500 |
| 1100M / n8 / pos3 | 1.379 ± 0.034 (n=500) | 0.218 ± 0.024 (n=500) | 0.255 ± 0.013 (n=500) | 0.188 [-0.015, 0.359] | -0.047 [-0.159, 0.051] | 500 |
| 1110M / n5 / pos1 | 1.575 ± 0.030 (n=500) | 0.384 ± 0.021 (n=500) | 0.237 ± 0.013 (n=500) | 0.426 [0.271, 0.561] | 0.083 [0.000, 0.164] | 500 |
| 1110M / n5 / pos2 | 1.580 ± 0.029 (n=500) | 0.381 ± 0.020 (n=500) | 0.225 ± 0.014 (n=500) | 0.286 [0.064, 0.458] | 0.055 [-0.060, 0.161] | 500 |
| 1110M / n5 / pos3 | 1.521 ± 0.033 (n=499) | 0.323 ± 0.018 (n=499) | 0.212 ± 0.014 (n=499) | 0.077 [-0.273, 0.333] | 0.038 [-0.098, 0.164] | 499 |
| 1110M / n8 / pos1 | 1.574 ± 0.035 (n=499) | 0.367 ± 0.024 (n=499) | 0.294 ± 0.012 (n=499) | 0.025 [-0.144, 0.171] | -0.108 [-0.207, -0.023] | 499 |
| 1110M / n8 / pos2 | 1.533 ± 0.034 (n=500) | 0.329 ± 0.024 (n=500) | 0.285 ± 0.013 (n=500) | 0.160 [-0.031, 0.319] | -0.047 [-0.167, 0.057] | 500 |
| 1110M / n8 / pos3 | 1.377 ± 0.032 (n=499) | 0.219 ± 0.024 (n=499) | 0.253 ± 0.014 (n=499) | 0.102 [-0.123, 0.286] | 0.000 [-0.108, 0.093] | 499 |

Source: `experiments/round8/T4/results/summary.json` and 6,000 per-hand NPZ files. Dependence persists without an opening lead. LC0's positive gap closure replicates at n=5 in positions 1 and 2; the dealer effect fails replication, and every n=8 LC0 interval includes zero. TPB has little explanatory power. Compared with M10's opening leader, these are descriptive differences across seeds and bidding contexts, not a paired leader intervention. These remain per-hand rollout diagnostics rather than deployable bidders.

### T5: exact hidden-team certificate in a small toy

Four players, 2 suits × 4 ranks, n=2; all myopic bids and TA hero play frozen. Each regime enumerates all **2,520 ordered deals**. Spoilers keep separate private information sets and optimize play only.

| Regime | TA baseline hero score | Independent-BR heuristic | Certified hidden team | PI minimum |
|---|---|---|---|---|
| Trump | 9.6024 | 8.9786 | 1257/140 | 1257/140 |
| NT | 9.9889 | 9.1032 | 946/105 | 946/105 |

All are full-population values, so no statistical CI applies. Exact integer support DP proves zero pointwise excess over the PI lower bound on every deal: the saved feasible hidden strategy has **joint optimality gap exactly zero**, in both regimes. This is stronger than the unilateral BR residual; multiplayer CFR convergence alone would not supply the certificate. Source: `experiments/round8/T5/results/certificate.json`, certified NPZ profiles, `T5/certify.py`, and M12 in `math/NOTES.md`. The independent-BR heuristic is the toy-specific construction, not the 52-card spoiler. This tiny fixed-bid play-only toy cannot bound the full-game hidden/PI ratio; a larger toy is a later-round question.

### T6: guide shape/lead feature grid gave no gain

Training seed 1,250M, n=60,000 S1-style true-deal counterfactual situations: none of 143 nonzero integer weight vectors beats the unchanged guide in this finite grid. The selected weights are (0,0,0,0). The best nonzero candidate's training difference is −0.002 ± 0.006 points/round (n=60,000); this is a selected, descriptive training estimate, not an independent replication of a feature effect.

The frozen zero-weight rule was screened at 1,300M (n=2,000 paired games) and validated unchanged at 1,400M (n=4,000). Both score and ties-as-win deltas are 0 ± 0 by identity; these games confirm implementation identity and supply no evidence for improved features. At validation, either guide scores 159.74 ± 0.85 points/game with ties-as-win probability 0.2423 ± 0.0133, n=4,000 against 3×TA, hero g%4. Source: `experiments/round8/T6/results/fit_sweep.json`, `fit.json`, screen/validation JSONL and summaries, and `verification.json` (all 60,000 vector/deployed bids agree; all 6,000 paired game records check). The original guide stays unchanged. This result excludes neither other encodings of shape/lead nor benefits under other opponent pools; no human playtest occurred.

### Reproduction and validation

From `engine/`, use `python experiments/round8/run_tasks.py T3` for the original/fresh grids (completed statuses skip by design; direct `T3/run.py` also resumes). Run each task's `analyze.py` where present, and `T3/verify.py`, `T5/certify.py`, `T6/verify.py` for the saved-data audits. T4 and T5 each provide a standalone `worker.py`; T6 `run.py` enforces its one-candidate freeze. Historical runners retain the overnight 07:30 ET hard deadline and therefore deliberately will not start compute after that date; use saved raw data for analysis, and preregister a fresh deadline and seeds for a new experiment. Exact commands and all seed bases are recorded in each STATUS.txt.

The final integrated suite passed **256 tests, 1 optional OpenSpiel module skipped** using `python -m pytest -q -p no:cacheprovider` (16.08 s). Existing core modules and existing policies are unchanged; only the new experimental guide-v2 policy was added.

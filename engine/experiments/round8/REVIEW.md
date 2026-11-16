# Round 8 reviews (Claude, head of research)

One section per task when it reaches `ready-for-review`: verdict (validated / changes-requested), what was checked, independent recomputations, requested changes.

## T1 — audit integration: VALIDATED (Claude)
Checked the full diff of AGENTS.md (both copies identical), engine/AGENTS.md, math/NOTES.md against round7_audit/AUDIT.md §5 and results/*.json.
All quoted numbers match the audit recomputations (0.296 [0.264, 0.327]; PIMC 0.356; LC1 .98, LCX .82/.30; LC0 .58 vs TPB .04; hook F3 10.17–6.96 incl. ±0.99 at n=6; bots +7.98/+17.07/+20.74; RB +6.51/+2.45/+2.72/+3.58/+2.37; C2 replication +2.94 ± 3.28; guide 0.808, −2.01). P1/P3/P4/W/L status changes are worded correctly. No new empirical claims.
Minor (no action needed): F3 counts are seat-observations, not deals; fine to leave.
Action: commit AGENTS.md (both), math/NOTES.md, experiments/round7_audit/, experiments/round8/T1/ now. Then T2 in background.

## T2 — top-pool replication at equal depth: VALIDATED (Claude)
Independent recomputation from the 1,200 raw `results/*.json` (challenger seat g%4 asserted for all; scores re-derived from bids/tricks on a spot game): identical to `summary.json`.
- Sanity (BI K32c6 as challenger, n=300): Δ = 0.00 exactly, 300/300 identical total vectors. Note: Hybrid-BI's sampling appears seeded from game state, not the agent seed, so this is a determinism/plumbing check, not a noise calibration. Fine for CRN.
- **RB K32c6 − pool: −0.74 ± 2.96 (n=300)**; ΔP(win, ties-as-win) +0.017 ± 0.057. At equal search depth the top pool is not beaten on fresh seeds.
- **RB K32c7 − pool: +4.64 ± 3.23 (n=300)**, ΔP(win) +0.033 ± 0.057 (n.s.). Depth contrast c7 − c6 (same RB): **+5.38 ± 3.12**.
- Interpretation for AGENTS/EVIDENCE: the earlier RB c7 edge over the pool is a depth effect, not an opponent-modelling effect. Equal-depth exploitability of the top pool by RB is not detected (CI excludes gains > +2.2 pts/game). c7 costs ~1.4× wall time (12.8 s vs 9.1 s/game). Score gain does not establish a win-rate gain.
- Wording to use: "Top pool (3×Hybrid-BI K32c6), fresh seeds 1,000M, n=300 paired: RB at equal depth −0.74 ± 2.96; RB with one more search card +4.64 ± 3.23 (depth contrast +5.38 ± 3.12)."
Action: commit T2 (scripts, results, STATUS). Proceed with T4.

## T4 — lead effect for non-leading heroes: VALIDATED (Claude)
Checked: STATUS ready block, summary values, raw per-hand NPZ structure (cards, 300 rollouts, fixed prior/own bids, legal set). Spot-checked dealer (pos3) files: stored legal set excludes exactly the hook bid n − Σothers (4/4 checked). Discovery (1,100M) and replication (1,110M) are separate seed bases, 500 hands × R=300 per cell, ≥498 non-degenerate hands everywhere.
Findings (NT, TA play; ranges over pos1–3 and both seed bases):
- Dependence persists without the opening lead: δ 1.38–1.58 (±0.03), cross-suit dispersion 0.22–0.38, lead persistence 0.21–0.30 (all ±≈0.013). Both shrink toward the dealer seat (pos3) and slightly with n.
- **LC0 gap closed, n=5:** pos1 0.31 [0.15, 0.45] → rep 0.43 [0.27, 0.56]; pos2 0.29 [0.12, 0.44] → rep 0.29 [0.06, 0.46]; pos3 0.44 [0.13, 0.71] → rep 0.08 [−0.27, 0.33] (**not replicated**).
- **n=8:** LC0 0.03–0.19, every CI includes 0 in both bases.
- TPB ≈ 0 everywhere (−0.11 to +0.08); one replication cell (n=8 pos1) is significantly negative, −0.11 [−0.21, −0.02].
Interpretation: trick dependence is not a leader-only artefact, but the lead-state chain explains much less of it for non-leaders (n=5 pos1/2: ≈0.3–0.4 vs 0.58 [0.49, 0.65] for the TA leader in M10; n=8: not detectable). Cross-design caveat: the M10 leader runs used different seeds/bid context, so leader vs non-leader is descriptive, not a paired contrast.
Wording to use: "For non-leading heroes (NT, TA), dependence persists (δ≈1.4–1.6), but LC0 closes only 0.29–0.43 of the PB-to-floor gap at n=5 for positions 1–2 (replicated, 500 hands per cell) and none detectably at n=8 or for the dealer."
Action: commit T4 (incl. tests/test_round8_lead.py after a green pytest run).

## T5 — hidden-information spoiler coalition, toy: VALIDATED (Claude), with a scope note
Checked: results/trump_{0,-1}.json, certificate.json, certificate logic. The fractions match the reported PI values (1257/140 = 8.97857, trump; 946/105 = 9.00952, NT).
Exact values: complete chance space, 4 players, 2 suits × 4 ranks, n=2, 2,520 ordered deals per regime, TA hero and myopic bids frozen, play-only team.
| regime | TA baseline | independent BR (heuristic) | hidden team (certified) | PI min |
|---|---|---|---|---|
| trump | 9.6024 | 8.9786 | 8.9786 | 8.9786 |
| NT | 9.9889 | 9.1032 | 9.0095 | 9.0095 |
- Certificate logic is sound. With the hero and bids fixed, the PI paranoid value is a lower bound for any spoiler team. The saved hidden-information profile attains it on every deal (zero pointwise excess, exact integer DP). So the hidden-team gap to PI is exactly 0.
- Result: in this toy, the hidden coalition loses nothing to information. Surplus share hidden/PI = 1.000 in both regimes. Independent BR gets 1.000 (trump) and 0.904 (NT) of the PI punishment.
- Scope note (important for the paper): the toy is too small to discriminate. At n=2 there is almost no hidden-information leverage, so this does **not** inform the 52-card ratio 0.296. It shows only that hidden-team optimality can be exactly certified, and that the gap appears in NT first. Do not cite it as evidence that hidden coalitions approach PI in the real game.
- Suggested follow-up (later round): the same certificate on a larger toy (n=3, 3×4 or 2×6 deck), where PI and hidden values may separate. Not tonight.
Wording to use: "In an exhaustively solved 8-card toy (n=2, fixed TA hero and bids), a hidden-information spoiler team attains the perfect-information punishment exactly (certified, 2,520 deals per regime). The toy is too small to bound the full-game hidden/PI ratio."
Action: commit T5 (incl. tests/test_round8_team.py).

## T6 — human bidding v2 (shape + NT lead): VALIDATED as a NULL result (Claude)
Checked: the feature code in `human_guide_v2.py` (trump: voids/singletons capped by trump length − 1, n ≥ 5; NT: long-suit units with control + outside ace, plus an opening-leader ace term), the 144-cell sweep (`fit_sweep.json`), and the screen/validation summaries.
- Training (1,250M, 60,000 situations): **no non-zero weight vector beats the current guide.** The best alternatives are NT-control +1: −0.002 ± 0.006, and lead +1: −0.009 ± 0.004 points/round. The selected weights are (0,0,0,0), so v2 ≡ current guide.
- Screen (1,300M, n=2,000) and validation (1,400M, n=4,000) are therefore exactly Δ=0 by construction. The protocol was followed (no tuning after screening), but these runs carry no information beyond confirming identity. Guide vs 3×TA level: 159.74 ± 0.85 points/game; P(win, ties-as-win) 0.242 ± 0.013 (n=4,000).
- Conclusion: as encoded (integer point bumps on top of the guide count), shape and NT-lead adjustments do not improve guide bidding. The training CIs bound each single-feature gain to under ~0.01 points/round. So the guide's −2.01 points/game bidding loss vs TA is not about shape.
- Do **not** promote `human_guide_v2` as an improvement. Keep it as the documented negative result (module + test can stay; it is new and does not touch the existing guide).
- Suggested follow-up (later): decompose the guide's bidding loss vs ModelBidder by n, trump/NT and seat (bid error = guide − model bid). Find where the 2 points go before proposing features.
Wording to use: "Adding shape (voids/singletons with trumps) and an NT long-suit/lead adjustment to the guide's bid count gave no gain: none of 143 candidate weightings beat the current rule on 60,000 training situations (best −0.002 ± 0.006 points/round)."
Action: commit T6 as a negative result.

## T3 — oracle bid vs mode under TA/TA+ play: VALIDATED, with an interpretation caveat (Claude)
Checked: the ready block, summary.json and verification.json. Verification passed on both seeds (laws, counts, oracle choices and moments all reconstructed). The full hook audit covers all 4,160 replication rows; the discovery rows lack that metadata, which is disclosed. Greedy reproduces the phase-1 distribution exactly (4,160/4,160).
Results: n=4,160 situations per seed, R=300 worlds per bid, Wilson 95% CIs. Discovery 4242 / confirmatory replication 4243 (worlds paired across policies and bids):
| play | oracle = mode | oracle ∈ {⌊μ⌋, ⌈μ⌉} | = round(μ) |
|---|---|---|---|
| Greedy | 86.8% [85.8, 87.8] / 87.0% [86.0, 88.0] | 92.8 / 93.0 | 63.7 / 63.8 |
| TA | 94.6% [93.9, 95.3] / 94.7% [94.0, 95.3] | 97.2 / 97.5 | 74.9 / 75.6 |
| TA+ | 95.6% [95.0, 96.2] / 95.6% [95.0, 96.2] | 96.9 / 97.2 | 77.4 / 77.7 |
Caveat that must travel with the number: under TA/TA+ the trick law depends on the bid (target-aware play steers toward it). The mode here is the mode of the law **at the oracle-selected bid**, so part of the agreement is the play hitting its own target, not a property of the hand. Read it as: "with target-aware play, the best bid is almost always the most likely trick count given that bid." It is not a bid-free 'bid the mode' rule. Greedy remains the only bid-independent law (86.8%).
Minor: 4243 is the next phase-1 base rather than a round-8 base. That's acceptable, since it is fresh for this question; record it in the board's seed list.
Wording to use: "Oracle bid = mode of the trick law: Greedy 86.8% [85.8, 87.8] (bid-independent law); TA 94.6% [93.9, 95.3] and TA+ 95.6% [95.0, 96.2], replicated on fresh worlds (n=4,160 each; mode taken in the law at the chosen bid, so not a bid-free rule)."
Action: commit T3. Then T7.

## T7 — evidence map and handoff integration: VALIDATED (Claude)
Checked the full diff of `paper/EVIDENCE.md` (+80 lines, round-8 section) and both AGENTS.md copies (byte-identical). Every number matches the T2–T6 verdicts above and my recomputations. Each task's caveats travel with its numbers:
- T2: the depth contrast is not opponent modelling.
- T3: TA/TA+ figures are bid-conditional laws.
- T4: the leader comparison is descriptive; the dealer cell did not replicate.
- T5: the toy does not bound the 52-card gap.
- T6: Δ=0 holds by identity.
No new estimates were added. "Next steps" were rewritten sensibly. The M11 cross-policy ratio is kept as history, with the same-hero 0.296 preferred.
Action: commit T7 (EVIDENCE.md, both AGENTS.md, BOARD/STATUS updates, T7/). No push. The round is closed; Codex can stop.

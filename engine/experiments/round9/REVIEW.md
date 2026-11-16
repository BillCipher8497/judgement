# Released round9 reviews

D0 was Claude-validated; D1,D2,D4 separately self-validated (Codex). Self-review is not independent external validation. Other local work is excluded.


## D0 — AGENTS.md duplicate removal: VALIDATED (Claude)
Both copies are byte-identical. Each of the two sentences now appears exactly once. No numbers changed. Commit locally.


## D1 — self-validated (Codex), 2026-11-16

**Verdict: passes for the implemented, fixed-hero intervention. External validation remains required before strong causal paper language.** Review performed separately after the experiment; reviewer and implementer are the same person, not independent investigators.

Fresh audit: `python engine/experiments/round9/reviews/audit_d1.py` from the repository root. It imports neither the runner nor analyzer. All 2,000 raw files were hashed, their seeded hands regenerated, and all 10,000 arm trajectories independently checked for turn order, card ownership, follow-suit, trick winners, hook legality and exact-hit scoring. Fixed bids match across arms; every all-information trajectory equals the omniscient reference. Each stage has 1,000 deals, balanced over S/NT and dealer. Hero is fixed seat 0 as in the specified a1 round design; g%4 applies to the full-game experiments, not this fixed-round design. Eight prespecified deals per stage were replayed in all five arms (80 trajectories), using the original PIMC/omniscient references where applicable; every saved history reproduced.

The fresh paired-deal bootstrap has 10,000 draws, stratified by S/NT, review seed 1,650,000,001 (PCG64DXSM). Discovery primary revealed-hero minus own/public surplus share is **0.369315 [0.315995, 0.427185]**; fresh replication is **0.341204 [0.288919, 0.394365]**, n=1,000 deals each, 95% CIs. The independent-stage difference is 0.028110 [−0.047286, 0.103561], n=1,000 per stage; lack of detected difference is not equivalence. Original point estimates match exactly; independently seeded bootstrap endpoints differ by less than 0.003. The original intervals remain the experiment's primary report; these are verification intervals. Full arm scores, shares, reference ratios and hashes are in `reviews/results/D1_raw_audit.json`; review bootstrap draws are retained in NPZ.

Source checks: only the world sampler changes across own/hero/all. Solver, K=16 per active decision, fallback, target objective and tie-breaking are identical. Hero revelation removes exactly that remaining hand from the unknown-card allocation and restores it; counts/voids/played-card constraints are preserved. The own branch reads no other private hand; the hero branch reads only the authorized hero hand; public state can depend on earlier play. Existing boundary tests swap unrevealed hands, check mid-trick void constraints and compare original own/all references. The STATUS chronology predeclares the primary contrast before compute, but no immutable prelaunch Git hash exists; the current code/raw hashes do not independently notarize that chronology.

Required caveats: this is K16 PIMC against a fixed TA+ hero and fixed ModelBidder bids, at five cards with S/NT equally weighted. Own information includes public bids/history/voids; sampling omits bid/play likelihoods. The total number of active decisions is endogenous, so equal K per decision is not equal total compute per round. The omniscient paranoid reference optimizes against a maximizing hero in its internal solver, while the actual hero is TA+; it is not certified optimal punishment against that fixed policy. No optimal hidden-team value, full-game information effect, or general 52-card theorem follows. No strong causal paper claim is authorized until external D1 review.

Checks already green: pause suite 262 passed, 1 optional skip; detailed log `results/pause_tests.txt`. Review adds only an audit script/data/documentation and removes a redundant tautological audit assertion; experiment behavior is unchanged.


## D2 — self-validated (Codex), 2026-11-16

**Verdict: passes as a replicated whole-bidder score decomposition. No new guide rule is validated.** Fresh command from repository root: `python engine/experiments/round9/reviews/audit_d2.py`. The audit imports neither runner nor analyzer. It independently reconstructs all200,000 seeded hero hands across8,000 paired25-round games, verifies all shared hand/prior/hook-legal contexts, seatg%4, bid/trick ranges, trick totals, exact-hit scores and game totals. Full-game replay on five prespecified indices per stage and both arms (20 games) reproduces all saved bids, tricks and totals. Every decomposition cell, bid-error frequency and CI matches the independent estimator to1e-10. The discovery raw SHA and selected top3 match the frozen packet. The primary contrast and freeze-then-replicate sequence are documented before their respective compute; as with D1, no immutable prelaunch commit notarizes the chronology.

Primary ModelBidder−guide hero points/game: **5.29425 ±0.82665** discovery (1500M), **5.05950 ±0.81963** replication (1510M), n=4,000 paired games each,95% CIs. Both use unchanged guide play against3×ModelBidder/TA. Ties-as-win differences are **0.04975 ±0.01510** and **0.03375 ±0.01533**, same n and95% CI. Score decomposition sums exactly back to the game contrast. Independent units are full paired games, stratified by rotated hero seat; repeated rounds remain in the same game cluster. Conditional per-round/bid-error ratios use cluster influence functions; structurally absent cells are not estimated.

Frozen cells were selected by positive contribution to total game loss, not by per-round loss. Contributions below are points per full game, n=4,000 independent game clusters per stage; nominal95% CIs (selected discovery intervals are descriptive/selection-biased):

| Cell | Discovery contribution | Fresh replication contribution |
|---|---|---|
| n2, trump, bid position2 | 0.32425 ±0.08435 | 0.23875 ±0.07760 |
| n7, trump, bid position3 (dealer) | 0.26000 ±0.11491 | 0.15175 ±0.11261 |
| n10, trump, bid position1 | 0.24325 ±0.13344 | 0.13625 ±0.12954 |

All three replication nominal intervals exclude zero, but this is three testing opportunities. A secondary reviewer sensitivity (`python engine/experiments/round9/reviews/multiplicity_d2.py`, `reviews/results/D2_multiplicity.json`) applies Bonferroni familywise95% intervals: the first two remain positive; the n10 cell does not. Do not describe all three as familywise-confirmed or as the replication's three largest cells. Full counts, distributions and absent cells are retained in `reviews/results/D2_raw_audit.json`.

Required caveats: ModelBidder is a comparator, not an oracle; guide−model bid differences are not measured truth errors. Replacing the bidder changes later bids/target-aware play as well, so this is not isolated direct bid EV. The guide's historical overall deficit versus TA is not its isolated bidding cost. No new feature, fitted adjustment, human playtest or seed reuse is validated. The post-review multiplicity check is a sensitivity analysis, not a preregistered endpoint. Existing core/policies remain unchanged; prior full suite262pass1skip, final integration suite will be recorded separately.


## D4 — self-validated (Codex), 2026-11-16

**Verdict: passes as a faithful compact source card, with a visually checked one-page PDF.** Read `paper/guide_card.md` against GUIDE.txt B1–B7, definitions and P0–P9. The eight rules combine P1/P2 and P4/P5 without changing their order. Trump/NT counting, the unequal above/below earlier-bid thresholds, and dealer hook parity are retained. No new strategic rule or empirical result is added. Existing source edge-case ambiguities are not claimed to be resolved; this is a condensation, not a new complete policy specification.

Additional layout verification: `D4/render_card.py` (reportlab+pypdf, bundled runtime) renders the unchanged Markdown to `D4/results/guide_card.pdf`. A4 page count is exactly1; all395 normalized word tokens are preserved in source order; Markdown whitespace word count407 uses a different tokenization. Source/PDF/font hashes are recorded in `D4/results/render_audit.json`. Rendered with Poppler `pdftoppm -f 1 -singlefile -scale-to 1500 -png ...`; inspected `guide_card_preview.png` visually. Table, definitions, all eight numbered rules and source/limitations footer are legible, unclipped and non-overlapping. No pagination claim is made for arbitrary Markdown renderers; the supplied PDF fixes a one-page layout.

No stochastic seed, empirical n or95% CI applies to source/layout verification. No human playtest or improved-performance claim. PDF skill used for render-and-inspect verification. Local scoped commit is authorized by this verdict; no push.


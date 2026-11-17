# Score incentives and policy-dependent punishment in Judgement

Local working draft, 2026-11-16. This document assembles validated project evidence; it is not a submission-ready paper. D1/D2 and the M13/M14 proofs/lower-bound enumeration are externally validated (Claude, 2026-11-16); other additions retain their separate validation labels. The fresh E0 full-suite gate passed locally; external validation did not include a test-suite rerun. A [bounded related-work draft](RELATED_WORK_DRAFT.md) is available; broader coverage and external review remain outstanding. Quantitative claim IDs map to `experiments/round10/R0/claim_ledger.json` and the [evidence map](EVIDENCE.md).

## Abstract

Judgement conserves tricks while its exact-hit scoring permits players to change the total score through play. We study expected score incentives under four-player rules using exact accounting, finite-game certificates and paired policy experiments. If M players miss their bids and U tricks are won by those players, total round score is W = 40 + n − 10M − U. The dealer's hook gives the sharp global ceiling W ≤ 30 + n. In 4,000 paired full games, replacing three self-interested opponents with specified targeted spoilers reduces table score by 171.708 ± 1.985 points/game (95% CI), including losses to the spoilers themselves. The evidence establishes general-sum score incentives and an operational dependence on opponents' policies. Hidden-information policy comparisons, lead-state diagnostics and measured hook costs provide further distinctions between game rules, information, policy choice and computational approximation. They do not solve optimal hidden-information coalition punishment.

## 1. Rules, utility and the research question

Four players use a standard 52-card deck and a 25-round ladder 13,12,…,1,…,13. Trump rotates spades, hearts, diamonds, clubs and no-trump. The player left of the dealer bids and leads first; the dealer bids last and cannot make the total bids equal the hand size. Players must follow suit when able, with no obligation to trump when void. An exact bid earns 10+bid; a miss earns zero. Trick winners lead the next trick. These are the owner's confirmed rules implemented by the engine.

The principal utility studied here is score. Highest final total determines victory, with every tied leader counted as a winner in reported victory indicators. Score comparisons and victory comparisons answer different questions. In particular, the engine's historical split-tie win field must be replaced by indicators computed from totals for the requested victory convention.

We call the hero's score difference between specified self-interested and targeted-hostile opponent policies a *cooperation surplus*. The term is operational. It requires neither communication nor a claim that an equilibrium, a bargain or an optimal coalition has been found. Every comparison must identify the hero, bidders, opponent policies and information available.

## 2. Exact score accounting

Let n denote tricks in a round. Successful bidders receive four potential hit bonuses minus the bonuses lost by the M missers. Successful bids sum to the tricks taken by successful bidders, n−U. Thus

    W = 40 + n − 10M − U.

The hook prevents everyone from hitting simultaneously, so M≥1 and W≤30+n. This ceiling is sharp across legal deals and histories; it need not be attainable for an arbitrary fixed deal/bid vector. The identity and sharpness are proved in M11. A legal two-card witness also holds deal and bids fixed while different discards produce different table scores, 30 and 31. These are exact constructions, so sampling CIs do not apply. [C01]

For paired rounds, the difference in table score is exactly ten times the increase in missed bonuses plus the increase in tricks held by missers. These terms are an accounting decomposition. They are not separately randomized causal mechanisms.

## 3. Experimental and computational methods

Full-game comparisons pair policies on the same seeded deals and rotate the challenger through seat g%4. Fixed-size hero experiments instead fix the hero and balance dealer/trump as specified. All bidders and play policies are identified in the saved task protocols. Ties-as-win are recomputed from totals.

For H1, baseline play is `ta_plus_v2()` in all seats, with `ModelBidder(model_ta_final.json)` throughout. The hostile arm replaces the three opponents' play with `SpoilerPolicy(target=hero)`, retaining the same bidder and TA+ fallback. Full-game seed is133,000,000+g; the implementation and sampling schedule are in `experiments/surplus_hidden/sim.py`. D2 instead uses guide hero play against three TA opponents, with the bidder contrast recorded separately. These opponent pools are not interchangeable.

Deal is the uncertainty unit for fixed-round results; whole game is the unit for ladder results. Repeated rounds or seats within a deal are not independent observations. Reported intervals are 95% unless explicitly labelled otherwise. Paired ratios preserve covariance; they are not obtained by dividing interval endpoints. The intervals describe variation under the saved sampling designs, not approximation error relative to optimal policies.

Perfect-information solvers, hidden-information PIMC and private-information team searches solve different computational problems. Max^n, paranoid objectives and fixed-hero play must be distinguished. A small unilateral best-response residual is not a certificate of joint team optimality. Exact chance enumeration eliminates chance-sampling error within a finite toy, while leaving representation, implementation and model-scope questions.

## 4. Targeted punishment destroys score for the whole table

The H1 experiment pairs an unchanged TA+ hero and fitted bidding regime with self-interested or targeted-hostile opponent play. In n=4,000 paired full games, the three-spoiler intervention produces the following losses, in points/game; all intervals are 95%. [C02]

| Quantity lost relative to self-interested opponents | Mean ±95% half-width |
|---|---|
| Hero score | 51.661 ± 1.105 |
| Mean score across the three spoilers | 40.015 ± 0.605 |
| Total table score | 171.708 ± 1.985 |
| Additional missed-bonus component | 154.425 ± 1.628 |
| Additional tricks-held-by-missers component | 17.282 ± 0.440 |

The table loss includes the opponents' own losses. It therefore supplies direct evidence that conserving tricks does not conserve score and that the specified score-destroying policies sacrifice their own returns. It does not show that these policies are equilibrium best responses, win-maximizers or optimal punishers.

## 5. Information, policies and exact finite-game certificates

The preferred historical information comparison uses the same TA+ hero and fixed bids, with five cards and an equal spades/no-trump mixture. The heuristic-hidden/omniscient surplus ratio is **0.296 [0.264, 0.327]**, n=4,000 paired deals; the hidden-PIMC(K16)/omniscient ratio is **0.356 [0.313, 0.398]**, n=2,000 paired deals,95% CIs. These are relative score reductions achieved by the named policies. Their differences do not identify an optimal hidden coalition or a policy-independent information effect. [C03]

The earlier cross-policy hidden/PI ratio is retained in the evidence archive as descriptive history. It mixes hero and bidding regimes and is not the preferred information estimate.

In an eight-card finite toy, 4 players with 2 cards each, fixed myopic bids and a fixed toy-TA hero, saved private-information team policies attain the PI lower bound on every deal. The certified mean hero scores are 1257/140 (trump) and 946/105 (NT), each over all 2,520 ordered deals. Exact integer support checks certify zero pointwise excess, so no sampling CI applies. This is a joint optimality certificate within that game. The tiny toy does not bound the 52-card hidden/PI ratio. [C04]

The externally audited round9 information intervention and larger-toy bound are detailed in Appendix A. D1 supports the implemented information intervention only; M13 certifies a strict toy information cost, with no material-magnitude claim or solved hidden-team optimum.

## 6. Lead diagnostics and hook costs

Winning a trick changes who leads next, providing a mechanism through which card outcomes can become dependent. In five-card NT TA rollouts, cross-fit bid match is **0.739 ± 0.027** for the independent-card model and **0.978 ± 0.009** for LC1, with empirical split-half reference **0.982 ± 0.008**, each n=1,000 hands and95% CIs. These are per-hand rollout-fitted diagnostics, not deployable hand-feature bidders or proof of a unique lead mechanism. Flexible alternative fits and failure at larger hand sizes limit the interpretation. [C05]

The hook has a sharp global ceiling effect and a separate policy-dependent average cost. Representative measured dealer costs are below, in points/round, each with a95% CI. [C06]

| Regime | Cards | Dealer cost | Independent units |
|---|---|---|---|
| Hidden cards, fitted bidder/TA | 1 | 5.41 ± 0.11 | 10,000 rounds |
| Hidden cards, fitted bidder/TA | 13 | 0.27 ± 0.06 | 20,000 rounds |
| PI rational bidding | 1 | 10.17 ± 0.02 | 2,000 deals |
| PI rational bidding | 5 | 7.82 ± 0.37 | 600 deals |

The regimes change policies and bidding as well as information. They should not be described as an information-only randomized contrast. At one card, when both bids are legal and the relevant win probability is p, score comparison gives bid1 iff p>10/21. Forced hook choices and bid-conditioned information require separate treatment; the threshold is exact, with no sampling CI. [C07]

## 7. Translation to a human guide

Against the TA bot pool, the existing guide retains **0.808 [0.794, 0.822]** of TA+3's edge over greedy, n=6,000 paired games,95% CI. This is a bot evaluation under correct rule execution, not human-learning evidence. [C08]

Round9 analysis, independently recomputed and externally validated by Claude, holds guide play fixed while replacing only the hero bidder. On fresh 1,510M seeds, ModelBidder−guide score is **5.05950 ±0.81963 points/game**, n=4,000 paired games,95% CI. ModelBidder is a comparator rather than an oracle; later bids and target-aware play can respond to the changed bid. This is the whole-bidder score cost, distinct from the guide's historical overall deficit versus TA. The discovery-selected cells and their multiple-testing qualification are in EVIDENCE; no new rule was fitted. [C09]

The source-faithful table card has a checked one-page A4 PDF. Its printed form adds no strategy or measured improvement. Human playtesting remains necessary.

## 8. Limitations and completion requirements

The evidence supports general-sum score incentives and measured dependence on specified opponents. It does not establish general multiplayer CFR convergence, optimal decentralized punishment in the full game, or universal superiority of a bot or guide. Finite-rollout fits and policy selection need their own out-of-sample checks. Negative and failed-replication results remain part of the evidence map.

Before submission, extend the bounded primary-source related-work audit and perform a claim-by-claim reproduction pass. External D1/M13/M14 review is complete; a fresh feasible-upper forward replay also passed as a separate Codex self-check (E5). Make no novelty or priority claim from the current literature coverage. The external living report has not been rewritten. The particle-MCMC experiment supplies no validated improvement to this draft and remains outside the publication scope.

## Appendix A. Externally audited information intervention and finite-toy bounds

**External status:** D1 and the M13/M14 proofs/lower-bound enumeration were independently checked and externally validated by Claude on 2026-11-16. The feasible uppers were also confirmed by a fresh forward replay in E5, self-validated (Codex); Claude checked their ordering only. Validation is limited to the stated interventions and finite games.

D1 holds hero/bids fixed and changes the information available to a K16 PIMC sampler. The primary revealed-hero minus own/public surplus-share contrast is **0.369315 [0.315244,0.424218]** on 1,600M discovery and **0.341204 [0.289874,0.394787]** on fresh 1,610M, each n=1,000 paired deals,95% CIs. The omniscient paranoid reference is not certified optimal against the actual fixed hero; equal K per active decision does not equal equal total compute. The supported causal statement is only that revealing the hero hand to this implemented sampler changes its punishment performance: n=5, equal S/NT, fixed TA+v2 hero, fixed bids and K16 per active decision. It does not identify a policy-independent information effect, optimal coalition punishment, human performance or a general effect across the full 25-round ladder using the standard 52-card deck. [C10]

D3 fixes all bids and hero moves in a 12-card, 4-player, n=3 toy. The complete 369,600-deal populations give exact optimization intervals **[446669/73920,574247/92400]** (trump) and **[213259/30800,109709/15400]** (NT). They are bounds, not CIs or solved team optima. Lower endpoints use M13's private first-decision relaxation; upper endpoints are feasible saved policies, confirmed by a fresh forward replay in E5 (self-validated, Codex). Claude independently reconstructed every deal and exactly reproduced the PI/J1 lower values, and rechecked M13. The certified J1−PI gaps are 25819/369600 ≈ 0.070 and 421/30800 ≈ 0.014 hero points, respectively. They certify a strict information cost and nothing about its magnitude: these lower bounds do not establish a material cost, nor estimate the actual hidden-team optimum minus PI. No 52-card extrapolation. [C11]

M14, also externally validated, shows that team randomization cannot improve the global minimum against fixed non-team policies: conditioning on tapes gives a mixture of feasible pure private-information tuples, whose mean cost cannot beat the best tuple. Team tapes may be shared, but must be independent of chance and non-team tapes. The fixed-hero premise and full own/public histories are essential; fixing the team tapes must not change hero observations outside those histories. This is not a strategic-adversary team-maxmin or CFR convergence theorem. No novelty claim is made.

## Evidence references

- C01,C07: [proof/status notes](../math/NOTES.md), M11 and one-card result; [fixed-bid welfare witness](../tests/test_round7_welfare.py).
- C02: [H1 welfare reanalysis](../experiments/round7_paper/results/welfare.json), paired-game NPZ and source hashes; [round7 audit](../experiments/round7_audit/AUDIT.md).
- C03: [same-hero audit](../experiments/round7_audit/results/a1_surplus.json).
- C04: [round8 exact certificate](../experiments/round8/T5/results/certificate.json), M12 and [Claude review](../experiments/round8/REVIEW.md).
- C05: [M10 diagnostic summary](../math/results/m10_summary.json), [audit qualification](../experiments/round7_audit/AUDIT.md).
- C06: [hidden hook tables](../experiments/phase1/results/phase1_tables.md), [PI summaries](../experiments/phase2/results/summary_main.md).
- C08: [guide summary](../experiments/guide/results/s3_summary.txt), [guide source](../experiments/guide/GUIDE.txt).
- C09–C11: [separate round9 reviews](../experiments/round9/REVIEW.md), task raw files and independent scripts under `experiments/round9/reviews/`.

- External audit: [Claude verdicts and retained limits](../../CLAUDE_AUDIT_2026-11-16.md); [M14 proof](../experiments/round10/R3/PROOF.md).

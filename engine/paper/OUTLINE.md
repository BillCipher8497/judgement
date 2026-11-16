# Information and general sum incentives in Judgement

Judgement conserves tricks while allowing players to create or destroy score through exact bidding. The paper should explain how this distinction produces a dependence on opponents pursuing their own targets, why hidden cards restrict punishment, and how lead control changes the distribution that a bidder must predict.

## Proposed abstract

Judgement is a trick-taking game in which each player scores only by taking exactly the number of tricks bid. We study its general-sum score incentives using exact small-round solutions, hidden-information policy interventions, and paired full-game experiments. An exact accounting identity separates table score into successful-bid bonuses and tricks won by successful bidders. In a saved experiment with 4,000 paired full games, replacing three self-interested opponents with targeted spoilers reduces the hero's score by 51.66 ± 1.10 points and total table score by 171.71 ± 1.99 points per game, including losses to the spoilers themselves. A descriptive comparison at five cards, standardized to equal spades and no-trump rounds, gives surplus shares of 22.98% ± 2.75 percentage points under hidden information and 81.09% ± 1.47 under perfect-information solutions, using 4,000 and 1,000 deals respectively. Because policies and bidding differ between these experiments, this comparison does not identify a causal effect of information. Further experiments with the same hidden-information hero and more informed spoilers support an information-based explanation. We connect these incentives to the dealer's hook restriction and show that a lead-state model explains much of the failure of independent-card bidding models in short no-trump rounds. All ± intervals are 95% intervals.

## Argument and scope

The primary object is expected score under the standard exact-hit rule. Distinguish this from tournament victory: split-tie win shares sum to one, while the requested ties-as-win convention counts every tied leader as a winner. A result about score incentives does not by itself establish a result about win-maximizing policies. Recompute victory indicators from totals throughout.

Use “cooperation surplus” as an operational name for the hero's score difference between specified self-interested and hostile opponent policies. It does not imply communication, collusion, an equilibrium welfare gap, or the exact value of an optimal hidden-information coalition. Identify the hero policy, bidding policy, adversary and information available in every comparison.

## Introduction

Open with a legal two-card position where the deal and bids are fixed but a discard changes total score. Follow with the accounting identity and the paired finding that targeted punishment destroys the spoilers' own score as well as the hero's. This gives the general-sum claim a direct foundation before introducing solver-based surplus measures.

State the three substantive contributions: information limits attainable punishment; lead control creates dependence between card wins; and the hook imposes a policy-dependent dealer cost whose magnitude changes with available information. Position adaptive bid inference and the human guide as applications rather than additional main theses.

## Rules and exact score accounting

Define the four-player rules, legal bids, hook, public history, private hands and exact-hit scoring. State the M11 identity from [the proof notes](../math/NOTES.md): if M players miss and U tricks are won by players who miss, table score is W = 40 + n − 10M − U. With the hook, M ≥ 1 and W ≤ 30 + n. The bound is sharp over legal deals and histories; it need not be attainable on a particular fixed deal and bid vector.

For a paired intervention on the same round, W_self − W_hostile = 10(M_hostile − M_self) + (U_hostile − U_self). Sum this identity over rounds before estimating full-game uncertainty. Its two components are exact accounting terms, not independently identified causal mechanisms.

## Experimental methods

Describe the rules engine, cross-validation, information states and policies. Exact max^n solutions use specified tie-breaking; paranoid values optimize against a coalition. Explain the rational perfect-information bidding scheme separately from fitted bidding with hidden cards. Report sensitivity to max^n tie-breaking.

Pair policies on the same deal seeds and rotate the challenger seat with g mod 4 for full-game challenger experiments. For H1 fixed-size rounds, hero seat is fixed while dealer and trump rotate. Deal is the uncertainty unit for fixed-size rounds; whole game is the unit for the ladder. Average seats within a perfect-information deal before computing a CI. Ratios require covariance-aware intervals; never divide two reported CI endpoints.

The M11 reanalysis makes no new policy decisions or random draws. It validates score reconstruction, trick conservation and hook legality for every included H1 outcome, saves paired components, and records source hashes. Its intervals quantify variation over the saved sampling design, not uncertainty about how well the policies approximate optimal play.

## Information and the cooperation surplus

Lead with the paired welfare loss under three spoilers. For 4,000 games, hero loss is 51.66 ± 1.10, mean loss per spoiler is 40.02 ± 0.60, and table loss is 171.71 ± 1.99 points per game. The table loss separates into 154.43 ± 1.63 points from additional missed-hit bonuses and 17.28 ± 0.44 points from additional tricks won by missers. Source: [M11 welfare results](../experiments/round7_paper/results/welfare.md).

Then compare surplus shares. The original hidden-information five-card sample mixes all five trump states; the perfect-information reference uses only spades and no-trump. Prefer the standardized S/NT comparison in the abstract. Its hidden/perfect surplus-share ratio is 0.283 ± 0.034, with 4,000 hidden-information deals and 1,000 perfect-information deals. The policies still differ, so label this ratio descriptive. Preserve the original all-trump H1 result separately for reproduction.

Use the same-hero strong-spoiler study as the more focused evidence about information. Report the paired heuristic, hidden-information PIMC and omniscient spoiler outcomes on their common subset of deals, with the corresponding counts and CIs. PIMC is an approximation; its inability to reproduce the omniscient result does not prove optimal hidden-information punishment is weak. Include the indistinguishable-deals counterexample as the exact result showing that perfect-information punishments can be incompatible with a hidden information set.

## The hook under different information regimes

Separate the sharp global score ceiling from the measured dealer cost. The hook lowers the largest possible table score from 40 + n to 30 + n, but this is not a prediction of the average dealer tax.

The hidden-information fitted-bidder experiment gives dealer costs of 5.41 ± 0.11 points at one card (10,000 rounds) and 0.27 ± 0.06 at thirteen cards (20,000 rounds). The perfect-information rational-bidding experiment gives 10.17 ± 0.02 at one card (2,000 deals) and 7.82 ± 0.37 at five cards (600 deals). These involve different bidding and play procedures; describe the empirical contrast without claiming that information alone was randomized. Explain the forbidden-bid probability and the gap between the best and next-best legal score as the mechanism to test.

Include the exact one-card score threshold P(win) > 10/21 where both bids are legal. The dealer's forced legal choice and followers' bid-conditioned information require separate treatment.

## Lead control and bidding distributions

Start with the covariance identity linking variance of total tricks to dependence between individual card wins. Establish the independent-card bidding lemmas and distinguish them from their applicability to actual play. Bid-dependent play means a fixed trick-count distribution is not generally valid across candidate bids.

In five-card no-trump TA rollouts, the independent-card model's cross-fit bid match is 0.739 ± 0.027 and the time-varying lead chain's is 0.978 ± 0.009; the empirical split-half reference is 0.982 ± 0.008, over 1,000 hands. These are diagnostic models fitted to per-hand rollout statistics. They do not establish that a player can infer those transition probabilities from hand features at bidding time.

Keep the failure at thirteen cards in the main results: lead alone is insufficient, and candidate models trade distribution fit against bid match. Per-suit control is a proposed explanation to test. The lead-state diagnostic supports a mechanism, not an exact Markov property of Judgement.

## Applications and limits

Use adaptive bidding inference to demonstrate the value and fragility of modeling opponents. Hybrid-RB exceeds plain Hybrid by 6.51 ± 1.81 points per game in the model-bidding pool (800 games), 2.45 ± 2.28 in the noisy pool (500), 2.72 ± 2.14 in the honour pool (500), and 3.58 ± 1.83 in the mixed pool (800). RB2 has no clear score advantage over RB in the completed comparisons. Report score and ties-as-win effects separately.

Summarize the guide as a translation exercise. It retains 0.808 [0.794, 0.822] of TA+3's edge over greedy against the TA pool (6,000 paired games). Its largest remaining loss is bidding. Evaluation against bots with perfect rule execution does not predict human learning or error rates.

Place toy equilibria in an appendix. Exact best-response calculations validate the finite-toy NashConv results, but observed CFR+ convergence in multiplayer general-sum toys is not a general convergence theorem or a certificate for the full game.

## Planned figures and tables

1. Score accounting: paired hero and opponent losses, with the exact missed-bonus/trick decomposition, using game-level CIs.
2. Surplus versus cards per player: hidden heuristic, hidden PIMC and omniscient spoilers, distinguishing the separate perfect-information hero experiment. Use matched deal subsets for paired comparisons.
3. Hook cost versus hand size, with explicit labels for information and bidding regime.
4. No-trump distribution and cross-fit bid match: independent cards, suit blocks, lead chain and empirical split-half reference.
5. A compact evidence table containing policy, information, seed design, unit of uncertainty and source for every headline result.

## Work required before a full manuscript

- Audit proof assumptions and the mapping from exact solutions to equilibrium terminology. Review the independent audit agent's findings before promoting any claim.
- Add a paired intervention that changes adversary information while keeping the hero, bids, deal population and computational method as comparable as possible. Keep it separate from the descriptive PI/hidden ratio.
- Recover a current export of the living report and review the local reference paper for literature context. Build a primary-source related-work section before making novelty claims.
- Generate publication figures from the versioned raw results and compare computed values with the evidence map.
- Finish the hand-feature bidder and particle-rejuvenation tracks independently; include them only if validation supports a material contribution.

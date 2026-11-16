# Deployable no-trump bidder: next-session protocol

**Status: protocol only. No fitted model, simulation or performance result.** The design may be reviewed before implementation. Reserved seed bases are unused and must be checked against the complete project seed ledger before launch.

## Target and acceptance

The first target is actual full-game NT bidding with TA+v2 play. Under the default25-round ladder and S/H/D/C/NT rotation, NT appears at zero-based rounds4,9,14,19,24, with hand sizes9,4,3,8,13. Each occurs once per game. The frequently studied n5 NT diagnostic is outside this default NT schedule; it is not the initial deployment target. These are deterministic schedule facts, with no sampling CI.

Primary performance contrast, to be frozen before games: **B−ModelBidder hero points per full game**, with all arms using TA+v2 play against three ModelBidder/TA+v2 opponents. Candidate B uses the new bidder in NT and canonical ModelBidder in trump rounds. Positive means improvement of this whole bidder intervention, not proof of a causal lead mechanism. Full-game challenger is seatg%4.

Secondary A uses the same learning procedure without lead/control interaction features; B−A and ties-as-win are descriptive secondary contrasts. No lead-feature benefit claim follows from B beating ModelBidder alone. No guide/HumanPlayPolicy improvement or new human advice is claimed by this design.

## Observable inputs and candidate-specific labels

Features are a pure function of own initial hand, n, relative bidding position, ordered earlier public bids, the candidate bid and the legal-bid list. Opening-lead status follows from bidding position. Allowed hand facts include suit lengths, A/K/Q/J holdings, contiguous top holdings within each suit, and long-suit/control/entry aggregates computable from that hand. Use suit-symmetric aggregates or a declared canonical ordering. No unknown holdings, actual future winners/lead persistence, realized tricks, fitted per-hand rollout transition rates or hidden `info._s` access enters prediction.

The label generator samples complete fresh deals, computes actual earlier ModelBidder bids, then evaluates each currently legal hero bid on that same deal. Later bidders respond normally to the forced hero bid; all play uses TA+v2. Store raw hit/score outcomes for every bid. Reset cloned post-prior-bidding RNG states per arm for CRN. Every distinct context/deal gets one counterfactual outcome per bid, rather than a hindsight best-bid classification label.

This direct-deal supervision integrates hidden-world variation across contexts and retains the pool's actual joint distribution of hands and earlier bids. It avoids silently treating public bids as uninformative in a conditional-world sampler. A candidate bid changes target-aware play and later bids, so labels estimate candidate-specific hit probabilities, not one exogenous trick-count law. In particular, a mode from a bid-selected law is not used as a bid-free label.

At each NT hand size and bid position, training deals are independent. Training hero may be seat0 with dealer=(−position−1)%4; final games rotate seats. Deal seed schema is `SeedSequence([base,n,position,g])`, with separate deal/policy child streams. Save seed, own hand, public context, legal bids, candidate outcomes and source/model fingerprints; the full deal can be regenerated for auditing. Features and outcome arrays have an explicit separation boundary.

## Fixed candidate family and fitting

Use numpy-only penalized binomial logistic models for P(hit|observable features,candidate bid), selecting the legal bid maximizing(10+bid)×predicted P(hit), deterministic lowest-bid tie-break. Standardization uses training data only. Every deal has total training weight1, split across its legal candidate rows, keeping hand sizes with more bids from dominating merely by row count.

Before reading training outcomes, freeze an explicit feature-name/basis manifest. A includes hand/control summaries, n, candidate-bid terms, prior-bid summaries and position main effects. B adds declared opening-lead×control/length interactions and corresponding candidate-bid interactions. No M10 per-hand fitted parameter is an input. Architecture changes after inspecting training outcomes require a documented exploratory revision before the untouched selection stage; changes after selection require a new protocol/seed allocation.

Ridge grid for each model:1e−4,1e−3,1e−2,1e−1,1,10, with an unpenalized intercept. Select on held-out actual score of the predicted legal bid, using paired candidate returns in the selection data, with deterministic ties favoring stronger regularization. Selection results are descriptive and do not validate an improvement. Freeze final A/B code, feature manifest, coefficients and all source/model hashes before screen games. Numerical fitting failure is reported rather than repaired on screen/confirmation outcomes.

## Stages, seeds and fixed counts

| Stage | Reserved base | Planned independent contexts/games | Use |
|---|---|---|---|
| Resource pilot | 1840M | 160 contexts,8 per size×position cell | Runtime/memory feasibility only; no game-value selection |
| Training | 1800M | 20,000 contexts,1,000 per cell | Fit A/B on raw bid-specific outcomes |
| Selection | 1810M | 8,000 contexts,400 per cell | Freeze regularization once |
| Screen | 1820M | 2,000 paired full-game triplets | Frozen B/A/ModelBidder, CRN |
| Fresh confirmation | 1830M | 2,000 paired full-game triplets | Same frozen candidates, no retuning |

These are design counts, not executed n values. There are five NT sizes and four bidding positions. No draw on these bases has occurred in R4. A resource-only pilot may reject the scope before training; it cannot choose candidates by score. If budget/capacity prevents the fixed counts, save complete contexts or game triplets and report a shortfall; do not convert a favorable partial run into confirmation. A new execution session needs an authorized hard deadline and at most two machine compute processes.

Full-game seeds are base+g, challenger seatg%4; all three arms share seeded deals and rotated arm order. Raw results save incrementally by complete triplet. All bidders/play models are frozen/stateless during evaluation. Check every score from bids/tricks, trick conservation, hook legality, paired deals/earlier hero contexts and ties-as-win from totals. Non-NT round outcomes should be identical across arms under these frozen deterministic policies; any mismatch requires investigation before a claim of an NT-only intervention.

## Inference and decision rule

The independent uncertainty unit is the full paired game, stratified by rotated hero seat. Report primary mean points difference and95% paired cluster CI at each stage, with exact n and seed base. All five NT rounds remain within each game's cluster; do not call10,000 rounds10,000 independent games. Ties-as-win use totals. Report both screen and fresh confirmation, including nulls/negative results. Promote only a predeclared B−ModelBidder gain with positive95% intervals in both stages; disagreement permits no improvement claim.

Report secondary contrasts and cell diagnostics with n/95% CIs as descriptive, without presenting a selected cell or feature as independently confirmed. No effect-based early stopping, pooling to rescue failed replication, or candidate reselection from game outcomes. Diagnostic calibration cannot substitute for game score. Training/selection accuracy and flexible feature fits do not establish a mechanistic lead effect.

## Leakage and reproducibility checks before launch

- Mutating unrevealed hands while holding own hand/public context fixed must leave features and predictions identical.
- Features never receive raw full deals or outcome fields; store and audit the exact named schema.
- Scores and legal-bid argmax reproduce on saved contexts, including dealer hook and ties; all non-NT behavior remains canonical.
- Check new bases against all prior registrations. Do not tune on1500M/1510M guide data or change the frozen1700M/1710M MCMC experiment.
- Preserve fitting failures, source hashes, model hashes, seed manifests and raw paired outputs. Add only new modules; core/existing policies remain unchanged.

This protocol offers a feasible deployable target without assuming the per-hand LC diagnostics are available at bidding. The final comparison measures a trained bidder under specified play/opponents; it does not isolate information causality, prove an optimal bidder, or predict human performance.

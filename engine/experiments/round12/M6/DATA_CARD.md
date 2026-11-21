# Round12 research artifact card — local, review pending

Purpose: audit learning from a fully informed teacher when a deployed decision
must use private/public information. A second, separate experiment tests a
baseline-preserving residual bidder in the full standard game. Not a general
RL leaderboard, human-performance study, equilibrium solver or52-card optimum.

## Exact root benchmark

Four-player S2R4n2/S3R4n3, trump0/NT, all2520/369600ordered deals per condition.
Fixed myopic bids, fixed private/public TAhero. Root is the first non-forced
spoiler action after legal forced/hero moves. Hero-impossible branches are
pruned to constant zero; their omitted actions cannot affect score. Terminal
roots have no student action. After the root action, continuation is omniscient
minimization. This makes the task a one-decision contextual learning benchmark;
it deliberately excludes student-induced pre-root occupancy shift and later
sequential information gathering. It is not full private-team execution.
The first-free-decision selection is part of the relaxation: a teammate cannot
generally infer whether an earlier teammate had multiple legal actions. No
executable root-only intervention is presumed; identical private keys may also
occur later under other worlds. Completed M9 acts at every private key without a
hidden root-stage flag. All40profiles were evaluated exactly; Claude independently
replayed8profiles over all deals. Every large deployed profile is worse than
the existing feasible private policy. Root keys recur later in9164/41712trump
and16192/42440NTkeys (exact counts). This mixed-hand deployment is not held-out
team generalization.

Portable NPZ stores integer per-world counterfactual costs, private-group IDs,
lexicographic deal indices, group chance counts, legal masks, exact target
counts, public payoff, and own-hand split. Separate JSONL exposes strictly
named private observations; full-world costs are labels only. Legal action
indices refer to sorted original own-hand cards, not pooled team cards.

Own initial hands are split60/20/10/10 by a frozen SHA256 ordering, shared across
seats/histories and across trump conditions within a deck. Small control has
28possible hands, including only1 reached hand per held-out set (3possible partition hands); large has220possible
hands,20/16 distinct reached hands in discovery/confirmation (22possible partition hands). All worlds for an own hand stay together. The same
five fitted learners evaluate both disjoint held-out sets; these are not two
independent training runs. t95%CIs across optimization seeds quantify fitting
variability only, not uncertainty over arbitrary future hands/games. Conditional
held-out free-root populations are evaluated completely and exactly per fit;
their values exclude terminal-root constants and have no world-IID CI.

Large labels inherit the externally audited tree and match exact PI/J1 rationals;
fresh manual recursion checks64fixed large deals/regime and all small deals.
That is not a second independent large full-population reimplementation. Raw
source hashes depend on recorded line endings; a clean-platform reproduction
needs an explicit derivative manifest rather than rewriting original freezes.

Root supervision differs in information retained from the common exactQdataset:
hardCE uses one action; uniformCE/setBCE the full optimal set; Q/direct all costs.
Equal worlds/parameters/optimization do not equalize label richness or oracle
query cost. UniformCE is the same-set control; no label-efficiency or universal
loss-superiority claim follows from the primary hard-label comparison.

## Full52-card bidder experiment

Fresh contexts have balanced actual NT sizes9,4,3,8,13 and four bid positions.
All legal hero bids are counterfactually rolled out on common worlds/RNG states
with canonical remaining bidders and TA+v2play. Synthetic deals and algorithmic
labels only; no personal data, human subjects or user gameplay collected.
Baseline probabilities, own hand, public prior bids, hook and rules are prediction
inputs. Full hidden hands exist only in raw replay records, never the predictor.

New1900M–1980Mstage seeds and2030Moptimizer seeds; originalD5/R4 results,
models and old1700M/1710M/1800M–1840Mstages remain untouched. Selection is
optimistic by construction; only frozen fullgame CRN screen/confirmation supports
performance claims. Specified human/greedy-bidder shifts are algorithmic pools,
not evidence about people or arbitrary distribution shifts. Every player uses
the same fixedTA+v2play, so this is a whole-bidder intervention in a fixed pool.

M10 reuses these frozen training/selection labels. Fresh2060Mpaired optimizer
seeds fit5models per family;2040Mscreen and2050Mconfirmation each contain
2000CRNquadruplets, hero g%4. R=MLP explicit offset,L=linear offset,N=MLP
without explicit offset but retaining canonical predictions as inputs.
Different epoch/blend choices confound an attribution to nonlinearity alone.
M4/M5/M10 first Git commits are post-run, with in-run source/data hashes
rather than pre-run commit provenance. Original freezes remain unchanged.

## Release and limitations

All new proof claims externally checked by Claude2026-11-17, corrections applied. Round12 remains local until separate
scientific/source/reproduction review. Do not upload private master or mixed
logbooks. Existing public lineage and11-group paper unchanged. Original raw data,
negative results and failed runs preserved. Repository licensing has not been
established here; do not invent third-party redistribution rights or redistribute
the local reference PDF. No claim that a NumPy MLP is a novel ML algorithm.

## Learning-curve seeds and gate coverage

M7uses2020M fresh paired optimization seeds and nested25/50/100%training-key
prefixes; every fraction contains all117 reached training hands. It is a
key-coverage diagnostic, not independent-hand sample efficiency. Five-fit
t95%CIs exclude hand-sampling uncertainty; chance populations are exact.
The full pytest command covers engine/tests only (testpaths); standalone
verify scripts have their own saved JSON receipts.

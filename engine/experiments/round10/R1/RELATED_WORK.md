# Bounded primary-source map

This is a source audit, not an exhaustive review or a novelty finding. The exact queries are in queries.txt; source metadata, inspected sections and limits are in source_map.json. Search snippets from secondary aggregators were not used as technical evidence.

| ID | Verified connection | Boundary for this project |
|---|---|---|
| L0 | Parmar et al.(2015) already discuss a Judgment AI system. SectionsIV-V describe modules and future algorithm work. | This supplied paper is a proposal, not a completed matched benchmark. It does not establish our scoring/hook protocol. Its broad priority statements should not be adopted. [Publisher copy](https://ijcsmc.com/docs/papers/April2015/V4I4201538.pdf) |
| L1 | Solinas et al.(2023) give MCMC history generation using RingSwap and a joint-policy reach target; Oh Hell evaluation is value estimation. | Frozen D5 uses bid-only weights and symmetric pair swaps. Their irreducibility/stationarity results cannot simply be imported; their empirical task is not our full-game score comparison. No broad MCMC-in-Oh Hell novelty claim. [Primary paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/87ee1bbac4635e7c948f3eea83c1f262-Paper-Conference.pdf) |
| L2 | Long et al.(2010) study when PIMC succeeds despite strategy-fusion/non-locality concerns. | Method context rather than a certificate for our implemented spoiler or hidden-team optimum. [AAAI paper](https://cdn.aaai.org/ojs/7562/7562-13-11092-1-2-20201228.pdf) |
| L3 | Cowling et al.(2012) search information sets and study variant-specific determinization issues. | Avoid saying all ISMCTS variants eliminate strategy fusion or imply optimal play. [Repository primary copy](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) |
| L4/L5 | The original CFR and MCCFR sources connect counterfactual regret, sampling and equilibrium approximation under stated assumptions. | The inspected equilibrium connection is two-player zero-sum/perfect recall. It does not certify general-sum multiplayer or decentralized-team convergence. [CFR](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf), [MCCFR](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf) |
| L6 | von Stengel/Koller(1997) motivate separate decision-makers with identical payoffs. | Abstract-only terminology connection: full theorem use remains unaudited. Our fixed-hero task is not automatically their strategic-adversary equilibrium problem. [Author copy](https://ai.stanford.edu/~koller/Papers/von Stengel%2BKoller%3AGEB97.pdf) |

The supplied OtherPaper.pdf has template metadata. Identification uses its visible title/authors, journal header and all four pages; its SHA is recorded. Publisher byte identity was not tested. Extraction/rendering intermediates remain in work/ and are excluded from commits.

Next: inspect the full team-maxmin source and CFR+ reference, broaden game-specific bidding search, and compare rule variants. No imported numerical result is used to calibrate this project's game-value estimates.

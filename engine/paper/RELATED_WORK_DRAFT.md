# Related work — bounded draft

This section records methodological connections from the first primary-source pass. Coverage remains incomplete, and no priority or novelty claim follows.

Parmar et al. describe a Judgment AI system proposal, including prediction and play modules. It establishes relevant game-specific precedent, with algorithm completion left to future work. [Parmar et al.,2015](https://ijcsmc.com/docs/papers/April2015/V4I4201538.pdf)

PIMC and information-set search provide distinct approaches to hidden-information decisions. Their limitations motivate identifying the implemented method rather than equating its value with optimal punishment. [Long et al.,2010](https://cdn.aaai.org/ojs/7562/7562-13-11092-1-2-20201228.pdf), [Cowling et al.,2012](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)

MCMC filtering already has direct Oh Hell precedent. The published RingSwap/history target differs from our frozen bid-weighted particle experiment; no playing-strength or mixing theorem is transferred. [Solinas et al.,2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/87ee1bbac4635e7c948f3eea83c1f262-Paper-Conference.pdf)

CFR/MCCFR are relevant algorithm families; the inspected equilibrium results retain two-player zero-sum assumptions. Our multiplayer/team searches require their own verification. [Zinkevich et al.,2007](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf), [Lanctot et al.,2009](https://proceedings.neurips.cc/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)

Team-maxmin work provides terminology for coincident payoffs with separate decision-makers. Only its primary abstract was available in this pass, so theorem application is deferred. [von Stengel and Koller,1997](https://www.sciencedirect.com/science/article/pii/S0899825697905273)

The source map and inspection limits are in `experiments/round10/R1/source_map.json`. Further comparison of rule variants, bidding methods, CFR+ and decentralized-team models is needed before a submission-ready related-work section.

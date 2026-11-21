# N0 pilot: learned card-location beliefs (Claude, 2026-11-17). PILOT, not a validated result.

Setup: TA tables (ModelBidder + TA play). n ∈ {5, 8, 11} × {spades, NT}. One recorded play decision per round, at a random seat and trick ≥ 1. Train: 6,000 rounds on base 2100M. Test: 600 rounds on base 2100M + 10M offset (100 per cell), 7,138 unseen held cards. Baselines are K=200 world marginals. Metric: mean log-loss (nats) of the true holder per card, averaged per decision; 95% CI over decisions (independent rounds).

**Leakage bug found and fixed in the analysis.** Cards are selected as "held by some opponent" using the true deal, which is privileged information when undealt cards exist (n < 13). The prop and net models normalize over the 3 opponents and silently benefit from this. The samplers correctly put mass on "undealt", so the raw fit.json comparison (net 0.88 vs bi 1.99) is **invalid**. Valid comparison: every method scored on the same conditional question, "given an opponent holds card c, which one?" (sampler marginals renormalized over opponents).

| method | conditional log-loss | − BI (paired) |
|---|---|---|
| prop (∝ cards left, voids zeroed) | 1.0103 ± 0.0088 | +0.0483 ± 0.0198 |
| uniform sampler | 1.0219 ± 0.0117 | +0.0598 ± 0.0210 |
| bid-informed SIR (Hybrid-BI) | 0.9620 ± 0.0232 | — |
| **residual MLP (numpy, 64 hidden)** | **0.8847 ± 0.0164** | **−0.0773 ± 0.0204** |

- Net − prop: −0.1256 ± 0.0128.
- The net beats BI in all 6 cells: −0.03 to −0.14 per cell, n=100 decisions each. The intervals exclude 0 in 4 of 6 cells.
- BI gave the true holder zero mass in 0.34% of cards (Monte Carlo sparsity; 1e-3 floor).

Reading: a small learned belief model using bids **and play history** (tricks won, need vs bid, the opponent's cards played in the suit, leads) beats the bid-only SIR posterior that Hybrid-BI uses for search. That supports N1/N4. Caveats:
- one seed and one table type (TA);
- the comparison is conditional only;
- the net's gain may come from play history, which BI ignores by design.

N1 must ablate bids-only vs history features.

Requirements for N1, from this pilot:
1. Model 4 holder classes (3 opponents + undealt) for every method, so the full unconditional log-loss is valid. Never select cards with the true deal.
2. Report both the unconditional and the conditional scores.
3. Ablate the features: bids-only net vs full net, which separates "better likelihood" from "uses play information".
4. Use larger K or analytic marginals for the samplers, so their Monte Carlo error doesn't decide the comparison.

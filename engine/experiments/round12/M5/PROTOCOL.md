# Frozen out-of-training bidder pools — before outcomes

Run the two frozenM4 ensembles without adaptation against three HumanBidder()
opponents, and separately three GreedyPolicy() bidders. Every player still uses
TA+v2play; hero canonical/ensemble unchanged fromM4. NonNThero canonical in all
arms. These are specified algorithmic shifts, not human-subject experiments or
a claim of broad OOD robustness. Existing policies are read-only dependencies.

Humanbid screen1950M and replication1970M; greedybid screen1960M and replication
1980M. Each has1000complete CRN triplets, hero g%4, base+g, canonical/BCE/direct,
rotated execution order. Save complete triplets incrementally. Primary within
each specified pool: BCE−canonical score, with equal-seat paired95%CIs. Require
positive lowerCI on both stages of a pool to claim a gain there. Four primary
stage contrasts across two pools; no omnibus/general robustness claim, no
selection of pools or changed n after outcomes. Report Directsecondary and
ties-as-win. Differences between pools descriptive; no mechanism inference.

M4 frozen before all these games, no fit/epoch/alpha changes. Hard stop07:00ET
17Nov2026; per-triplet save, one worker. Document incomplete scope without
promoting partial unreplicated results if the deadline arrives. No reduced target.

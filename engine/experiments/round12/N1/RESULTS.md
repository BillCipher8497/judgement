> Current interpretation, 2026-11-19: numerical result externally reviewed; [review corrections](REVIEW_CORRECTIONS_20261119.md) govern. BI below means K256resampled SIR/max2048proposals. Higher-budget diagnostics are post-hoc weighted estimates, not converged BI or a measured64×compute comparison.

# N1 completed — self-reviewed (Codex), pending external result audit

The fixed source was Claude-approved and committed as4c610703 before any data. The completed **results** have separate Codex raw/source/forward/selection/CI and manual prefix/sampler checks; no external N1result verdict is claimed. New results remain local and uncommitted pending later external review. No PyTorch, frozen source/data/seed edits or tuning.

## Primary and design

Four holders include undealt stock. Every unseen card is scored; no oracle filter. One independent round/decision per record, hero/observer g%4,44balanced n3..13 × S/NT × TA/TA+ cells and four observer seats. Counts30800training/4400validation/3520confirmation, plus176resource pilot. Namespaced2100Mtrain/validation/pilot and2110Mconfirmation; exact entropy addresses in PROTOCOL/STATUS and seed_audit. Five paired fit seeds per feature arm,50epochs, frozen epoch/blend choices before confirmation.

Primary full-history ensemble minus K256BI unconditional log-loss/decision: **-0.048316 [-0.053126, -0.043506] nats**, n=3520, paired equal176stratum normal95%CI. Negative is better. This is a fixed-ensemble prediction comparison on the specified synthetic policy/deal distribution, not a posterior exactness, arbitrary-hand, human or game-score guarantee. Every interval below is conditional on the selected fixed five-fit ensemble; it does not quantify refitting uncertainty.

|method|mean unconditional nats/decision [95%CI]|n independent rounds|
|---|---:|---:|
|prop|0.911919 [0.906436, 0.917402]|3520|
|uniform|0.909791 [0.903980, 0.915602]|3520|
|bi|0.885351 [0.877176, 0.893526]|3520|
|net_bids|0.874245 [0.868339, 0.880151]|3520|
|net_full|0.837036 [0.830957, 0.843114]|3520|

Bids-only retains structural capacities/voids/public card removal; full adds the declared behavioral-history summaries. Full is not a recurrent/injective full-history encoder. Both were selected on the same fixed validation distribution; compare whole trained pipelines, not Bayes information value.

|secondary paired contrast|difference [95%CI]|n|
|---|---:|---:|
|net_full-net_bids|-0.037209 [-0.038853, -0.035565]|3520|
|net_full-prop|-0.074883 [-0.077139, -0.072627]|3520|
|net_full-uniform|-0.072755 [-0.075082, -0.070429]|3520|
|net_bids-bi|-0.011106 [-0.015819, -0.006394]|3520|

## Predeclared Monte Carlo sparsity sensitivity

The declared primary floors all finite-sampler probabilities with .001prop. Its effective pseudocount is256*.001/.999. Additional **pre-confirmation** analyses use q_s=(count+a*prop)/(256+a), a1/4/16, with the same paired stratum CI. They are secondary; the primary is unchanged. No new deals, training or tuning.

|sampler|smoothing|net_full minus sampler [95%CI]|n|
|---|---|---:|---:|
|bi|declared_floor|-0.048316 [-0.053126, -0.043506]|3520|
|bi|1|-0.046276 [-0.050880, -0.041672]|3520|
|bi|4|-0.043154 [-0.047438, -0.038870]|3520|
|bi|16|-0.037928 [-0.041657, -0.034200]|3520|
|uniform|declared_floor|-0.072755 [-0.075082, -0.070429]|3520|
|uniform|1|-0.072716 [-0.075041, -0.070391]|3520|
|uniform|4|-0.072589 [-0.074911, -0.070267]|3520|
|uniform|16|-0.072165 [-0.074476, -0.069853]|3520|

The nominal advantage survives all three predeclared BI smoothing settings.

|sampler|true-holder zero count, round-equal fraction [95%CI]|pooled-card fraction [round-cluster95%CI]|n rounds|n card labels|
|---|---:|---:|---:|---:|
|bi|0.001163 [0.000925, 0.001401]|0.001225 [0.000977, 0.001472]|3520|112696|
|uniform|1/112696 cards in1/3520rounds; see corrected round-level interval|0.000009 [0.000000, 0.000026]|3520|112696|

Zero fractions describe the **unfloored K256sampler counts**; they do not assert zero true posterior probability. Pooled-card estimates use a round-cluster delta method, not card independence. Normal intervals are approximate; no equality is inferred from a null.

## Calibration and conditional diagnostics

|method|ECE [stratified bootstrap95%CI]|conditional-active card log-loss [95%CI]|n rounds|
|---|---:|---:|---:|
|prop|0.018239 [0.017403, 0.019391]|1.000429 [0.996302, 1.004556]|3520|
|uniform|0.006519 [0.005218, 0.008889]|1.006872 [1.001312, 1.012433]|3520|
|bi|0.017100 [0.015108, 0.019123]|0.934697 [0.924896, 0.944499]|3520|
|net_bids|0.004313 [0.003521, 0.007134]|0.943802 [0.938094, 0.949511]|3520|
|net_full|0.007324 [0.005987, 0.009412]|0.881485 [0.874778, 0.888192]|3520|

Conditional-active scoring uses truth **only for this labelled diagnostic** and renormalizes every method over opponents identically. Primary inputs/scoring remain all unseen cards/four holders. Reliability-bin masses/confidence/accuracy CIs and supporting round/card counts are in sparsity_diagnostics.json; ten fixed bins,2000paired stratified round resamples. Empty bins have no estimate, sparse-bin ratios use nonempty-resample intervals and are exploratory; intervals are marginal, not simultaneous.

## BI resource/degeneracy by n

|n|ESS mean [95%CI]|M mean [95%CI]|n rounds|
|---|---:|---:|---:|
|3|250.863738 [244.707765, 257.019710]|1032.800000 [967.037891, 1098.562109]|320|
|4|214.578944 [206.572722, 222.585165]|1512.600000 [1451.394527, 1573.805473]|320|
|5|182.141955 [172.235510, 192.048401]|1668.400000 [1613.044293, 1723.755707]|320|
|6|174.385388 [164.649355, 184.121422]|1676.800000 [1617.917806, 1735.682194]|320|
|7|156.570879 [146.330599, 166.811158]|1750.400000 [1693.837210, 1806.962790]|320|
|8|139.872308 [129.220293, 150.524323]|1752.400000 [1695.766049, 1809.033951]|320|
|9|127.162766 [116.606194, 137.719338]|1825.400000 [1775.773667, 1875.026333]|320|
|10|144.513449 [133.783678, 155.243220]|1775.400000 [1721.216387, 1829.583613]|320|
|11|120.863841 [110.035701, 131.691980]|1820.600000 [1768.828733, 1872.371267]|320|
|12|131.491058 [120.599651, 142.382465]|1757.400000 [1699.337249, 1815.462751]|320|
|13|141.693476 [130.287948, 153.099004]|1681.000000 [1613.190202, 1748.809798]|320|

Full ESS/M observed ranges and quartiles with2000stratified bootstrap95%CIs are retained in sparsity_diagnostics.json. Observed min/max are exact log-sample extrema, not population bounds. New secondary bootstrap namespace91was declared before confirmation and collision-checked against123544known roots; ECE/bin diagnostics replay the predeclared stream90.

## All predeclared cells (marginal intervals)

|cell|net_full minus BI nats/decision [95%CI]|n rounds|
|---|---:|---:|
|n3/NT0/table0|0.002899 [-0.005985, 0.011782]|80|
|n3/NT0/table1|0.006569 [-0.000223, 0.013362]|80|
|n3/NT1/table0|-0.003237 [-0.011247, 0.004773]|80|
|n3/NT1/table1|-0.000830 [-0.006688, 0.005028]|80|
|n4/NT0/table0|-0.003521 [-0.010913, 0.003872]|80|
|n4/NT0/table1|-0.000432 [-0.010930, 0.010066]|80|
|n4/NT1/table0|-0.005069 [-0.014414, 0.004275]|80|
|n4/NT1/table1|-0.004798 [-0.015386, 0.005791]|80|
|n5/NT0/table0|-0.024326 [-0.040280, -0.008372]|80|
|n5/NT0/table1|-0.019279 [-0.035898, -0.002659]|80|
|n5/NT1/table0|-0.018161 [-0.028343, -0.007980]|80|
|n5/NT1/table1|-0.023700 [-0.042671, -0.004729]|80|
|n6/NT0/table0|-0.042738 [-0.059336, -0.026139]|80|
|n6/NT0/table1|-0.018638 [-0.034061, -0.003216]|80|
|n6/NT1/table0|-0.020293 [-0.030209, -0.010377]|80|
|n6/NT1/table1|-0.016405 [-0.031964, -0.000846]|80|
|n7/NT0/table0|-0.043840 [-0.067300, -0.020380]|80|
|n7/NT0/table1|-0.032705 [-0.049887, -0.015522]|80|
|n7/NT1/table0|-0.024327 [-0.038108, -0.010546]|80|
|n7/NT1/table1|-0.019924 [-0.038091, -0.001757]|80|
|n8/NT0/table0|-0.074430 [-0.103296, -0.045564]|80|
|n8/NT0/table1|-0.064342 [-0.095478, -0.033207]|80|
|n8/NT1/table0|-0.037648 [-0.055346, -0.019951]|80|
|n8/NT1/table1|-0.045116 [-0.064975, -0.025257]|80|
|n9/NT0/table0|-0.088982 [-0.128304, -0.049660]|80|
|n9/NT0/table1|-0.055530 [-0.092762, -0.018297]|80|
|n9/NT1/table0|-0.059592 [-0.093963, -0.025221]|80|
|n9/NT1/table1|-0.085208 [-0.116851, -0.053564]|80|
|n10/NT0/table0|-0.091693 [-0.135005, -0.048381]|80|
|n10/NT0/table1|-0.064575 [-0.095235, -0.033916]|80|
|n10/NT1/table0|-0.080096 [-0.108885, -0.051307]|80|
|n10/NT1/table1|-0.111756 [-0.176638, -0.046875]|80|
|n11/NT0/table0|-0.124422 [-0.166065, -0.082778]|80|
|n11/NT0/table1|-0.077767 [-0.118496, -0.037038]|80|
|n11/NT1/table0|-0.107436 [-0.151480, -0.063391]|80|
|n11/NT1/table1|-0.104771 [-0.134532, -0.075010]|80|
|n12/NT0/table0|-0.138696 [-0.190624, -0.086769]|80|
|n12/NT0/table1|-0.046894 [-0.095803, 0.002015]|80|
|n12/NT1/table0|-0.079004 [-0.135428, -0.022581]|80|
|n12/NT1/table1|-0.090352 [-0.125234, -0.055469]|80|
|n13/NT0/table0|-0.022292 [-0.067942, 0.023359]|80|
|n13/NT0/table1|-0.009298 [-0.053020, 0.034424]|80|
|n13/NT1/table0|-0.089812 [-0.158391, -0.021233]|80|
|n13/NT1/table1|-0.063418 [-0.096648, -0.030187]|80|

All44cells shown; no selected-cell or multiplicity-adjusted superiority claim. No extension of confirmation, source/model changes or threshold/seed retuning.

## Separate validation and limitations

Independent numeric/raw-label audit covers30800training,4400validation,3520confirmation rounds and both model selections; primary CI reproduced exactly. Manual replay checks2816prefixes without RoundState/make_rngs/deal, plus880K256uniform/BIdecisions; all saved marginals/stats match. The frozen bidder/play functions and sampler algorithms are reused as definitions, not independently reimplemented. This is not an exact posterior certificate. All source/model/raw manifests and original results are retained.

Proper four-class prediction loss can improve without improving search or score. Sampling budgets, smoothing and the known fixed table type matter. Observational histories come from controlled TA/TA+ tables with canonical bidders, not humans or arbitrary opponents. The protocol and source preceded N1data; additional diagnostics were declared before confirmation. Separate diagnostic modules are post-source-freeze additions and have their own hashes; original approved source stays untouched. New tests/verification and later external audit are separate gates. No publication/push.


## Claude external audit addendum (2026-11-18)
See root `CLAUDE_AUDIT_2026-11-18.md`, section N1, for the verdict, the post-hoc diagnostics and the requested changes. That section governs the interpretation of this packet: where it conflicts with the text above, the audit wording applies.

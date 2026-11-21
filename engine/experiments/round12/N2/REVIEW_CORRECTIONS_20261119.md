# N2 independent verification and interpretation — 2026-11-19

Claude numbers reproduce. These are complete fixed test populations, not four fresh sampling replications. Optimizer-seed CIs and hand-sensitivity intervals answer different questions.

|Condition|n test hands|Hand-sensitivity t95% net−BI|Hands favouring net|Oracle plug-in joint KL|Numerical IPF candidate KL|
|---|---:|---|---:|---:|---:|
|S2R4n2_t0|6|+0.097377 [-0.112947, +0.307700]|2|0.056321501230042856|0.054807825405261065|
|S2R4n2_t-1|6|+0.250126 [+0.119466, +0.380786]|0|0.06848345577615075|0.06508407297267597|
|S3R4n3_t0|44|+0.500435 [+0.468840, +0.532030]|0|not enumerated|not enumerated|
|S3R4n3_t-1|44|+0.493414 [+0.439020, +0.547808]|0|not enumerated|not enumerated|

Hand intervals use t(dfH−1) over seed-mean hand contrasts; they are sensitivity to a hypothetical hand population, not sampling intervals over the exact enumerated deals. Small trump favours the net on2/6hands and its hand interval crosses zero. The positive fixed-population means and original n5optimizer-seed CIs remain correct. Avoid per-hand or key-wise dominance and a claim of four independent replications.

Independent exhaustive small-support calculations reproduce plug-in0.0563215/0.0684835 and IPF0.0548078/0.0650841. The oracle plug-in is a feasible product-family witness below matched soft-BI on aggregate. It uses exact posterior marginals; it is not a deployable network and is not restricted to the frozen35-feature neural family/floor. IPF is a numerical optimization diagnostic, not a rigorous certified global optimum. It need not beat BI on every hand/key.

|Condition|Prop card CE (exact population)|Net card CE [optimizer-seed t95%]|Full-info marginal entropy floor (exact population)|
|---|---:|---|---:|
|S2R4n2_t0|0.840903|0.671672 [0.666628, 0.676715], n5fits|0.501092|
|S2R4n2_t-1|0.770027|0.598239 [0.592108, 0.604369], n5fits|0.387543|
|S3R4n3_t0|0.813517|0.661155 [0.658813, 0.663496], n5fits|0.448047|
|S3R4n3_t-1|0.834495|0.656124 [0.653857, 0.658391], n5fits|0.436216|

These quantities use the entire declared reached test population:6hands/small condition and44hands/large, with4320/887040decision-world weight respectively. Exact means/floors receive no chance-sampling intervals. The full-info floor is conditioned on a richer history than the shared summary encoder may retain.

All20fits selected epoch20and alpha1at the upper grid boundaries; this suggests unresolved optimization/training headroom. It **does not causally establish** that training duration alone explains the gap: feature information, finite key coverage and approximation remain possible contributors. The matched BI/hard-bid methods know the exact data-generating bidder; they are oracle-likelihood baselines. A longer/richer/structured-loss follow-up needs a new protocol and unused fit streams, not a retune or extension of the frozen2100M/0x4E32 packet.

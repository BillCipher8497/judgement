# M7 learning curves — Claude external validated, corrections applied


This is a descriptive key-coverage curve, not a learning curve over independent hands. All three nested training fractions contain all117 reached training hands. Feature-compression diagnostics were not produced here; M2 contains the exact representation check. All30 fits and evaluations are preserved unchanged.

Each fixed policy is evaluated on the complete held-out root population; exact per-fit risks have no chance-sampling CI. Table intervals are five-fit t95% intervals (df4), paired for differences. They do not cover dominant hand-sampling uncertainty. Discovery and confirmation reuse the same fits.

| fraction | held-out population | hard CE excess | set BCE excess | paired set-hard | n fits |
|---|---|---:|---:|---:|---:|
|0.25|discovery|0.381292 +/- 0.004875|0.086450 +/- 0.011795|-0.294842 +/- 0.009599|5|
|0.25|confirmation|0.630896 +/- 0.034152|0.170173 +/- 0.065107|-0.460723 +/- 0.078688|5|
|0.5|discovery|0.375078 +/- 0.011196|0.025872 +/- 0.017964|-0.349207 +/- 0.013776|5|
|0.5|confirmation|0.590600 +/- 0.020523|0.051660 +/- 0.022051|-0.538940 +/- 0.026481|5|
|1.0|discovery|0.355276 +/- 0.018600|0.023467 +/- 0.016041|-0.331809 +/- 0.025310|5|
|1.0|confirmation|0.593377 +/- 0.030635|0.093426 +/- 0.067161|-0.499951 +/- 0.072534|5|

| held-out population | exact worlds | private keys | reached own hands | hard label decoder excess | set decoder excess | information floor |
|---|---:|---:|---:|---:|---:|---:|
|discovery|38753|4967|20|0.353391995|0|0.008825123|
|confirmation|24340|2549|16|0.565817584|0|0.031347576|
Hard CE remains near its label-decoding floor at every fraction. Set BCE avoids that label-induced cost. Discovery set-BCE excess declines from about0.086 to0.023; confirmation is not monotone. No trend claim is made. The population label decoder is a reference, not a nonnegative component of each finite learner risk: finite learners can beat it.
Source/seed provenance: 2020M five paired seeds at each fraction, 30 fits total. Frozen files as executed; first Git commit is post-run. Claude independently reproduced all30 fit summaries and independently forwarded two fitted models with zero action mismatches.

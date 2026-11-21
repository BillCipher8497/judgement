# Completed M5 packet — ready for Claude external review

Frozen data/models/source unchanged. Independent source/raw/selection/replay checks in M6/results. All contrasts below paired equal-seat normal95%CIs; no universal or human-benefit claim.

human_screen bce-canonical score: 0.443000 +/-0.601369, n=1000,95%CI [-0.15836893700421067, 1.0443689370042106].

human_screen bce-canonical ties_as_win: -0.003000 +/-0.018004, n=1000,95%CI [-0.021004445858807508, 0.01500444585880751].

human_screen direct-canonical score: 0.260000 +/-0.583854, n=1000,95%CI [-0.32385429832337687, 0.8438542983233769].

human_screen direct-canonical ties_as_win: -0.007000 +/-0.016961, n=1000,95%CI [-0.023961368248210567, 0.009961368248210568].

human_screen bce-direct score: 0.183000 +/-0.480210, n=1000,95%CI [-0.29720954207386924, 0.6632095420738693].

human_screen bce-direct ties_as_win: 0.004000 +/-0.014936, n=1000,95%CI [-0.010936214581482698, 0.0189362145814827].

human_confirmation bce-canonical score: 0.043000 +/-0.579335, n=1000,95%CI [-0.5363351547085026, 0.6223351547085024].

human_confirmation bce-canonical ties_as_win: -0.014000 +/-0.018144, n=1000,95%CI [-0.032144009023747075, 0.004144009023747076].

human_confirmation direct-canonical score: 0.204000 +/-0.569388, n=1000,95%CI [-0.36538831512774833, 0.7733883151277483].

human_confirmation direct-canonical ties_as_win: -0.010000 +/-0.018159, n=1000,95%CI [-0.02815930823302496, 0.00815930823302496].

human_confirmation bce-direct score: -0.161000 +/-0.467804, n=1000,95%CI [-0.6288043826714178, 0.30680438267141785].

human_confirmation bce-direct ties_as_win: -0.004000 +/-0.013602, n=1000,95%CI [-0.017601982373830365, 0.009601982373830366].

greedy_screen bce-canonical score: 0.031000 +/-0.572388, n=1000,95%CI [-0.5413875910464305, 0.6033875910464304].

greedy_screen bce-canonical ties_as_win: -0.006000 +/-0.018369, n=1000,95%CI [-0.02436880015888649, 0.012368800158886492].

greedy_screen direct-canonical score: -0.093000 +/-0.581993, n=1000,95%CI [-0.67499306176187, 0.48899306176187].

greedy_screen direct-canonical ties_as_win: 0.004000 +/-0.018768, n=1000,95%CI [-0.01476762657420793, 0.02276762657420793].

greedy_screen bce-direct score: 0.124000 +/-0.461739, n=1000,95%CI [-0.3377394034904786, 0.5857394034904786].

greedy_screen bce-direct ties_as_win: -0.010000 +/-0.015437, n=1000,95%CI [-0.02543705363401962, 0.005437053634019619].

greedy_confirmation bce-canonical score: 0.168000 +/-0.563565, n=1000,95%CI [-0.3955654835357926, 0.7315654835357926].

greedy_confirmation bce-canonical ties_as_win: 0.009000 +/-0.016746, n=1000,95%CI [-0.007746247340822361, 0.025746247340822363].

greedy_confirmation direct-canonical score: -0.066000 +/-0.547752, n=1000,95%CI [-0.6137521764022611, 0.481752176402261].

greedy_confirmation direct-canonical ties_as_win: 0.013000 +/-0.016039, n=1000,95%CI [-0.0030394443513608524, 0.029039444351360855].

greedy_confirmation bce-direct score: 0.234000 +/-0.454227, n=1000,95%CI [-0.2202270184632438, 0.6882270184632439].

greedy_confirmation bce-direct ties_as_win: -0.004000 +/-0.016415, n=1000,95%CI [-0.020415462590137164, 0.012415462590137163].

Canonical-pool score gain replicated; cross-pool differences are untested; M5shift intervals include zero and establish no gain. Nulls retained. Claude external review/public reproduction gate pending; no commit/push yet.


## Claude external17Nov corrections applied before commit

No gain detected in the shift pools; cross-pool differences are untested and descriptive. The canonical-pool score gain replicated, but the win gain did not; BCE versus Direct is null in both stages. This is a whole-bidder intervention versus3canonical ModelBidder opponents with fixedTA+v2play, not a human benefit or established bid-shift robustness.

Provenance: code was hashed in-run but never committed before the run; no predeclared selection-stage non-inferiority gate. Selection already showed+0.193±0.074points/NTcontext,n12000,95%CI(descriptive/optimistic); BCEalpha1is a grid endpoint and3/5seeds chose the lowest epoch. Epoch andalpha reused the same selection contexts. No retuning or fresh outcome selection.

screencanonical mean166.3465points/game,95%CI[165.0871870259162, 167.6058129740838],n2000; BCEgain fraction0.007310,paired delta-method95%CI[0.004945534165007282, 0.009674551018630486]. Posthocscale context only, about1.1–1.2points/game and under1%ofcanonical mean.

confirmationcanonical mean164.4895points/game,95%CI[163.23456036809213, 165.74443963190785],n2000; BCEgain fraction0.006760,paired delta-method95%CI[0.004276990604244756, 0.009243629252949777]. Posthocscale context only, about1.1–1.2points/game and under1%ofcanonical mean.

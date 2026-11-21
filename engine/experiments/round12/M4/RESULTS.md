# Completed M4 packet — ready for Claude external review

Frozen data/models/source unchanged. Independent source/raw/selection/replay checks in M6/results. All contrasts below paired equal-seat normal95%CIs; no universal or human-benefit claim.

screen bce-canonical score: 1.216000 +/-0.391756, n=2000,95%CI [0.8242435978329419, 1.607756402167058].

screen bce-canonical ties_as_win: 0.020500 +/-0.011924, n=2000,95%CI [0.008575606282453515, 0.03242439371754649].

screen direct-canonical score: 1.319000 +/-0.383391, n=2000,95%CI [0.9356090513465154, 1.7023909486534845].

screen direct-canonical ties_as_win: 0.022000 +/-0.011475, n=2000,95%CI [0.010525116248334868, 0.03347488375166513].

screen bce-direct score: -0.103000 +/-0.306238, n=2000,95%CI [-0.409238333260741, 0.20323833326074095].

screen bce-direct ties_as_win: -0.001500 +/-0.009948, n=2000,95%CI [-0.01144784866119162, 0.008447848661191621].

confirmation bce-canonical score: 1.112000 +/-0.406709, n=2000,95%CI [0.7052910961691267, 1.5187089038308734].

confirmation bce-canonical ties_as_win: 0.010000 +/-0.010908, n=2000,95%CI [-0.0009075258900487684, 0.020907525890048767].

confirmation direct-canonical score: 0.994500 +/-0.377119, n=2000,95%CI [0.6173813676943763, 1.3716186323056236].

confirmation direct-canonical ties_as_win: 0.010000 +/-0.010454, n=2000,95%CI [-0.0004543357067840637, 0.020454335706784066].

confirmation bce-direct score: 0.117500 +/-0.308126, n=2000,95%CI [-0.19062644290166086, 0.42562644290166085].

confirmation bce-direct ties_as_win: 0.000000 +/-0.008644, n=2000,95%CI [-0.008643999946213574, 0.008643999946213574].

Canonical-pool score gain replicated; cross-pool differences are untested; M5shift intervals include zero and establish no gain. Nulls retained. Claude external review/public reproduction gate pending; no commit/push yet.


## Claude external17Nov corrections applied before commit

No gain detected in the shift pools; cross-pool differences are untested and descriptive. The canonical-pool score gain replicated, but the win gain did not; BCE versus Direct is null in both stages. This is a whole-bidder intervention versus3canonical ModelBidder opponents with fixedTA+v2play, not a human benefit or established bid-shift robustness.

Provenance: code was hashed in-run but never committed before the run; no predeclared selection-stage non-inferiority gate. Selection already showed+0.193±0.074points/NTcontext,n12000,95%CI(descriptive/optimistic); BCEalpha1is a grid endpoint and3/5seeds chose the lowest epoch. Epoch andalpha reused the same selection contexts. No retuning or fresh outcome selection.

screencanonical mean166.3465points/game,95%CI[165.0871870259162, 167.6058129740838],n2000; BCEgain fraction0.007310,paired delta-method95%CI[0.004945534165007282, 0.009674551018630486]. Posthocscale context only, about1.1–1.2points/game and under1%ofcanonical mean.

confirmationcanonical mean164.4895points/game,95%CI[163.23456036809213, 165.74443963190785],n2000; BCEgain fraction0.006760,paired delta-method95%CI[0.004276990604244756, 0.009243629252949777]. Posthocscale context only, about1.1–1.2points/game and under1%ofcanonical mean.

#!/bin/sh
# waits for run_r3.sh, then H2 analysis + trump contrast, then the policy-aware MC spoiler (deadline-bounded)
cd "$(dirname "$0")"
while ! grep -q ALLDONE results/r3.log; do sleep 20; done
DL=$(TZ=America/New_York date -d "03:30" +%s)
python ../../math/m10_analyze.py > ../../math/results/m10.log 2>&1
python ../../math/m10_sim.py --ns 5,8 --trumps 0 --hands 500 --R 300 --deadline $DL
python ../../math/m10_analyze.py --cfg 5:S,8:S --tag _S > ../../math/results/m10_S.log 2>&1
SPOILER=mc:8 CONDS=self,s3 python sim.py rounds 2,3,4,5 1000 mc_2_5 $DL
echo ALLDONE_B

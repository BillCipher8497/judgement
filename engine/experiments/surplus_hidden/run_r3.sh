#!/bin/sh
# Round-3 chained queue (one process at a time).
cd "$(dirname "$0")"
DL=$(( $(date +%s) + 9000 ))
SPOILER=omni CONDS=self,s3 python sim.py rounds 2,3,4,5,6 4000 omni_2_6 $DL
SPOILER=omni CONDS=self,s3 python sim.py rounds 7 2000 omni_7 $DL
SPOILER=pimc:16 CONDS=self,s3 python sim.py rounds 2,3,4,5 2000 pimc_2_5 $DL
SPOILER=pimc:16 CONDS=self,s3 python sim.py rounds 6 2000 pimc_6 $DL
python ../../math/m10_sim.py --ns 5,8,13 --trumps -1 --hands 1000 --R 300 --deadline $DL
SPOILER=omni CONDS=self,s3 python sim.py rounds 8 1000 omni_8 $DL
SPOILER=pimc:16 CONDS=self,s3 python sim.py rounds 7 1000 pimc_7 $DL
echo ALLDONE

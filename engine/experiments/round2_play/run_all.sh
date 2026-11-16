#!/bin/bash
# Sequential queue (one process at a time).
# Seeds: TA+ screen 31,000,000+g; validation 41,000,000+g (1v3_f_*, TA+ with lead_high=2);
# final 1v3 61,000,000+g (1v3_x_*); mixed tables 51,000,000+b.
cd "$(dirname "$0")"
export R2_BASE=61000000
python one_vs_three.py ta 3000 x_ta
python one_vs_three.py greedy 3000 x_greedy
python one_vs_three.py ta_spite 3000 x_ta_spite
python one_vs_three.py ta_plus 3000 x_ta_plus
python one_vs_three.py "ta_plus:shed=1;lh=0;thm=0" 3000 x_abl_shed
python one_vs_three.py "ta_plus:shed=0;lh=3;thm=0" 3000 x_abl_leadhigh
python one_vs_three.py "ta_plus:lh=2" 3000 x_abl_lh_allrounds
python one_vs_three.py "ta_plus:thm=0" 3000 x_abl_nothm
python one_vs_three.py "ta_plus:duck=1;dm=1" 3000 x_abl_plusduck
python one_vs_three.py "hybrid:k=16,c=5" 800 x_hybrid16
python pi_agree.py 500 5
python e1_taplus.py 500
python one_vs_three.py "hybrid:k=32,c=5" 400 x_hybrid32
python mixed.py 100000 1794308700
echo ALLDONE

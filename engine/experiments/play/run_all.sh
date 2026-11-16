#!/bin/bash
# Sequential queue (one process at a time). 1v3 seeds 71,000,000+g; mixed 81,000,000+b.
# (screen.sh: win_high screen on 91,000,000+g, run before this.)
cd "$(dirname "$0")"
export PLAY_BASE=71000000
D1=1794375300   # 00:35 ET hard stop
D2=1794376800   # 01:00 ET hard stop
G=4000
python one_vs_three.py ta $G x_ta
python one_vs_three.py ta_plus $G x_ta_plus
python one_vs_three.py "ta_plus:shed=0;lh=0;thm=0;wh=0" $G x_none
python one_vs_three.py "ta_plus:shed=0" $G x_no_shed
python one_vs_three.py "ta_plus:lh=0" $G x_no_lh
python one_vs_three.py "ta_plus:thm=0" $G x_no_thm
python one_vs_three.py "ta_plus:wh=0" $G x_no_wh
python one_vs_three.py "ta_plus:lh=0;thm=0;wh=0" $G x_shed_only
python one_vs_three.py "ta_plus:shed=0;thm=0;wh=0" $G x_lh_only
python one_vs_three.py "ta_plus:shed=0;lh=0;wh=0" $G x_thm_only
python one_vs_three.py "ta_plus:shed=0;lh=0;thm=0" $G x_wh_only
python one_vs_three.py "ta_plus:lh=2" $G x_lh_all
python one_vs_three.py "ta_plus:duck=1" $G x_plus_duck
python one_vs_three.py "ta_plus:duck=1;dlo=1" $G x_plus_duck_last
python one_vs_three.py "ta_plus:sc=1" $G x_plus_scarce
python one_vs_three.py greedy $G x_greedy
python one_vs_three.py ta_spite $G x_ta_spite
echo T1DONE
python pi_agree.py 1000 5
echo T4DONE
python one_vs_three.py "hybrid:k=16;c=4" 1000 x_hyb_c4 $D1
python one_vs_three.py "hybrid:k=16;c=5" 1000 x_hyb_c5 $D1
python one_vs_three.py "hybrid:k=16;c=4;nt=1" 1000 x_hyb_c4_nt $D1
python one_vs_three.py "hybrid:k=16;c=5;nt=1" 1000 x_hyb_c5_nt $D1
echo T2DONE
python one_vs_three.py "tap_oracle:K=24;nmax=5" 2000 x_orc_n5 $D1
python mixed.py 60 $D1
echo MIXEDDONE
python one_vs_three.py "tap_oracle:K=24;nmax=13" 800 x_orc_all $D2
python one_vs_three.py "hybrid:k=32;c=5" 400 x_hyb_k32_c5 $D2
echo ALLDONE

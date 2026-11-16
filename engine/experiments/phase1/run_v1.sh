#!/bin/sh
# Rerun of every TA-dependent experiment with the corrected TA (joint P(win)).
cd "$(dirname "$0")"
export PYTHONPATH=../..
R=results
python fixed_point.py --play ta --iters 5 --rounds 8000 > $R/log_fp_ta.txt 2>&1
python fp_continue.py --tag ta --start 5 --iters 4 --rounds 8000 > $R/log_fp_continue_ta.txt 2>&1
python bidder_eval.py --play ta --sits 390 --samples 160 > $R/log_eval_ta.txt 2>&1
python games.py --configs ta,spite1,spite_all,ta_nohook,greedy_play_ta_bids --games 10000 > $R/log_games.txt 2>&1
python e1_shift.py > $R/log_e1.txt 2>&1
python e1_extend.py > $R/log_e1_extend.txt 2>&1
python e2_specific.py > $R/log_e2_specific.txt 2>&1
python e2_ltv.py > $R/log_e2_ltv.txt 2>&1
for m in neg_bid neg_10 neg_diff; do
  python fixed_point.py --play ta --miss $m --iters 3 --rounds 6000 > $R/log_fp_ta_$m.txt 2>&1
  python fp_continue.py --tag ta_$m --miss $m --start 3 --iters 2 --rounds 6000 > $R/log_fp_continue_ta_$m.txt 2>&1
  python games.py --configs ta_$m --games 6000 >> $R/log_games.txt 2>&1
done
echo done > $R/v1_done.txt

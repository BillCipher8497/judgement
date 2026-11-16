#!/bin/sh
# Phase 1 queue: strictly sequential (one simulation process at a time).
cd "$(dirname "$0")"
export PYTHONPATH=../..
R=results
while pgrep -f "fixed_point.py --play greedy" > /dev/null; do sleep 5; done
python fp_continue.py --tag ta --start 5 --iters 4 --rounds 8000 > $R/log_fp_continue_ta.txt 2>&1
python bidder_eval.py --play ta --sits 390 --samples 160 > $R/log_eval_ta.txt 2>&1
python bidder_eval.py --play greedy --sits 390 --samples 300 > $R/log_eval_greedy.txt 2>&1
python games.py --configs ta,spite1,spite_all,greedy,ta_nohook,greedy_play_ta_bids --games 10000 > $R/log_games.txt 2>&1
python e1_shift.py > $R/log_e1.txt 2>&1
python e3_shading.py > $R/log_e3.txt 2>&1
for m in neg_bid neg_10 neg_diff; do
  python fixed_point.py --play ta --miss $m --iters 3 --rounds 6000 > $R/log_fp_ta_$m.txt 2>&1
  python fp_continue.py --tag ta_$m --miss $m --start 3 --iters 2 --rounds 6000 > $R/log_fp_continue_ta_$m.txt 2>&1
  python games.py --configs ta_$m --games 10000 >> $R/log_games.txt 2>&1
done
echo QUEUE_DONE > $R/queue_done.txt

#!/bin/bash
# Phase 3 run queue (one process at a time).  P1 is launched separately first.
cd "$(dirname "$0")"
while pgrep -f p1_main.py >/dev/null; do sleep 10; done
python p2_main.py --games 10000 --L 3000 > results/log_p2.txt 2>&1
python p3_main.py --deals 3000 --K 150 > results/log_p3.txt 2>&1
python p4_main.py --arm ta_spite --games 3000 > results/log_p4_ta_spite.txt 2>&1
python p4_main.py --arm greedy --games 3000 > results/log_p4_greedy.txt 2>&1
python p4_main.py --arm oracle_i --games 2000 > results/log_p4_oracle_i.txt 2>&1

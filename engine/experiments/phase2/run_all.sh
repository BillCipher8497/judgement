#!/bin/sh
# Phase 2: every run that produced results/ (strictly sequential, one process).
# Seeds are fixed inside each script; PH2_START offsets the deal index so later
# runs extend earlier ones without overlap.  Total wall time ~2.5 h on one core.
cd "$(dirname "$0")"
python bench_solver.py 1 2 3 4 5          > results/bench_solver_1_5_final.log
python bench_solver.py 6 7                > results/bench_solver_6_7.log
python exp_main.py 1000 1 2 3             >> results/main.log     # F1-F4, F6 data
python exp_main.py 600 4                  >> results/main.log
python exp_main.py 250 5                  >> results/main.log
PH2_START=250 python exp_main.py 250 5    >> results/main.log
PH2_TRUMPS=S python exp_main.py 100 6     >> results/main.log
PH2_TRUMPS=NT python exp_main.py 50 6     >> results/main.log
python exp_enum.py 200 1 2 3              > results/enum.log      # F1/F2 (a) all bid vectors
python exp_onecard.py                     > results/onecard.log   # stretch: 1-card game
python exp_pimc.py 400 16 2 3             >> results/pimc.log     # F5
python exp_pimc.py 200 16 4               >> results/pimc.log
python exp_pimc.py 120 16 5               >> results/pimc.log
PH2_START=400 python exp_pimc.py 600 16 2 3 >> results/pimc.log
PH2_START=200 python exp_pimc.py 800 16 4   >> results/pimc.log
PH2_START=120 python exp_pimc.py 880 16 5   >> results/pimc.log
python exp_pimc.py 100 16 6                 >> results/pimc.log
PH2_TAG=_K4 python exp_pimc.py 300 4 4      > results/pimc_k.log  # K sensitivity
PH2_TAG=_K64 python exp_pimc.py 300 64 4    >> results/pimc_k.log
python exp_ta_pi.py 500 2 3 4 5             > results/ta_pi.log   # F4b vs Phase-1 TA policy
PH2_HEUR=ta PH2_TAG=_TA python exp_pimc.py 500 16 2 3 4 5 > results/pimc_ta.log   # F5b
# analysis
python analyze_main.py > /dev/null; python analyze_pimc.py > /dev/null; python analyze_pimc.py K > /dev/null
python analyze_rules.py > /dev/null; python analyze_ta.py > /dev/null

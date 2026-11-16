#!/bin/bash
# stop Q2 at 03:25 ET (it saves every chunk), then run Q3 until 04:05 ET (one process at a time)
cd "$(dirname "$0")"
T1=$(TZ=America/New_York date -d '2026-11-11 03:25' +%s); T2=$(TZ=America/New_York date -d '2026-11-11 04:05' +%s)
while [ $(date +%s) -lt $T1 ] && pgrep -f q2_ladder.py >/dev/null; do sleep 20; done
pkill -f q2_ladder.py; sleep 2
python q3_mixed.py 2000 $T2 > results/q3.log 2>&1
python q3_mixed.py analyze >> results/q3.log 2>&1

#!/bin/bash
# Sequential job queue (one process). Re-reads jobs.txt each step so jobs can be appended.
cd "$(dirname "$0")"
D=$(TZ=America/New_York date -d '2026-11-11 06:30' +%s)
i=1
while true; do
  line=$(sed -n "${i}p" jobs.txt)
  [ -z "$line" ] && break
  python run.py $line $D
  i=$((i+1))
done
echo QUEUEDONE

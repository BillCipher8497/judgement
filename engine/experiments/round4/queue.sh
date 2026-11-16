#!/bin/bash
# runs jobs file sequentially (one process). usage: queue.sh JOBS DEADLINE_EPOCH
cd "$(dirname "$0")"
while read -r pool chal start G tag base; do
  [ -z "$pool" ] && continue
  echo "$(TZ=America/New_York date +%H:%M:%S) start $tag" >> results/queue.log
  python run.py "$pool" "$chal" "$start" "$G" "$tag" "$base" "$2" >> results/queue.log 2>&1
done < "$1"
echo "$(TZ=America/New_York date +%H:%M:%S) QUEUEDONE $1" >> results/queue.log

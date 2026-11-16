#!/bin/bash
# shared job list for several workers: each job claimed atomically (mkdir results/claims/TAG). usage: queue2.sh JOBS DEADLINE_EPOCH
cd "$(dirname "$0")"; mkdir -p results/claims
while read -r pool chal start G tag base; do
  [ -z "$pool" ] && continue
  [ "$(date +%s)" -gt "${2%.*}" ] && break
  mkdir "results/claims/$tag" 2>/dev/null || continue
  echo "$(TZ=America/New_York date +%H:%M:%S) start $tag" >> results/queue.log
  python run.py "$pool" "$chal" "$start" "$G" "$tag" "$base" "$2" >> results/queue.log 2>&1
done < "$1"
echo "$(TZ=America/New_York date +%H:%M:%S) QUEUEDONE $1" >> results/queue.log

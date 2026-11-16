#!/bin/bash
# S3: sequential (one process). G games per job, BASE 71,000,000, games 0..G-1, challenger seat g%4.
cd "$(dirname "$0")"
G=${1:-6000}
DL=${2:-0}
while read pool chal; do
  tag="f_${pool}__$(echo $chal | tr ':,=' '__-')"
  [ -f results/g_${tag}.npz ] && continue
  if [ "$DL" != "0" ]; then python run.py $pool $chal 0 $G $tag 71000000 $DL; else python run.py $pool $chal 0 $G $tag; fi
done < s3_jobs.txt
echo ALLDONE

#!/bin/bash
# Extra queue after run_all.sh (same seeds 71,000,000+g).
cd "$(dirname "$0")"
export PLAY_BASE=71000000
D=$(TZ=America/New_York date -d '2026-11-11 00:40' +%s)
python one_vs_three.py "hybrid:k=16;c=6" 500 x_hyb_c6 $D
python one_vs_three.py "hyb_oracle:k=16;c=5;K=24;nmax=13" 500 x_hyb_orc $D
echo EXTRADONE

#!/bin/bash
# win_high screen on separate seeds (91,000,000+g), 3000 games each
cd "$(dirname "$0")"
export PLAY_BASE=91000000
for spec in "ta scr_ta" "ta_plus scr_tap" "ta_plus:wh=1 scr_wh1" "ta_plus:wh=2 scr_wh2" "ta_plus:wh=1;wm=0.5 scr_wh1_m05" "ta_plus:wh=1;wm=-0.5 scr_wh1_mm05" "ta_plus:wh=1;wm=1 scr_wh1_m1"; do
  set -- $spec; python one_vs_three.py "$1" 3000 "$2"
done
echo SCREENDONE

#!/bin/bash
cd /home/user/sensor/docs/src/oned
while pgrep -f "queue.sh" >/dev/null || pgrep -f "e035_full" >/dev/null; do sleep 30; done
R=../../results/2026-11-09_E039_fastpath
for m in e039 e039_s2; do
  python3 benchmark.py real:cq:proto $m 21 120 $R/heldout_$m.json 2 > $R/heldout_$m.log 2>&1
done
python3 benchmark.py real:allpw:proto e039 21 60 $R/heldout_allpw_e039.json 2 > $R/heldout_allpw_e039.log 2>&1

#!/bin/bash
# waits for the E064 test walls, then starts N=5 and N=7 runs as independent (nohup) workers; resumable
cd /home/user/sensor/docs/results/2026-11-11_E063_Nlayers
until [ "$(cat ../2026-11-10_E060_heldout/prof_*.jsonl 2>/dev/null | wc -l)" -ge 3 ]; do sleep 30; done
for N in 5 7; do for i in 0 1; do
  OMP_NUM_THREADS=1 nohup python3 -u run.py $N $i 2 900 919 0 20 n${N}_$i.jsonl >> n${N}_$i.log 2>&1 &
done; done

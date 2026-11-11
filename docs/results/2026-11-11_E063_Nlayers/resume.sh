#!/bin/bash
# idempotent: (re)starts the checkpointed N=5 budget diagnosis for walls that have no result yet
S=/home/user/sensor/docs/results/2026-11-11_E063_Nlayers
D=/home/user/sensor/docs/results/2026-11-11_E063_Nlayers
for sd in 902 916 904 910; do
  [ -f $D/ckpt/${sd}_12000_result.json ] && continue
  pgrep -f "n5ckpt.py $sd" >/dev/null && continue
  OMP_NUM_THREADS=1 nohup python3 -u $S/n5ckpt.py $sd 12000 >> $D/ckpt_$sd.log 2>&1 &
done

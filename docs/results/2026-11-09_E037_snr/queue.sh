#!/bin/bash
# waits for E035 to finish, then runs E037 (SNR sweep) and E038 (handheld motion), 2 workers
cd /home/user/sensor/docs/src/oned
while pgrep -f "e035_" >/dev/null; do sleep 20; done
R=../../results
for v in 35 30 25 20; do
  python3 benchmark.py real:cqsnr$v:proto e035_none 1 20 $R/2026-11-09_E037_snr/snr$v.json 2 > $R/2026-11-09_E037_snr/snr$v.log 2>&1
done
for K in 1 4; do
  python3 benchmark.py motion:$K:cqsnr25:proto e038 1 20 $R/2026-11-09_E038_motion/k${K}_snr25.json 2 > $R/2026-11-09_E038_motion/k${K}_snr25.log 2>&1
done
python3 benchmark.py motion:4:cq:proto e038 1 10 $R/2026-11-09_E038_motion/k4_snr40.json 2 > $R/2026-11-09_E038_motion/k4_snr40.log 2>&1

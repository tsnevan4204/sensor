#!/bin/bash
cd /home/user/sensor/docs/src/oned
sleep 60
while pgrep -f "queue2.sh" >/dev/null; do sleep 30; done
R=../../results/2026-11-09_E040_stress
for b in cqtilt2 cqtilt4; do for m in e035_none e035_full; do
  python3 benchmark.py real:$b:proto $m 1 10 $R/${b}_$m.json 2 > $R/${b}_$m.log 2>&1
done; done
python3 benchmark.py real:cqdriftref:proto e040_sig 1 10 $R/driftref_sighat.json 2 > $R/driftref_sighat.log 2>&1
for v in 15 10; do
  python3 benchmark.py real:cqsnr$v:proto e035_none 1 20 ../../results/2026-11-09_E037_snr/snr$v.json 2 > ../../results/2026-11-09_E037_snr/snr$v.log 2>&1
done

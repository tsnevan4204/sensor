cd /home/user/sensor/docs/src/oned
while pgrep -f "queue_e024.sh" > /dev/null; do sleep 20; done
R=/home/user/sensor/docs/results/2026-11-09_E025_speed && mkdir -p $R
for b in real:cq:dual60 real:itucausal:dualuwb; do
  for m in e025_base e025_s1 e025_s2 e025_s3 e025_s4 e025_s5; do
    python3 benchmark.py $b $m 1 20 $R/$(echo $b | tr ':' '_')_$m.json 3 > /dev/null
  done
done

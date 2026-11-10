cd /home/user/sensor/docs/src/oned
while pgrep -f "heldout.json" > /dev/null; do sleep 20; done
R=/home/user/sensor/docs/results/2026-11-09_E030_gap && mkdir -p $R
for b in real:cqgap:dual60 real:allpw:dual60; do
  python3 benchmark.py $b e030 1 10 $R/$(echo $b | tr ':' '_')_e030.json 3 > /dev/null
done

cd /home/user/sensor/docs/src/oned
while pgrep -f "queue_e030.sh" > /dev/null; do sleep 20; done
R=/home/user/sensor/docs/results/2026-11-09_E031_fmcw && mkdir -p $R
for job in "fmcw:fm100:naive e027_s4" "fmcw:fm100:chain e031" "fmcw:fm250:naive e027_s4" "fmcw:fm250:chain e031" "fmcw:fm100nl:chain e031"; do
  set -- $job; python3 benchmark.py $1 $2 1 10 $R/$(echo $1 | tr ':' '_')_$2.json 3 > /dev/null
done

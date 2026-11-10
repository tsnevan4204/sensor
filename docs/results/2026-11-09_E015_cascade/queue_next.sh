cd /home/user/sensor/docs/src/oned
while pgrep -f "benchmark.py v2 e015b" > /dev/null; do sleep 15; done
R=/home/user/sensor/docs/results
mkdir -p $R/2026-11-09_E017_dual_band $R/2026-11-09_E016_posterior
python3 benchmark.py v2dual e017 1 20 $R/2026-11-09_E017_dual_band/v2dual_e017.json 3
python3 benchmark.py v2 e016 1 20 $R/2026-11-09_E016_posterior/v2_e016.json 3

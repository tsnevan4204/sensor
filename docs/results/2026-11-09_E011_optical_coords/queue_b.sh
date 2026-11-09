cd /home/user/sensor/docs/src/oned
while pgrep -f "benchmark.py v2 e012 " > /dev/null; do sleep 15; done
R=/home/user/sensor/docs/results
python3 benchmark.py v2 e011b 1 20 $R/2026-11-09_E011_optical_coords/v2_e011b.json 3
python3 benchmark.py v2 e012b 1 20 $R/2026-11-09_E012_integer_ambiguity/v2_e012b.json 3

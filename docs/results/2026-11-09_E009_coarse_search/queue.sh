# Runs after E008 finishes. Each experiment: same benchmark (v2, seeds 1-20), one switch added.
cd /home/user/sensor/docs/src/oned
while pgrep -f "results/2026-11-09_E008_benchmark/run.sh" > /dev/null; do sleep 20; done
R=/home/user/sensor/docs/results
python3 benchmark.py v2 e009 1 20 $R/2026-11-09_E009_coarse_search/v2_e009.json 3
python3 benchmark.py v2 e010 1 20 $R/2026-11-09_E010_pencil/v2_e010.json 3
python3 benchmark.py v2 e011 1 20 $R/2026-11-09_E011_optical_coords/v2_e011.json 3
python3 benchmark.py v2 e012 1 20 $R/2026-11-09_E012_integer_ambiguity/v2_e012.json 3

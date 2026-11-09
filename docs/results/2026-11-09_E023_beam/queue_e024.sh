cd /home/user/sensor/docs/src/oned
while pgrep -f "benchmark.py lambot:c" > /dev/null; do sleep 20; done
R=/home/user/sensor/docs/results/2026-11-09_E024_facings && mkdir -p $R
for job in "real:cqpaper:dual60 e021b" "real:cqpaper:dual60 e024" "real:cqrough:dual60 e024" "real:cqoffset:dual60 e024" "lambot:all:dual60 e024beam"; do
  set -- $job; python3 benchmark.py $1 $2 1 10 $R/$(echo $1 | tr ':' '_')_$2.json 3 > /dev/null
done

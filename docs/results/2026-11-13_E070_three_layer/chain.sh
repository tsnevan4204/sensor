#!/bin/bash
# E070 run chain (each run uses 4 worker processes, one wall per process)
cd "$(dirname "$0")"
run() { echo "=== $1 start $(date -u +%H:%M)"; shift_name=$1; shift; python3 run.py --out $shift_name.jsonl "$@" > $shift_name.log 2>&1; echo "=== $shift_name done $(date -u +%H:%M)"; }
run main_ref     --seeds 2000 2059 --truth in  --solver in  --balanced --mode reference
run outscope_ref --seeds 3000 3049 --truth out --solver in  --balanced --mode reference
run main_fast    --seeds 2000 2059 --truth in  --solver in  --balanced --mode fast
run main_3band   --seeds 2000 2059 --truth in  --solver in  --balanced --mode reference --bands 3
run map_noisefree --seeds 1000 1079 --truth all --solver all --balanced --mode reference --noisefree
run all80_ref    --seeds 1000 1079 --truth all --solver all --balanced --mode reference
run stress_ref   --seeds 4000 4029 --truth in  --solver in  --balanced --mode reference --thick stress
echo "=== CHAIN DONE"

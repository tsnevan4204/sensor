#!/bin/bash
# E070 resume after the second container restart (runs resume: run.py skips seeds already written)
cd "$(dirname "$0")"
run() { nm=$1; shift; echo "=== $nm start $(date -u +%H:%M)"; python3 run.py --out $nm.jsonl "$@" >> $nm.log 2>&1; echo "=== $nm done $(date -u +%H:%M)"; }
run stress_ref    --seeds 4000 4029 --truth in  --solver in  --balanced --mode reference --thick stress
run main_3band    --seeds 2000 2059 --truth in  --solver in  --balanced --mode reference --bands 3
run all80_ref     --seeds 1000 1079 --truth all --solver all --balanced --mode reference
echo "=== ALL DONE 2 $(date -u +%H:%M)"

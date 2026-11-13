#!/bin/bash
# E070 resume after container restart: every run resumes (run.py skips seeds already written)
cd "$(dirname "$0")"
run() { nm=$1; shift; echo "=== $nm start $(date -u +%H:%M)"; env $ENVX python3 run.py --out $nm.jsonl "$@" >> $nm.log 2>&1; echo "=== $nm done $(date -u +%H:%M)"; }
ENVX=""
run map_noisefree --seeds 1000 1079 --truth all --solver all --balanced --mode reference --noisefree
run main_3band    --seeds 2000 2059 --truth in  --solver in  --balanced --mode reference --bands 3
ENVX="TRI3_WOOD=chipboard"
run wood_chipboard_ref --seeds 2000 2059 --truth in --solver in --balanced --mode reference
ENVX=""
run main_fast     --seeds 2000 2059 --truth in  --solver in  --balanced --mode fast
run stress_ref    --seeds 4000 4029 --truth in  --solver in  --balanced --mode reference --thick stress
run all80_ref     --seeds 1000 1079 --truth all --solver all --balanced --mode reference
echo "=== ALL DONE $(date -u +%H:%M)"

#!/bin/bash
cd "$(dirname "$0")"
until grep -q "map_noisefree done" chain.log; do sleep 10; done
pkill -f chain.sh; sleep 1; pkill -f "run.py --out all80_ref"; sleep 3
rm -f all80_ref.jsonl all80_ref.log
nohup ./chain2.sh > chain2.log 2>&1 &

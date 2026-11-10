#!/bin/bash
# resumable E041 held-out runs; relaunch after a container restart with: nohup ./go.sh &
R=/home/user/sensor/docs/results/2026-11-09_E041_peel
cd /home/user/sensor/docs/src/oned
python3 $R/run.py 3 101 120 2 $R/n3_heldout.jsonl >> $R/n3_heldout.log 2>&1
python3 $R/run.py 5 1 10 2 $R/n5.jsonl >> $R/n5.log 2>&1
python3 $R/run.py 7 1 10 2 $R/n7.jsonl >> $R/n7.log 2>&1

#!/bin/bash
# resumable E042 runs (results kept per wall; relaunch continues)
R=/home/user/sensor/docs/results/2026-11-09_E044_known_n
cd /home/user/sensor/docs/src/oned
python3 $R/run.py 3 101 108 2 $R/n3_heldout.jsonl >> $R/log.txt 2>&1
python3 $R/run.py 3 1 6 2 $R/n3_dev.jsonl >> $R/log.txt 2>&1
python3 $R/run.py 5 1 10 2 $R/n5.jsonl >> $R/log.txt 2>&1
python3 $R/run.py 7 1 10 2 $R/n7.jsonl >> $R/log.txt 2>&1

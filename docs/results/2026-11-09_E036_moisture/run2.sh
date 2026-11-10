#!/bin/bash
cd /home/user/sensor/docs/src/oned; R=../../results
OUT=$R/2026-11-09_E036_moisture/wet_n2_v2.jsonl P_WET=0.7 MOIST=1 python3 $R/2026-11-09_E034_stacks/run_trial.py 2 1 12 3 > $R/2026-11-09_E036_moisture/wet_n2_v2.txt 2>/dev/null
OUT=$R/2026-11-09_E034_stacks/n3_s11-30.jsonl MOIST=1 python3 $R/2026-11-09_E034_stacks/run_trial.py 3 11 30 3 > $R/2026-11-09_E034_stacks/n3_s11-30.txt 2>/dev/null

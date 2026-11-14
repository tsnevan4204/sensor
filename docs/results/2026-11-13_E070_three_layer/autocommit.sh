#!/bin/bash
# commit + push E070 outputs whenever a run finishes (container restarts lose uncommitted results)
cd "$(dirname "$0")"
seen=$(grep -c "done" chain4.log)
while true; do
  n=$(grep -c "done" chain4.log)
  if [ "$n" != "$seen" ]; then
    seen=$n; last=$(grep "done" chain4.log | tail -1)
    cd /home/user/sensor && git add docs/results/2026-11-13_E070_three_layer && git commit -q -m "E070 run finished: $last

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017DGgcmTZEMRvB5FFcE2UF8" && for i in 1 2 3 4; do git push -q origin claude/cool-bohr-v3xzec && break || sleep $((2**i)); done
    cd - >/dev/null
  fi
  grep -q "ALL DONE 2" chain4.log && break
  sleep 30
done

"""Diagnostic for E008-E012: chi2 of the best fit *inside the truth's basin* (LM started at the
truth, all frequencies, tan d free), per seed. Classifies each silent failure as search failure
(truth basin better), enumeration failure (truth basin within dchi2 4 but not listed) or data
ambiguity (truth basin worse by > 4)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, benchmark as B, scenarios as S, search as SR
out = {}
for seed in range(1, 21):
    f, yc, s, d0, tv = B.make_case("v2", seed)
    m = S.stud_wall_model(free=("t", "eps_r", "tan_d"), standoff_guess=d0)
    for n, _, p in m.items():
        if n in tv:
            p.value = tv[n]
    m.sync_ties()
    mo, c = SR.lm(m, f, yc, s)
    out[seed] = c
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(out)

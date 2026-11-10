"""E052 dev set: python3 run.py N seed0 seed1 out.jsonl [MAT_RESCUE]  (resumable)."""
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, wallsolve as WS
N, s0, s1, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
WS.MAT_RESCUE = int(sys.argv[5]) if len(sys.argv) > 5 else 0
done = {json.loads(l)["seed"] for l in open(out)} if os.path.exists(out) else set()
for seed in range(s0, s1 + 1):
    if seed in done:
        continue
    f, y, s, tr = WS.make_problem(seed, N)
    t = time.time()
    try:
        res = WS.solve(f, y, s, N, tr["standoff"] + 0.001)
    except Exception as e:
        open(out, "a").write(json.dumps(dict(seed=seed, crash=repr(e))) + "\n"); continue
    sc = WS.score(res, tr, N); sol, gaps = WS.report(res, N)
    tag = "OK" if sc["ok"] else ("flag" if res["alarm"] else ("unobs" if res["unobserved"] else ("weak" if any(res["weak"]) else "SILENT")))
    rec = dict(seed=seed, tag=tag, wall_s=time.time() - t, chi2n=res["chi2n"], route=res["route"],
               truth=[(L["name"], L["t"]) for L in tr["layers"]], gaps_true=tr["gaps"], back=tr["back"],
               fit=[(x["label"], x["t"]) for x in sol], gaps_fit=gaps, back_fit=res["back"],
               t_err_mm=sc["t_err_mm"], mat_ok=sc["mat_ok"])
    open(out, "a").write(json.dumps(rec, default=float) + "\n")
    print(seed, tag, f"{rec['wall_s']:.0f}s", flush=True)

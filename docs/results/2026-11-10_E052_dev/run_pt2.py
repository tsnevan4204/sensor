"""E054: PT solver on the E052 dev walls that the echo cascade did not solve. python3 run_pt.py shard nshard out.jsonl"""
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, wallsolve as WS, pt
sh, nsh, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
R = {json.loads(l)["seed"]: json.loads(l) for f in ("a.jsonl", "b.jsonl") for l in open(f)}
seeds = [int(x) for x in os.environ["SEEDS"].split(",")][sh::nsh]
done = {json.loads(l)["seed"] for l in open(out)} if os.path.exists(out) else set()
for seed in seeds:
    if seed in done:
        continue
    f, y, s, tr = WS.make_problem(seed, 3)
    t = time.time()
    res = pt.solve(f, y, s, 3, tr["standoff"] + 0.001)
    sc = WS.score(res, tr, 3)
    rec = dict(seed=seed, ok=sc["ok"], alarm=res["alarm"], chi2n=res["chi2n"], wall_s=time.time() - t,
               t_err_mm=sc["t_err_mm"], mat_ok=sc["mat_ok"], back_ok=sc["back_ok"], gap_ok=sc["gap_ok"],
               echo_tag=R[seed].get("tag"), truth=[L["name"] for L in tr["layers"]])
    open(out, "a").write(json.dumps(rec, default=float) + "\n")
    print(seed, sc["ok"], round(res["chi2n"], 2), flush=True)

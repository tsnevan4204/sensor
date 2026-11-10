import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, wallsolve as WS, diagnose as DG
R = {json.loads(l)["seed"]: json.loads(l) for f in ("a.jsonl", "b.jsonl") for l in open(f)}
seeds = [s for s, r in R.items() if r.get("tag") != "OK"][int(sys.argv[1])::4]
for sd in seeds:
    f, y, s, tr = WS.make_problem(sd, 3)
    d = DG.diagnose(f, y, s, tr, 3, dict(chi2n=R[sd].get("chi2n", 99), chi2=R[sd].get("chi2n", 99) * 2 * f.size))
    print(json.dumps(dict(seed=sd, tag=R[sd].get("tag"), chi_solver=round(R[sd].get("chi2n", 0), 2),
          chi_truth=round(d["1_truth_chi2n"], 2), crb=d["1_truth_crb_t_mm"], ifc=d["2_true_interfaces_mm"],
          miss=d["2_missed_interfaces_mm"], first=d.get("first_failure")), default=float), flush=True)

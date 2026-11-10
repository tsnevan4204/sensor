"""E057 full pipeline (echo cascade -> PT): python3 run.py N shard nshard seed0 seed1 out.jsonl (resumable)."""
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, wallsolve as WS
N, sh, nsh, s0, s1, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
WS.MAT_RESCUE = 3
done = {json.loads(l)["seed"] for l in open(out)} if os.path.exists(out) else set()


def alt_ok(alt, tr):
    sol = [t for t, _ in alt["solids"]]
    if len(sol) != len(tr["layers"]) or alt["back"] != tr["back"]:
        return False
    return all(abs(t - L["t"] * 1e3) <= max(0.3, 0.02 * L["t"] * 1e3) + 0.5 for t, L in zip(sol, tr["layers"]))


for seed in range(s0, s1 + 1)[sh::nsh]:
    if seed in done:
        continue
    f, y, s, tr = WS.make_problem(seed, N)
    t = time.time()
    try:
        res = WS.solve_full(f, y, s, N, tr["standoff"] + 0.001)
    except Exception as e:
        open(out, "a").write(json.dumps(dict(seed=seed, tag="crash", err=repr(e))) + "\n"); continue
    sc = WS.score(res, tr, N)
    amb_true = any(alt_ok(a, tr) for a in res.get("alternatives", []))
    if sc["ok"]:
        tag = "OK"
    elif res["alarm"]:
        tag = "flag"
    elif res["unobserved"] or any(res["weak"]):
        tag = "unobs/weak"
    elif res.get("ambiguous"):
        tag = "ambig(truth listed)" if amb_true else "ambig(truth not listed)"
    else:
        tag = "SILENT"
    rec = dict(seed=seed, tag=tag, wall_s=time.time() - t, chi2n=res["chi2n"], route=res["route"],
               truth=[(L["name"], L["t"]) for L in tr["layers"]], gaps_true=tr["gaps"], back=tr["back"],
               t_err_mm=sc["t_err_mm"], mat_ok=sc["mat_ok"], gap_ok=sc["gap_ok"], back_ok=sc["back_ok"],
               crb_mm=[float(1e3 * e) for e in res["t_err"]], n_alt=len(res.get("alternatives", [])))
    open(out, "a").write(json.dumps(rec, default=float) + "\n")
    print(seed, tag, f"{rec['wall_s']:.0f}s", flush=True)

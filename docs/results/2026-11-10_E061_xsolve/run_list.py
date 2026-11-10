"""E061: xsolve (PT + LS polish + mmWave cycle scan + label sets) on N-layer walls.
python3 run.py N shard nshard seed0 seed1 angles(comma deg) boost_db out.jsonl  (resumable)"""
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, xsolve as XS
XS.FAST = os.environ.get("FAST") == "1"
N, sh, nsh, s0, s1 = (int(a) for a in sys.argv[1:6])
angles = tuple(float(a) for a in sys.argv[6].split(","))
boost, out = float(sys.argv[7]), sys.argv[8]
done = {json.loads(l)["seed"] for l in open(out)} if os.path.exists(out) else set()


def alt_ok(alt, tr):
    if len(alt["solids"]) != len(tr["layers"]) or alt["back"] != tr["back"]:
        return False
    return all(abs(t - L["t"] * 1e3) <= max(0.3, 0.02 * L["t"] * 1e3) + 0.5 for t, L in zip(alt["solids"], tr["layers"]))


for seed in [int(x) for x in os.environ["SEEDS"].split(",")][sh::nsh]:
    if seed in done:
        continue
    views, tr = XS.make_views(seed, N, angles, boost)
    t = time.time()
    res = XS.solve(views, N, tr["standoff"] + 0.001)
    sc = XS.score(res, tr, N)
    amb = any(alt_ok(a, tr) for a in res["alternatives"])
    if sc["ok"]:
        tag = "OK"
    elif res["alarm"]:
        tag = "flag"
    elif any(res["weak"]) or res["unobserved"]:
        tag = "weak/unobs"
    elif amb:
        tag = "ambig(truth listed)"
    else:
        tag = "SILENT"
    rec = dict(seed=seed, tag=tag, geo_ok=sc["geo_ok"], mat_ok=sc["mat_ok"], mat_top=sc["mat_top"],
               t_err_mm=sc["t_err_mm"], g_ok=sc["g_ok"], back_ok=sc["back_ok"], chi2n=res["chi2n"],
               crb_mm=[1e3 * e for e in res["crb_t"]], labels=res["labels"], n_alt=len(res["alternatives"]),
               truth=[(L["name"], L["t"]) for L in tr["layers"]], gaps=tr["gaps"], back=tr["back"],
               wall_s=time.time() - t)
    open(out, "a").write(json.dumps(rec, default=float) + "\n")
    print(seed, tag, round(rec["wall_s"]), flush=True)

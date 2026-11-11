"""Checkpointed 3x-budget solve for one N=5 wall: each PT run (backing x restart) is saved to disk and
skipped if present, so a container restart loses at most one PT run. Then polish/finish as xsolve."""
import sys, os, warnings, time, json; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, xsolve as XS, pt as PT
sd = int(sys.argv[1]); N = 5; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
out = "/home/user/sensor/docs/results/2026-11-11_E063_Nlayers/ckpt"
os.makedirs(out, exist_ok=True)
v, tr = XS.make_views(sd, N, (0.0,), 20.0)
dg = tr["standoff"] + 0.001
cands = []
for back in ("air", "metal"):
    mp = XS.MultiProblem(v, N, back, dg)
    for rs in range(3):
        fn = f"{out}/{sd}_{back}_{rs}_{steps}.npz"
        if not os.path.exists(fn):
            t = time.time()
            best, cand = PT.run(mp, steps=steps, seed=101 * rs)
            np.savez(fn, C=np.array(cand))
            print(sd, back, rs, "PT done", round(best[2] / sum(2 * q[0].size for q in mp.views), 2), round(time.time() - t), "s", flush=True)
        C = np.load(fn)["C"]
        cands += [(mp.energy(c[None, :])[0][0], c, back) for c in C]
XS.PROFILE = False
XS.LAYER_SWEEP = True
res = XS._finish(cands, v, N, dg, 10, time.time())
sc = XS.score(res, tr, N)
json.dump(dict(seed=sd, steps=steps, ok=sc["ok"], geo_ok=sc["geo_ok"], chi2n=res["chi2n"], t_err_mm=sc["t_err_mm"]),
          open(f"{out}/{sd}_{steps}_sweep_result.json", "w"), default=float)
print(sd, "RESULT ok", sc["ok"], "chi2n", round(res["chi2n"], 3), np.round(sc["t_err_mm"], 1), flush=True)

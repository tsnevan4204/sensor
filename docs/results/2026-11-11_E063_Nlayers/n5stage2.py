"""E066 test: stage-2 tempering at the TRUE sigma (no floor), seeded from the stage-1 (floored) PT candidates."""
import sys, os, glob, warnings, time, json; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, xsolve as XS, pt as PT
sd = int(sys.argv[1]); N = 5; steps2 = int(sys.argv[2])
D = "/home/user/sensor/docs/results/2026-11-11_E063_Nlayers"
v, tr = XS.make_views(sd, N, (0.0,), 20.0); dg = tr["standoff"] + 0.001
cands = []
for back in ("air", "metal"):
    XS.PT_SIGMA_FLOOR = True
    mp1 = XS.MultiProblem(v, N, back, dg)
    C = []
    for fn in sorted(glob.glob(f"{D}/ckpt/{sd}_{back}_*_12000.npz")):
        C += list(np.load(fn)["C"])
    C.sort(key=lambda c: mp1.energy(c[None, :])[0][0])
    XS.PT_SIGMA_FLOOR = False
    mp2 = XS.MultiProblem(v, N, back, dg)
    fn2 = f"{D}/ckpt/{sd}_{back}_stage2_{steps2}.npz"
    if not os.path.exists(fn2):
        t = time.time()
        best, cand = PT.run(mp2, steps=steps2, seed=7, init=C[:8])
        np.savez(fn2, C=np.array(cand))
        print(sd, back, "stage2 PT chi2n", round(best[2] / (2 * mp2.views[0][0].size), 2), round(time.time() - t), "s", flush=True)
    C2 = list(np.load(fn2)["C"])
    cands += [(mp2.energy(c[None, :])[0][0], c, back) for c in C2] + [(mp2.energy(c[None, :])[0][0], c, back) for c in C[:8]]
XS.PROFILE = False
res = XS._finish(cands, v, N, dg, 10, time.time())
sc = XS.score(res, tr, N)
json.dump(dict(seed=sd, ok=sc["ok"], geo_ok=sc["geo_ok"], chi2n=res["chi2n"], t_err_mm=sc["t_err_mm"]), open(f"{D}/ckpt/{sd}_stage2_result.json", "w"), default=float)
print(sd, "RESULT ok", sc["ok"], "chi2n", round(res["chi2n"], 3), np.round(sc["t_err_mm"], 2), flush=True)

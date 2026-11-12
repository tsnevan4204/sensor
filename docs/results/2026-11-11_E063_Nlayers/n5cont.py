"""E067 test: noise continuation. Stage-1 candidates (floored sigma, 12000 steps, checkpointed) -> PT at floor x 0.316
-> PT at true sigma -> PT at true sigma INCLUDING mmWave; each stage seeded with the previous stage's best 8."""
import sys, os, glob, warnings, time, json; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, xsolve as XS, pt as PT
sd = int(sys.argv[1]); N = 5
D = "/home/user/sensor/docs/results/2026-11-11_E063_Nlayers"
v, tr = XS.make_views(sd, N, (0.0,), 20.0); dg = tr["standoff"] + 0.001
STAGES = [("s316", True, 0.316, 11e9, 3000), ("strue", False, 1.0, 11e9, 3000), ("smm", False, 1.0, 70e9, 2000)]
cands = []
for back in ("air", "metal"):
    XS.PT_SIGMA_FLOOR, XS.SIGMA_FLOOR_SCALE, PT.FMAX = True, 1.0, 11e9
    mp = XS.MultiProblem(v, N, back, dg)
    C = []
    for fn in sorted(glob.glob(f"{D}/ckpt/{sd}_{back}_[0-9]_12000.npz")):
        C += list(np.load(fn)["C"])
    for name, floor, scale, fmax, steps in STAGES:
        XS.PT_SIGMA_FLOOR, XS.SIGMA_FLOOR_SCALE, PT.FMAX = floor, scale, fmax
        mp = XS.MultiProblem(v, N, back, dg)
        C.sort(key=lambda c: mp.energy(c[None, :])[0][0])
        fn = f"{D}/ckpt/{sd}_{back}_{name}.npz"
        if not os.path.exists(fn):
            t = time.time()
            best, cand = PT.run(mp, steps=steps, seed=11, init=C[:8])
            np.savez(fn, C=np.array(cand))
            print(sd, back, name, "chi2n", round(best[2] / sum(2 * q[0].size for q in mp.views), 2), round(time.time() - t), "s", flush=True)
        C = list(np.load(fn)["C"]) + C[:8]
    XS.PT_SIGMA_FLOOR, XS.SIGMA_FLOOR_SCALE, PT.FMAX = False, 1.0, 11e9
    mpf = XS.MultiProblem(v, N, back, dg)
    cands += [(mpf.energy(c[None, :])[0][0], c, back) for c in C]
XS.PT_SIGMA_FLOOR, XS.SIGMA_FLOOR_SCALE, PT.FMAX = True, 1.0, 11e9
XS.PROFILE = False
res = XS._finish(cands, v, N, dg, 10, time.time())
sc = XS.score(res, tr, N)
json.dump(dict(seed=sd, ok=sc["ok"], geo_ok=sc["geo_ok"], chi2n=res["chi2n"], t_err_mm=sc["t_err_mm"]), open(f"{D}/ckpt/{sd}_cont_result.json", "w"), default=float)
print(sd, "RESULT ok", sc["ok"], "chi2n", round(res["chi2n"], 3), np.round(sc["t_err_mm"], 2), flush=True)

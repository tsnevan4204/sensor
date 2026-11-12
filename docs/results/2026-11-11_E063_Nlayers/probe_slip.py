"""Probe: from the stage-2 best answer, set the slipped layer(s) to the TRUE thickness and polish; compare chi2."""
import sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, xsolve as XS
sd = int(sys.argv[1]); N = 5
D = "/home/user/sensor/docs/results/2026-11-11_E063_Nlayers"
v, tr = XS.make_views(sd, N, (0.0,), 20.0); dg = tr["standoff"] + 0.001
back = tr["back"]
pol = XS.Polisher(v, N, back, dg); n = 2 * v[0][0].size
XS.PT_SIGMA_FLOOR = False
mp = XS.MultiProblem(v, N, back, dg)
C = list(np.load(f"{D}/ckpt/{sd}_{back}_stage2_4000.npz")["C"])
x0 = min(C, key=lambda c: mp.energy(c[None, :])[0][0])
x, c, _ = pol.polish(x0, max_nfev=200)
tt = lambda z: [np.exp(z[1 + 5 * i]) / np.sqrt(1 + np.exp(z[2 + 5 * i])) for i in range(N)]
err = np.array(tt(x)) - np.array([L["t"] for L in tr["layers"]])
print(sd, "start chi2n", round(c / n, 2), "err mm", np.round(err * 1e3, 2), flush=True)
bad = [i for i in range(N) if abs(err[i]) > 0.5e-3]
y = x.copy()
for i in bad:
    er = 1 + np.exp(y[2 + 5 * i]); y[1 + 5 * i] = np.log(tr["layers"][i]["t"] * np.sqrt(er))
print(" set bad layers", bad, "to truth (rest rigid): raw chi2n", round(pol.chi2(y) / n, 2), flush=True)
yp, cp, _ = pol.polish(y, max_nfev=300)
print(" -> polished", round(cp / n, 3), "err mm", np.round((np.array(tt(yp)) - [L["t"] for L in tr["layers"]]) * 1e3, 2), flush=True)
xs, cs = XS.cycle_scan(pol, x)
print(" cycle_scan from start ->", round(cs / n, 3), "err mm", np.round((np.array(tt(xs)) - [L["t"] for L in tr["layers"]]) * 1e3, 2), flush=True)

"""E010 calibration: pencil error distribution on seeds 1001-1050 (disjoint from benchmark seeds 1-20)."""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys, json, time
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, benchmark as B, pencil as PC, wall_toy as W
lam = W.C0 / 79e9
rows = []
t0 = time.perf_counter()
for seed in range(1001, 1051):
    f, yc, sigma, d0, tv = B.make_case("v2", seed)
    e = PC.wall_estimate(f, yc, sigma, d0)
    row = dict(seed=seed, M=e["M"], n_t=len(e.get("t", [])))
    for i, L in enumerate(["drywall_front", "insulation", "drywall_back"]):
        if i < len(e.get("tau", [])):
            er = tv["drywall_front.eps_r" if L == "drywall_back" else f"{L}.eps_r"]
            td = tv["drywall_front.tan_d" if L == "drywall_back" else f"{L}.tan_d"]
            tau_true = np.sqrt(er * (1 - 1j * td)).real * tv[f"{L}.t"]
            row[f"{L}.tau_err_periods"] = (e["tau"][i] - tau_true) / (lam / 2)
    for i, L in enumerate(["drywall_front", "insulation"]):
        if i < len(e.get("eps", [])):
            row[f"{L}.eps_err"] = e["eps"][i] - tv[f"{L}.eps_r"]
    row["standoff_err_mm"] = (e.get("standoff", np.nan) - tv["standoff"]) * 1e3
    rows.append(row)
dt = (time.perf_counter() - t0) / 50
out = [f"pencil calibration on seeds 1001-1050 (v2), {dt*1e3:.1f} ms per wall"]
out.append("model order M: " + str(np.bincount([r['M'] for r in rows]).tolist()) + "  (counts of M=0,1,2,...)")
for k in ["drywall_front.tau_err_periods", "insulation.tau_err_periods", "drywall_back.tau_err_periods",
          "drywall_front.eps_err", "insulation.eps_err", "standoff_err_mm"]:
    v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
    if v.size:
        out.append(f"{k:30s} n={v.size:2d}  mean {v.mean():+.2f}  std {v.std():.2f}  max|.| {np.abs(v).max():.2f}  "
                   f"95% |.| {np.quantile(np.abs(v), .95):.2f}")
print("\n".join(out))
open(sys.argv[1], "w").write("\n".join(out) + "\n")
json.dump(rows, open(sys.argv[1].replace(".txt", ".json"), "w"), indent=1, default=float)

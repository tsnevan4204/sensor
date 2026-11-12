"""E053 runner (fused leaf kernel + fast refit path; add -fast): wallsolve2.solve on wallsolve.make_problem.  python3 run.py N seed0 seed1 out.jsonl
(one wall at a time: the compiled kernels use all cores)"""
import sys, json, time, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/oned"))
import numpy as np
import wallsolve as WS, wallsolve2 as W2

N = int(sys.argv[1])
VERBOSE = "-v" in sys.argv
FAST = "-fast" in sys.argv
if "-front" in sys.argv:
    import wallsearch as _LS
    _LS.FRONT_SIGMA = True


def one(seed):
    f, y, s, tr = WS.make_problem(seed, N)
    t0 = time.time()
    try:
        res = W2.solve(f, y, s, N, tr["standoff"] + 0.0005, verbose=VERBOSE, fast=FAST)
        sc = WS.score(res, tr, N)
    except Exception as e:
        import traceback; traceback.print_exc()
        return dict(seed=seed, crash=repr(e), wall_s=time.time() - t0)
    sol, gaps = WS.report(res, N)
    # statistical check: same structure, and every thickness (solids and gaps) within
    # max(fixed tolerance, 3 Cramer-Rao sigma) of the truth
    tt, tol = [], []
    for i, L in enumerate(tr["layers"]):
        tt.append(L["t"]); tol.append(max(0.3e-3, 0.02 * L["t"]))
        if i < len(tr["gaps"]) and tr["gaps"][i] > 0:
            tt.append(tr["gaps"][i]); tol.append(max(0.3e-3, 0.05 * tr["gaps"][i]))
    ft = [L.t.value for L in res["model"].layers]
    kinds_fit = [k for k, _ in res["slots"]]
    kinds_true = []
    for i, L in enumerate(tr["layers"]):
        kinds_true.append("solid")
        if i < len(tr["gaps"]) and tr["gaps"][i] > 0:
            kinds_true.append("gap")
    same = kinds_fit == kinds_true and res["back"] == tr["back"]
    ok3 = bool(same and all(abs(a - b) <= max(tl, 3 * (c if np.isfinite(c) else 0.0))
                            for a, b, tl, c in zip(ft, tt, tol, res["t_err"])))
    return dict(seed=seed, ok=sc["ok"], ok3=ok3, same_structure=bool(same), alarm=res["alarm"], chi2n=res["chi2n"], wall_s=time.time() - t0,
                search_s=res["search_s"], truth=[(L["name"], round(1e3 * L["t"], 2)) for L in tr["layers"]],
                gaps_true=[round(g, 2) for g in sc["gaps_true_mm"]], back=tr["back"],
                fit=[(s_["label"], round(1e3 * s_["t"], 2)) for s_ in sol], mat_ok=sc["mat_ok"],
                gaps_fit=[round(g, 2) for g in sc["gaps_fit_mm"]], back_fit=res["back"], ambiguous=res["ambiguous"], unresolved=res.get("unresolved"), partial=res.get("partial"),
                weak=res["weak"], t_err_mm=[round(e, 3) for e in sc["t_err_mm"]],
                crb_mm=[round(1e3 * e, 4) for e in res["t_err"]], n_eval=res["n_eval"],
                search_chi2n=res["search_chi2n"], alternatives=res["alternatives"], attempts=res["attempts"],
                bnb={k: v for k, v in res["bnb"].items() if k != "passes"}, passes=res["bnb"]["passes"])


if __name__ == "__main__":
    s0, s1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["seed"] for l in open(out) if l.strip()}
    for seed in range(s0, s1 + 1):
        if seed in done:
            continue
        r = one(seed)
        open(out, "a").write(json.dumps(r, default=float) + "\n")
        tag = "CRASH" if "crash" in r else ("OK" if r["ok"] else ("flag" if r["alarm"] else ("PART" if r.get("partial") else ("OK3s" if r.get("ok3") else "SILENT"))))
        print(r["seed"], tag, f"{r['wall_s']:.0f}s", f"chi2/n {r.get('chi2n', 0):.2f}", r.get("truth"), r.get("gaps_true"), r.get("back"),
              "->", r.get("fit"), r.get("gaps_fit"), r.get("back_fit"), flush=True)

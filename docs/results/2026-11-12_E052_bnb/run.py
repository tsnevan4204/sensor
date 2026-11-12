"""E052 runner (branch-and-bound structure search): wallsolve2.solve on wallsolve.make_problem.  python3 run.py N seed0 seed1 out.jsonl
(one wall at a time: the compiled kernels use all cores)"""
import sys, json, time, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/oned"))
import numpy as np
import wallsolve as WS, wallsolve2 as W2

N = int(sys.argv[1])
VERBOSE = "-v" in sys.argv


def one(seed):
    f, y, s, tr = WS.make_problem(seed, N)
    t0 = time.time()
    try:
        res = W2.solve(f, y, s, N, tr["standoff"] + 0.0005, verbose=VERBOSE)
        sc = WS.score(res, tr, N)
    except Exception as e:
        import traceback; traceback.print_exc()
        return dict(seed=seed, crash=repr(e), wall_s=time.time() - t0)
    sol, gaps = WS.report(res, N)
    return dict(seed=seed, ok=sc["ok"], alarm=res["alarm"], chi2n=res["chi2n"], wall_s=time.time() - t0,
                search_s=res["search_s"], truth=[(L["name"], round(1e3 * L["t"], 2)) for L in tr["layers"]],
                gaps_true=[round(g, 2) for g in sc["gaps_true_mm"]], back=tr["back"],
                fit=[(s_["label"], round(1e3 * s_["t"], 2)) for s_ in sol], mat_ok=sc["mat_ok"],
                gaps_fit=[round(g, 2) for g in sc["gaps_fit_mm"]], back_fit=res["back"], ambiguous=res["ambiguous"],
                weak=res["weak"], t_err_mm=[round(e, 3) for e in sc["t_err_mm"]],
                crb_mm=[round(1e3 * e, 4) for e in res["t_err"]], n_eval=res["n_eval"],
                search_chi2n=res["search_chi2n"], alternatives=res["alternatives"],
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
        tag = "CRASH" if "crash" in r else ("OK" if r["ok"] else ("flag" if r["alarm"] else "SILENT"))
        print(r["seed"], tag, f"{r['wall_s']:.0f}s", f"chi2/n {r.get('chi2n', 0):.2f}", r.get("truth"), r.get("gaps_true"), r.get("back"),
              "->", r.get("fit"), r.get("gaps_fit"), r.get("back_fit"), flush=True)

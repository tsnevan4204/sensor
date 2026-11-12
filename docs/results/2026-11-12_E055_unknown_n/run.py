"""E055 runner: the number of layers is NOT given. True N = 1 + seed % 3; wallsolve2.solve_unknown_n
(fast refit + front-excluded discrepancy model) tries N = 1..4.  python3 run.py seed0 seed1 out.jsonl"""
import sys, json, time, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/oned"))
import numpy as np
import wallsolve as WS, wallsolve2 as W2, wallsearch as LS
LS.FRONT_SIGMA = True


def one(seed):
    N = 1 + seed % 3
    f, y, s, tr = WS.make_problem(seed, N)
    t0 = time.time()
    try:
        res = W2.solve_unknown_n(f, y, s, tr["standoff"] + 0.0005, n_max=4, fast=True)
        Nf = res["N"]
        sc = WS.score(res, tr, N) if Nf == N else dict(ok=False, gaps_true_mm=[1e3 * g for g in tr["gaps"]], gaps_fit_mm=[])
    except Exception as e:
        import traceback; traceback.print_exc()
        return dict(seed=seed, N_true=N, crash=repr(e), wall_s=time.time() - t0)
    sol, gaps = WS.report(res, Nf)
    return dict(seed=seed, N_true=N, N_fit=Nf, n_ok=bool(Nf == N), ok=bool(sc["ok"]), alarm=res["alarm"],
                partial=res.get("partial"), chi2n=res["chi2n"], wall_s=time.time() - t0, tried=res["tried"],
                truth=[(L["name"], round(1e3 * L["t"], 2)) for L in tr["layers"]],
                gaps_true=[round(1e3 * g, 2) for g in tr["gaps"]], back=tr["back"],
                fit=[(s_["label"], round(1e3 * s_["t"], 2)) for s_ in sol], gaps_fit=[round(1e3 * g, 2) for g in gaps],
                back_fit=res["back"])


if __name__ == "__main__":
    s0, s1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    done = {json.loads(l)["seed"] for l in open(out)} if os.path.exists(out) else set()
    for seed in range(s0, s1 + 1):
        if seed in done:
            continue
        r = one(seed)
        open(out, "a").write(json.dumps(r, default=float) + "\n")
        tag = "CRASH" if "crash" in r else ("OK" if r["ok"] else ("flag" if r["alarm"] else ("PART" if r.get("partial") else "SILENT")))
        print(r["seed"], tag, f"N {r.get('N_true')}->{r.get('N_fit')}", f"{r['wall_s']:.0f}s", f"chi2/n {r.get('chi2n', 0):.2f}", r.get("truth"), r.get("gaps_true"),
              r.get("back"), "->", r.get("fit"), r.get("gaps_fit"), r.get("back_fit"), [(t["N"], round(t["chi2n"], 2), t["alarm"]) for t in r.get("tried", [])], flush=True)

"""E050 baseline: wallsolve.solve_echo (E049d state) on wallsolve.make_problem.
python3 run.py N seed0 seed1 workers out.jsonl"""
import sys, json, time, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src/oned"))
import numpy as np
import wallsolve as WS
from multiprocessing import Pool

N = int(sys.argv[1])


def one(seed):
    f, y, s, tr = WS.make_problem(seed, N)
    t0 = time.time()
    try:
        res = WS.solve_echo(f, y, s, N, tr["standoff"] + 0.0005)
        sc = WS.score(res, tr, N)
    except Exception as e:
        return dict(seed=seed, crash=repr(e), wall_s=time.time() - t0)
    sol, gaps = WS.report(res, N)
    return dict(seed=seed, ok=sc["ok"], alarm=res["alarm"], chi2n=res["chi2n"], wall_s=time.time() - t0,
                truth=[(L["name"], round(1e3 * L["t"], 2)) for L in tr["layers"]], gaps_true=sc["gaps_true_mm"],
                back=tr["back"], fit=[(s_["label"], round(1e3 * s_["t"], 2)) for s_ in sol],
                gaps_fit=sc["gaps_fit_mm"], back_fit=res["back"], ambiguous=res["ambiguous"],
                n_echoes=res["n_echoes"], n_cands=res["n_cands"])


if __name__ == "__main__":
    s0, s1, wk, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["seed"] for l in open(out) if l.strip()}
    todo = [s for s in range(s0, s1 + 1) if s not in done]
    with Pool(wk) as p:
        for r in p.imap_unordered(one, todo):
            open(out, "a").write(json.dumps(r, default=float) + "\n")
            tag = "CRASH" if "crash" in r else ("OK" if r["ok"] else ("flag" if r["alarm"] else "SILENT"))
            print(r["seed"], tag, f"{r['wall_s']:.0f}s", r.get("truth"), r.get("gaps_true"), r.get("back"), "->", r.get("fit"), r.get("gaps_fit"), r.get("back_fit"), flush=True)

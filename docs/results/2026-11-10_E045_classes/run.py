"""E045 benchmark runner (material-class search, known N): python3 run.py N seed0 seed1 workers out.jsonl  (N = exact layer count)."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np
import stack as ST, known_n as KN
from multiprocessing import Pool

N = int(sys.argv[1])


def one(seed):
    f, y, s, d0, tr = ST.make_case(seed, n_exact=N)
    t0 = time.time()
    try:
        m, c, info = KN.invert_cls(f, y, s, N)
    except Exception as e:
        return dict(seed=seed, crash=repr(e))
    tl = tr["layers"]
    fl = ST.describe(m)
    struct = len(fl) == len(tl) and m.back == tr["back"]
    terr = [abs(a["t"] - b["t"]) for a, b in zip(tl, fl)] if struct else []
    ok = struct and all(e < max(0.5e-3, 0.03 * a["t"]) for e, a in zip(terr, tl))
    return dict(seed=seed, ok=bool(ok), struct=bool(struct), alarm=info["alarm"], chi2n=c / (2 * f.size),
                wall_s=time.time() - t0, truth=[(a["name"], a["t"]) for a in tl], back=tr["back"],
                fit=[(b["t"], b["er"], b["water"]) for b in fl], back_fit=m.back,
                classes=info["labels"])


if __name__ == "__main__":
    s0, s1, wk, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    import os
    done = set()
    if os.path.exists(out):                       # resumable: container restarts kill long runs
        done = {json.loads(l)["seed"] for l in open(out) if l.strip()}
    todo = [s for s in range(s0, s1 + 1) if s not in done]
    with Pool(wk) as p:
        for r in p.imap_unordered(one, todo):
            open(out, "a").write(json.dumps(r, default=float) + "\n")
            print(r.get("seed"), "OK" if r.get("ok") else ("flag" if r.get("alarm") else "SILENT"), f"{r.get('wall_s', 0):.1f}s", flush=True)

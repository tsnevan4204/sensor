"""E055 test B: reduced-budget PT (1500 steps, 1 restart per backing) with vs without NN seeds.
python3 runB.py shard nshard seed0 seed1 use_nn(0/1) out.jsonl"""
import sys, json, time, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, xsolve as XS, e055_nn as NN, pt as PT
sh, nsh, s0, s1, use = (int(a) for a in sys.argv[1:6]); out = sys.argv[6]
model = NN.load()
done = {json.loads(l)["seed"] for l in open(out)} if os.path.exists(out) else set()
for sd in list(range(s0, s1 + 1))[sh::nsh]:
    if sd in done:
        continue
    views, tr = XS.make_views(sd, 3, (0.0,))
    f, y, s, _ = views[0]
    dg = tr["standoff"] + 0.001
    t = time.time()
    init = None
    if use:
        xa, xm, pm = NN.predict(f, y, dg, model)
        init = {}
        rng = np.random.default_rng(sd)
        for back, x0 in (("air", xa), ("metal", xm)):
            mp = XS.MultiProblem(views, 3, back, dg)
            rows = [x0]
            for r in range(7):                 # NN front layer + standoff, deeper layers from the prior
                z = mp.sample_prior(1, rng)[0]
                z[0] = x0[0]; z[1:6] = x0[1:6]
                rows.append(z)
            init[back] = rows
    res = XS.solve(views, 3, dg, steps=1500, restarts=1, init=init)
    sc = XS.score(res, tr, 3)
    r = dict(seed=sd, use_nn=use, ok=sc["ok"], geo_ok=sc["geo_ok"], chi2n=res["chi2n"], alarm=res["alarm"], s=time.time() - t)
    open(out, "a").write(json.dumps(r, default=float) + "\n")
    print(sd, r["ok"], r["geo_ok"], round(r["chi2n"], 2), round(r["s"]), flush=True)

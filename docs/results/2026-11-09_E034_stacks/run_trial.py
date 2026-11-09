import sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np, stack as ST
from multiprocessing import Pool
def one(seed):
    f, y, s, d0, tr = ST.make_case(seed, nmax=int(sys.argv[1]))
    t0 = time.time()
    try:
        m, c, info = ST.invert(f, y, s, d0)
    except Exception as e:
        return f"seed {seed}: CRASH {e!r}"
    return (f"seed {seed}: truth {[ (L['name'], round(L['t']*1e3,1)) for L in tr['layers']]} {tr['back']} d {tr['standoff']:.4f}\n"
            f"   fit  {[ (round(L.t.value*1e3,1), round(L.eps_r.value,2), 'W' if L.theta.value>0.5 else '') for L in m.layers]} {m.back} d {m.standoff.value:.4f} chi2 {c:.0f}/{2*f.size} alarm {info['alarm']} {time.time()-t0:.0f}s\n"
            f"   class {[list(k)[0] for k in info['classes']]}")
with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 3) as p:
    for r in p.imap(one, range(int(sys.argv[2]), int(sys.argv[3]) + 1)): print(r, flush=True)

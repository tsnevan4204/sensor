"""Development harness for wallsearch.search with truth comparison (observability)."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import wallsolve as WS, treesolve as TS, wallsearch as LS, devtree as DT


def truth_cmp(ctx, S, tr):
    """(classes match, max interface-position error in mm) of a state against the true wall."""
    ti = DT.truth_interfaces(tr)
    tab = LS._table(ctx, S)
    tau = LS._taus(S, tab[1], ctx)
    names = [ctx.lib.base[c] for c in S.cls] + [ctx.lib.base[S.back]]
    tn = [c for _, c in ti]
    if names != tn:
        return False, None
    return True, float(np.max(np.abs(tau - np.array([z for z, _ in ti]))) * 1e3)


def run(seed, N=3, verbose=True):
    f, y, s, tr = WS.make_problem(seed, N)
    fs, ys, ss, segs = TS.search_subset(f, y, s)
    print(f"seed {seed}: truth", [(L['name'], round(1e3 * L['t'], 2)) for L in tr["layers"]],
          "gaps", [round(1e3 * g, 2) for g in tr["gaps"]], tr["back"], flush=True)
    holder = {}

    def trace(name, n, S0, S1):
        ok, err = truth_cmp(holder["ctx"], S1, tr) if "ctx" in holder else (None, None)
        print(f"    {name:10s} n={n:8d} cost {S0.cost:10.1f} -> {S1.cost:10.1f}  {S1.describe(holder['ctx'])} truth={ok} err={err}", flush=True)
    orig = LS.Ctx.__init__

    def init(self, *a, **k):
        orig(self, *a, **k); holder["ctx"] = self
    LS.Ctx.__init__ = init
    t0 = time.time()
    sols, ctx = LS.search(fs, ys, ss, N, tr["standoff"] + 0.0005, trace=trace if verbose else None, verbose=True)
    LS.Ctx.__init__ = orig
    n = 2 * fs.size
    for S in sols[:4]:
        ok, err = truth_cmp(ctx, S, tr)
        print(f"  -> cost {S.cost:9.1f} chi2/n {S.chi / n:6.2f} {S.describe(ctx)} truth={ok} err_mm={err}")
    print(f"  {time.time() - t0:.1f}s, {ctx.n_eval:.2e} evals, {ctx.n_polish} polishes", flush=True)
    return sols, ctx, tr


if __name__ == "__main__":
    for sd in sys.argv[1:]:
        run(int(sd))

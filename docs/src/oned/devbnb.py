"""Development harness for bnb.solve_grid: grid optimum, node counts, and where the truth's grid
representative ranks (observability)."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import wallsolve as WS, treesolve as TS, bnb as BB, devtree as DT, fastbatch as FB, wall_inverse as I


def truth_rep(tr, lib, d0, h, m):
    ti = DT.truth_interfaces(tr)
    cls = tuple(lib.names.index(c) for _, c in ti[:-1])
    back = lib.names.index(ti[-1][1])
    q = tuple(int(round((z - d0) / h)) - 1 for z, _ in ti[1:])
    return cls, q, back


PER_BAND = (24, 64)
FRONT = False


def run(seed, N=3, h=5e-3, rho=1.5, extent=None, **kw):
    f, y, s, tr = WS.make_problem(seed, N)
    fs, ys, ss, segs = TS.search_subset(f, y, s, per_band=PER_BAND)
    lib = TS.Library(fs)
    reps, members = BB.class_clusters(lib)
    print('  clusters:', [[lib.names[c] for c in v] for v in members.values()])
    d0 = tr["standoff"] + 0.0005
    f0, y0, s0, sg0 = TS.search_subset(f, y, s)
    ext = TS._data_extent(f0, y0, s0, sg0, d0) if extent is None else extent
    print(f"seed {seed}: truth", [(L['name'], round(1e3 * L['t'], 1)) for L in tr["layers"]],
          "gaps", [round(1e3 * g, 1) for g in tr["gaps"]], tr["back"], f"extent {1e3 * ext:.0f} mm", flush=True)
    ss_true = ss
    ss = BB.model_sigma(fs, ys, ss, segs, h, d0=(d0 if FRONT else None))
    print("  sigma_eff/sigma at band edges:", [(round(fs[i0] / 1e9, 1), round(float(ss[i0] / ss_true[i0]), 1), round(fs[i1 - 1] / 1e9, 1), round(float(ss[i1 - 1] / ss_true[i1 - 1]), 1)) for i0, i1 in segs])
    leaves, info = BB.solve_grid(fs, ys, ss, segs, lib, N, d0, ext, h=h, rho=rho, verbose=True, reps=reps, **kw)
    tc, tq, tb = truth_rep(tr, lib, d0, h, info["m"])
    rep_of = {c: r for r, v in members.items() for c in v}
    tc = tuple(rep_of.get(c, c) for c in tc)
    k0 = 2 * np.pi * fs / TS.C0
    # truth representative: best nominal grid cost over +-1 index shifts of every interface
    import itertools
    best_rep = np.inf
    for sh in itertools.product((-1, 0, 1), repeat=len(tq)):
        q = np.array(tq) + np.array(sh)
        if q[0] < 0 or np.any(np.diff(q) <= 0):
            continue
        t = h * np.diff(np.concatenate([[-1], q])) / lib.nref[list(tc)]
        c = FB.chi_batch(np.array([tc]), np.array([t]), np.array([len(tc)]), np.array([tb]), np.array([d0]), lib.ntab, k0, ys, 1 / ss ** 2)[0]
        best_rep = min(best_rep, c)
    rank = sum(1 for L in leaves if L[0] < best_rep)
    seqs = []
    for L in leaves:
        if (L[1], L[3]) not in seqs:
            seqs.append((L[1], L[3]))
    srank = seqs.index((tc, tb)) if (tc, tb) in seqs else None
    print(f"  truth rep cost {best_rep:.0f} (chi2/n {best_rep / (2 * fs.size):.2f}) = {best_rep / info['best']:.2f} x best; leaves better: {rank}; "
          f"true class sequence rank among sequences: {srank} of {len(seqs)}")
    for L in leaves[:3]:
        print("    ", round(L[0]), [(lib.names[c], round(float(1e3 * h * dq / lib.nref[c]), 1)) for c, dq in zip(L[1], np.diff(np.concatenate([[-1], L[2]])))], lib.names[L[3]])
    return leaves, info


if __name__ == "__main__":
    for sd in sys.argv[1:]:
        run(int(sd))

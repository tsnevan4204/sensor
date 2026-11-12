"""Development harness for treesolve.march with truth tracing (observability):
at every gate, the rank and score of the best survivor consistent with the true wall."""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import wallsolve as WS, treesolve as TS, stack as ST

TOL = 4e-3


def truth_interfaces(tr):
    """[(optical range, class name behind)] for the true wall, front face first."""
    f5 = np.array([TS.F_NREF])
    z = tr["standoff"]
    out = []
    it = iter(tr["_layers"])
    seq = []
    for i, L in enumerate(tr["layers"]):
        seq.append((L["name"], next(it)))
        if i < len(tr["gaps"]) and tr["gaps"][i] > 0:
            seq.append(("air", next(it)))
    for nm, (t, e) in seq:
        out.append((z, nm))
        z += float(np.sqrt(e(f5) + 0j)[0].real) * t
    out.append((z, "metal" if tr["back"] == "metal" else "air"))
    if out[-1][1] == "air" and len(out) >= 2 and out[-2][1] == "air":
        out.pop()                                   # gap then air backing cannot occur
    return out


def consistent(beam, lib, ti, T, i):
    nl = int(beam.nl[i])
    pos = beam.d[i] + np.concatenate([[0.0], np.cumsum(beam.t[i, :nl] * lib.nref[beam.cls[i, :nl]])])
    cl = [lib.base[c] for c in beam.cls[i, :nl]] + [lib.base[beam.cur[i]]]
    m = len(pos)
    need_lo = sum(1 for z, _ in ti if z <= T - TOL)
    need_hi = sum(1 for z, _ in ti if z <= T + TOL)
    if not (need_lo <= m <= need_hi) or m > len(ti):
        return False
    return all(cl[k] == ti[k][1] and abs(pos[k] - ti[k][0]) < TOL for k in range(m))


def make_trace(tr, rows):
    ti = truth_interfaces(tr)

    def trace(j, T, beam, sc, lib, K):
        order = np.argsort(sc)
        rank = None
        for r, i in enumerate(order):
            if consistent(beam, lib, ti, T, i):
                rank = r; best = float(sc[i]); break
        rows.append((1e3 * (T - tr["standoff"]), K, len(beam), float(sc[order[0]]), rank, best if rank is not None else None))
    return trace, ti


def run(seed, N=3, verbose=True, **kw):
    f, y, s, tr = WS.make_problem(seed, N)
    fs, ys, ss, segs = TS.search_subset(f, y, s)
    rows = []
    trace, ti = make_trace(tr, rows)
    t0 = time.time()
    cands, info, lib = TS.march(fs, ys, ss, N, tr["standoff"] + 0.0005, trace=trace, **kw)
    if verbose:
        print(f"seed {seed}: truth", [(L['name'], round(1e3 * L['t'], 1)) for L in tr["layers"]],
              "gaps", [round(1e3 * g, 1) for g in tr["gaps"]], tr["back"])
        print("  interfaces (mm):", [(round(1e3 * (z - tr['standoff']), 1), c) for z, c in ti])
        lost = None
        for r in rows:
            if r[4] is None and lost is None:
                lost = r[0]
        for r in rows[::4]:
            print(f"   T={r[0]:6.1f} K={r[1]:3d} beam={r[2]:5d} best={r[3]:9.1f} truth rank={r[4]} score={r[5] if r[5] is None else round(r[5],1)}")
        print(f"  LOST at {lost}" if lost is not None else "  truth survived", f"| evals {info['n_eval']:.2e}, {time.time() - t0:.1f}s, t_end {info['t_end']*1e3:.0f} mm")
        for c in cands[:5]:
            print("   ", round(c[0], 1), [(lib.names[k], round(1e3 * t, 1)) for k, t in zip(c[1], c[2])], lib.names[c[3]])
    return rows, cands, info, lib, tr


if __name__ == "__main__":
    for sd in sys.argv[1:]:
        run(int(sd))

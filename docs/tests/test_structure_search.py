"""Gates for the E051/E052 structure search (compiled kernels and branch-and-bound)."""
import itertools
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "oned"))
import numpy as np
import stack as ST
import wall_toy as W
import wallsolve as WS
import treesolve as TS
import fastbatch as FB
import polish as PL
import bnb as BB

C0 = W.C0


def _freqs():
    return np.concatenate([np.linspace(a, b, 40) for a, b in WS.BANDS3])


def test_fastbatch_equals_reference():
    """Table-driven batched recursion == independent transfer-matrix reference."""
    f = _freqs()
    k0 = 2 * np.pi * f / C0
    rng = np.random.default_rng(0)
    names = list(ST.ALL_MATERIALS)
    laws = [ST.material_law(nm, rng, 20.0)[0] for nm in names]
    nt = np.array([np.sqrt(e(f) + 0j) for e in laws] + [np.sqrt(np.full(f.size, ST.METAL_EPS))])
    nt = np.where(nt.imag > 0, -nt, nt)
    AIR, MET = names.index("air"), len(names)
    worst = 0.0
    for _ in range(40):
        nl = int(rng.integers(1, 8))
        c = rng.integers(0, len(names), nl)
        t = np.exp(rng.uniform(np.log(1e-3), np.log(0.1), nl))
        bk = MET if rng.random() < 0.5 else AIR
        ref = ST.gamma_stack(f, [(t[i], laws[c[i]]) for i in range(nl)], "metal" if bk == MET else "air") * np.exp(-2j * k0 * 0.3)
        g = FB.gamma_batch(c[None, :].astype(np.int64), t[None, :], np.array([nl]), np.array([bk]), np.array([0.3]), nt, k0)[0]
        worst = max(worst, float(np.abs(g - ref).max()))
    assert worst < 1e-12


def test_polish_forward_equals_reference():
    """The compiled polish forward (Jonscher solids, LHM water, air, metal) == reference."""
    f = _freqs()
    k0 = 2 * np.pi * f / C0
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        er, td, pp = rng.uniform(1.5, 8), rng.uniform(1e-3, 0.2), rng.uniform(0, 0.99)
        Tw, t = rng.uniform(5, 35), rng.uniform(2e-3, 0.05, 3)
        ap = 2 / np.pi * np.arctan(td / (1 - pp))
        solid = lambda x: er * (pp + (1 - pp - 1j * td) * (np.asarray(x, float) / ST.F_REF) ** (-ap))
        water = lambda x: W.eps_water_lhm(x, Tw)
        air = lambda x: np.ones(np.shape(x), complex)
        for back, bk in (("air", 2), ("metal", 3)):
            ref = ST.gamma_stack(f, [(t[0], solid), (t[1], water), (t[2], air)], back) * np.exp(-2j * k0 * 0.31)
            g = PL.gamma_one(np.array([0, 1, 2] + [0] * 9), np.array([er, 1, 1] + [1.0] * 9), np.array([td, 0, 0] + [0.0] * 9),
                             np.array([pp, 0, 0] + [0.0] * 9), np.concatenate([t, np.zeros(9)]), 3, bk, 0.31, Tw, f, k0)
            worst = max(worst, float(np.abs(g - ref).max()))
    assert worst < 1e-12


def _all_walls(lib, N, m, dqmin):
    """Every admissible grid wall (python reference for the tree): yields (classes, q, back)."""
    AIR, MET = lib.AIR, lib.METAL
    solids = [int(c) for c in lib.is_solid_idx]

    def rec(cls, qs, ns):
        cur = cls[-1]
        last = qs[-1] if qs else -1
        if ns == N and cur == AIR:
            yield tuple(cls[:-1]), tuple(qs), AIR
        if cur == MET:
            yield tuple(cls[:-1]), tuple(qs), MET
            return
        opts = []
        if cur != AIR:
            opts.append((AIR, ns))
        if ns < N:
            opts += [(c, ns + 1) for c in solids if c != cur]
        if ns == N:
            opts.append((MET, ns))
        start = last + int(dqmin[cur]) if qs else int(dqmin[cur]) - 1
        for q in range(start, m):
            for c, ns2 in opts:
                yield from rec(cls + [c], qs + [q], ns2)
    for c1 in solids:
        yield from rec([c1], [], 1)


def test_bnb_equals_brute_force():
    """rho = 1 branch-and-bound returns the optimum of the grid problem (N = 2, all gap patterns,
    both backings), found independently by enumerating every admissible wall."""
    f = np.concatenate([np.linspace(0.4e9, 0.96e9, 12), np.linspace(3.1e9, 10.6e9, 32)])
    segs = [(0, 12), (12, 44)]
    lib = TS.Library(f)
    keep = [lib.names.index(n) for n in ("plasterboard", "brick", "glass", "water")]
    k0 = 2 * np.pi * f / C0
    h, m, N, d0 = 5e-3, 9, 2, 0.30
    rng = np.random.default_rng(3)
    for trial, (cls, q, back) in enumerate([((keep[0], lib.AIR, keep[2]), (2, 4, 6), lib.METAL),
                                           ((keep[1], keep[3]), (5, 7), lib.AIR)]):
        t = h * np.diff(np.concatenate([[-1], q])) / lib.nref[list(cls)] * rng.uniform(0.97, 1.03, len(cls))
        y = FB.gamma_batch(np.array([cls]), t[None, :], np.array([len(cls)]), np.array([back]), np.array([d0 + 3e-4]), lib.ntab, k0)[0]
        sig = np.full(f.size, 10 ** (-40 / 20) / np.sqrt(2))
        y = y + sig * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
        s_eff = BB.model_sigma(f, y, sig, segs, h)
        leaves, info = BB.solve_grid(f, y, s_eff, segs, lib, N, d0, m * h, h=h, rho=1.0, reps=keep, max_nodes=10 ** 9, budget=10 ** 12)
        lib2_solid = lib.solid.copy()
        tmin = np.where(lib.solid, BB.T_SOLID, BB.T_THIN); tmin[lib.WATER] = BB.T_THIN
        dqmin = np.maximum(1, np.ceil(tmin * lib.nref / h - 1e-9)).astype(int)

        class L2:                                   # the library restricted to the branched classes
            AIR, METAL, is_solid_idx = lib.AIR, lib.METAL, keep
        best = np.inf
        walls = list(_all_walls(L2, N, m, dqmin))
        for nl in sorted({len(w[0]) for w in walls}):
            sub = [w for w in walls if len(w[0]) == nl]
            C = np.array([w[0] for w in sub], np.int64)
            Q = np.array([w[1] for w in sub])
            T = np.ascontiguousarray(h * np.diff(np.concatenate([-np.ones((len(sub), 1)), Q], axis=1), axis=1) / lib.nref[C])
            chi = FB.chi_batch(C, T, np.full(len(sub), nl), np.array([w[2] for w in sub], np.int64), np.full(len(sub), d0),
                               lib.ntab, k0, y, 1 / s_eff ** 2)
            best = min(best, float(chi.min()))
        assert not info["truncated"]
        assert len(walls) > 1000
        assert abs(info["best"] - best) <= 1e-6 * best, (trial, info["best"], best)

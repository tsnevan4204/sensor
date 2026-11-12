"""
E052: branch-and-bound over ALL class-discrete walls on a position grid (known N; any admissible
gap pattern; both backings), with time-gate lower bounds.

Search space (the "grid problem")
  A wall = N solid layers (library classes incl. water), optional air gap behind every solid layer
  (behind the last one only before a metal backing), backing air or metal; every interface at an
  optical depth d0 + h (q + 1), q = 0..m-1 behind the front face; nominal class members.
  Cost = sum_f |Gamma(f) - y(f)|^2 / sigma_f^2 on the search bands (exact reflection recursion).

Tree
  A node is a wall prefix: k complete layers and the class of the half-space behind them. Its
  response G_node is exact for every completion at two-way delays earlier than the completion's
  next interface (causality). So for a child whose next interface is at grid index q,

      cost(any completion)  >=  ( || Q_q (y - G_node) ||  -  A_q )^2          (gate bound)

  where the rows of Q_q are orthonormal, noise-whitened, and concentrated in the delay window
  ending at that interface (Slepian sequences of the window, per band), and A_q bounds what the
  completion's later echoes can leak into the gate (doc 16, Lemma 2). Children are scanned in
  increasing q: as soon as the bound exceeds the pruning threshold, no later q can work either
  (the half-space model has failed: an interface must come earlier) and the node is closed.

Result
  Every grid wall with cost <= rho * (best grid cost) is visited (rho >= 1); the best `cap` of each
  root task are returned. rho = 1 returns the global optimum of the grid problem.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange
import wall_toy as W
import treesolve as TS

C0 = W.C0


def build_gates(f, s, segs, d0, h, m, eps=1e-5, lead=0.05, a_tail=1.0):
    """Qpad (m, Kmax, F) whitened gate rows for the windows [d0 - lead, d0 + h (q + 1)], their
    row counts Kn (m,), Qy is computed by the caller; A (m,) leakage allowances:
    A_q = a_tail * sqrt( sum_bands sup_{rho >= window end} || Q_qb e(rho) / sigma ||^2 ), the
    largest gated norm of a unit non-dispersive echo from behind the window (numerically, on a
    fine range grid over one unambiguous period)."""
    Qs, A = [], np.zeros(m)
    for q in range(m):
        T = d0 + h * (q + 1)
        rows, leak = [], 0.0
        for (i0, i1) in segs:
            fb = f[i0:i1]
            Qb, _ = TS.slepian(fb, d0 - lead, T, eps)
            full = np.zeros((Qb.shape[0], f.size), complex)
            full[:, i0:i1] = Qb / s[i0:i1][None, :]
            rows.append(full)
            if Qb.shape[0]:
                Ru = C0 / (2 * (fb[1] - fb[0]))
                rho = np.linspace(T, d0 - lead + Ru, 6 * fb.size)
                E = np.exp(-2j * 2 * np.pi * fb[:, None] / C0 * rho[None, :]) / s[i0:i1][:, None]
                leak += float(np.max(np.sum(np.abs(Qb @ E) ** 2, axis=0)))
        Qs.append(np.vstack(rows))
        A[q] = a_tail * np.sqrt(leak)
    Kn = np.array([Q.shape[0] for Q in Qs], np.int64)
    Qpad = np.zeros((m, max(int(Kn.max()), 1), f.size), np.complex128)
    for q, Q in enumerate(Qs):
        Qpad[q, : Q.shape[0]] = Q
    return Qpad, Kn, A


@njit(cache=True, fastmath=True)
def _node_response(lc, qp, k, Etab, rtab, ph, G):
    """Response of the prefix: layers 0..k-1 (classes lc[0..k-1], back interfaces at grid indices
    qp[0..k-1]) in front of a half-space of class lc[k]. Etab[c, dq, f] = exp(-2j k0 n_c h dq /
    nref_c) and rtab[a, b, f] = (n_a - n_b)/(n_a + n_b) (a = C: air in front) are tabulated, so a
    node costs k Moebius steps per frequency and no transcendental function."""
    F = ph.shape[0]
    C = rtab.shape[0] - 1
    for f in range(F):
        g = rtab[lc[k - 1] if k > 0 else C, lc[k], f]
        for i in range(k - 1, -1, -1):
            dq = qp[i] - (qp[i - 1] if i > 0 else -1)
            r = rtab[lc[i - 1] if i > 0 else C, lc[i], f]
            ge = g * Etab[lc[i], dq, f]
            g = (r + ge) / (1.0 + r * ge)
        G[f] = g * ph[f]


@njit(cache=True, fastmath=True)
def _full_cost(G, y, w, stop):
    s = 0.0
    for f in range(G.shape[0]):
        dr = G[f].real - y[f].real
        di = G[f].imag - y[f].imag
        s += (dr * dr + di * di) * w[f]
        if s > stop:
            break
    return s


@njit(cache=True, fastmath=True)
def _gate_bound(G, Q, Qy, K, A):
    s = 0.0
    for k in range(K):
        z = -Qy[k]
        for f in range(G.shape[0]):
            z += Q[k, f] * G[f]
        s += z.real * z.real + z.imag * z.imag
    r = np.sqrt(s) - A
    return r * r if r > 0 else 0.0


@njit(cache=True, fastmath=True)
def _record(cost, lc, qp, k, back, o_cost, o_nl, o_cls, o_q, o_back):
    cap = o_cost.shape[0]
    j = 0
    worst = -1.0
    for i in range(cap):
        if o_cost[i] > worst:
            worst = o_cost[i]; j = i
    if cost < worst:
        o_cost[j] = cost; o_nl[j] = k; o_back[j] = back
        for i in range(k):
            o_cls[j, i] = lc[i]; o_q[j, i] = qp[i]


@njit(cache=True, fastmath=True)
def _valid(c, cur, ns, N, AIR, METAL, solid):
    if c == METAL:
        return ns == N
    if c == AIR:
        return cur != AIR
    return solid[c] and ns < N and c != cur


@njit(cache=True, fastmath=True)
def _dfs(c1, c2, q1, N, m, Etab, rtab, ph, y, w, Qpad, Kn, Qy, A, best, rho, thr, AIR, METAL, solid,
         o_cost, o_nl, o_cls, o_q, o_back, stats, max_nodes):
    F = ph.shape[0]
    C = rtab.shape[0] - 1
    LM = 2 * N + 1
    lc = np.zeros(LM + 1, np.int64)
    qp = np.zeros(LM + 1, np.int64)
    ns = np.zeros(LM + 1, np.int64)
    qi = np.zeros(LM + 1, np.int64)
    oi = np.zeros(LM + 1, np.int64)
    G = np.zeros((LM + 1, F), np.complex128)
    # root of this task: half-space c1 behind the front face; its bound for a first interface at q1
    lc[0] = c1
    _node_response(lc, qp, 0, Etab, rtab, ph, G[0])
    if Kn[q1] > 0 and _gate_bound(G[0], Qpad[q1], Qy[q1], Kn[q1], A[q1]) > min(thr, rho * best[0]):
        stats[1] += 1
        return
    qp[0] = q1; lc[1] = c2
    ns[0] = 1
    ns[1] = 1 + (1 if solid[c2] else 0)
    level = 1
    _node_response(lc, qp, 1, Etab, rtab, ph, G[1])
    stats[0] += 1
    if c2 == METAL or (c2 == AIR and ns[1] == N):
        lim = min(thr, rho * best[0])
        cost = _full_cost(G[1], y, w, lim)
        stats[2] += 1
        if cost <= lim and cost < best[0]:
            best[0] = cost
        if cost <= lim:
            _record(cost, lc, qp, 1, c2, o_cost, o_nl, o_cls, o_q, o_back)
    if c2 == METAL:
        return
    qi[1] = q1 + 1; oi[1] = 0
    while level >= 1:
        k = level
        if qi[k] >= m or k >= LM or stats[0] > max_nodes:
            level -= 1
            continue
        q = qi[k]
        if oi[k] == 0 and Kn[q] > 0:
            if _gate_bound(G[k], Qpad[q], Qy[q], Kn[q], A[q]) > min(thr, rho * best[0]):
                stats[1] += 1
                level -= 1                      # no interface at q or later can rescue this prefix
                continue
        c = oi[k]
        while c < C and not _valid(c, lc[k], ns[k], N, AIR, METAL, solid):
            c += 1
        if c >= C:
            qi[k] = q + 1; oi[k] = 0
            continue
        oi[k] = c + 1
        qp[k] = q; lc[k + 1] = c
        ns[k + 1] = ns[k] + (1 if solid[c] else 0)
        _node_response(lc, qp, k + 1, Etab, rtab, ph, G[k + 1])
        stats[0] += 1
        if c == METAL or (c == AIR and ns[k + 1] == N):
            lim = min(thr, rho * best[0])
            cost = _full_cost(G[k + 1], y, w, lim)
            stats[2] += 1
            if cost <= lim and cost < best[0]:
                best[0] = cost
            if cost <= lim:
                _record(cost, lc, qp, k + 1, c, o_cost, o_nl, o_cls, o_q, o_back)
        if c != METAL and k + 1 < LM:
            level = k + 1
            qi[level] = q + 1; oi[level] = 0


@njit(parallel=True, cache=True, fastmath=True)
def run_tasks(tasks, N, m, Etab, rtab, ph, y, w, Qpad, Kn, Qy, A, best, rho, thr, AIR, METAL, solid, cap, max_nodes):
    T = tasks.shape[0]
    LM = 2 * N + 1
    o_cost = np.full((T, cap), np.inf)
    o_nl = np.zeros((T, cap), np.int64)
    o_cls = np.zeros((T, cap, LM), np.int64)
    o_q = np.zeros((T, cap, LM), np.int64)
    o_back = np.zeros((T, cap), np.int64)
    stats = np.zeros((T, 3), np.int64)
    for t in prange(T):
        _dfs(tasks[t, 0], tasks[t, 1], tasks[t, 2], N, m, Etab, rtab, ph, y, w, Qpad, Kn, Qy, A, best, rho, thr,
             AIR, METAL, solid, o_cost[t], o_nl[t], o_cls[t], o_q[t], o_back[t], stats[t], max_nodes)
    return o_cost, o_nl, o_cls, o_q, o_back, stats


def solve_grid(f, y, s, segs, lib, N, d0, extent, h=5e-3, rho=1.5, eps=1e-3, cap=16, incumbent=np.inf,
               max_nodes=2_000_000, a_tail=1.0, thr0=0.125, reps=None, verbose=False):
    """Branch-and-bound over the grid problem. Returns leaves [(cost, classes, grid indices, back)]
    sorted by cost (all with cost <= rho * best found), and a statistics dict."""
    import time
    t0 = time.time()
    m = int(np.ceil(extent / h))
    k0 = 2 * np.pi * f / C0
    w = 1.0 / s ** 2
    Qpad, Kn, A = build_gates(f, s, segs, d0, h, m, eps, a_tail=a_tail)
    Qy = np.einsum("qkf,f->qk", Qpad, y)
    t_g = time.time() - t0
    solid = np.ascontiguousarray(lib.solid).copy()
    if reps is not None:                       # only cluster representatives are branched on
        solid[:] = False
        solid[np.asarray(reps, np.int64)] = True
    AIR, METAL = lib.AIR, lib.METAL
    tasks = []
    for c1 in np.where(solid)[0]:
        for c2 in range(len(lib.names)):
            if not _valid(c2, c1, 1, N, AIR, METAL, solid):
                continue
            for q1 in range(m):
                tasks.append((c1, c2, q1))
    tasks = np.array(tasks, np.int64)
    tasks = tasks[np.random.default_rng(0).permutation(len(tasks))]      # balance the threads (static chunks)
    C = len(lib.names)
    base = np.exp(-2j * k0[None, :] * lib.ntab * (h / lib.nref[:, None]))
    Etab = np.ascontiguousarray(np.cumprod(np.concatenate([np.ones((C, 1, f.size), complex),
                                                           np.repeat(base[:, None, :], m + 1, axis=1)], axis=1), axis=1))
    nfront = np.vstack([lib.ntab, np.ones((1, f.size), complex)])            # row C: air in front of the wall
    rtab = np.ascontiguousarray((nfront[:, None, :] - nfront[None, :, :]) / (nfront[:, None, :] + nfront[None, :, :]))
    ph = np.exp(-2j * k0 * d0)
    best = np.array([incumbent], float)
    n_res = 2 * f.size
    # threshold continuation: search under an absolute misfit threshold, doubled until a wall is
    # found; each pass prunes hard, and the first successful pass ends at the grid optimum
    passes = []
    thr = thr0 * n_res
    nodes_total = 0
    while True:
        t_p = time.time()
        out = run_tasks(tasks, N, m, Etab, rtab, ph, y, w, Qpad, Kn, Qy, A, best, 1.0, thr, AIR, METAL, solid, 1, max_nodes)
        nodes_total += int(out[5][:, 0].sum())
        passes.append((thr / n_res, float(best[0]), int(out[5][:, 0].sum()), time.time() - t_p))
        if verbose:
            print(f"    pass chi2/n <= {thr / n_res:6.1f}: nodes {passes[-1][2]:.2e}, best {best[0]:.0f} ({passes[-1][3]:.1f}s)", flush=True)
        if np.isfinite(best[0]) and best[0] <= thr:
            break
        if thr > 1e4 * n_res:
            break
        thr *= 2
    out = run_tasks(tasks, N, m, Etab, rtab, ph, y, w, Qpad, Kn, Qy, A, best, rho, np.inf, AIR, METAL, solid, cap, max_nodes)
    o_cost, o_nl, o_cls, o_q, o_back, stats = out
    leaves = []
    lim = rho * best[0]
    for t, j in zip(*np.where(o_cost <= lim)):
        nl = int(o_nl[t, j])
        leaves.append((float(o_cost[t, j]), tuple(int(c) for c in o_cls[t, j, :nl]),
                       tuple(int(x) for x in o_q[t, j, :nl]), int(o_back[t, j])))
    leaves.sort(key=lambda t: t[0])
    info = dict(passes=passes, nodes_search=nodes_total, nodes=int(stats[:, 0].sum()), pruned=int(stats[:, 1].sum()), leaves_eval=int(stats[:, 2].sum()),
                best=float(best[0]), m=m, h=h, K_end=int(Kn[-1]), gate_s=t_g, wall_s=time.time() - t0,
                truncated=bool(np.any(stats[:, 0] > max_nodes)), n_tasks=len(tasks), A_max=float(A.max()))
    if verbose:
        print(f"  bnb: m={m} h={1e3 * h:.1f} mm, K_end={info['K_end']}, nodes {info['nodes']:.2e}, pruned {info['pruned']:.2e}, "
              f"leaves {info['leaves_eval']:.2e}, best {best[0]:.0f}, kept {len(leaves)}, truncated {info['truncated']}, "
              f"A_max {info['A_max']:.1f}, {info['wall_s']:.1f}s (gates {t_g:.1f}s)", flush=True)
    return leaves, info


def model_sigma(f, y, s, segs, h, eps_mat=0.07, d_err=0.5e-3):
    """Effective per-frequency error level for the GRID problem: measurement noise plus the model
    discrepancy of a wall restricted to the grid with nominal class members. An interface at a
    position uniformly within +-h/2 of its grid point changes its echo a exp(-2j k0 tau) by
    a (exp(-2j k0 delta) - 1): variance |a|^2 2 (1 - sinc(k0 h / pi)) ~ |a|^2 (2 k0 h)^2 / 12;
    the standoff guess adds (2 k0 d_err)^2; class variability changes the echo amplitude by a
    relative eps_mat. |a|^2 summed over the echoes is the band's mean |y|^2. Per real component."""
    k0 = 2 * np.pi * f / C0
    out = np.empty_like(s)
    for i0, i1 in segs:
        P = float(np.mean(np.abs(y[i0:i1]) ** 2))
        x = k0[i0:i1] * h
        var = P * (2 * (1 - np.sinc(x / np.pi)) + (2 * k0[i0:i1] * d_err) ** 2 + eps_mat ** 2)
        out[i0:i1] = np.sqrt(s[i0:i1] ** 2 + var / 2)
    return out


def class_clusters(lib, tol=0.045):
    """Greedy clusters of solid classes whose nominal refractive indices differ by less than tol
    (relative, complex, worst search frequency): members are interchangeable at the grid problem's
    resolution. Returns (representatives, {rep: members})."""
    idx = [int(c) for c in lib.is_solid_idx]
    reps, members = [], {}
    for c in idx:
        for r in reps:
            if np.max(np.abs(lib.ntab[c] - lib.ntab[r]) / np.abs(lib.ntab[r])) < tol:
                members[r].append(c)
                break
        else:
            reps.append(c); members[c] = [c]
    return reps, members

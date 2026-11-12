"""
E051: structure search for the known-N wall problem by LARGE EXHAUSTIVE NEIGHBOURHOODS on the
exact model, with class-discrete materials and continuous refinement while ranking.

Why this design (measured, results/2026-11-12_E051_*/rank_test.txt):
  - with the true material classes, an exhaustive scan of all interface positions on a 2.5 mm
    optical grid puts the true wall at / next to the global minimum of the exact misfit;
  - with the true positions, an exhaustive scan of all class sequences ranks the true one first;
  - the exceptions are walls whose actual material sample is far from its class's nominal member,
    so candidates must be refined continuously (materials within their class) before ranking;
  - a front-to-back time-gated beam cannot prune thin walls: the UWB gate holds ~1 complex number
    per 20 mm of optical depth.
So: a state is a complete wall (class per layer, continuous thickness and material per layer,
gaps, backing). A move enumerates EXHAUSTIVELY a large block of the wall with everything else
fixed (compiled exact forward, 1e5-1e6 walls/s), polishes the best few with a compiled LM
(thicknesses, materials within the class prior, standoff, water temperature) and is accepted when
the MAP cost drops. Blocks: two adjacent layers (both classes x both back-interface positions),
one layer, all class sequences, gap insertion / removal at every slot, the backing.
The answer is then refitted by the exact adjoint LM on all bands (wallsolve2.py).
"""
from __future__ import annotations
import itertools
import time
import numpy as np
import wall_toy as W
import wall_inverse as I
import stack as ST
import fastbatch as FB
import polish as PL
import treesolve as TS

C0 = W.C0
KIND_DRY, KIND_WATER, KIND_AIR, KIND_METAL = 0, 1, 2, 3
GAPS = np.array([1.0, 1.6, 2.5, 4.0, 6.0, 9.0, 13.0, 19.0, 27.0, 38.0]) * 1e-3
DMIN = 1.5e-3            # smallest optical thickness enumerated for a layer
EXTENT_CAP = 0.40        # first search depth (optical, behind the front face); the chi2 alarm escalates it


class Ctx:
    def __init__(self, f, y, s, N, d_guess, w_prior=1.0):
        self.f, self.y, self.s, self.N, self.d0 = f, y, s, N, d_guess
        self.k0 = 2 * np.pi * f / C0
        self.w = 1.0 / s ** 2
        self.isig = 1.0 / s
        self.lib = TS.Library(f)
        lib = self.lib
        self.C = len(lib.names)
        self.kind = np.array([KIND_WATER if b == "water" else KIND_AIR if b == "air" else KIND_METAL if b == "metal"
                              else KIND_DRY for b in lib.base])
        st = ST.library_stats()
        self.mu = np.zeros((self.C, 2)); self.Li = np.zeros((self.C, 2, 2))
        self.er0 = np.ones(self.C); self.td0 = np.zeros(self.C); self.pp0 = np.zeros(self.C)
        for c, b in enumerate(lib.base):
            if b in st:
                mu, Cv = st[b]
                self.mu[c] = mu
                self.Li[c] = np.linalg.cholesky(np.linalg.inv(Cv))
                self.er0[c], self.td0[c], self.pp0[c] = lib.laws[c].er, lib.laws[c].td, lib.laws[c].pp
        self.solids = [int(c) for c in lib.is_solid_idx]
        self.AIR, self.METAL, self.WATER = lib.AIR, lib.METAL, lib.WATER
        segs = I.band_segments(f)
        self.extent = min(TS._data_extent(f, y, s, segs, d_guess), EXTENT_CAP)
        self.w_prior = w_prior
        self.n_eval = 0
        self.n_polish = 0
        self.log = []


class State:
    """cls[j] class index of layer j (AIR = gap); t, er, td per layer; back = AIR or METAL."""

    def __init__(self, cls, t, er, td, back, d, Tw=20.0, cost=np.inf, chi=np.inf):
        self.cls = np.asarray(cls, np.int64); self.t = np.asarray(t, float)
        self.er = np.asarray(er, float); self.td = np.asarray(td, float)
        self.back, self.d, self.Tw, self.cost, self.chi = int(back), float(d), float(Tw), cost, chi

    @property
    def nl(self):
        return len(self.cls)

    def describe(self, ctx):
        return ([(ctx.lib.names[c], round(float(1e3 * t), 2)) for c, t in zip(self.cls, self.t)], ctx.lib.names[self.back])

    def key(self):
        return (tuple(self.cls), self.back, tuple(np.round(self.t * 1e3 / 0.5).astype(int)))


# ---------------------------------------------------------------------------- tables for a state
def _table(ctx, S):
    """Refractive-index table: nominal classes (rows 0..C-1) + the state's fitted layers (rows
    C + j). Returns ntab, nref (Re n at F_NREF) and for every row its class."""
    F = ctx.f.size
    rows = np.empty((S.nl, F), complex)
    nref = np.empty(S.nl)
    f5 = np.array([TS.F_NREF])
    o5 = np.empty(1, complex)
    for j in range(S.nl):
        PL._nrow(ctx.kind[S.cls[j]], S.er[j], S.td[j], ctx.pp0[S.cls[j]], S.Tw, ctx.f, rows[j])
        PL._nrow(ctx.kind[S.cls[j]], S.er[j], S.td[j], ctx.pp0[S.cls[j]], S.Tw, f5, o5)
        nref[j] = max(o5[0].real, 1.0)
    ntab = np.ascontiguousarray(np.vstack([ctx.lib.ntab, rows]))
    return ntab, np.concatenate([ctx.lib.nref, nref]), np.concatenate([np.arange(ctx.C), S.cls])


def _taus(S, nref_ext, ctx):
    return S.d + np.concatenate([[0.0], np.cumsum(S.t * nref_ext[ctx.C + np.arange(S.nl)])])


# ---------------------------------------------------------------------------- candidate blocks
class Block:
    """P candidates with the same number of layers: rows (P, nl), tau (P, nl + 1), back (P,)."""

    def __init__(self, rows, tau, back):
        self.rows, self.tau, self.back = np.asarray(rows, np.int64), np.asarray(tau, float), np.asarray(back, np.int64)


def _eval(ctx, blocks, ntab, nref):
    out = []
    for b in blocks:
        P, nl = b.rows.shape
        if P == 0:
            out.append(np.zeros(0)); continue
        Tt = np.ascontiguousarray(np.diff(b.tau, axis=1) / nref[b.rows])
        chi = FB.chi_batch(np.ascontiguousarray(b.rows), Tt, np.full(P, nl), b.back, b.tau[:, 0].copy(), ntab,
                           ctx.k0, ctx.y, ctx.w)
        ctx.n_eval += P
        out.append(chi)
    return out


def _top_diverse(blocks, chis, k):
    """Best candidate of every distinct (class rows, backing) group, then the k best groups."""
    best = {}
    for bi, (b, chi) in enumerate(zip(blocks, chis)):
        if chi.size == 0:
            continue
        order = np.argsort(chi)[: 40 * k]
        for i in order:
            key = (tuple(b.rows[i]), int(b.back[i]))
            if key not in best or chi[i] < best[key][0]:
                best[key] = (float(chi[i]), bi, int(i))
    return sorted(best.values())[:k]


def _polish(ctx, S, cands, row_cls, iters=15, isig=None):
    """cands: list of (rows, tau, back) built on state S's table. Returns polished States."""
    P = len(cands)
    if P == 0:
        return []
    NP = 2 + 3 * FB.LMAX
    X = np.zeros((P, NP)); LO = np.zeros((P, NP)); HI = np.zeros((P, NP))
    kind = np.zeros((P, PL.LMAX), np.int64); cls = np.zeros((P, PL.LMAX), np.int64); pp = np.zeros((P, PL.LMAX))
    nl = np.zeros(P, np.int64); back = np.zeros(P, np.int64); has_w = np.zeros(P, np.bool_); npar = np.zeros(P, np.int64)
    for p, (rows, tau, bk, nref) in enumerate(cands):
        n = len(rows)
        nl[p] = n
        back[p] = KIND_METAL if bk == ctx.METAL else KIND_AIR
        X[p, 0] = tau[0]; LO[p, 0] = ctx.d0 - 4e-3; HI[p, 0] = ctx.d0 + 4e-3
        k = 1
        for j, r in enumerate(rows):
            c = int(row_cls[r])
            cls[p, j] = c; kind[p, j] = ctx.kind[c]; pp[p, j] = ctx.pp0[c]
            t = max((tau[j + 1] - tau[j]) / nref[r], 0.35e-3)
            X[p, k] = np.log(t); LO[p, k] = np.log(0.3e-3); HI[p, k] = np.log(0.6); k += 1
            if ctx.kind[c] == KIND_DRY:
                er, td = (ctx.er0[c], ctx.td0[c]) if r < ctx.C else (S.er[r - ctx.C], S.td[r - ctx.C])
                X[p, k] = np.log(er); LO[p, k] = 0.0; HI[p, k] = np.log(12.0)
                X[p, k + 1] = np.log10(max(td, 1.1e-5)); LO[p, k + 1] = -5.0; HI[p, k + 1] = np.log10(0.6)
                k += 2
            elif ctx.kind[c] == KIND_WATER:
                has_w[p] = True
        if has_w[p]:
            X[p, k] = S.Tw; LO[p, k] = 0.0; HI[p, k] = 40.0; k += 1
        npar[p] = k
    cost, chi = PL.polish_batch(X, LO, HI, npar, kind, cls, pp, nl, back, has_w, ctx.f, ctx.k0, ctx.y,
                                ctx.isig if isig is None else isig, ctx.mu, ctx.Li, ctx.w_prior, iters)
    ctx.n_polish += P
    out = []
    for p in range(P):
        n = nl[p]
        t = np.zeros(n); er = np.ones(n); td = np.zeros(n)
        k = 1
        for j in range(n):
            t[j] = np.exp(X[p, k]); k += 1
            if kind[p, j] == KIND_DRY:
                er[j], td[j] = np.exp(X[p, k]), 10 ** X[p, k + 1]; k += 2
        Tw = X[p, k] if has_w[p] else S.Tw
        out.append(State(cls[p, :n].copy(), t, er, td, ctx.METAL if back[p] == KIND_METAL else ctx.AIR, X[p, 0], Tw,
                         float(cost[p]), float(chi[p])))
    return out


def _grid(lo, hi, h):
    if hi <= lo:
        return np.zeros(0)
    n = max(1, int(np.floor((hi - lo) / h)) + 1)
    return lo + h * np.arange(n)


def _class_opts(ctx, S, j):
    """Table rows a layer may take in a block move: the gap stays a gap; a solid may become any
    solid class (nominal member) or stay what it is (fitted row)."""
    if S.cls[j] == ctx.AIR:
        return [ctx.C + j]
    nb = {int(S.cls[i]) for i in (j - 1, j + 1) if 0 <= i < S.nl}
    return [c for c in ctx.solids if c not in nb] + [ctx.C + j]


# ---------------------------------------------------------------------------- moves
def mv_pair(ctx, S, tau, i, budget=6e5):
    """Layers i and i+1 jointly: both classes, the interface between them and the back interface
    of layer i+1; (a) later interfaces fixed, (b) later interfaces shifted rigidly."""
    nl = S.nl
    cur = ctx.C + np.arange(nl)
    A, B = _class_opts(ctx, S, i), _class_opts(ctx, S, i + 1)
    pairs = [(a, b) for a in A for b in B
             if not (a < ctx.C and b < ctx.C and a == b)]
    blocks = []
    end = max(ctx.d0 + ctx.extent, tau[-1] + 0.03)
    for mode in ("fixed", "rigid"):
        last = i + 2 == nl
        if mode == "fixed" and last:
            continue
        hi = tau[i + 3] - DMIN if mode == "fixed" else end
        span = hi - tau[i]
        if span <= 2 * DMIN:
            continue
        npos = np.sqrt(2 * budget / max(len(pairs), 1))
        h = float(np.clip(span / npos, 2.5e-3, 10e-3))
        g1 = _grid(tau[i] + DMIN, hi - DMIN, h)
        P1, P2 = np.meshgrid(g1, _grid(tau[i] + 2 * DMIN, hi, h), indexing="ij")
        ok = P2 >= P1 + DMIN
        p1, p2 = P1[ok], P2[ok]
        if p1.size == 0:
            continue
        npair, npos_ = len(pairs), p1.size
        rows = np.tile(cur, (npair * npos_, 1))
        pr = np.array(pairs)
        rows[:, i] = np.repeat(pr[:, 0], npos_)
        rows[:, i + 1] = np.repeat(pr[:, 1], npos_)
        T = np.tile(tau, (npair * npos_, 1))
        T[:, i + 1] = np.tile(p1, npair)
        newb = np.tile(p2, npair)
        if mode == "rigid":
            T[:, i + 3:] += (newb - tau[i + 2])[:, None]
        T[:, i + 2] = newb
        blocks.append(Block(rows, T, np.full(rows.shape[0], S.back)))
        if last and S.cls[-1] != ctx.AIR:                 # the block reaches the back: try the other backing too
            blocks.append(Block(rows, T, np.full(rows.shape[0], ctx.METAL if S.back == ctx.AIR else ctx.AIR)))
    return blocks


def mv_single(ctx, S, tau, i):
    """Layer i alone: class x back-interface position, later interfaces fixed or shifted."""
    nl = S.nl
    cur = ctx.C + np.arange(nl)
    A = _class_opts(ctx, S, i)
    blocks = []
    end = max(ctx.d0 + ctx.extent, tau[-1] + 0.03)
    for mode in ("fixed", "rigid"):
        last = i + 1 == nl
        if mode == "fixed" and last:
            continue
        hi = tau[i + 2] - DMIN if mode == "fixed" else end
        g = _grid(tau[i] + DMIN, hi, 1.25e-3 if hi - tau[i] < 0.25 else 2.5e-3)
        if g.size == 0:
            continue
        rows = np.tile(cur, (len(A) * g.size, 1))
        rows[:, i] = np.repeat(A, g.size)
        T = np.tile(tau, (len(A) * g.size, 1))
        newb = np.tile(g, len(A))
        if mode == "rigid":
            T[:, i + 2:] += (newb - tau[i + 1])[:, None]
        T[:, i + 1] = newb
        blocks.append(Block(rows, T, np.full(rows.shape[0], S.back)))
        if last and S.cls[-1] != ctx.AIR:
            blocks.append(Block(rows, T, np.full(rows.shape[0], ctx.METAL if S.back == ctx.AIR else ctx.AIR)))
    return blocks


def mv_classes(ctx, S, tau):
    """Every class sequence at the present interface positions, both backings (metal only where
    the present structure allows it without changing the interface count)."""
    nl = S.nl
    opts = [_class_opts(ctx, S, j) for j in range(nl)]
    seqs = [s for s in itertools.product(*opts)
            if not any(a == b and a < ctx.C for a, b in zip(s[:-1], s[1:]))]
    rows = np.array(seqs, np.int64)
    T = np.tile(tau, (rows.shape[0], 1))
    out = [Block(rows, T, np.full(rows.shape[0], S.back))]
    other = ctx.METAL if S.back == ctx.AIR else ctx.AIR
    if not (other == ctx.AIR and S.cls[-1] == ctx.AIR):
        out.append(Block(rows, T, np.full(rows.shape[0], other)))
    return out


def mv_gaps(ctx, S, tau):
    """Insert a gap at every slot that has none (three ways to make room), remove every gap
    (three ways to close it), and toggle the backing (with / without a gap before metal)."""
    nl = S.nl
    cur = list(ctx.C + np.arange(nl))
    blocks = []
    AIRROW = ctx.AIR

    def add(rows, T, back):
        blocks.append(Block(np.array(rows, np.int64).reshape(len(T), -1), np.array(T), np.full(len(T), back)))
    for j in range(nl):                      # slot behind layer j
        solid_j = S.cls[j] != ctx.AIR
        nxt_solid = j + 1 < nl and S.cls[j + 1] != ctx.AIR
        at_metal = j + 1 == nl and S.back == ctx.METAL
        if solid_j and (nxt_solid or at_metal):
            rows = cur[: j + 1] + [AIRROW] + cur[j + 1:]
            Ts = []
            for g in GAPS:
                # rigid: everything behind moves back by g
                Ts.append(np.concatenate([tau[: j + 2], [tau[j + 1] + g], tau[j + 2:] + g]))
                # carve from the layer in front
                if tau[j + 1] - g > tau[j] + DMIN:
                    Ts.append(np.concatenate([tau[: j + 1], [tau[j + 1] - g], tau[j + 1:]]))
                # carve from the layer behind
                if j + 2 <= nl and j + 1 < nl and tau[j + 1] + g < tau[j + 2] - DMIN:
                    Ts.append(np.concatenate([tau[: j + 2], [tau[j + 1] + g], tau[j + 2:]]))
            add([rows] * len(Ts), Ts, S.back)
    for j in range(1, nl):                   # remove gap layer j
        if S.cls[j] != ctx.AIR:
            continue
        rows = cur[:j] + cur[j + 1:]
        g = tau[j + 1] - tau[j]
        Ts = [np.concatenate([tau[: j + 1], tau[j + 2:] - g])]                 # rigid close
        Ts.append(np.concatenate([tau[:j], tau[j + 1:]]))                      # front layer grows
        if j + 1 < nl:
            Ts.append(np.concatenate([tau[: j + 1], tau[j + 2:]]))             # back layer grows
        bk = S.back
        if j + 1 == nl and S.back == ctx.METAL:
            add([rows] * len(Ts), Ts, ctx.METAL)
            add([rows] * 1, [np.concatenate([tau[: j + 1]])], ctx.AIR)        # gap + metal -> air backing
        else:
            add([rows] * len(Ts), Ts, bk)
    # backing toggles
    if S.back == ctx.AIR:
        add([cur], [tau], ctx.METAL)
        Ts = [np.concatenate([tau, [tau[-1] + g]]) for g in GAPS]
        add([cur + [AIRROW]] * len(Ts), Ts, ctx.METAL)
        Ts = [np.concatenate([tau[:-1], [tau[-1] - g, tau[-1]]]) for g in GAPS if tau[-1] - g > tau[-2] + DMIN]
        if Ts:
            add([cur + [AIRROW]] * len(Ts), Ts, ctx.METAL)
    else:
        if S.cls[-1] != ctx.AIR:
            add([cur], [tau], ctx.AIR)
    return blocks


def mv_relocate(ctx, S, tau):
    """Take one solid layer out (its optical thickness goes to the layer in front, or behind, or
    the wall closes up) and put a new solid layer of any class and thickness somewhere else:
    behind the last layer or between two layers. N stays fixed. This repairs the typical wrong
    optimum "a sliver beside a strong interface, and a weak deep layer missing"."""
    nl = S.nl
    cur = list(ctx.C + np.arange(nl))
    solids_j = [j for j in range(nl) if S.cls[j] != ctx.AIR]
    if len(solids_j) < 2:
        return []
    end = ctx.d0 + ctx.extent
    blocks = []
    for j in solids_j:
        g = tau[j + 1] - tau[j]
        variants = [np.concatenate([tau[: j + 1], tau[j + 2:] - g])]            # close up
        if j > 0:
            variants.append(np.concatenate([tau[:j], tau[j + 1:]]))            # front neighbour grows
        if j + 1 < nl:
            variants.append(np.concatenate([tau[: j + 1], tau[j + 2:]]))       # back neighbour grows
        rows0 = cur[:j] + cur[j + 1:]
        if any(S.cls[r - ctx.C] == ctx.AIR and S.cls[q - ctx.C] == ctx.AIR for r, q in zip(rows0[:-1], rows0[1:])):
            continue                                                            # would leave two gaps in a row
        if S.cls[rows0[0] - ctx.C] == ctx.AIR:
            continue
        for T0 in variants:
            n0 = len(rows0)
            for slot in range(1, n0 + 1):                                       # new layer becomes layer `slot`
                if slot < n0 and S.cls[rows0[slot] - ctx.C] == ctx.AIR and S.cls[rows0[slot - 1] - ctx.C] == ctx.AIR:
                    continue
                span = max(end - T0[-1], 0.30) if slot == n0 else 0.08    # deep layers may lie beyond the UWB extent
                th = np.arange(5e-3, span + 1e-9, 5e-3)
                if th.size == 0:
                    continue
                R, TT = [], []
                for c in ctx.solids:
                    for x in th:
                        R.append(rows0[:slot] + [c] + rows0[slot:])
                        TT.append(np.concatenate([T0[: slot + 1], T0[slot:] + x]))
                blocks.append(Block(np.array(R, np.int64), np.array(TT), np.full(len(R), S.back)))
    return blocks


def mv_positions(ctx, S, tau, budget=1.5e6):
    """All interface positions at once for the present classes (thin walls: few resolution cells)."""
    nl = S.nl
    cur = ctx.C + np.arange(nl)
    end = max(ctx.d0 + ctx.extent, tau[-1] + 0.03)
    span = end - tau[0]
    h = 2.5e-3
    while True:
        m = int(span / h)
        from math import comb
        if comb(m, nl) <= budget or h > 40e-3:
            break
        h *= 1.25
    g = tau[0] + DMIN + h * np.arange(m)
    comb_idx = np.array(list(itertools.combinations(range(m), nl)), np.int64)
    if comb_idx.size == 0:
        return []
    T = np.concatenate([np.full((comb_idx.shape[0], 1), tau[0]), g[comb_idx]], axis=1)
    rows = np.tile(cur, (T.shape[0], 1))
    return [Block(rows, T, np.full(T.shape[0], S.back))]


# ---------------------------------------------------------------------------- the search
def _apply(ctx, S, name, blocks, k, trace):
    ntab, nref, row_cls = S._tab
    chis = _eval(ctx, blocks, ntab, nref)
    top = _top_diverse(blocks, chis, k)
    cands = [(blocks[bi].rows[i], blocks[bi].tau[i], int(blocks[bi].back[i]), nref) for _, bi, i in top]
    pol = _polish(ctx, S, cands, row_cls)
    pol = [p for p in pol if np.isfinite(p.cost)]
    if not pol:
        return S, False
    best = min(pol, key=lambda p: p.cost)
    n_c = sum(c.size for c in chis)
    if best.cost < S.cost - max(2.0, 1e-4 * S.cost):
        ctx.log.append(dict(move=name, n=n_c, cost0=S.cost, cost1=best.cost, chi1=best.chi, wall=best.describe(ctx)))
        if trace:
            trace(name, n_c, S, best)
        return best, True
    return S, False


def _prep(ctx, S):
    S._tab = _table(ctx, S)
    S._tau = _taus(S, S._tab[1], ctx)
    return S


def descend(ctx, S, k=8, max_rounds=12, trace=None, t_budget=None, light=False):
    """Large-neighbourhood descent from state S until no block move improves the MAP cost."""
    t0 = time.time()
    S = _prep(ctx, S)
    for rnd in range(max_rounds):
        improved = False
        moves = []
        nl = S.nl
        if nl <= 4 and not light:
            moves.append(("positions", lambda S: mv_positions(ctx, S, S._tau)))
        if not light:
            moves += [(f"pair{i}", (lambda S, i=i: mv_pair(ctx, S, S._tau, i) if i + 1 < S.nl else [])) for i in range(nl - 1)]
        moves += [(f"single{i}", (lambda S, i=i: mv_single(ctx, S, S._tau, i) if i < S.nl else [])) for i in range(nl)]
        moves.append(("classes", lambda S: mv_classes(ctx, S, S._tau)))
        moves.append(("gaps", lambda S: mv_gaps(ctx, S, S._tau)))
        moves.append(("relocate", lambda S: mv_relocate(ctx, S, S._tau)))
        for name, gen in moves:
            blocks = gen(S)
            if not blocks:
                continue
            S2, ok = _apply(ctx, S, name, blocks, k, trace)
            if ok:
                S = _prep(ctx, S2)
                improved = True
            if t_budget is not None and time.time() - t0 > t_budget:
                return S
        if not improved:
            break
    return S


def candidate_positions(ctx, fine=5e-3, coarse=20e-3, halo=15e-3):
    """Optical depths (behind the front face) where an interface may be: a fine grid wherever the
    widest band's range profile shows energy above the noise (+- halo), a coarse grid elsewhere
    (interfaces seen only by the low band, whose half wavelength is > 15 cm)."""
    f, y = ctx.f, ctx.y
    i0, i1 = max(I.band_segments(f), key=lambda ab: f[ab[1] - 1] - f[ab[0]])
    fb, yb = f[i0:i1], y[i0:i1]
    n = 8 * fb.size
    r = np.arange(n) * C0 / (2 * (fb[1] - fb[0]) * n)
    P = np.abs(np.fft.ifft(yb * np.hanning(fb.size), n))
    far = (r > ctx.d0 + 1.5) & (r < r.max() - 0.3)
    lvl = 4 * np.max(P[far])
    act = r[P > lvl] - ctx.d0
    g = np.arange(fine, ctx.extent + 1e-9, fine)
    keep = np.array([np.any(np.abs(act - x) < halo) for x in g]) if act.size else np.zeros(g.size, bool)
    gc = np.arange(coarse, ctx.extent + 1e-9, coarse)
    out = np.unique(np.round(np.concatenate([g[keep], gc]) / fine).astype(int)) * fine
    return out[out > DMIN]


def _patterns(ctx, n_gap):
    """Layer-kind patterns with N solids and n_gap gaps, with the backings each allows:
    a gap sits between two solids, or between the last solid and a metal backing."""
    N = ctx.N
    out = []
    for slots in itertools.combinations(range(N), n_gap):          # slot j = behind solid j
        pat = []
        for j in range(N):
            pat.append("s")
            if j in slots:
                pat.append("g")
        backs = [ctx.METAL] if (N - 1) in slots else [ctx.AIR, ctx.METAL]
        out.append((pat, backs))
    return out


def joint_start(ctx, n_gap=0, n_keep=48, budget=4e7, verbose=False):
    """Exhaustive enumeration of ALL class sequences x ALL interface-position combinations with
    nominal class members on the candidate positions (n_gap air gaps in every admissible slot
    pattern, both backings); the best member of every (sequence, backing) is kept and the n_keep
    best are polished."""
    from math import comb
    N = ctx.N
    L = N + n_gap
    lib = ctx.lib
    pats = _patterns(ctx, n_gap)
    n_seq = len(ctx.solids) * (len(ctx.solids) - 1) ** (N - 1) * sum(len(b) for _, b in pats)
    fine = 5e-3
    while True:
        g = candidate_positions(ctx, fine=fine)
        if comb(len(g), L) * n_seq <= budget or fine > 30e-3:
            break
        fine *= 1.25
    q = np.round(g / fine).astype(np.int64) - 1
    cidx = np.array(list(itertools.combinations(range(len(g)), L)), np.int64)
    if cidx.size == 0:
        return []
    combos = np.ascontiguousarray(q[cidx])
    best = []
    n0 = ctx.n_eval
    for pat, backs in pats:
        seqs = []
        for seq in itertools.product(ctx.solids, repeat=N):
            if any(a == b for a, b in zip(seq[:-1], seq[1:])):
                continue
            it = iter(seq)
            seqs.append([next(it) if k == "s" else ctx.AIR for k in pat])
        seqs = np.array(seqs, np.int64)
        bc, bi = FB.joint_enum(seqs, np.array(backs, np.int64), combos, fine, ctx.d0, lib.nref, lib.ntab,
                               ctx.k0, ctx.y, ctx.w, 2)
        ctx.n_eval += seqs.shape[0] * len(backs) * combos.shape[0]
        for si in range(seqs.shape[0]):
            for b, back in enumerate(backs):
                for j in range(2):
                    if np.isfinite(bc[si, b, j]):
                        best.append((float(bc[si, b, j]), tuple(seqs[si]), back, int(bi[si, b, j])))
    best.sort(key=lambda t: t[0])
    S0 = State([ctx.solids[0]] * L, np.ones(L), np.ones(L), np.ones(L), ctx.AIR, ctx.d0)
    cands = []
    for _, seq, back, i in best[:n_keep]:
        tau = np.concatenate([[ctx.d0], ctx.d0 + g[cidx[i]]])
        cands.append((np.array(seq, np.int64), tau, back, lib.nref))
    pol = _polish(ctx, S0, cands, np.arange(ctx.C))
    pol = sorted([p for p in pol if np.isfinite(p.cost)], key=lambda p: p.cost)
    if verbose and pol:
        print(f"  joint start ({n_gap} gaps): {len(g)} positions (step {1e3 * fine:.1f} mm), {combos.shape[0]} combos, "
              f"{ctx.n_eval - n0:.2e} evals; best polished cost {pol[0].cost:.1f} chi2/n {pol[0].chi / (2 * ctx.f.size):.2f} "
              f"{pol[0].describe(ctx)}", flush=True)
    return pol


def polish_state(ctx, S, iters=25, isig=None):
    tab = _table(ctx, S)
    tau = _taus(S, tab[1], ctx)
    return _polish(ctx, S, [(ctx.C + np.arange(S.nl), tau, S.back, tab[1])], tab[2], iters, isig)[0]


GOOD_CHI2N = 1.5        # search-band fit this close to the noise: stop exploring further starts


def bnb_start(ctx, n_keep=24, rho=1.5, h=5e-3, extra_extent=0.0, verbose=False):
    """E052: branch-and-bound over all class-discrete grid walls (any gap pattern, both backings)
    under the model-discrepancy weights, then: best leaf of every distinct (class sequence, backing),
    cluster members expanded, compiled polish first under the same smooth weights and then under
    the measurement noise. Returns polished States (best first) and the B&B statistics."""
    import bnb as BB
    fb, yb, sb, segs = TS.search_subset(ctx.f, ctx.y, ctx.s, per_band=(24, 64))
    libb = TS.Library(fb)
    reps, members = BB.class_clusters(ctx.lib)
    s_eff = BB.model_sigma(fb, yb, sb, segs, h)
    leaves, info = BB.solve_grid(fb, yb, s_eff, segs, libb, ctx.N, ctx.d0, ctx.extent + extra_extent, h=h, rho=rho,
                                 reps=reps, verbose=verbose)
    ctx.n_eval += info["nodes"] + info["nodes_search"]
    info["fallback"] = False
    if not leaves:                             # node budget spent without a single wall: exhaustive no-gap start
        info["fallback"] = True
        return joint_start(ctx, n_gap=0, verbose=verbose), info
    seen, picked = set(), []
    for cost, cls, q, back in leaves:
        if (cls, back) in seen:
            continue
        seen.add((cls, back))
        picked.append((cost, cls, q, back))
        if len(picked) >= n_keep:
            break
    lib = ctx.lib
    cands = []
    for cost, cls, q, back in picked:
        tau = ctx.d0 + h * (np.concatenate([[-1], np.array(q)]) + 1)
        opts = [members.get(c, [c]) for c in cls]
        for combo in itertools.islice(itertools.product(*opts), 8):
            if any(a == b for a, b in zip(combo[:-1], combo[1:])):
                continue
            cands.append((np.array(combo, np.int64), tau, back, lib.nref))
    S0 = State([ctx.solids[0]], np.ones(1), np.ones(1), np.ones(1), ctx.AIR, ctx.d0)
    segs_full = I.band_segments(ctx.f)
    s_eff_full = BB.model_sigma(ctx.f, ctx.y, ctx.s, segs_full, h)
    pol = _polish(ctx, S0, cands, np.arange(ctx.C), iters=20, isig=1.0 / s_eff_full)      # smooth cost first
    out = []
    for P in pol:
        if np.isfinite(P.cost):
            out.append(polish_state(ctx, P, iters=30))                                       # then the noise
    out = sorted([p for p in out if np.isfinite(p.cost)], key=lambda p: p.cost)
    info["n_sequences"] = len(picked)
    info["n_polished"] = len(out)
    if verbose and out:
        for P in out[:3]:
            print(f"   polished: cost {P.cost:.1f} chi2/n {P.chi / (2 * ctx.f.size):.2f} {P.describe(ctx)}", flush=True)
    return out, info


def search(f, y, s, N, d_guess, k=8, n_desc=3, trace=None, verbose=False, t_budget=None, extra_extent=0.0,
           rho=1.5, n_keep=24):
    """Returns the list of distinct local optima (States, best first) and the context."""
    ctx = Ctx(f, y, s, N, d_guess)
    t0 = time.time()
    n = 2 * f.size
    starts, info = bnb_start(ctx, verbose=verbose, extra_extent=extra_extent, rho=rho, n_keep=n_keep)
    ctx.bnb = info
    found = {}
    n_run = 0
    for S0 in starts[:n_desc]:
        S = descend(ctx, S0, k=k, trace=trace, t_budget=t_budget, light=True)
        n_run += 1
        key = S.key()
        if key not in found or S.cost < found[key].cost:
            found[key] = S
        if verbose:
            print(f"  start {n_run}: cost {S0.cost:.1f} -> {S.cost:.1f} chi2/n {S.chi / n:.2f} {S.describe(ctx)} "
                  f"({time.time() - t0:.1f}s, {ctx.n_eval:.2e} evals)", flush=True)
    for S0 in starts[n_desc:]:
        key = S0.key()
        if key not in found:
            found[key] = S0
    out = sorted(found.values(), key=lambda S: S.cost)
    ctx.wall_s = time.time() - t0
    return out, ctx

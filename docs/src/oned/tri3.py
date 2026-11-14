"""
E070: the three-layer, five-class wall problem (scope agreed with the owner, 2026-11-13).

  Given calibrated complex reflection data at normal incidence (3.1-10.6 GHz and 60-64 GHz),
  recover the ordered material classes and physical thicknesses of three homogeneous planar layers,
  air-backed, no internal gaps, plus the radar-to-wall distance, with uncertainty.

Classes (all models explicit, section MATERIALS):
  gypsum      ITU-R P.2040 plasterboard law (eps' x U(0.92,1.08), conductivity x U(0.6,1.6), Jonscher)
  wood_board  ONE BROAD CLASS spanning ITU chipboard (OSB / particle board) to ITU plywood: a mixing
              parameter m ~ U(0,1) interpolates eps', log conductivity at 10 GHz and the conductivity
              exponent between the two ITU entries, then the same per-sample variability as ITU.
              (Owner decision pending; option (c) of 2026-11-13. Note: the ITU plywood entry,
              constant 0.33 S/m, is very lossy - ~35 dB two-way per 50 mm - and should be checked.)
  fibrous     mineral wool / fibreglass batt (repo 'batt': eps' 1 + 0.12 x U(0.6,1.4), constant Q)
  foam        rigid XPS/EPS board (repo 'foam': eps' 1 + 0.05 x U(0.6,1.4), constant Q)
  tile        glazed ceramic tile (eps' 6 x U(0.95,1.05), tan d 0.03 x U(0.6,1.6), constant Q)
Every layer draws its own material sample (layers 1 and 3 of the same class are independent).

Scopes: SEQ_ALL = the 80 ordered sequences without adjacent repeats; SEQ_IN = the 30 whose outer
layers are boards (gypsum, wood_board, tile); SEQ_OUT = the other 50 (insulation at an outer face).
Thickness: 'real' = per-class realistic ranges (T_REAL), 'stress' = 5-50 mm for every class.

Solver (reuses the verified kernels: fastbatch.chi_batch grid forward, polish.polish_batch compiled LM,
search.lm exact adjoint LM + search.cycle_hop, wall_inverse.fisher CRB):
  A  standoff: earliest range-profile peak (stack.front_guess), then a one-sided standoff grid
     d_est - 36 mm ... d_est + 6 mm (merged front echoes bias the peak late)
  B  every allowed sequence: exhaustive grid over all three thicknesses (class bounds), nominal class
     members, model-discrepancy weights (bnb.model_sigma), sub-12-GHz data; K distinct minima kept
  C  compiled polish (thicknesses, standoff, eps' and loss of every layer under ITS class prior),
     first under the smooth weights, then the noise, low band(s) only
  D  compiled polish on all bands (adds 60-64 GHz)
  E1 exact adjoint LM on all bands (dispersion shape released), class-fixed prior, bounds kept, from
     the KE best distinct polished minima of EVERY sequence (the polish fixes the dispersion shape and
     ranks poorly: 1702 vs 1010 on one wall, so no sequence is dropped on polished cost)
  E2 cycle hops on the best fit of every sequence within HOP_WINDOW of the best E1 cost
  F  decision: chi2 alarm, material-out-of-class alarm (MAT_ALARM); ABSTAIN on labels if another sequence is within AMBIG of the best MAP cost;
     ABSTAIN on thickness if a distinct minimum of the same sequence is within AMBIG or 3 CRB > tol.
reference mode: KE = 4, HOP_WINDOW = 500 (calibrated with hop_all, calib_hopall.jsonl); fast: KE = 2, HOP_WINDOW = 150.
Both modes fit every allowed sequence; the confusion map is the per-sequence cost table (per_seq).
"""
from __future__ import annotations
import itertools
import time
import numpy as np
import wall_toy as W
import stack as ST
import search as SR
import wall_inverse as I
import matlib as ML
import polish as PL
import fastbatch as FB
import bnb as BB
import treesolve as TS

C0 = W.C0
CLASSES = ("gypsum", "wood_board", "fibrous", "foam", "tile")
BOARDS = ("gypsum", "wood_board", "tile")
SEQ_ALL = [s for s in itertools.product(CLASSES, repeat=3) if s[0] != s[1] and s[1] != s[2]]
SEQ_IN = [s for s in SEQ_ALL if s[0] in BOARDS and s[2] in BOARDS]
SEQ_OUT = [s for s in SEQ_ALL if s not in SEQ_IN]
assert len(SEQ_ALL) == 80 and len(SEQ_IN) == 30

T_REAL = {"gypsum": (9.5e-3, 16e-3), "wood_board": (6e-3, 19e-3), "fibrous": (30e-3, 90e-3),
          "foam": (10e-3, 50e-3), "tile": (6e-3, 12e-3)}
T_STRESS = (5e-3, 50e-3)
BANDS2 = ((3.1e9, 10.6e9), (60e9, 64e9))
SNR2 = (40.0, 40.0)
BANDS3 = ((0.4e9, 0.96e9),) + BANDS2
SNR3 = (30.0, 40.0, 40.0)
NPTS = 256
D_BOX = (0.20, 0.40)          # stated standoff interval (m)
TOL = (0.5e-3, 0.02)          # proposed thickness criterion max(0.5 mm, 2 %)
AMBIG = 10.0                  # MAP-cost window for 'another answer is as good' (pre-registered)


# ============================================================================ MATERIALS
_ORIG_LAW = ST.material_law
_MAP = {"gypsum": "plasterboard", "fibrous": "batt", "foam": "foam", "tile": "tile"}


import os
WOOD = os.environ.get("TRI3_WOOD", "broad")      # broad (default) | chipboard | plywood  (owner decision pending)


def _wood_board(rng):
    a0, c0, d0 = ST.ITU["chipboard"][:3]
    a1, c1, d1 = ST.ITU["plywood"][:3]
    m = rng.uniform(0, 1)                         # drawn in every mode: same random stream for the rest of the wall
    m = {"broad": m, "chipboard": 0.0, "plywood": 1.0}[WOOD]
    a = np.exp((1 - m) * np.log(a0) + m * np.log(a1))
    s10 = np.exp((1 - m) * np.log(c0 * 10 ** d0) + m * np.log(c1 * 10 ** d1))
    d = (1 - m) * d0 + m * d1
    er = a * rng.uniform(0.92, 1.08)
    sig10 = s10 * rng.uniform(0.6, 1.6)
    tan10 = sig10 / (2 * np.pi * ST.F_REF * ST.EPS0 * er)
    acq = 2 / np.pi * np.arctan(tan10)
    ap = float(np.clip(max(1 - d, acq), acq, 0.98))
    pp = 1 - tan10 / np.tan(np.pi * ap / 2)

    def eps(f, er=er, t=tan10, ap=ap, pp=pp):
        g = (np.asarray(f, float) / ST.F_REF) ** (-ap)
        return er * (pp + (1 - pp - 1j * t) * g)
    return eps, dict(eps10=er, tan10=tan10, alpha=ap, p=pp, mix=m)


def material_law(name, rng, T=None):
    if name == "wood_board":
        return _wood_board(rng)
    if name in _MAP:
        return _ORIG_LAW(_MAP[name], rng, T)
    return _ORIG_LAW(name, rng, T)


def install():
    """Restrict the repo's global material world to the five classes (library, priors, labels)."""
    if getattr(ST, "_TRI3", False):
        return
    ST.material_law = material_law
    ST.ALL_MATERIALS = list(CLASSES) + ["air", "water"]
    ST._LIB = None
    ST._TRI3 = True


_SAVED = None


def uninstall():
    """Restore the repo's full material world (tests run other modules in the same process)."""
    if not getattr(ST, "_TRI3", False):
        return
    ST.material_law = _ORIG_LAW
    ST.ALL_MATERIALS = list(ST.ITU) + list(ST.OTHER) + list(ST.SPECIAL)
    ST._LIB = None
    ST._TRI3 = False


def bounds(name, thick="real"):
    return T_REAL[name] if thick == "real" else T_STRESS


# ============================================================================ world
def make3(seed, scope="in", thick="real", bands=BANDS2, snr=SNR2, seq=None, noise=True):
    """Truth generator (exact recursion stack.gamma_stack, independent of the solver's kernels)."""
    install()
    rng = np.random.default_rng(seed)
    pool = {"in": SEQ_IN, "out": SEQ_OUT, "all": SEQ_ALL}[scope]
    if seq is None:
        seq = pool[int(rng.integers(len(pool)))]
    layers, desc = [], []
    for nm in seq:
        lo, hi = bounds(nm, thick)
        t = float(rng.uniform(lo, hi))
        e, dsc = material_law(nm, rng)
        layers.append((t, e)); desc.append(dict(name=nm, t=t, **dsc))
    d = float(rng.uniform(D_BOX[0] + 0.03, D_BOX[1] - 0.03))
    fs, ys, ss, Gs = [], [], [], []
    for k, ((a, b), sn) in enumerate(zip(bands, snr)):
        f = np.linspace(a, b, NPTS)
        G = ST.gamma_stack(f, layers, "air") * np.exp(-2j * 2 * np.pi * f / C0 * d)
        sig = 10 ** (-sn / 20) / np.sqrt(2)
        nr = np.random.default_rng(seed + 7919 * (k + 1))
        y = G + (sig * (nr.standard_normal(f.size) + 1j * nr.standard_normal(f.size)) if noise else 0)
        fs.append(f); ys.append(y); ss.append(np.full(f.size, sig)); Gs.append(G)
    # two-way loss down to the last interface (reported, never used to discard a wall)
    loss = {}
    for fq in (0.7e9, 5e9, 62e9):
        k0 = 2 * np.pi * fq / C0
        loss[f"{fq / 1e9:g}GHz"] = float(sum(2 * k0 * abs(np.sqrt(e(np.array([fq]))[0] + 0j).imag) * t * 20 / np.log(10)
                                             for t, e in layers[:-1]))
    truth = dict(seq=list(seq), layers=desc, standoff=d, loss_db=loss, scope=scope, thick=thick)
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ss), truth


# ============================================================================ class-fixed prior
class ClassPrior(ML.LibPrior):
    """Library prior with the class of every layer FIXED (the candidate's sequence), not the nearest
    class: each of the 80 hypotheses is fitted inside its own classes."""

    def __init__(self, seq, weight=1.0):
        super().__init__(weight)
        d = {nm: (nm, mu, Lc) for nm, mu, Lc in self.cls}
        self.seq_cls = [d[s] for s in seq]
        self._k = 0

    def rows(self, model):
        self._idx = {L.name: j for j, L in enumerate(model.layers)}
        return super().rows(model)

    def _layer_rows(self, L):
        j = self._idx.get(L.name, 0) if hasattr(self, "_idx") else 0
        nm, mu, Lc = self.seq_cls[j]
        er, td = L.eps_r.value, max(L.tan_d.value, 1e-7)
        x = np.array([np.log(er), np.log10(td)])
        dx = np.array([[1 / er, 0.0], [0.0, 1 / (td * np.log(10))]])
        z = Lc.T @ (x - mu)
        return (z @ z, z, Lc.T @ dx, nm)

    def labels(self, model):
        return [s[0] for s in self.seq_cls]


# ============================================================================ solver context
class Ctx:
    def __init__(self, f, y, s, seqs, thick, d_box=D_BOX):
        install()
        self.f, self.y, self.s = f, y, s
        self.seqs, self.thick = seqs, thick
        self.n_all = 2 * f.size
        # low-band data (everything below 12 GHz), thinned
        idx = []
        for a, b in I.band_segments(f):
            if f[a] > 12e9:
                continue
            step = max(1, (b - a) // 96)          # UNIFORM thinning: band_segments detects bands by their
            idx += list(range(a, b, step))        # spacing (linspace+int split a band into 3-point pieces)
        idx = np.array(idx)
        self.fl, self.yl, self.sl = f[idx], y[idx], s[idx]
        self.segl = I.band_segments(self.fl)
        self.lib_l = TS.Library(self.fl)
        self.lib_f = TS.Library(f)
        lib = self.lib_l
        self.ci = {nm: lib.names.index(nm) for nm in CLASSES}
        st = ST.library_stats()
        C = len(lib.names)
        self.mu = np.zeros((C, 2)); self.Li = np.zeros((C, 2, 2)); self.pp0 = np.zeros(C)
        self.er0 = np.ones(C); self.td0 = np.zeros(C)
        for nm, c in self.ci.items():
            mu, Cv = st[nm]
            self.mu[c] = mu
            self.Li[c] = np.linalg.cholesky(np.linalg.inv(Cv))
            self.er0[c], self.td0[c], self.pp0[c] = lib.laws[c].er, lib.laws[c].td, lib.laws[c].pp
        # standoff from the WIDEST band only: front_guess builds a range profile assuming a uniform
        # frequency grid, which sub-GHz + UWB concatenated is not (3-band ablation: errors up to 116 mm)
        a, b = max(I.band_segments(f), key=lambda ab: f[ab[1] - 1] - f[ab[0]] if f[ab[0]] < 20e9 else -1)
        d = ST.front_guess(f[a:b], y[a:b], lo=d_box[0], hi=d_box[1])
        self.d_est = float(np.clip(d, d_box[0], d_box[1]))
        self.d_box = d_box
        self.n_grid = 0
        self.n_polish = 0
        self.n_exact = 0


# ============================================================================ B: exhaustive grid
H_OPT = 2.0e-3        # optical grid step (m)
# standoff grid: the earliest range-profile peak can only be LATE (a front echo merged with a stronger
# interface behind it within one resolution cell, ~20 mm; wall 2049: +12.6 mm), by at most the front
# layer's optical thickness (<= ~31 mm for these classes): grid d_est - D_BACK ... d_est + D_FWD
D_BACK = 36.0e-3
D_FWD = 6.0e-3
D_STEP = 1.5e-3
D_LOCAL = 8.0e-3      # polish box around each candidate's own standoff
NMS_OPT = 6.0e-3      # two grid minima are the same basin if every optical depth is within this


def grid_sequence(ctx, seq, K):
    """All thickness triples (class bounds, optical step H_OPT) x standoff grid; nominal members.
    Returns up to K distinct minima: list of (cost, t (3,), d)."""
    lib = ctx.lib_l
    rows = np.array([ctx.ci[s] for s in seq], np.int64)
    grids = []
    for s, c in zip(seq, rows):
        lo, hi = bounds(s, ctx.thick)
        n = lib.nref[c]
        m = max(2, int(np.ceil((hi - lo) * n / H_OPT)) + 1)
        grids.append(np.linspace(lo, hi, m))
    # solvers that admit insulation at the FRONT: a weak foam / fibrous front echo can be missed
    # altogether, so the estimate may be late by that layer's whole optical thickness
    front = {s[0] for s in ctx.seqs}
    back = D_BACK if front <= set(BOARDS) else max(bounds(c, ctx.thick)[1] * 1.08 for c in front) + 6e-3
    # ... and with a weak insulation front the 'earliest peak' can be a range SIDELOBE of the strong
    # second echo, i.e. EARLY (map wall 1004: -14 mm): one main-lobe width forward as well
    fwd = D_FWD if front <= set(BOARDS) else 40e-3
    dg = ctx.d_est + np.arange(-back, fwd + 1e-9, D_STEP)
    dg = dg[(dg >= ctx.d_box[0]) & (dg <= ctx.d_box[1])]
    T = np.array(list(itertools.product(*grids)))                  # (M, 3)
    M = T.shape[0]
    P = M * dg.size
    Tt = np.ascontiguousarray(np.tile(T, (dg.size, 1)))
    D = np.repeat(dg, M)
    cls = np.ascontiguousarray(np.tile(rows, (P, 1)))
    w = 1.0 / ctx.s_eff ** 2
    k0 = 2 * np.pi * ctx.fl / C0
    chi = FB.chi_batch(cls, Tt, np.full(P, 3, np.int64), np.full(P, lib.AIR, np.int64), D, lib.ntab, k0, ctx.yl, w)
    ctx.n_grid += P
    order = np.argsort(chi)
    nref = lib.nref[rows]
    out = []
    for i in order[: 400 * K]:
        tau = np.cumsum(Tt[i] * nref) + D[i]
        if any(np.all(np.abs(tau - t2) < NMS_OPT) and abs(D[i] - d2) < NMS_OPT for _, _, _, t2, d2 in out):
            continue
        out.append((float(chi[i]), Tt[i].copy(), float(D[i]), tau, float(D[i])))
        if len(out) >= K:
            break
    return [(c, t, d) for c, t, d, _, _ in out]


# ============================================================================ C/D: compiled polish
def _pack(ctx, cands, lib, pp_override=None):
    P = len(cands)
    NP = 1 + 3 * 3
    X = np.zeros((P, NP)); LO = np.zeros((P, NP)); HI = np.zeros((P, NP))
    kind = np.zeros((P, PL.LMAX), np.int64); cls = np.zeros((P, PL.LMAX), np.int64); pp = np.zeros((P, PL.LMAX))
    for p, (seq, t, d, er, td, ppv) in enumerate(cands):
        X[p, 0] = d; LO[p, 0] = max(ctx.d_box[0], d - D_LOCAL); HI[p, 0] = min(ctx.d_box[1], d + D_LOCAL)
        X[p, 0] = np.clip(X[p, 0], LO[p, 0], HI[p, 0])
        k = 1
        for j, s in enumerate(seq):
            c = ctx.ci[s]
            lo, hi = bounds(s, ctx.thick)
            cls[p, j] = c; kind[p, j] = 0; pp[p, j] = ppv[j]
            X[p, k] = np.log(np.clip(t[j], lo, hi)); LO[p, k] = np.log(lo); HI[p, k] = np.log(hi); k += 1
            X[p, k] = np.log(er[j]); LO[p, k] = 0.0; HI[p, k] = np.log(12.0)
            X[p, k + 1] = np.log10(max(td[j], 1.1e-5)); LO[p, k + 1] = -5.0; HI[p, k + 1] = np.log10(0.6)
            k += 2
    npar = np.full(P, NP, np.int64)
    return X, LO, HI, npar, kind, cls, pp


def polish(ctx, cands, f, y, isig, iters):
    """cands: list of (seq, t, d, er, td, pp). Returns list of (cost, chi, seq, t, d, er, td, pp)."""
    if not cands:
        return []
    lib = ctx.lib_l
    X, LO, HI, npar, kind, cls, pp = _pack(ctx, cands, lib)
    P = len(cands)
    nl = np.full(P, 3, np.int64); back = np.full(P, 2, np.int64); has_w = np.zeros(P, np.bool_)
    k0 = 2 * np.pi * f / C0
    cost, chi = PL.polish_batch(X, LO, HI, npar, kind, cls, pp, nl, back, has_w, f, k0, y, isig,
                                ctx.mu, ctx.Li, 1.0, iters)
    ctx.n_polish += P
    out = []
    for p, (seq, *_r, ppv) in enumerate(cands):
        t = np.exp(X[p, [1, 4, 7]]); er = np.exp(X[p, [2, 5, 8]]); td = 10 ** X[p, [3, 6, 9]]
        out.append((float(cost[p]), float(chi[p]), seq, t, float(X[p, 0]), er, td, ppv))
    return out


def _pp_variants(ctx, seq):
    """Dispersion shape is fixed in the compiled polish: wood_board spans chipboard-like (p ~ 0.75)
    to plywood-like (p ~ 0.99) shapes and gypsum p ~ 0.15-0.7, so those layers get two shapes each."""
    opts = []
    for s in seq:
        c = ctx.ci[s]
        if s == "wood_board":
            opts.append((0.75, 0.99))
        elif s == "gypsum":                     # ITU plasterboard law: p ~ 0.15-0.7 with the sample's loss
            opts.append((0.3, 0.6))
        else:
            opts.append((ctx.pp0[c],))
    return list(itertools.product(*opts))


# ============================================================================ E: exact refit
def exact_refit(ctx, cand):
    """E1: exact adjoint LM on all bands under the class-fixed prior, thickness bounds enforced;
    dispersion shape at the candidate's value first, then released. Returns dict."""
    _, _, seq, t, d, er, td, ppv = cand
    lay = [dict(t=float(t[j]), er=float(er[j]), td=float(td[j]), water=False, pinf=float(min(ppv[j], 0.998)))
           for j in range(3)]
    ST.SBOX = (max(ctx.d_box[0], d - 0.004), min(ctx.d_box[1], d + 0.004))
    mod = ST.build(lay, d, back="air")
    ST.SBOX = None
    for s, L in zip(seq, mod.layers):
        lo, hi = bounds(s, ctx.thick)
        L.t.lo, L.t.hi = lo, hi
        # start strictly inside the box: search.lm works in logit coordinates, where a start ON a
        # bound cannot move (wall 2052: gypsum stuck at 9.50 mm, truth 9.87 mm -> silent wrong label)
        m_ = 1e-3 * (hi - lo)          # 3 % moved a start by a whole cycle (wall 2034: fibrous 30.31 -> 31.8 mm)
        L.t.value = float(np.clip(L.t.value, lo + m_, hi - m_))
    pri = ClassPrior(seq)
    old = SR._PRIORS
    SR._PRIORS = (pri,)
    try:
        # dispersion shape free from the first iteration: with the shape fixed at the polish value a
        # thickness can run into its bound and stay there (wall 2052)
        for L in mod.layers:
            L.pinf.value, L.pinf.free = float(np.clip(L.pinf.value, 0.02, 0.998)), True
        mod, c = ST._fit(mod, ctx.f, ctx.y, ctx.s)
        # a thickness still on a bound: one restart from the middle of its box
        for j, (s, L) in enumerate(zip(seq, mod.layers)):
            lo, hi = bounds(s, ctx.thick)
            if min(L.t.value - lo, hi - L.t.value) < 0.01 * (hi - lo):
                m2 = mod.copy()
                m2.layers[j].t.value = 0.5 * (lo + hi)
                m2, c2 = ST._fit(m2, ctx.f, ctx.y, ctx.s)
                if c2 < c:
                    mod, c = m2, c2
    finally:
        SR._PRIORS = old
    ctx.n_exact += 1
    return _result(ctx, seq, mod, c, pri)


def exact_hop(ctx, r):
    """E2: cycle hops (single and paired thickness / standoff phase-period jumps) from an E1 fit."""
    pri = ClassPrior(r["seq"])
    old = SR._PRIORS
    SR._PRIORS = (pri,)
    try:
        mod, c = SR.cycle_hop(r["model"], ctx.f, ctx.y, ctx.s, SR.Opts(search_step=1, search_atol=0.0))
    finally:
        SR._PRIORS = old
    ctx.n_hop += 1
    out = _result(ctx, r["seq"], mod, c, pri)
    out["hopped"] = True
    out["hop_gain"] = r["cost"] - out["cost"]
    return out if out["cost"] <= r["cost"] else dict(r, hopped=True, hop_gain=0.0)


def _result(ctx, seq, mod, c, pri):
    pri.rows(mod)
    prior_pen = pri.penalty(mod)
    G, _ = I.forward(mod, ctx.f, jacobian=False)
    chi = float(np.sum(np.abs((G - ctx.y) / ctx.s) ** 2))
    tt = np.array([L.t.value for L in mod.layers])
    inb = all(bounds(s, ctx.thick)[0] - 1e-9 <= x <= bounds(s, ctx.thick)[1] + 1e-9 for s, x in zip(seq, tt))
    return dict(seq=tuple(seq), cost=float(c), chi2=chi, prior=float(prior_pen), t=tt, d=float(mod.standoff.value),
                er=[L.eps_r.value for L in mod.layers], td=[L.tan_d.value for L in mod.layers],
                pinf=[L.pinf.value for L in mod.layers], in_bounds=bool(inb), model=mod, hopped=False, hop_gain=0.0)


def crb(mod, f, s):
    try:
        F = I.fisher(mod, f, s)
        d = dict(zip(F["names"], F["crb_std"]))
        return [float(d.get(f"{L.name}.t", np.nan)) for L in mod.layers], float(d.get("standoff", np.nan))
    except Exception:
        return [np.nan] * 3, np.nan


# ============================================================================ the solver
K_GRID = 6            # distinct grid minima per sequence (all polished)
KE = {"reference": 4, "fast": 2}          # E1 exact fits per sequence (distinct polished minima)
HOP_WINDOW = {"reference": 500.0, "fast": 150.0}  # E2 hops within this of the best E1 cost (calibration: a 109 hop on wall 2 found the truth; nothing beyond 300 came within 333)


def _distinct(polished):
    """Distinct polished minima of one sequence (optical depths differ by > 1 mm somewhere)."""
    out = []
    for P in sorted(polished, key=lambda p: p[0]):
        t, d = P[3], P[4]
        tau = d + np.cumsum(t * np.sqrt(P[5]))
        if any(np.all(np.abs(tau - q) < 1e-3) for _, q in out):
            continue
        out.append((P, tau))
    return [o[0] for o in out]


def solve3(f, y, s, scope="in", thick="real", mode="reference", d_box=D_BOX, verbose=False, hop_all=False):
    t0 = time.time()
    seqs = {"in": SEQ_IN, "out": SEQ_OUT, "all": SEQ_ALL}[scope]
    ctx = Ctx(f, y, s, seqs, thick, d_box)
    ctx.n_hop = 0
    ctx.s_eff = BB.model_sigma(ctx.fl, ctx.yl, ctx.sl, ctx.segl, H_OPT, d0=ctx.d_est)
    tim = {}
    # B: grid
    starts = {seq: grid_sequence(ctx, seq, K_GRID) for seq in seqs}
    tim["grid"] = time.time() - t0
    # C: low-band polish (smooth weights, then the noise); D: all bands
    cands = []
    for seq, lst in starts.items():
        for _, t, d in lst:
            er = [ctx.er0[ctx.ci[x]] for x in seq]; td = [ctx.td0[ctx.ci[x]] for x in seq]
            for ppv in _pp_variants(ctx, seq):
                cands.append((seq, t, d, er, td, ppv))
    pol = polish(ctx, cands, ctx.fl, ctx.yl, 1.0 / ctx.s_eff, 25)
    pol = polish(ctx, [p[2:] for p in pol], ctx.fl, ctx.yl, 1.0 / ctx.sl, 30)
    pol = polish(ctx, [p[2:] for p in pol], f, y, 1.0 / s, 30)
    tim["polish"] = time.time() - t0 - tim["grid"]
    by_seq = {}
    for p in pol:
        if np.isfinite(p[0]):
            by_seq.setdefault(p[2], []).append(p)
    # E1: exact LM from the KE best distinct polished minima of EVERY sequence
    fits = []
    for sq, v in by_seq.items():
        for P in _distinct(v)[: KE[mode]]:
            try:
                r = exact_refit(ctx, P)
                r["polished_cost"] = P[0]
                fits.append(r)
            except Exception as e:
                if verbose:
                    print("refit failed", sq, repr(e))
    tim["exact"] = time.time() - t0 - tim["grid"] - tim["polish"]
    # E2: cycle hops on the best fit of every sequence near the best (all of them if hop_all)
    best = {}
    for r in fits:
        if r["seq"] not in best or r["cost"] < best[r["seq"]]["cost"]:
            best[r["seq"]] = r
    cb = min(r["cost"] for r in best.values())
    hop_targets = [r for r in best.values() if hop_all or r["cost"] <= cb + HOP_WINDOW[mode]]
    lead = min(best.values(), key=lambda r: r["cost"])
    hop_targets += [r for r in fits if r["seq"] == lead["seq"] and r is not lead][:1]   # leader's 2nd minimum
    hopped = []
    for r in hop_targets:
        try:
            hopped.append(exact_hop(ctx, r))
        except Exception as e:
            if verbose:
                print("hop failed", r["seq"], repr(e))
    ids = {id(r) for r in hop_targets}
    fits = [r for r in fits if id(r) not in ids] + hopped
    tim["hop"] = time.time() - t0 - tim["grid"] - tim["polish"] - tim["exact"]
    res = decide(ctx, fits)
    res.update(timing=tim, wall_s=time.time() - t0, n_grid=ctx.n_grid, n_polish=ctx.n_polish, n_exact=ctx.n_exact,
               n_hop=ctx.n_hop, d_est=ctx.d_est, mode=mode, scope=scope,
               hop_gains=[dict(seq="|".join(r["seq"]), gain=r["hop_gain"], dcost_before=r["cost"] + r["hop_gain"] - cb)
                          for r in hopped if r["hop_gain"] > 0.5])
    return res


MAT_ALARM = 16.0      # squared Mahalanobis distance of a fitted layer material to ITS class; true samples of
                      # every class stay below 6.2 (20000 draws each); out-of-scope walls fitted with an
                      # in-scope sequence put foam/batt into a 'wood_board' slot at 250-350 (walls 3007, 3031)


def mat_d2(seq, er, td):
    st = ST.library_stats()
    out = []
    for nm, e, t in zip(seq, er, td):
        mu, C = st[nm]
        x = np.array([np.log(e), np.log10(max(t, 1e-7))]) - mu
        out.append(float(x @ np.linalg.solve(C, x)))
    return out


def decide(ctx, fits):
    fits = sorted(fits, key=lambda r: r["cost"])
    b = fits[0]
    crb_t, crb_d = crb(b["model"], ctx.f, ctx.s)
    n = ctx.n_all
    md2 = mat_d2(b["seq"], b["er"], b["td"])
    chi_alarm = bool(b["chi2"] > ST.chi2_alarm(n))
    mat_alarm = bool(max(md2) > MAT_ALARM)
    alarm = chi_alarm or mat_alarm
    rivals = [r for r in fits[1:] if r["seq"] != b["seq"] and r["cost"] <= b["cost"] + AMBIG]
    tol = [max(TOL[0], TOL[1] * t) for t in b["t"]]
    same_alt = [r for r in fits[1:] if r["seq"] == b["seq"] and r["cost"] <= b["cost"] + AMBIG
                and any(abs(a - c) > tl for a, c, tl in zip(r["t"], b["t"], tol))]
    weak_t = [bool(not np.isfinite(e) or 3 * e > tl) for e, tl in zip(crb_t, tol)]
    status = "answer"
    if alarm:
        status = "alarm"
    elif rivals:
        status = "abstain_labels"
    elif same_alt or any(weak_t):
        status = "abstain_thickness"
    per_seq = {}
    for r in fits:
        k = "|".join(r["seq"])
        if k not in per_seq or r["cost"] < per_seq[k]["cost"]:
            per_seq[k] = dict(cost=r["cost"], chi2=r["chi2"], t_mm=[round(1e3 * x, 3) for x in r["t"]])
    return dict(seq=list(b["seq"]), t=[float(x) for x in b["t"]], d=b["d"], chi2=b["chi2"], chi2n=b["chi2"] / n,
                cost=b["cost"], alarm=alarm, status=status, chi_alarm=chi_alarm, mat_alarm=mat_alarm, mat_d2=md2, crb_t=crb_t, crb_d=crb_d, weak_t=weak_t,
                rivals=[dict(seq=list(r["seq"]), dcost=r["cost"] - b["cost"], t_mm=[round(1e3 * x, 3) for x in r["t"]])
                        for r in rivals],
                same_seq_alt=[dict(dcost=r["cost"] - b["cost"], t_mm=[round(1e3 * x, 3) for x in r["t"]]) for r in same_alt],
                in_bounds=b["in_bounds"], er=[float(x) for x in b["er"]], td=[float(x) for x in b["td"]],
                pinf=[float(x) for x in b["pinf"]],
                per_seq=per_seq)


def score(res, truth):
    tseq = truth["seq"]
    tt = [L["t"] for L in truth["layers"]]
    seq_ok = list(res["seq"]) == list(tseq)
    t_err = [a - b for a, b in zip(res["t"], tt)]
    t_ok = [abs(e) <= max(TOL[0], TOL[1] * t) for e, t in zip(t_err, tt)]
    joint = bool(seq_ok and all(t_ok))
    return dict(seq_ok=bool(seq_ok), layer_ok=[a == b for a, b in zip(res["seq"], tseq)], t_ok=t_ok,
                t_err_mm=[1e3 * e for e in t_err], joint=joint, d_err_mm=1e3 * (res["d"] - truth["standoff"]),
                silent=bool(res["status"] == "answer" and not joint))

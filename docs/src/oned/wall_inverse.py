"""
Inversion of a layered wall from calibrated radar data, using the exact-star DEC model.

Pieces (each usable on its own):
  WallModel          adjustable parameter spec: any number of layers, each with
                     thickness t, eps_r, tan_d, moisture theta; every parameter can be
                     free or fixed, with bounds. Plus the standoff distance.
  forward()          exact-star DEC solve (tridiagonal, LAPACK gtsv with pivoting) and the
                     full Jacobian dGamma/dp by the adjoint method (one extra right-hand side).
  calibrate()        metal-plate calibration: removes the radar response A(f).
  levenberg_marquardt()  bounded damped least squares, optional regularization rows.
  invert()           multi-start LM driver.
  fisher()           Fisher information, Cramer-Rao standard deviations, correlations,
                     singular values (identifiability).
  choose_lambda()    regularization weight by Morozov's discrepancy principle.

Model of one layer (CRIM moisture mixing on a lossy dielectric host):
  h = eps_r (1 - j tan_d),   eps = ((1 - theta) sqrt(h) + theta sqrt(eps_water(f)))^2
Exact stars on a cell of length l, k = k0 sqrt(eps):
  b = -j k / (w mu0 sin(k l))            (edge star  S = (1/jw) *mu^-1, exact)
  M =  j k tan(k l / 2) / (w mu0)         (half-cell star  jw *eps, exact)
Each cell contributes  b*[[1,-1],[-1,1]] + M*I  to the tridiagonal matrix.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import solve_banded
from wall_toy import C0, ETA0, eps_water_debye, _sqrt_branch

MU0 = ETA0 / C0   # 2019 SI: mu0 = eta0 / c0 (4*pi*1e-7 is off by ~1e-10)


# ============================================================================ parameters
@dataclass
class P:
    value: float
    lo: float
    hi: float
    free: bool = True
    tie: str | None = None    # name of another parameter this one always equals (e.g. same material)


@dataclass
class LayerSpec:
    name: str
    t: P
    eps_r: P
    tan_d: P
    theta: P = field(default_factory=lambda: P(0.0, 0.0, 0.3, False))
    pinf: P = field(default_factory=lambda: P(0.0, 0.0, 0.9, False))   # E021b Jonscher: eps_inf / eps_r


@dataclass
class WallModel:
    layers: list[LayerSpec]
    standoff: P
    eps_back: complex = 1.0 + 0j
    water_T: float = 20.0
    law: str = "const"          # host material law: "const" (E007) or "cq" causal constant-Q (E021)
    f_ref: float = 79e9         # reference frequency of the "cq" law (eps_r, tan_d are the values there)
    beam: tuple | None = None   # E023: Gaussian beam waist (m) per band segment; None = plane wave
    beam_K: int = 10            # E023: Gauss-Laguerre nodes of the k_rho integral
    beam_dref: float | None = 0.30   # E023: distance fixing the rotated contour (None = real nodes)
    min_cells: bool = False     # E027: fewest cells per layer that keep the stars away from their poles
    facing: "LayerSpec | None" = None   # E024: thin facing (paper) on both faces of the layers in facing_on
    facing_on: tuple = ()
    rough: "P | None" = None            # E024: rms roughness of the front face (graded transition layer)
    band_offset: "P | None" = None      # E024: phase-centre/standoff offset of the 2nd and later bands
    gap: "P | None" = None              # E030: air gap behind the layer gap_after (cavity depth kept as its t)
    gap_after: str = "insulation"

    # ---- flat views over parameters -------------------------------------------------
    def items(self):
        for i, L in enumerate(self.layers):
            for key in ("t", "eps_r", "tan_d", "theta", "pinf"):
                yield f"{L.name}.{key}", (i, key), getattr(L, key)
        if self.facing is not None:
            for key in ("t", "eps_r", "tan_d"):
                yield f"facing.{key}", (None, key), getattr(self.facing, key)
        if self.rough is not None:
            yield "rough", (None, "rough"), self.rough
        if self.band_offset is not None:
            yield "band_offset", (None, "band_offset"), self.band_offset
        if self.gap is not None:
            yield "gap", (None, "gap"), self.gap
        yield "standoff", (None, "standoff"), self.standoff

    def free_names(self):
        return [n for n, _, p in self.items() if p.free and p.tie is None]

    def get(self):
        return np.array([p.value for _, _, p in self.items() if p.free and p.tie is None])

    def set(self, v):
        for x, (_, _, p) in zip(v, [it for it in self.items() if it[2].free and it[2].tie is None]):
            p.value = float(x)
        self.sync_ties()

    def sync_ties(self):
        vals = {n: p for n, _, p in self.items()}
        for n, _, p in self.items():
            if p.tie is not None:
                p.value = vals[p.tie].value

    def owner(self, name):
        """The free parameter that controls `name` (itself, or the one it is tied to), or None."""
        vals = {n: p for n, _, p in self.items()}
        p = vals[name]
        src = p.tie if p.tie is not None else name
        return src if (vals[src].free and vals[src].tie is None) else None

    def bounds(self):
        fr = [p for _, _, p in self.items() if p.free and p.tie is None]
        return np.array([p.lo for p in fr]), np.array([p.hi for p in fr])

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def summary(self):
        return {n: p.value for n, _, p in self.items()}


# ============================================================================ materials
def layer_eps_and_derivs(L: LayerSpec, f, T, law="const", f_ref=79e9):
    """eps(f) and d eps / d(eps_r, tan_d, theta), all exact (analytic).
    Host law "const": h = eps_r (1 - j tan_d) at every f (not causal over wide bands).
    Host law "cq" (E021, Kjartansson constant-Q, exactly causal and passive):
        h(f) = eps_r (1 - j tan_d) (f / f_ref)^(-a),  a = (2/pi) arctan(tan_d),
    i.e. eps_r sqrt(1+tan_d^2) (j f/f_ref)^(-a) since tan(pi a / 2) = tan_d. Same two parameters,
    equal to "const" at f_ref; eps' falls as f^-a, the loss angle is constant."""
    f = np.asarray(f, float)
    if law == "jonscher":
        # E021b: eps = eps_r [ p + (1 - p - j t) g ],  g = (f/f_ref)^(-a'),  a' = (2/pi) atan(t/(1-p))
        # = eps_inf + C (j f/f_ref)^(-a') (Jonscher universal response): causal, passive, p = 0 is "cq"
        er, t, pp = L.eps_r.value, L.tan_d.value, L.pinf.value
        u = t / (1 - pp)
        a = 2 / np.pi * np.arctan(u)
        lg = -np.log(f / f_ref)
        g = np.exp(a * lg)
        core = (1 - pp - 1j * t) * g
        h = er * (pp + core)
        da_dt = 2 / np.pi / (1 + u**2) / (1 - pp)
        da_dp = 2 / np.pi / (1 + u**2) * t / (1 - pp) ** 2
        dh_der = h / er
        dh_dtd = er * (-1j * g + core * lg * da_dt)
        dh_dp = er * (1 - g + core * lg * da_dp)
        eps, d = _crim(L, f, T, h, dh_der, dh_dtd)
        d["pinf"] = d["eps_r"] / dh_der * dh_dp
        return eps, d
    if law == "cq":
        t = L.tan_d.value
        a = 2 / np.pi * np.arctan(t)
        g = (f / f_ref) ** (-a)
        h = L.eps_r.value * (1 - 1j * t) * g
        dh_der = h / L.eps_r.value
        dh_dtd = L.eps_r.value * (-1j) * g + h * (-np.log(f / f_ref)) * (2 / np.pi) / (1 + t**2)
    else:
        h = L.eps_r.value * (1 - 1j * L.tan_d.value) * np.ones_like(f)
        dh_der = (1 - 1j * L.tan_d.value) * np.ones_like(f)
        dh_dtd = (-1j * L.eps_r.value) * np.ones_like(f)
    return _crim(L, f, T, h, dh_der, dh_dtd)


def _crim(L, f, T, h, dh_der, dh_dtd):
    w = eps_water_debye(f, T)
    sh, sw = np.sqrt(h), np.sqrt(w)
    th = L.theta.value
    s = (1 - th) * sh + th * sw
    eps = s ** 2
    de_dh = s * (1 - th) / sh
    return eps, {
        "eps_r": de_dh * dh_der,
        "tan_d": de_dh * dh_dtd,
        "theta": 2 * s * (sw - sh),
        "pinf": 0 * eps,
    }


def n_cells_for(L: LayerSpec, fmax, T):
    """Cells so that |k| l <= pi/2 at the top frequency (keeps stars positive and
    far from sin(kl)=0). The answer is exact for ANY cell count; this is only conditioning."""
    eps, _ = layer_eps_and_derivs(L, np.array([fmax]), T)          # conditioning only
    k = 2 * np.pi * fmax / C0 * abs(np.sqrt(eps[0]))
    return max(1, int(np.ceil(k * L.t.value / (0.9 * np.pi))))


# ============================================================================ forward + adjoint
ROUGH_SUB = 4          # sublayers of the graded roughness layer
# taper thickness / rms height. A graded layer reflects ~ sinc(k_bar h) ~ 1 - (k_bar h)^2/6 with k_bar = n_bar k0
# the wavenumber INSIDE the taper; Kirchhoff/Ament: exp(-2 (k0 sigma)^2) ~ 1 - 2 (k0 sigma)^2. Equal for
# h = 2 sqrt(3) sigma / n_bar; n_bar = (1 + n_drywall)/2 ~ 1.33 (E024 check: fitted 1/0.74). sigma is an
# effective roughness for other front materials.
ROUGH_H = 2 * np.sqrt(3.0) / 1.33


def _effective(model: WallModel, f):
    """Sublayers actually discretized (E024), each as (length, eps (F,), {owner: d length},
    {owner: d eps (F,)}, ncell). Without facing/rough these are exactly model.layers (same cells).
      facing   : layer L in facing_on -> [facing t_p, core t - 2 t_p, facing t_p]; t keeps meaning
                 the total sheet thickness.
      rough    : graded layer of thickness h = 2 sqrt(3) sigma in front, ROUGH_SUB sublayers with
                 eps_i = 1 + w_i (eps_front - 1), w_i = (i + 1/2)/ROUGH_SUB. A linear taper reduces
                 the reflection by 1 - (k h)^2/6 + ..., the Kirchhoff coherent factor exp(-2 (k sigma)^2)
                 by 1 - 2 (k sigma)^2 + ...: equal at second order for h = 2 sqrt(3) sigma."""
    T = model.water_T
    own = model.owner
    subs = []

    known = {n for n, _, _ in model.items()}

    def dmap(L, d, prefix):
        out = {}
        for key in ("eps_r", "tan_d", "theta", "pinf"):
            if f"{prefix}.{key}" not in known:
                continue
            o = own(f"{prefix}.{key}")
            if o is not None:
                out[o] = out.get(o, 0) + d[key]
        return out

    fp = None
    if model.facing is not None:
        ep, dp = layer_eps_and_derivs(model.facing, f, T, model.law, model.f_ref)
        fp = (ep, dmap(model.facing, dp, "facing"))
    kmax = 2 * np.pi * f.max() / C0

    def ncell_for(length, eps_fmax):
        return max(1, int(np.ceil(kmax * abs(np.sqrt(eps_fmax)) * length / (0.9 * np.pi))))

    def ncell_min(length, eps):
        """E027: the stars are exact for ANY cell count; cells only keep b = -jk/(w mu sin kl) and
        M ~ tan(kl/2) away from their poles. Smallest n with |sin(k l/n)| >= 0.25 and
        |cos(k l/2n)| >= 0.25 at every frequency (k complex for lossy layers)."""
        kk = 2 * np.pi * f / C0 * np.sqrt(eps.astype(complex))
        for n in range(1, 400):
            x = kk * length / n
            if np.min(np.abs(np.sin(x))) >= 0.25 and np.min(np.abs(np.cos(x / 2))) >= 0.25:
                return n
        return ncell_for(length, eps[-1])

    for L in model.layers:
        e, d = layer_eps_and_derivs(L, f, T, model.law, model.f_ref)
        dm = dmap(L, d, L.name)
        ot = own(f"{L.name}.t")
        if fp is not None and L.name in model.facing_on:
            tp = model.facing.t.value
            otp = own("facing.t")
            core = L.t.value - 2 * tp
            dl_core = {}
            if ot is not None:
                dl_core[ot] = 1.0
            if otp is not None:
                dl_core[otp] = dl_core.get(otp, 0) - 2.0
            dl_face = {otp: 1.0} if otp is not None else {}
            ef = layer_eps_and_derivs(model.facing, np.array([f.max()]), T, model.law, model.f_ref)[0][0]
            ec = layer_eps_and_derivs(L, np.array([f.max()]), T, model.law, model.f_ref)[0][0]
            nf = ncell_min(tp, fp[0]) if model.min_cells else ncell_for(tp, ef)
            nc = ncell_min(core, e) if model.min_cells else ncell_for(core, ec)
            subs += [(tp, fp[0], dl_face, fp[1], nf),
                     (core, e, dl_core, dm, nc),
                     (tp, fp[0], dl_face, fp[1], nf)]
        elif model.gap is not None and L.name == model.gap_after:
            # E030: [layer material over t - g, air gap g]; t stays the cavity depth
            g = model.gap.value
            og = own("gap")
            dl_core = {ot: 1.0} if ot is not None else {}
            if og is not None:
                dl_core[og] = dl_core.get(og, 0) - 1.0
            core = L.t.value - g
            n_c = ncell_min(core, e) if model.min_cells else max(1, n_cells_for(L, f.max(), T))
            air = np.ones(f.size, complex)
            n_g = ncell_min(g, air) if model.min_cells else ncell_for(g, 1.0 + 0j)
            subs += [(core, e, dl_core, dm, n_c), (g, air, {og: 1.0} if og is not None else {}, {}, n_g)]
        else:
            n_l = ncell_min(L.t.value, e) if model.min_cells else n_cells_for(L, f.max(), T)
            subs.append((L.t.value, e, {ot: 1.0} if ot is not None else {}, dm, n_l))
    shift = (0.0, {})
    if model.rough is not None:
        # the taper straddles the MEAN surface (Kirchhoff's reference): h/2 comes out of the front
        # material and h/2 out of the air gap, so the air path is d - h/2 (returned as `shift`)
        h = ROUGH_H * model.rough.value
        orr = own("rough")
        first = subs[0]
        taper = []
        for i in range(ROUGH_SUB):
            wi = (i + 0.5) / ROUGH_SUB
            taper.append((h / ROUGH_SUB, 1 + wi * (first[1] - 1),
                          {orr: ROUGH_H / ROUGH_SUB} if orr is not None else {},
                          {o: wi * dd for o, dd in first[3].items()}, 1))
        dl_first = dict(first[2])
        if orr is not None:
            dl_first[orr] = dl_first.get(orr, 0) - ROUGH_H / 2
        subs = taper + [(first[0] - h / 2, first[1], dl_first, first[3], first[4])] + subs[1:]
        shift = (-h / 2, {orr: -ROUGH_H / 2} if orr is not None else {})
    return subs, shift


def forward(model: WallModel, f, jacobian=True):
    """Gamma(f) at the radar plane and (optionally) dGamma/dp for all FREE parameters.
    With model.beam set (E023) the result is the beam-received signal S(f) of forward_beam()."""
    f = np.asarray(f, float)
    model.sync_ties()
    if model.band_offset is not None and len(band_segments(f)) > 1:
        return _forward_band_offset(model, f, jacobian)
    if model.beam is not None:
        return forward_beam(model, f, jacobian)
    if USE_COMPILED:
        import fastkern as FK
        if FK.HAVE_NUMBA:
            out = _forward_fast(model, f, jacobian, FK)
            if out is not None:
                return out
    names = model.free_names()
    J = np.zeros((f.size, len(names)), complex) if jacobian else None
    subs, shift = _effective(model, f)
    ncell = [sb[4] for sb in subs]
    layer_of_cell = np.repeat(np.arange(len(subs)), ncell)
    ell = np.concatenate([[sb[0] / n] * n for sb, n in zip(subs, ncell)])
    Nc = ell.size
    eps_all = [sb[1] for sb in subs]
    n_back = _sqrt_branch(np.array([model.eps_back]))[0]
    col = {nm: j for j, nm in enumerate(names)}

    w = 2 * np.pi * f[:, None]                                   # (F,1)
    k0 = w / C0
    eps_l = np.stack(eps_all, axis=1)                              # (F, layers)
    n_l = _sqrt_branch(eps_l)
    k = k0 * n_l[:, layer_of_cell]                                 # (F, cells)
    x = k * ell[None, :]
    s, c = np.sin(x), np.cos(x)
    b = -1j * k / (w * MU0 * s)
    M = 1j * k * np.tan(x / 2) / (w * MU0)
    diag = np.zeros((f.size, Nc + 1), complex)
    diag[:, :-1] += b + M
    diag[:, 1:] += b + M
    diag[:, 0] += 1 / ETA0
    diag[:, -1] += n_back / ETA0
    off = -b
    rhs = np.zeros((f.size, Nc + 1, 2), complex)
    rhs[:, 0, 0] = 2 / ETA0          # forward: incident amplitude a = 1
    rhs[:, 0, 1] = 1.0               # adjoint: picks out e_0 (A is symmetric, so A^T = A)
    sol = _tridiag_solve(diag, off, rhs)
    e, lam = sol[..., 0], sol[..., 1]
    phase = np.exp(-2j * k0[:, 0] * (model.standoff.value + shift[0]))
    G = (e[:, 0] - 1) * phase
    if not jacobian:
        return G, None
    # adjoint:  dGamma/dp = -phase * sum_cells [ db/dp * Q1 + dM/dp * Q2 ]
    Q1 = (lam[:, :-1] - lam[:, 1:]) * (e[:, :-1] - e[:, 1:])
    Q2 = lam[:, :-1] * e[:, :-1] + lam[:, 1:] * e[:, 1:]
    db_dk = -1j / (w * MU0) * (s - x * c) / s**2
    db_dl = 1j * k**2 * c / (w * MU0 * s**2)
    sec2 = 1 / np.cos(x / 2) ** 2
    dM_dk = 1j / (w * MU0) * (np.tan(x / 2) + (x / 2) * sec2)
    dM_dl = 1j * k**2 * sec2 / (2 * w * MU0)
    Tk = db_dk * Q1 + dM_dk * Q2                 # per-cell sensitivity to k
    Tl = db_dl * Q1 + dM_dl * Q2                 # per-cell sensitivity to cell length
    for li, sb in enumerate(subs):
        sel = layer_of_cell == li
        tl = -phase * Tl[:, sel].sum(1) / ncell[li]
        for o, coef in sb[2].items():
            J[:, col[o]] += coef * tl
        tk = -phase * Tk[:, sel].sum(1) * k0[:, 0] / (2 * n_l[:, li])
        for o, de in sb[3].items():
            J[:, col[o]] += tk * de
    if model.owner("standoff") is not None:
        J[:, col["standoff"]] += G * (-2j * k0[:, 0])
    for o, coef in shift[1].items():
        J[:, col[o]] += coef * G * (-2j * k0[:, 0])
    return G, J


def _forward_fast(model, f, jacobian, FK):
    """E026: plane-wave forward + adjoint Jacobian with stars computed once per (frequency,
    sublayer) and a compiled chain solve. Identical to the cell-wise path (gate); returns None
    (caller falls back) if any row fails the residual check."""
    names = model.free_names()
    subs, shift = _effective(model, f)
    ncell = np.array([sb[4] for sb in subs], np.int64)
    w = 2 * np.pi * f[:, None]
    k0 = w / C0
    eps_s = np.stack([sb[1] for sb in subs], axis=1)                    # (F, S)
    n_s = _sqrt_branch(eps_s)
    k = k0 * n_s
    lc = np.array([sb[0] for sb in subs]) / ncell                      # cell length per sublayer
    x = k * lc[None, :]
    sx, cx = np.sin(x), np.cos(x)
    b = -1j * k / (w * MU0 * sx)
    tx2 = np.tan(x / 2)
    M = 1j * k * tx2 / (w * MU0)
    n_back = _sqrt_branch(np.array([model.eps_back]))[0]
    Fn = f.size
    e0, SQ1, SQ2, res = FK.chain(np.ascontiguousarray(b), np.ascontiguousarray(M), ncell,
                                 np.full(Fn, 1 / ETA0 + 0j), np.full(Fn, n_back / ETA0 + 0j),
                                 np.full(Fn, 2 / ETA0 + 0j))
    if not np.all(res < 1e-10):
        return None
    phase = np.exp(-2j * k0[:, 0] * (model.standoff.value + shift[0]))
    G = (e0 - 1) * phase
    if not jacobian:
        return G, None
    J = np.zeros((Fn, len(names)), complex)
    col = {nm: j for j, nm in enumerate(names)}
    db_dk = -1j / (w * MU0) * (sx - x * cx) / sx**2
    db_dl = 1j * k**2 * cx / (w * MU0 * sx**2)
    sec2 = 1 / np.cos(x / 2) ** 2
    dM_dk = 1j / (w * MU0) * (tx2 + (x / 2) * sec2)
    dM_dl = 1j * k**2 * sec2 / (2 * w * MU0)
    Tk = db_dk * SQ1 + dM_dk * SQ2
    Tl = db_dl * SQ1 + dM_dl * SQ2
    for li, sb in enumerate(subs):
        tl = -phase * Tl[:, li] / ncell[li]
        for o, coef in sb[2].items():
            J[:, col[o]] += coef * tl
        tk = -phase * Tk[:, li] * k0[:, 0] / (2 * n_s[:, li])
        for o, de in sb[3].items():
            J[:, col[o]] += tk * de
    if model.owner("standoff") is not None:
        J[:, col["standoff"]] += G * (-2j * k0[:, 0])
    for o, coef in shift[1].items():
        J[:, col[o]] += coef * G * (-2j * k0[:, 0])
    return G, J


def _forward_band_offset(model, f, jacobian):
    """E024: bands after the first see standoff d + delta (separate phase centre). Each band is
    solved with its own standoff; d Gamma / d delta = d Gamma / d standoff on those bands."""
    names = model.free_names()
    inner = model.copy()
    inner.band_offset = None
    inner.standoff.free = True
    inner_names = inner.free_names()
    G = np.zeros(f.size, complex)
    J = np.zeros((f.size, len(names)), complex) if jacobian else None
    ob = model.owner("band_offset")
    for k, (a, b) in enumerate(band_segments(f)):
        inner.standoff.value = model.standoff.value + (model.band_offset.value if k > 0 else 0.0)
        if model.beam is not None:
            inner.beam = (model.beam[k],)
        Gk, Jk = forward(inner, f[a:b], jacobian)
        G[a:b] = Gk
        if jacobian:
            for j, nm in enumerate(inner_names):
                if nm in names:
                    J[a:b, names.index(nm)] += Jk[:, j]
            if k > 0 and ob is not None:
                J[a:b, names.index(ob)] += Jk[:, inner_names.index("standoff")]
    return G, J


EPS0 = 1 / (MU0 * C0**2)


def forward_oblique(model: WallModel, f, krho, jacobian=True):
    """E023. Exact-star DEC reflection at the front face (no air path) for oblique plane waves.
    f (F,), krho (F, K). Returns Gamma_TE, Gamma_TM (F, K) and, if jacobian, their derivatives
    (F, K, n_free). Every (f, k_rho, polarisation) is an independent 1D problem: in each cell
        k_z = sqrt(eps k0^2 - k_rho^2) (Im <= 0),  Y = k_z/(w mu0) (TE) or w eps0 eps / k_z (TM),
        b = -j Y / sin(k_z l),  M = j Y tan(k_z l / 2)          (exact 2-port of a uniform line)
    which at k_rho = 0 is the normal-incidence star of forward(). Ports: air admittance Y0 with
    source 2 Y0 (unit incident field), back half-space admittance; Gamma = e_0 - 1 (tangential E).
    Adjoint as in forward(): dGamma/dp = -sum_cells [db/dp Q1 + dM/dp Q2]."""
    f = np.asarray(f, float)
    F, K = krho.shape
    names = model.free_names()
    subs, _ = _effective(model, f)        # the air-path shift is applied by forward_beam
    ncell = [sb[4] for sb in subs]
    layer_of_cell = np.repeat(np.arange(len(subs)), ncell)
    ell = np.concatenate([[sb[0] / n] * n for sb, n in zip(subs, ncell)])
    eps_l = np.stack([sb[1] for sb in subs], axis=1)                    # (F, sublayers)
    # rows: (pol, f, k): R = 2 F K
    w = np.tile(np.repeat(2 * np.pi * f, K), 2)[:, None]
    k0 = w / C0
    kr = np.tile(krho.reshape(-1), 2)[:, None]
    tm = np.repeat([False, True], F * K)[:, None]
    epsc = np.tile(np.repeat(eps_l[:, layer_of_cell], K, axis=0), (2, 1))   # (R, cells)

    def kz_of(eps):
        q = np.sqrt(eps * k0**2 - kr**2 + 0j)
        return np.where(q.imag > 0, -q, q)

    def Y_of(eps, kz):
        return np.where(tm, w * EPS0 * eps / kz, kz / (w * MU0))

    kz = kz_of(epsc)
    Y = Y_of(epsc, kz)
    x = kz * ell[None, :]
    sx, cx = np.sin(x), np.cos(x)
    b = -1j * Y / sx
    M = 1j * Y * np.tan(x / 2)
    one = np.ones((w.shape[0], 1), complex)
    kz0 = kz_of(one)
    Y0 = Y_of(one, kz0)
    eb = model.eps_back * one
    Yb = Y_of(eb, kz_of(eb))
    Nc = ell.size
    diag = np.zeros((w.shape[0], Nc + 1), complex)
    diag[:, :-1] += b + M
    diag[:, 1:] += b + M
    diag[:, 0] += Y0[:, 0]
    diag[:, -1] += Yb[:, 0]
    rhs = np.zeros((w.shape[0], Nc + 1, 2), complex)
    rhs[:, 0, 0] = 2 * Y0[:, 0]
    rhs[:, 0, 1] = 1.0
    sol = _tridiag_solve(diag, -b, rhs)
    e, lam = sol[..., 0], sol[..., 1]
    G = (e[:, 0] - 1).reshape(2, F, K)
    if not jacobian:
        return G[0], G[1], None
    col = {nm: j for j, nm in enumerate(names)}
    J = np.zeros((w.shape[0], len(names)), complex)
    Q1 = (lam[:, :-1] - lam[:, 1:]) * (e[:, :-1] - e[:, 1:])
    Q2 = lam[:, :-1] * e[:, :-1] + lam[:, 1:] * e[:, 1:]
    db_dx = 1j * Y * cx / sx**2
    dM_dx = 1j * Y / (2 * np.cos(x / 2) ** 2)
    Tx = db_dx * Q1 + dM_dx * Q2                      # per-cell sensitivity to x = k_z l
    TY = (b / Y) * Q1 + (M / Y) * Q2                  # per-cell sensitivity to Y
    dkz = k0**2 / (2 * kz)                            # d k_z / d eps
    dY = np.where(tm, w * EPS0 / kz - w * EPS0 * epsc / kz**2 * dkz, dkz / (w * MU0))
    Te = Tx * ell[None, :] * dkz + TY * dY            # d/d eps (per cell)
    Tl = Tx * kz                                      # d/d cell length
    for li, sb in enumerate(subs):
        sel = layer_of_cell == li
        tl = -Tl[:, sel].sum(1) / ncell[li]
        for o, coef in sb[2].items():
            J[:, col[o]] += coef * tl
        te = -Te[:, sel].sum(1)
        for o, de in sb[3].items():
            J[:, col[o]] += te * np.tile(np.repeat(de, K), 2)
    J = J.reshape(2, F, K, len(names))
    return G[0], G[1], J


def beam_nodes(f, w0, K, d_ref=None):
    """Quadrature for the Gaussian beam: int_0^inf exp(-u) g(u) du, u = (k_rho w0)^2 / 2.
    Without d_ref: plain Gauss-Laguerre in u (evanescent nodes dropped, weights renormalised).
    With d_ref (contour rotation / steepest descent): g(u) carries the Fresnel phase
    exp(-2j k_z0 d) ~ exp(-2j k0 d) exp(j beta u), beta = 2 d_ref / (k0 w0^2), which makes the real
    nodes inefficient. The integrand is analytic in u, so substitute u = v/(1 - j beta):
        int e^-u g(u) du = 1/(1 - j beta) int e^-v [g(u(v)) e^{-j beta u(v)}] dv,
    i.e. Gauss-Laguerre in v at complex nodes u(v); the integral is unchanged (Cauchy).
    Returns k_rho (possibly complex) and weights for sum_q w_q g(u_q)."""
    v, wq = np.polynomial.laguerre.laggauss(K)
    k0 = 2 * np.pi * f[:, None] / C0
    if d_ref is None:
        kr = np.sqrt(2 * v)[None, :] / w0 * np.ones((f.size, 1))
        wts = np.where(kr < 0.999 * k0, wq[None, :], 0.0)
        return kr, wts / wts.sum(1, keepdims=True)
    beta = 2 * d_ref / (k0 * w0**2)                                  # (F, 1)
    u = v[None, :] / (1 - 1j * beta)
    kr = np.sqrt(2 * u + 0j) / w0
    wts = wq[None, :] / (1 - 1j * beta) * np.exp(-1j * beta * u)
    return kr, wts


def forward_beam(model: WallModel, f, jacobian=True):
    """E023. Received co-polar signal of a Gaussian beam (waist model.beam[band]) on the layered
    wall at standoff d:  S(f) = sum_q w_q (Gamma_TE + Gamma_TM)/2 exp(-2j k_z0 d).
    Each node is an exact oblique-star solve, so the only approximation is the quadrature."""
    segs = band_segments(f)
    K = model.beam_K
    krs, wts = [], []
    for (a, bnd), w0 in zip(segs, model.beam):
        kr, wt = beam_nodes(f[a:bnd], w0, K, model.beam_dref)
        krs.append(kr); wts.append(wt)
    kr, wt = np.vstack(krs), np.vstack(wts)
    GTE, GTM, Jo = forward_oblique(model, f, kr, jacobian)
    k0 = 2 * np.pi * f[:, None] / C0
    kz0 = np.sqrt(k0**2 - kr**2 + 0j)
    _, shift = _effective(model, f[:1]) if model.rough is not None else (None, (0.0, {}))
    ph = np.exp(-2j * kz0 * (model.standoff.value + shift[0]))
    Gco = 0.5 * (GTE + GTM)
    S = np.sum(wt * Gco * ph, axis=1)
    if not jacobian:
        return S, None
    names = model.free_names()
    J = np.einsum("fk,fkp->fp", wt * ph, 0.5 * (Jo[0] + Jo[1]))
    dS_dd = np.sum(wt * Gco * ph * (-2j * kz0), axis=1)
    if model.owner("standoff") is not None:
        J[:, names.index("standoff")] += dS_dd
    for o, coef in shift[1].items():
        J[:, names.index(o)] += coef * dS_dd
    return S, J


USE_COMPILED = True     # E026: compiled Thomas kernel (fastkern); False = numpy reference path


def _tridiag_solve(diag, off, rhs, check=1e-10):
    """Solve symmetric tridiagonal systems, one per frequency (rows), vectorized Thomas.
    Every solution is verified by its residual; any frequency that fails the check is
    re-solved with LAPACK's pivoting tridiagonal solver, so accuracy is guaranteed."""
    if USE_COMPILED:
        import fastkern as FK
        if FK.HAVE_NUMBA and rhs.shape[-1] == 2 and diag.shape[1] > 1:
            x, res = FK.thomas_batch(np.ascontiguousarray(diag), np.ascontiguousarray(off), np.ascontiguousarray(rhs))
            for m in np.where(~(res < check))[0]:
                n = diag.shape[1]
                ab = np.zeros((3, n), complex)
                ab[0, 1:], ab[1], ab[2, :-1] = off[m], diag[m], off[m]
                x[m] = solve_banded((1, 1), ab, rhs[m])
            return x
    return _tridiag_solve_numpy(diag, off, rhs, check)


def _tridiag_solve_numpy(diag, off, rhs, check=1e-10):
    """Reference (numpy) implementation of _tridiag_solve."""
    F, n = diag.shape
    cp = np.zeros((F, n - 1), complex)
    dp = np.zeros_like(rhs)
    den = diag[:, 0]
    cp[:, 0] = off[:, 0] / den
    dp[:, 0] = rhs[:, 0] / den[:, None]
    for i in range(1, n):
        den = diag[:, i] - off[:, i - 1] * cp[:, i - 1]
        if i < n - 1:
            cp[:, i] = off[:, i] / den
        dp[:, i] = (rhs[:, i] - off[:, i - 1, None] * dp[:, i - 1]) / den[:, None]
    x = np.zeros_like(rhs)
    x[:, -1] = dp[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i, None] * x[:, i + 1]
    Ax = diag[..., None] * x
    Ax[:, :-1] += off[..., None] * x[:, 1:]
    Ax[:, 1:] += off[..., None] * x[:, :-1]
    res = np.abs(Ax - rhs).max(axis=(1, 2)) / (np.abs(rhs).max(axis=(1, 2)) + np.abs(diag).max(1) * np.abs(x).max(axis=(1, 2)))
    bad = np.where(~(res < check))[0]
    for m in bad:
        ab = np.zeros((3, n), complex)
        ab[0, 1:], ab[1], ab[2, :-1] = off[m], diag[m], off[m]
        x[m] = solve_banded((1, 1), ab, rhs[m])
    return x


# ============================================================================ calibration
def calibrate(y_wall, y_plate, f, plate_distance):
    """Metal plate (Gamma = -1) at a known distance gives A(f); divide it out of the wall scan."""
    A_est = -y_plate * np.exp(+2j * 2 * np.pi * f / C0 * plate_distance)
    return y_wall / A_est, A_est


# ============================================================================ regularization
@dataclass
class Regularizer:
    names: list[str]          # ordered parameter names forming a 1D profile
    lam: float
    kind: str = "smooth"      # "smooth" (Tikhonov on first differences) or "tv"
    beta: float = 1e-3        # TV smoothing

    def rows(self, model: WallModel):
        """Residual rows r_reg and their Jacobian w.r.t. the free physical parameters."""
        free = model.free_names()
        vals = model.summary()
        v = np.array([vals[n] for n in self.names])
        d = np.diff(v)
        Jd = np.zeros((d.size, len(free)))
        for i in range(d.size):
            Jd[i, free.index(self.names[i + 1])] += 1
            Jd[i, free.index(self.names[i])] -= 1
        sl = np.sqrt(self.lam)
        if self.kind == "smooth":
            return sl * d, sl * Jd
        q = d**2 + self.beta**2                      # TV: sum sqrt(d^2+beta^2) = sum r^2
        r = sl * q**0.25
        dr = sl * 0.5 * d * q**-0.75
        return r, dr[:, None] * Jd


# ============================================================================ optimizer
def _to_u(p, lo, hi):
    s = np.clip((p - lo) / (hi - lo), 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def _from_u(u, lo, hi):
    s = 1 / (1 + np.exp(-np.clip(u, -40, 40)))
    return lo + (hi - lo) * s, (hi - lo) * s * (1 - s)


_GATE_CACHE = {}


def band_segments(f, rtol=1e-6):
    """Split a frequency vector into uniformly spaced bands: list of (start, stop) indices.
    A step that differs from both neighbouring steps is a jump between bands. A single uniform
    grid gives one segment, so single-band code paths are unchanged."""
    f = np.asarray(f, float)
    d = np.diff(f)
    if d.size < 3:
        return [(0, f.size)]
    same = lambda x, y: abs(x - y) <= rtol * max(abs(x), abs(y))
    jumps = [i for i in range(d.size)
             if (i == 0 or not same(d[i], d[i - 1])) and (i == d.size - 1 or not same(d[i], d[i + 1]))
             and 0 < i < d.size - 1]
    cuts = [0] + [i + 1 for i in jumps] + [f.size]
    return list(zip(cuts[:-1], cuts[1:]))


def carrier(f):
    """Centre frequency of the highest band: sets the phase-cycle period (finest basins)."""
    segs = band_segments(f)
    a, b = max(segs, key=lambda ab: f[ab[1] - 1])
    return float(np.mean(f[a:b]))


@dataclass
class Gate:
    """Linear time gate: windowed, zero-padded inverse FFT, keep range bins in [r_lo, r_hi].
    Applied identically to data and model, so it selects echoes without biasing the fit."""
    r_lo: float
    r_hi: float
    pad: int = 4

    def op(self, f):
        segs = band_segments(f)
        if len(segs) > 1:                    # several bands: block-diagonal, one gate per band
            key = (tuple((f[a], f[b - 1], b - a) for a, b in segs), self.r_lo, self.r_hi, self.pad)
            if key not in _GATE_CACHE:
                blocks = [self.op(f[a:b]) for a, b in segs]
                Op = np.zeros((sum(B.shape[0] for B in blocks), f.size), complex)
                i = 0
                for (a, b), B in zip(segs, blocks):
                    Op[i:i + B.shape[0], a:b] = B
                    i += B.shape[0]
                _GATE_CACHE[key] = Op
            return _GATE_CACHE[key]
        key = (f.size, f[0], f[-1], self.r_lo, self.r_hi, self.pad)
        if key in _GATE_CACHE:
            return _GATE_CACHE[key]
        F = f.size
        n = self.pad * F
        r = np.arange(n) * C0 / (2 * (f[1] - f[0]) * n)
        keep = (r >= self.r_lo) & (r <= self.r_hi)
        w = np.hanning(F)
        E = np.exp(2j * np.pi * np.outer(np.arange(n)[keep], np.arange(F)) / n) / np.sqrt(F)
        _GATE_CACHE[key] = E * w[None, :]            # (bins kept, F)
        return _GATE_CACHE[key]


def residuals(model, f, y, sigma, regs=(), gate=None):
    G, Jc = forward(model, f)
    if gate is not None:
        Op = gate.op(f)
        noise = sigma * np.sqrt(np.mean(np.abs(Op) ** 2) * f.size)
        d = Op @ (G - y)
        Jc = Op @ Jc
        r = np.concatenate([d.real, d.imag]) / noise
        J = np.vstack([Jc.real, Jc.imag]) / noise
    else:
        r = np.concatenate([(G - y).real, (G - y).imag]) / sigma
        J = np.vstack([Jc.real, Jc.imag]) / sigma
    for R in regs:
        rr, jj = R.rows(model)
        r = np.concatenate([r, rr])
        J = np.vstack([J, jj])
    return r, J


def levenberg_marquardt(model, f, y, sigma, regs=(), max_iter=200, tol=1e-12, verbose=False, gate=None):
    """Bounded LM in u = logit((p-lo)/(hi-lo)). Returns the model at the optimum and the cost."""
    model = model.copy()
    lo, hi = model.bounds()
    u = _to_u(model.get(), lo, hi)
    p, dpdu = _from_u(u, lo, hi)
    model.set(p)
    r, Jp = residuals(model, f, y, sigma, regs, gate)
    cost = r @ r
    mu = 1e-3
    for it in range(max_iter):
        Ju = Jp * dpdu
        A = Ju.T @ Ju
        g = Ju.T @ r
        step_taken = False
        for _ in range(30):
            D = np.diag(np.diag(A)) + 1e-12 * np.eye(len(u))
            du = np.linalg.solve(A + mu * D, -g)
            u_new = u + du
            p_new, dpdu_new = _from_u(u_new, lo, hi)
            trial = model.copy(); trial.set(p_new)
            r_new, Jp_new = residuals(trial, f, y, sigma, regs, gate)
            c_new = r_new @ r_new
            if c_new < cost:
                rel = (cost - c_new) / max(cost, 1e-300)
                u, model, r, Jp, dpdu, cost = u_new, trial, r_new, Jp_new, dpdu_new, c_new
                mu = max(mu / 3, 1e-12)
                step_taken = True
                break
            mu *= 4
        if verbose:
            print(f"  it {it:3d}  cost {cost:.6e}  mu {mu:.1e}")
        if not step_taken or rel < tol:
            break
    return model, cost


def invert(model0, f, y, sigma, n_starts=24, regs=(), rng=None, keep_init_for=(), verbose=False, gate=None):
    """Multi-start LM. Free parameters are drawn uniformly within their bounds, except
    names in keep_init_for, which start at model0's value (e.g. a good standoff guess)."""
    rng = np.random.default_rng(rng)
    lo, hi = model0.bounds()
    names = model0.free_names()
    best, best_cost, costs = None, np.inf, []
    for s in range(n_starts):
        m = model0.copy()
        p = lo + (hi - lo) * rng.random(len(lo)) if s > 0 else m.get()
        for i, nm in enumerate(names):
            if nm in keep_init_for:
                p[i] = model0.get()[i]
        m.set(p)
        m_opt, c = levenberg_marquardt(m, f, y, sigma, regs, gate=gate)
        costs.append(c)
        if c < best_cost:
            best, best_cost = m_opt, c
        if verbose:
            print(f"start {s:2d}: cost {c:.4e}")
    return best, best_cost, np.array(costs)


# ============================================================================ identifiability
def fisher(model, f, sigma):
    """Fisher information for the free parameters (data term only) and derived quantities."""
    _, Jc = forward(model, f)
    J = np.vstack([Jc.real, Jc.imag]) / sigma
    F = J.T @ J
    cov = np.linalg.inv(F)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    scale = np.abs(model.get()) + 1e-12            # relative sensitivities
    sv = np.linalg.svd(J * scale, compute_uv=False)
    return dict(F=F, cov=cov, crb_std=std, corr=corr, singular_values=sv, names=model.free_names())


def choose_lambda(model0, f, y, sigma, make_regs, lams, n_starts=6, rng=0):
    """Morozov discrepancy principle: largest lambda whose data misfit chi^2 is <= the
    expected noise level (number of real residuals)."""
    target = 2 * len(f)
    out = []
    for lam in lams:
        regs = make_regs(lam)
        m, _, _ = invert(model0, f, y, sigma, n_starts=n_starts, regs=regs, rng=rng)
        r, _ = residuals(m, f, y, sigma)
        out.append((lam, r @ r, m))
    ok = [o for o in out if o[1] <= target * 1.05]
    chosen = max(ok, key=lambda o: o[0]) if ok else min(out, key=lambda o: o[1])
    return chosen, out, target


# ============================================================================ cycle-skip escape
def _phase_periods(model, f):
    """For each free thickness (and the standoff), the shift that changes the round-trip
    phase at band centre by exactly 2*pi: lambda_c / (2 Re n). False minima sit at these
    spacings, so hopping by whole periods jumps directly between basins."""
    fc = carrier(f)                      # = f.mean() for a single uniform band
    lam_c = C0 / fc
    out = {}
    for L in model.layers:
        if L.t.free:
            eps, _ = layer_eps_and_derivs(L, np.array([fc]), model.water_T, model.law, model.f_ref)
            out[f"{L.name}.t"] = lam_c / (2 * np.sqrt(eps[0]).real)
    if model.standoff.free:
        out["standoff"] = lam_c / 2
    return out


def cycle_hop(model, f, y, sigma, regs=(), max_rounds=6, hops=(1, -1, 2, -2), verbose=False):
    """Greedy basin hopping: shift one thickness by +-k periods (or two neighbouring
    thicknesses in opposite directions, which keeps the total depth fixed), re-run LM,
    keep any improvement. Repeats until a full round finds nothing better."""
    best = model
    r, _ = residuals(best, f, y, sigma, regs)
    best_cost = r @ r
    for rnd in range(max_rounds):
        improved = False
        per = _phase_periods(best, f)
        names = list(per)
        moves = [((a, h),) for a in names for h in hops]
        th = [n for n in names if n != "standoff"]
        moves += [((a, h), (b, -h)) for a, b in zip(th[:-1], th[1:]) for h in (1, -1)]
        moves += [(("standoff", h), (th[0], -h)) for h in (1, -1)] if th and "standoff" in per else []
        for mv in moves:
            trial = best.copy()
            vals = dict(zip(trial.free_names(), trial.get()))
            lo, hi = trial.bounds()
            for nm, h in mv:
                vals[nm] += h * per[nm]
            v = np.array([vals[n] for n in trial.free_names()])
            if np.any(v <= lo) or np.any(v >= hi):
                continue
            trial.set(v)
            m_opt, c = levenberg_marquardt(trial, f, y, sigma, regs)
            if c < best_cost * (1 - 1e-9):
                best, best_cost, improved = m_opt, c, True
                if verbose:
                    print(f"  hop {mv}: cost -> {c:.2f}")
        if not improved:
            break
    return best, best_cost


# ============================================================================ layer stripping
@dataclass
class Stage:
    free: tuple            # parameter names fitted in this stage (others held fixed)
    gate: Gate | None      # time gate selecting this stage's echoes (None = all data)
    n_starts: int = 30     # random starts (used when no lattice is given)
    lattice: dict | None = None   # deterministic starts: {thickness name: range of period
                                  # offsets k}, {eps name: list of values}; full product
    profile_standoff: bool = True  # at each start, set standoff to its exact best value on a fine grid


def strip_invert(model0, f, y, sigma, stages, rng=0, final_hop=True, verbose=False):
    """Fit echo groups one at a time (each on its own time-gated data, other parameters held),
    then a joint LM on all data from the combined estimate, then cycle hopping."""
    rng = np.random.default_rng(rng)
    model = model0.copy()
    all_free = {n for n, _, p in model.items() if p.free}  # includes tied names
    for k, st in enumerate(stages):
        m = model.copy()
        for n, _, p in m.items():
            p.free = (n in st.free) or (p.tie is not None and p.tie in st.free)
        if st.lattice:
            best, c, costs = lattice_invert(m, f, y, sigma, st.lattice, gate=st.gate,
                                            profile_standoff=st.profile_standoff)
        else:
            best, c, costs = invert(m, f, y, sigma, n_starts=st.n_starts, rng=rng, gate=st.gate)
        for n, _, p in best.items():
            p.free = n in all_free
        model = best
        if verbose:
            print(f"stage {k}: {st.free}  gated chi2 {c:.1f}  ({(costs < c * 1.02).sum()}/{len(costs)} starts reached it)")
    model, cost = levenberg_marquardt(model, f, y, sigma)
    if final_hop:
        model, cost = cycle_hop(model, f, y, sigma)
    return model, cost


def best_standoff(model, f, y, sigma, gate=None, step_frac=1 / 16):
    """Standoff enters only as the phase ramp exp(-2j k0 d), so the misfit for every d on a
    fine grid (step = period * step_frac) costs one forward solve plus one matrix product.
    Returns the grid minimizer within the standoff bounds."""
    p = model.standoff
    period = C0 / carrier(f) / 2
    grid = np.arange(p.lo, p.hi, period * step_frac)
    m0 = model.copy(); m0.standoff.value = 0.0
    G0, _ = forward(m0, f, jacobian=False)
    k0 = 2 * np.pi * f / C0
    P = np.exp(-2j * np.outer(k0, grid))                      # (F, grid)
    R = y[:, None] - G0[:, None] * P
    if gate is not None:
        R = gate.op(f) @ R
    cost = np.sum(np.abs(R) ** 2, axis=0)
    return grid[np.argmin(cost)]


def lattice_invert(model0, f, y, sigma, lattice, gate=None, regs=(), profile_standoff=False):
    """LM from every point of a start lattice. Thickness entries are offsets in whole
    phase periods from the current value (one start per basin); other entries are values."""
    import itertools
    per = _phase_periods(model0, f)
    keys = list(lattice)
    best, best_cost, costs = None, np.inf, []
    for combo in itertools.product(*[lattice[k] for k in keys]):
        m = model0.copy()
        vals = dict(zip(m.free_names(), m.get()))
        for k, v in zip(keys, combo):
            vals[k] = vals[k] + v * per[k] if k in per else v
        lo, hi = m.bounds()
        x = np.clip([vals[n] for n in m.free_names()], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
        m.set(x)
        if profile_standoff and "standoff" in m.free_names() and "standoff" not in lattice:
            m.standoff.value = best_standoff(m, f, y, sigma, gate)
        mo, c = levenberg_marquardt(m, f, y, sigma, regs, gate=gate)
        costs.append(c)
        if c < best_cost:
            best, best_cost = mo, c
    return best, best_cost, np.array(costs)


def ambiguity_scan(model, f, y, sigma, names=None, ks=(-3, -2, -1, 1, 2, 3), dchi2=4.0, regs=(),
                   return_models=False):
    """Global uncertainty for thickness-like parameters: shift each by whole phase periods,
    re-optimize everything, and list every basin whose chi2 is within dchi2 of the best.
    Returns {name: [(value, chi2), ...]} sorted by chi2, plus the (possibly improved) best model.
    With return_models=True, also returns every plausible basin as (summary dict, chi2), so a
    benchmark can check whether one basin matches the truth in all parameters jointly."""
    r, _ = residuals(model, f, y, sigma, regs)
    best, best_cost = model, r @ r
    per = _phase_periods(model, f)
    names = names or [n for n in per if n != "standoff"]
    found = {n: [(model.summary()[n], best_cost)] for n in names}
    basins = [(model.summary(), best_cost)]
    for n in names:
        for k in ks:
            m = model.copy()
            vals = dict(zip(m.free_names(), m.get()))
            vals[n] += k * per[n]
            lo, hi = m.bounds()
            v = np.array([vals[q] for q in m.free_names()])
            if np.any(v <= lo) or np.any(v >= hi):
                continue
            m.set(v)
            mo, c = levenberg_marquardt(m, f, y, sigma, regs)
            found[n].append((mo.summary()[n], c))
            basins.append((mo.summary(), c))
            if c < best_cost:
                best, best_cost = mo, c
    out = {n: sorted([(v, c) for v, c in lst if c <= best_cost + dchi2], key=lambda t: t[1]) for n, lst in found.items()}
    if return_models:
        plaus = sorted([(sm, c) for sm, c in basins if c <= best_cost + dchi2], key=lambda t: t[1])
        return out, best, best_cost, plaus
    return out, best, best_cost

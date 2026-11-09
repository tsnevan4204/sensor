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


@dataclass
class WallModel:
    layers: list[LayerSpec]
    standoff: P
    eps_back: complex = 1.0 + 0j
    water_T: float = 20.0

    # ---- flat views over parameters -------------------------------------------------
    def items(self):
        for i, L in enumerate(self.layers):
            for key in ("t", "eps_r", "tan_d", "theta"):
                yield f"{L.name}.{key}", (i, key), getattr(L, key)
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
def layer_eps_and_derivs(L: LayerSpec, f, T):
    """eps(f) and d eps / d(eps_r, tan_d, theta), all exact (analytic)."""
    h = L.eps_r.value * (1 - 1j * L.tan_d.value)
    w = eps_water_debye(f, T)
    sh, sw = np.sqrt(h), np.sqrt(w)
    th = L.theta.value
    s = (1 - th) * sh + th * sw
    eps = s ** 2
    de_dh = s * (1 - th) / sh
    return eps, {
        "eps_r": de_dh * (1 - 1j * L.tan_d.value),
        "tan_d": de_dh * (-1j * L.eps_r.value),
        "theta": 2 * s * (sw - sh),
    }


def n_cells_for(L: LayerSpec, fmax, T):
    """Cells so that |k| l <= pi/2 at the top frequency (keeps stars positive and
    far from sin(kl)=0). The answer is exact for ANY cell count; this is only conditioning."""
    eps, _ = layer_eps_and_derivs(L, np.array([fmax]), T)
    k = 2 * np.pi * fmax / C0 * abs(np.sqrt(eps[0]))
    return max(1, int(np.ceil(k * L.t.value / (0.9 * np.pi))))


# ============================================================================ forward + adjoint
def forward(model: WallModel, f, jacobian=True):
    """Gamma(f) at the radar plane and (optionally) dGamma/dp for all FREE parameters."""
    f = np.asarray(f, float)
    model.sync_ties()
    names = model.free_names()
    J = np.zeros((f.size, len(names)), complex) if jacobian else None
    T = model.water_T
    ncell = [n_cells_for(L, f.max(), T) for L in model.layers]
    layer_of_cell = np.repeat(np.arange(len(model.layers)), ncell)
    ell = np.concatenate([[L.t.value / n] * n for L, n in zip(model.layers, ncell)])
    Nc = ell.size
    eps_all, deps_all = zip(*[layer_eps_and_derivs(L, f, T) for L in model.layers])
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
    phase = np.exp(-2j * k0[:, 0] * model.standoff.value)
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
    for li, L in enumerate(model.layers):
        sel = layer_of_cell == li
        o = model.owner(f"{L.name}.t")
        if o is not None:
            J[:, col[o]] += -phase * Tl[:, sel].sum(1) / ncell[li]
        dk_deps = k0[:, 0] / (2 * n_l[:, li])
        for key in ("eps_r", "tan_d", "theta"):
            o = model.owner(f"{L.name}.{key}")
            if o is not None:
                J[:, col[o]] += -phase * Tk[:, sel].sum(1) * dk_deps * deps_all[li][key]
    if model.owner("standoff") is not None:
        J[:, col["standoff"]] += G * (-2j * k0[:, 0])
    return G, J


def _tridiag_solve(diag, off, rhs, check=1e-10):
    """Solve symmetric tridiagonal systems, one per frequency (rows), vectorized Thomas.
    Every solution is verified by its residual; any frequency that fails the check is
    re-solved with LAPACK's pivoting tridiagonal solver, so accuracy is guaranteed."""
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


@dataclass
class Gate:
    """Linear time gate: windowed, zero-padded inverse FFT, keep range bins in [r_lo, r_hi].
    Applied identically to data and model, so it selects echoes without biasing the fit."""
    r_lo: float
    r_hi: float
    pad: int = 4

    def op(self, f):
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
    fc = f.mean()
    lam_c = C0 / fc
    out = {}
    for L in model.layers:
        if L.t.free:
            eps, _ = layer_eps_and_derivs(L, np.array([fc]), model.water_T)
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
    period = C0 / f.mean() / 2
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

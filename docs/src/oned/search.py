"""
Faster global search for the layered-wall inversion (E009-E012). The forward model is the same
exact-star DEC solve (wall_inverse.forward); only the search around it changes. Every option
defaults to the E007 behaviour so each experiment flips one switch.

  lm()            Levenberg-Marquardt identical to wall_inverse.levenberg_marquardt, plus
                  - atol: also stop when one accepted step lowers chi2 by less than atol (E009)
                  - coords="optical": iterate in (tau = Re(n) t, eps_r, tan_d, ...) instead of
                    (t, eps_r, tan_d, ...)  (E011). Exact change of variables; chain-rule Jacobian.
  invert()        layer stripping -> joint LM -> basin hops -> plausible set, with options:
                  search_step   use every k-th frequency for all *search* LM runs; the winners are
                                always re-converged on all frequencies at tol 1e-12 (E009)
                  init          "lattice" (E007) or "pencil" (matrix-pencil start, E010)
                  ambiguity     "scan" (E007 hops) or "ils" (integer least squares, E012)
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np
import wall_toy as W
import wall_inverse as I
from wall_inverse import _to_u, _from_u, layer_eps_and_derivs

C0 = W.C0


# ============================================================================ coordinates
class Coords:
    """Map between the model's free physical parameters p and the iteration coordinates q.
    kind="phys": q = p.  kind="optical": every free thickness L.t is replaced by
    tau_L = Re(n_L) t_L with n_L = sqrt(eps_r (1 - j tan_d)) (dry layers, frequency independent).
    Tied layers (e.g. back drywall eps tied to front) are handled through model.owner()."""

    def __init__(self, model, kind="phys"):
        self.kind = kind
        self.names = model.free_names()
        self.tidx = {}                       # index in q of tau_L -> layer index
        if kind == "optical":
            layer_names = [L.name for L in model.layers]
            for i, n in enumerate(self.names):
                if n.endswith(".t") and n[:-2] in layer_names:      # facing.t etc. stay physical
                    li = layer_names.index(n[:-2])
                    if model.layers[li].theta.free or model.layers[li].theta.value != 0:
                        raise ValueError("optical coordinates implemented for dry layers only")
                    self.tidx[i] = li

    @staticmethod
    def _n(L):
        return np.sqrt(L.eps_r.value * (1 - 1j * L.tan_d.value))

    def q_of(self, model):
        model.sync_ties()
        p = model.get().copy()
        for i, li in self.tidx.items():
            p[i] = self._n(model.layers[li]).real * model.layers[li].t.value
        return p

    def apply(self, model, q):
        """Return a copy of model with coordinates q (non-thickness first, then t = tau/Re n)."""
        m = model.copy()
        p = np.array(q, float).copy()
        if self.tidx:
            v = m.get().copy()
            for i in range(len(p)):
                if i not in self.tidx:
                    v[i] = p[i]
            m.set(v)                                   # sets eps/tan (and ties) first
            for i, li in self.tidx.items():
                v[i] = p[i] / self._n(m.layers[li]).real
            m.set(v)
        else:
            m.set(p)
        return m

    def dpdq(self, model):
        """Jacobian dp/dq at the model (n x n)."""
        n = len(self.names)
        P = np.eye(n)
        for i, li in self.tidx.items():
            L = model.layers[li]
            nn = self._n(L)
            rn = nn.real
            tau = rn * L.t.value
            P[i, i] = 1 / rn
            for key, dn in (("eps_r", (1 - 1j * L.tan_d.value) / (2 * nn)),
                            ("tan_d", (-1j * L.eps_r.value) / (2 * nn))):
                o = model.owner(f"{L.name}.{key}")
                if o is not None:
                    P[i, self.names.index(o)] += -tau / rn**2 * dn.real
        return P

    def bounds(self, model):
        lo, hi = model.bounds()
        lo, hi = lo.copy(), hi.copy()
        for i, li in self.tidx.items():
            L = model.layers[li]
            er_lo, er_hi = (L.eps_r.lo, L.eps_r.hi)
            lo[i] = L.t.lo * np.sqrt(er_lo)
            hi[i] = L.t.hi * np.sqrt(er_hi) * 1.01
        return lo, hi


# ============================================================================ priors (E013)
@dataclass
class GaussianPrior:
    """Independent Gaussian priors on named parameters: residual rows (v - mean)/sd, exactly a
    MAP term in the least-squares cost (chi2 + sum ((v - mean)/sd)^2). Rows only for parameters
    that are free in the model at hand (stage models see only their own)."""
    terms: tuple            # ((name, mean, sd), ...)

    def rows(self, model):
        free = model.free_names()
        vals = model.summary()
        r, J = [], []
        for name, mean, sd in self.terms:
            if name in free:
                row = np.zeros(len(free))
                row[free.index(name)] = 1.0 / sd
                r.append((vals[name] - mean) / sd)
                J.append(row)
        return np.array(r, float), (np.array(J, float) if J else np.zeros((0, len(free))))

    def penalty(self, model):
        vals = model.summary()
        return float(sum(((vals[n] - m) / s) ** 2 for n, m, s in self.terms))


_PRIORS = ()     # set by invert() for the duration of one inversion (single process)


# ============================================================================ LM
def lm(model, f, y, sigma, regs=(), max_iter=200, tol=1e-12, gate=None, atol=0.0, coords="phys"):
    """Bounded LM in u = logit((q-lo)/(hi-lo)). With coords="phys" and atol=0 this is
    step-for-step the same algorithm as wall_inverse.levenberg_marquardt."""
    regs = tuple(regs) + tuple(_PRIORS)
    model = model.copy()
    box = coords == "optical_box"          # E011b: optical coordinates + physical box kept
    C = Coords(model, "optical" if box else coords)
    lo, hi = C.bounds(model)
    plo, phi = model.bounds()
    u = _to_u(C.q_of(model), lo, hi)
    q, dqdu = _from_u(u, lo, hi)
    model = C.apply(model, q)
    if box:                                   # start strictly inside the physical box
        pv = model.get()
        model.set(np.clip(pv, plo + 1e-9 * (phi - plo), phi - 1e-9 * (phi - plo)))
        u = _to_u(C.q_of(model), lo, hi)
        q, dqdu = _from_u(u, lo, hi)
        model = C.apply(model, q)
    r, Jp = I.residuals(model, f, y, sigma, regs, gate)
    cost = r @ r
    mu = 1e-3
    n_eval = 1
    for it in range(max_iter):
        Ju = (Jp @ C.dpdq(model)) * dqdu if C.tidx else Jp * dqdu
        A = Ju.T @ Ju
        g = Ju.T @ r
        step_taken = False
        for _ in range(30):
            D = np.diag(np.diag(A)) + 1e-12 * np.eye(len(u))
            du = np.linalg.solve(A + mu * D, -g)
            u_new = u + du
            q_new, dqdu_new = _from_u(u_new, lo, hi)
            trial = C.apply(model, q_new)
            if box:
                pv = trial.get()
                if np.any(pv <= plo) or np.any(pv >= phi):   # outside the prior box: reject the
                    mu *= 4                                   # step exactly like a cost increase
                    continue
            r_new, Jp_new = I.residuals(trial, f, y, sigma, regs, gate)
            n_eval += 1
            c_new = r_new @ r_new
            if c_new < cost:
                rel = (cost - c_new) / max(cost, 1e-300)
                dec = cost - c_new
                u, model, r, Jp, dqdu, cost = u_new, trial, r_new, Jp_new, dqdu_new, c_new
                mu = max(mu / 3, 1e-12)
                step_taken = True
                break
            mu *= 4
        if not step_taken or rel < tol or dec < atol:
            break
    return model, cost


# ============================================================================ helpers
def phase_periods(model, f):
    return I._phase_periods(model, f)


def cost_of(model, f, y, sigma, gate=None):
    G, _ = I.forward(model, f, jacobian=False)
    d = G - y
    if gate is not None:
        Op = gate.op(f)
        noise = sigma * np.sqrt(np.mean(np.abs(Op) ** 2) * f.size)
        d = (Op @ d) / noise
    else:
        d = d / sigma
    return float(np.sum(d.real**2 + d.imag**2)) + sum(p.penalty(model) for p in _PRIORS)


def half(a, b):
    return np.arange(2 * a, 2 * b + 1) / 2


@dataclass
class Opts:
    search_step: int = 1          # E009: every k-th frequency in search LMs (1 = all)
    search_atol: float = 0.0      # E009: early stop of search LMs at this chi2 decrease
    n_polish: int = 3             # E009: distinct best starts re-converged on all frequencies
    init: str = "lattice"         # E010: "lattice" | "pencil"
    coords: str = "phys"          # E011: "phys" | "optical" | "optical_box" (E011b)
    ambiguity: str = "scan"       # E012: "scan" | "ils"
    fit_tan: bool = True
    priors: tuple = ()            # E013: ((name, mean, sd), ...) Gaussian priors
    standoff_halfwidth: float = 0.010   # E018: search box guess +- this; resolution cell c/2B = 0.0375
    law: str = "const"                  # E021: host material law of the model ("const" | "cq" | "jonscher")
    pinf_free: bool = False             # E021b: free Jonscher eps_inf fraction of the drywall (tied sheets)
    beam: tuple | None = None           # E023: per-band beam waists (m) from the antenna calibration
    beam_K: int = 2                     # E023: rotated Gauss-Laguerre nodes in the search (error ~1e-5 << noise)
    beam_K_final: int = 4               # E023: nodes for the final re-polish (error ~1e-8)
    facing: bool = False                # E024: shared paper facing on both drywall sheets (3 params)
    rough: bool = False                 # E024: effective front-face roughness (1 param)
    band_offset: bool = False           # E024: phase-centre offset of the second band (1 param)
    gap: bool = False                   # E030: air gap behind the insulation (1 param)
    chain: object = None                # E031: per-band measurement operators (IF chain x A_hat); search on full grid
    lat_step: float = 0.5               # E025: start-lattice spacing in phase periods
    lat0_eps: tuple = (2.0, 2.6, 3.2)   # E025: drywall eps start values
    lat1_eps: tuple = (1.05, 1.15, 1.3) # E025: insulation eps start values
    hops: tuple = (1, -1, 2, -2)        # E025: cycle-hop sizes
    amb_ks: tuple = (-3, -2, -1, 1, 2, 3)   # E025: ambiguity-scan shifts
    min_cells: bool = False             # E027: fewest pole-safe cells per layer (exact for any count)


def _subset(f, y, k):
    return (f, y) if k == 1 else (f[::k], y[::k])


def _distinct(cands, model_names, f, k):
    """Keep the best candidate per phase basin (thicknesses rounded to quarter periods)."""
    out, seen = [], set()
    for c, m in sorted(cands, key=lambda t: t[0]):
        per = phase_periods(m, f)
        key = tuple(int(np.round(m.summary()[n] / (per[n] / 2))) for n in model_names if n in per and n != "standoff")
        if key not in seen:
            seen.add(key)
            out.append((c, m))
        if len(out) >= k:
            break
    return out


# ============================================================================ stages
def stud_stages(d0, fit_tan=True, opts=None):
    """Stage definitions. The lattice density (E025) is configurable through opts:
    lat_step (offset spacing in phase periods; 0.5 = E007), lat0_eps / lat1_eps (eps start values)."""
    step = getattr(opts, "lat_step", 0.5) if opts is not None else 0.5
    e0 = getattr(opts, "lat0_eps", (2.0, 2.6, 3.2)) if opts is not None else (2.0, 2.6, 3.2)
    e1 = getattr(opts, "lat1_eps", (1.05, 1.15, 1.3)) if opts is not None else (1.05, 1.15, 1.3)
    off = lambda a: np.arange(-a, a + 1e-9, step)
    s0 = ("standoff", "drywall_front.t", "drywall_front.eps_r") + (("drywall_front.tan_d",) if fit_tan else ())
    if opts is not None and getattr(opts, "band_offset", False):
        s0 = s0 + ("band_offset",)      # E024b: the front echo pins the per-band phase centre
    if opts is not None and getattr(opts, "facing", False):
        s0 = s0 + ("facing.t", "facing.eps_r")   # E024b: facings shape the front echo too
    if opts is not None and getattr(opts, "rough", False):
        s0 = s0 + ("rough",)
    s1 = ("insulation.t", "insulation.eps_r", "drywall_back.t") + (("insulation.tan_d",) if fit_tan else ())
    return [I.Stage(s0, I.Gate(d0 - 0.03, d0 + 0.06),
                    lattice={"drywall_front.t": off(4), "drywall_front.eps_r": list(e0)}),
            I.Stage(s1, I.Gate(d0 + 0.06, d0 + 0.25),
                    lattice={"insulation.t": off(6), "insulation.eps_r": list(e1)})]


def _stage_model(model, st):
    m = model.copy()
    for n, _, p in m.items():
        p.free = (n in st.free) or (p.tie is not None and p.tie in st.free)
    return m


def _restore_free(m, all_free):
    for n, _, p in m.items():
        p.free = n in all_free
    return m


def lattice_stage(m, f, y, sigma, st, opts, starts=None):
    """LM from every lattice point (or the given start models) on the search frequencies, then
    re-converge the n_polish best distinct basins on all frequencies."""
    fs, ys = _subset(f, y, opts.search_step)
    per = phase_periods(m, f)
    if starts is not None and len(starts) == 0:      # pencil found nothing usable: regular lattice
        starts = None
    if starts is None:
        starts = []
        keys = list(st.lattice)
        for combo in itertools.product(*[st.lattice[k] for k in keys]):
            s = m.copy()
            vals = dict(zip(s.free_names(), s.get()))
            for k, v in zip(keys, combo):
                vals[k] = vals[k] + v * per[k] if k in per else v
            lo, hi = s.bounds()
            x = np.clip([vals[n] for n in s.free_names()], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
            s.set(x)
            starts.append(s)
    cands = []
    for s in starts:
        if st.profile_standoff and "standoff" in s.free_names() and "standoff" not in (st.lattice or {}):
            s.standoff.value = I.best_standoff(s, fs, ys, sigma, st.gate)
        mo, c = lm(s, fs, ys, sigma, gate=st.gate, atol=opts.search_atol, coords=opts.coords)
        if np.isfinite(c):
            cands.append((c, mo))
    if opts.search_step == 1 and opts.search_atol == 0:
        c, mo = min(cands, key=lambda t: t[0])
        return mo, c, len(starts)
    best = None
    for c, mo in _distinct(cands, m.free_names(), f, opts.n_polish):
        mp, cp = lm(mo, f, y, sigma, gate=st.gate, coords=opts.coords)
        if best is None or cp < best[1]:
            best = (mp, cp)
    return best[0], best[1], len(starts)


# ============================================================================ hops / ambiguity
def _shift(model, f, moves):
    per = phase_periods(model, f)
    t = model.copy()
    vals = dict(zip(t.free_names(), t.get()))
    lo, hi = t.bounds()
    names = t.free_names()
    for nm, h in moves:
        vals[nm] += h * per[nm]
        i = names.index(nm)
        if not (lo[i] < vals[nm] < hi[i]):         # only the shifted parameters are checked
            return None
    t.set(np.array([vals[n] for n in names]))
    return t


def _search_then_polish(trial, f, y, sigma, opts, ref_cost_coarse):
    """LM a hop candidate on the search frequencies; re-converge on all frequencies only if it
    is competitive there. Returns (model, full cost) or (None, inf)."""
    if opts.search_step == 1 and opts.search_atol == 0:
        return lm(trial, f, y, sigma, coords=opts.coords)
    fs, ys = _subset(f, y, opts.search_step)
    mo, c = lm(trial, fs, ys, sigma, atol=opts.search_atol, coords=opts.coords)
    # chi2 differences between basins scale ~ with the number of frequencies, so a coarse
    # dchi2 < 4 corresponds to a full-data dchi2 of about 4*step: a generous filter
    if c > ref_cost_coarse + 4.0:
        return None, np.inf
    return lm(mo, f, y, sigma, coords=opts.coords)


def cycle_hop(model, f, y, sigma, opts, max_rounds=6, hops=None):
    hops = opts.hops if hops is None else hops
    best = model
    best_cost = cost_of(best, f, y, sigma)
    fs, ys = _subset(f, y, opts.search_step)
    for rnd in range(max_rounds):
        improved = False
        per = phase_periods(best, f)
        names = list(per)
        th = [n for n in names if n != "standoff"]
        moves = [((a, h),) for a in names for h in hops]
        moves += [((a, h), (b, -h)) for a, b in zip(th[:-1], th[1:]) for h in (1, -1)]
        moves += [(("standoff", h), (th[0], -h)) for h in (1, -1)] if th and "standoff" in per else []
        ref_c = cost_of(best, fs, ys, sigma)
        for mv in moves:
            trial = _shift(best, f, mv)
            if trial is None:
                continue
            mo, c = _search_then_polish(trial, f, y, sigma, opts, ref_c)
            if c < best_cost * (1 - 1e-9):
                best, best_cost, improved = mo, c, True
                ref_c = cost_of(best, fs, ys, sigma)
        if not improved:
            break
    return best, best_cost


def ambiguity_scan(model, f, y, sigma, opts, ks=None, dchi2=4.0):
    ks = opts.amb_ks if ks is None else ks
    """As wall_inverse.ambiguity_scan, returning plausible basin summaries."""
    best, best_cost = model, cost_of(model, f, y, sigma)
    fs, ys = _subset(f, y, opts.search_step)
    ref_c = cost_of(best, fs, ys, sigma)
    per = phase_periods(model, f)
    names = [n for n in per if n != "standoff"]
    basins = [(model.summary(), best_cost)]
    for n in names:
        for k in ks:
            trial = _shift(model, f, [(n, k)])
            if trial is None:
                continue
            mo, c = _search_then_polish(trial, f, y, sigma, opts, ref_c)
            if mo is None:
                continue
            basins.append((mo.summary(), c))
            if c < best_cost:
                best, best_cost = mo, c
    plaus = sorted([(s, c) for s, c in basins if c <= best_cost + dchi2], key=lambda t: t[1])
    return best, best_cost, plaus


def posterior_scan(model, f, y, sigma, opts, joint=3, single=(-2, -1, 1, 2), mass=0.95, keep=30.0):
    """E016: enumerate phase basins and report a calibrated credible set.
    Basins: joint shifts (a, b) in [-joint, joint]^2 of (insulation.t, drywall_back.t), plus single
    shifts of drywall_front.t and standoff, each re-optimized (coarse search, then all frequencies).
    Basins are de-duplicated by their integer cycle offsets from the incumbent (optical thickness /
    period). Posterior over basins (flat prior over basins, equal Laplace volumes):
        w_b = exp(-(C_b - C_min)/2),  C = chi2 + prior penalty.
    Returns the best model, its cost, and the smallest set of basins holding >= `mass` of the
    posterior, as [(summary, C)], plus the posterior of the best basin."""
    lam2 = C0 / I.carrier(f) / 2
    fs, ys = _subset(f, y, opts.search_step)
    best, best_cost = model, cost_of(model, f, y, sigma)

    def key(m):
        v, v0 = m.summary(), model.summary()
        k = []
        for n in ("standoff",) + tuple(f"{L}.t" for L in ("drywall_front", "insulation", "drywall_back")):
            if n == "standoff":
                k.append(int(np.round((v[n] - v0[n]) / lam2)))
            else:
                L = model.layers[[x.name for x in model.layers].index(n[:-2])]
                Lm = m.layers[[x.name for x in m.layers].index(n[:-2])]
                tau = Coords._n(Lm).real * Lm.t.value
                tau0 = Coords._n(L).real * L.t.value
                k.append(int(np.round((tau - tau0) / lam2)))
        return tuple(k)

    basins = {key(model): (model, best_cost)}
    moves = [[("insulation.t", a), ("drywall_back.t", b)] for a in range(-joint, joint + 1)
             for b in range(-joint, joint + 1) if (a, b) != (0, 0)]
    moves += [[("drywall_front.t", k)] for k in single] + [[("standoff", k)] for k in (-1, 1)]
    ref_c = cost_of(model, fs, ys, sigma)
    for mv in moves:
        mv = [(n, k) for n, k in mv if k != 0]
        trial = _shift(model, f, mv)
        if trial is None:
            continue
        mo, c = _search_then_polish(trial, f, y, sigma, opts, ref_c + keep / max(opts.search_step, 1))
        if mo is None:
            continue
        k = key(mo)
        if k not in basins or c < basins[k][1]:
            basins[k] = (mo, c)
        if c < best_cost:
            best, best_cost = mo, c
    items = sorted(basins.values(), key=lambda t: t[1])
    C = np.array([c for _, c in items])
    w = np.exp(-(C - C.min()) / 2)
    w /= w.sum()
    n_keep = int(np.searchsorted(np.cumsum(w), mass) + 1)
    cred = [(m.summary(), c) for m, c in items[:n_keep]]
    info = dict(p_best_basin=float(w[0]), n_basins=len(items), n_plausible=len(cred),
                basin_posterior=w[:n_keep].tolist())
    return best, best_cost, cred, info


# ============================================================================ driver
def invert(f, yc, sigma, d0, opts: Opts = Opts()):
    """Layer stripping + joint LM + hops + plausible set, for the three-layer stud wall.
    With priors, all comparisons use the posterior cost chi2 + prior penalty; the returned cost
    is the data chi2 alone (for the chi2/n noise test)."""
    global _PRIORS
    _PRIORS = (GaussianPrior(tuple(opts.priors)),) if opts.priors else ()
    try:
        model, cost, plaus, info = _invert(f, yc, sigma, d0, opts)
        if opts.beam is not None and opts.beam_K_final > opts.beam_K:     # E023: final re-polish
            model.beam_K = opts.beam_K_final
            model, cost = lm(model, f, yc, sigma, coords=opts.coords)
            info["beam_K_final"] = opts.beam_K_final
    finally:
        pri = _PRIORS
        _PRIORS = ()
    if pri:
        info["prior_penalty"] = pri[0].penalty(model)
        cost = cost - info["prior_penalty"]
    return model, cost, plaus, info


def _invert(f, yc, sigma, d0, opts: Opts):
    import scenarios as S
    free = ("t", "eps_r", "tan_d") if opts.fit_tan else ("t", "eps_r")
    model = S.stud_wall_model(free=free, standoff_guess=d0, standoff_halfwidth=opts.standoff_halfwidth)
    model.law = opts.law
    model.min_cells = opts.min_cells
    if opts.beam is not None:
        model.beam, model.beam_K, model.beam_dref = tuple(opts.beam), opts.beam_K, float(d0)
    if opts.facing:
        model.facing = I.LayerSpec("facing", I.P(3e-4, 5e-5, 1.0e-3, True), I.P(2.2, 1.2, 4.0, True),
                                   I.P(0.05, 0.0, 0.3, True))
        model.facing_on = ("drywall_front", "drywall_back")
    if opts.rough:
        model.rough = I.P(5e-5, 1e-6, 4e-4, True)
    if opts.band_offset:
        model.band_offset = I.P(0.0, -1e-3, 1e-3, True)
    if opts.gap:
        model.gap = I.P(2e-3, 3e-4, 1e-2, True)
    if opts.chain is not None:
        model.chain = opts.chain
    if opts.pinf_free:
        model.layers[0].pinf = I.P(0.3, 0.0, 0.9, True)
        model.layers[2].pinf = I.P(0.3, 0.0, 0.9, True, tie="drywall_front.pinf")
        model.sync_ties()
    all_free = {n for n, _, p in model.items() if p.free}
    stages = stud_stages(d0, opts.fit_tan, opts)
    info = {"starts": []}
    pencil_starts = None
    if opts.init == "pencil":
        import pencil as PC
        pencil_starts = PC.stage_starts(model, f, yc, sigma, d0, stages)
        info["pencil"] = pencil_starts["estimate"]
    for k, st in enumerate(stages):
        m = _stage_model(model, st)
        starts = pencil_starts["stages"][k](m) if pencil_starts else None
        best, c, n = lattice_stage(m, f, yc, sigma, st, opts, starts)
        info["starts"].append(n)
        model = _restore_free(best, all_free)
    model, cost = lm(model, f, yc, sigma, coords=opts.coords)
    if opts.ambiguity == "posterior":
        model, cost = cycle_hop(model, f, yc, sigma, opts)
        model, cost, plaus, ainfo = posterior_scan(model, f, yc, sigma, opts)
        info.update(ainfo)
        return model, cost, plaus, info
    if opts.ambiguity == "scan":
        model, cost = cycle_hop(model, f, yc, sigma, opts)
        model, cost, plaus = ambiguity_scan(model, f, yc, sigma, opts)
    else:
        import ambiguity as AM
        model, cost, plaus, ainfo = AM.resolve(model, f, yc, sigma, opts)
        info.update(ainfo)
    return model, cost, plaus, info

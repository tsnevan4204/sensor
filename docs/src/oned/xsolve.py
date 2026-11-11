"""
E053: multi-angle known-N solver in parameter-vector space (same coordinates as wallsolve.gamma_batch).

Forward: exact TE plane-wave reflection of the layered wall at incidence angle th,
    kz_i = sqrt(n_i^2 - sin^2 th) (decaying branch), r_i = (kz_{i-1} - kz_i)/(kz_{i-1} + kz_i),
    e_i = exp(-2j k0 kz_i t_i), air path exp(-2j k0 cos(th) d); th = 0 reproduces gamma_batch exactly.
A measurement is a list of 'views' (f, y, sigma, th). Search: parallel tempering (pt.py moves) with the
energy summed over views. Polish: bounded least squares (scipy, trust-region reflective) with the
Jacobian from ONE batched forward call per view (central differences, D+... rows). Error bars: Fisher
from the same Jacobian (delta method for physical thickness). Alternatives: every polished answer with a
different structure within AMBIG_DCHI2.
Physics of the gain: a layer's delay is 2 k0 t sqrt(n^2 - sin^2 th): from several angles, n and t
separate (velocity analysis), which a single normal-incidence view cannot do for low-contrast layers.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
import wallsolve as WS
import stack as ST
import pt as PT

C0, F_REF = WS.C0, WS.F_REF


def gamma_ang(X, f, N, back, th):
    P = X.shape[0]
    k0 = 2 * np.pi * f / C0
    th = np.asarray(th, float)
    if th.ndim == 1:
        th = th[:, None]                               # E062b: one angle per row (P, 1)
    s2 = np.sin(th) ** 2
    lg = -np.log(f / F_REF)[None, :]
    NG = N - 1 + (1 if back == "metal" else 0)
    kz, ts = [], []

    def branch(z):
        return np.where(z.imag > 0, -z, z)
    for i in range(N):
        lt, la, lb, u = (X[:, 1 + 5 * i + j][:, None] for j in range(4))
        er, td, p = 1 + np.exp(la), np.exp(lb), 1 - np.exp(-u)
        a = 2 / np.pi * np.arctan(td / (1 - p))
        h = er * (p + (1 - p - 1j * td) * np.exp(a * lg))
        kz.append(branch(np.sqrt(h - s2 + 0j)))
        ts.append(np.exp(lt) / np.sqrt(er))
        if i < NG:
            g = np.maximum(X[:, 1 + 5 * N + i], 0.0)[:, None]
            kz.append(np.broadcast_to(np.sqrt(1 - s2) + 0j, (P, f.size)).copy()); ts.append(g)
    kb = branch(np.sqrt((ST.METAL_EPS if back == "metal" else 1.0) - s2 + 0j)) * np.ones((1, f.size))
    k_air = np.sqrt(1 - s2)
    G = (kz[-1] - kb) / (kz[-1] + kb)
    for i in range(len(kz) - 1, -1, -1):
        kp = kz[i - 1] if i > 0 else k_air
        e = np.exp(-2j * k0[None, :] * kz[i] * ts[i])
        r = (kp - kz[i]) / (kp + kz[i])
        G = (r + G * e) / (1 + r * G * e)
    return G * np.exp(-2j * k0[None, :] * k_air * X[:, :1])


def gamma_stack_ang(f, layers, back, th):
    """World (truth) side: exact TE reflection of (t, eps callable) layers."""
    k0 = 2 * np.pi * f / C0
    s2 = np.sin(th) ** 2
    br = lambda z: np.where(z.imag > 0, -z, z)
    kz = [np.sqrt(1 - s2) * np.ones_like(f, complex)] + [br(np.sqrt(e(f) - s2 + 0j)) for _, e in layers]
    kb = br(np.sqrt((ST.METAL_EPS if back == "metal" else 1.0) - s2 + 0j)) * np.ones_like(f)
    G = (kz[-1] - kb) / (kz[-1] + kb)
    for i in range(len(layers), 0, -1):
        e = np.exp(-2j * k0 * kz[i] * layers[i - 1][0])
        r = (kz[i - 1] - kz[i]) / (kz[i - 1] + kz[i])
        G = (r + G * e) / (1 + r * G * e)
    return G


def make_views(seed, N, angles_deg=(0.0,), snr_boost_db=0.0, pos_err=0.0, ang_err_deg=0.0):
    """Same wall as wallsolve.make_problem(seed, N); one noisy measurement per angle.
    E062: the TRUE sensor standoff of each view is d + N(0, pos_err) and its true angle the nominal
    + N(0, ang_err_deg); the returned views carry only the NOMINAL angle (what the solver knows)."""
    _, _, _, tr = WS.make_problem(seed, N)
    views = []
    er = np.random.default_rng(seed + 31337)
    for ai, adeg in enumerate(angles_deg):
        th_nom = np.radians(adeg)
        th = th_nom + np.radians(ang_err_deg) * er.standard_normal() if ai > 0 else th_nom
        dk = tr["standoff"] + (pos_err * er.standard_normal() if ai > 0 else 0.0)
        fs, ys, ss = [], [], []
        for k, ((a, b), sn) in enumerate(zip(WS.BANDS3, WS.SNR3)):
            f = np.linspace(a, b, WS.NPTS)
            G = gamma_stack_ang(f, tr["_layers"], tr["back"], th) * np.exp(-2j * 2 * np.pi * f / C0 * dk * np.cos(th))
            sig = 10 ** (-(sn + snr_boost_db) / 20) / np.sqrt(2)
            nr = np.random.default_rng(seed + 7919 * (k + 1) + 104729 * ai)
            ys.append(G + sig * (nr.standard_normal(f.size) + 1j * nr.standard_normal(f.size)))
            fs.append(f); ss.append(np.full(f.size, sig))
        views.append((np.concatenate(fs), np.concatenate(ys), np.concatenate(ss), th_nom))
    return views, tr


PT_SIGMA_FLOOR = True
VIEW_OFFSETS = False     # E062: each view (k >= 1) gets its own standoff offset (sensor position unknown)
OFF_TOL = 3e-3
ANG_TOL = np.radians(3.0)


def view_X(X, k, DL, th=0.0):
    """Parameter rows and angles for view k: view k >= 1 has standoff d + offset_k and angle
    th_nominal + dth_k (extra coordinates DL + (k-1) and DL + V + (k-1), V = #views - 1)."""
    if X.shape[1] <= DL or k == 0:
        return X, th
    V = (X.shape[1] - DL) // 2
    Y = X.copy()
    tk = th + X[:, DL + V + k - 1]
    # E062d: the offset is in AIR-PATH units: d_k cos(th_k) = d cos(th_nominal) + offset_k, so an angle
    # change no longer drags the large air phase with it (the (offset, angle) valley is decoupled)
    Y[:, 0] = (X[:, 0] * np.cos(th) + X[:, DL + k - 1]) / np.cos(tk)
    return Y, tk


class MultiProblem(PT.Problem):
    """pt.Problem whose energy sums the chi2 of every view (each tempered on f < FMAX, SUB per band)."""
    def __init__(self, views, N, back, d_guess, d_tol=3e-3):
        f, y, s, th = views[0]
        super().__init__(f, y, s, N, back, d_guess, d_tol)
        self.views = []
        for (f, y, s, th) in views:
            q = PT.Problem(f, y, s, N, back, d_guess, d_tol)
            # E061d: temper never colder than the nominal design SNR (30 dB low band, 40 dB above):
            # extra SNR (chirp averaging) sharpens the landscape; the LS polish uses the true sigma
            fl = np.where(q.f < 1e9, 10 ** (-30 / 20), 10 ** (-40 / 20)) / np.sqrt(2) if PT_SIGMA_FLOOR else 0.0
            self.views.append((q.f, q.y, np.maximum(q.s, fl), th))
        if VIEW_OFFSETS and len(views) > 1:
            V = len(views) - 1
            self.lo = np.concatenate([self.lo, np.full(V, -OFF_TOL), np.full(V, -ANG_TOL)])
            self.hi = np.concatenate([self.hi, np.full(V, OFF_TOL), np.full(V, ANG_TOL)])
            self.D = self.lo.size
            self.off_step = np.concatenate([np.full(V, 0.15e-3), np.full(V, np.radians(0.1))])

    def energy(self, X):
        chi = np.zeros(X.shape[0])
        for k, (f, y, s, th) in enumerate(self.views):
            Xk, tk = view_X(X, k, self.DL, th)
            G = gamma_ang(Xk, f, self.N, self.back, tk)
            chi += np.sum(np.abs((G - y[None, :]) / s[None, :]) ** 2, axis=1)
        pen = WS._prior_batch(X, self.N, self.lib)
        for i in range(self.N):
            lt, la = X[:, 1 + 5 * i], X[:, 2 + 5 * i]
            er = 1 + np.exp(la)
            k = np.argmin(np.abs(np.log(er)[:, None] - np.log(self.cer)[None, :]), axis=1)
            t = np.exp(lt) / np.sqrt(er)
            lo, hi = PT.T_WIDEN[0] * self.trng[k, 0], PT.T_WIDEN[1] * self.trng[k, 1]
            pen += (np.maximum(0, np.log(lo / t)) ** 2 + np.maximum(0, np.log(t / hi)) ** 2) / 0.05 ** 2
        out = 0.5 * (chi + pen)
        bad = np.any(X < self.lo[None, :] - 1e-12, axis=1) | np.any(X > self.hi[None, :] + 1e-12, axis=1)
        return np.where(np.isfinite(out) & ~bad, out, np.inf), chi


def free_mask(N, back):
    lo, hi, NG = WS.layout(N, back)
    m = np.ones(lo.size, bool)
    m[-1] = False                                       # water temperature (no water)
    for i in range(N):
        m[5 + 5 * i] = False                            # water coordinate c
    return m


class Polisher:
    """Bounded least squares on all views (all frequencies) + library prior as residuals."""
    def __init__(self, views, N, back, d_guess, d_tol=3e-3):
        self.views, self.N, self.back = views, N, back
        self.lo, self.hi, self.NG = WS.layout(N, back)
        self.lo[0], self.hi[0] = d_guess - d_tol, d_guess + d_tol
        self.m = free_mask(N, back)
        self.DL = self.lo.size
        if VIEW_OFFSETS and len(views) > 1:
            V = len(views) - 1
            self.lo = np.concatenate([self.lo, np.full(V, -OFF_TOL), np.full(V, -ANG_TOL)])
            self.hi = np.concatenate([self.hi, np.full(V, OFF_TOL), np.full(V, ANG_TOL)])
            self.m = np.concatenate([self.m, np.ones(2 * V, bool)])
        self.lib = WS._lib_arrays()
        ps = p_stats()
        self.pstat = [ps[nm] for nm in ST.library_stats()]

    def full(self, xf, x0):
        X = np.repeat(x0[None, :], xf.shape[0], 0)
        X[:, self.m] = xf
        return X

    def res_batch(self, X):
        rs = []
        for k, (f, y, s, th) in enumerate(self.views):
            Xk, tk = view_X(X, k, self.DL, th)
            G = gamma_ang(Xk, f, self.N, self.back, tk)
            r = (G - y[None, :]) / s[None, :]
            rs += [r.real, r.imag]
        pen = self.prior(X)
        rs.append(np.sqrt(np.maximum(pen, 0) + 1e-12)[:, None])
        return np.concatenate(rs, axis=1)

    force = None                                   # {layer: class index}: that layer's prior is class c

    def prior(self, X):
        if not self.force:
            return WS._prior_batch(X, self.N, self.lib)
        mus, Ls = self.lib
        pen = np.zeros(X.shape[0])
        for i in range(self.N):
            la, lb = X[:, 2 + 5 * i], X[:, 3 + 5 * i]
            v = np.stack([np.log(1 + np.exp(la)), lb / np.log(10)], axis=1)
            d = v[:, None, :] - mus[None, :, :]
            m = np.sum(np.einsum("pci,cij->pcj", d, Ls) ** 2, axis=2)
            if i in self.force:
                pen += m[:, self.force[i]]
            else:
                pen += np.min(m, axis=1)
        return pen

    def polish(self, x0, max_nfev=60):
        x0 = np.clip(x0, self.lo, self.hi)
        lo, hi = self.lo[self.m], self.hi[self.m] + 1e-12
        xf0 = np.clip(x0[self.m], lo, hi - 1e-12)

        def fun(xf):
            return self.res_batch(self.full(xf[None, :], x0))[0]

        def jac(xf):
            h = 1e-6 * np.maximum(np.abs(xf), 1e-3)
            P = np.vstack([xf + np.diag(h), xf - np.diag(h)])
            R = self.res_batch(self.full(P, x0))
            n = xf.size
            return ((R[:n] - R[n:]) / (2 * h[:, None])).T
        sol = least_squares(fun, xf0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev)
        x = x0.copy(); x[self.m] = sol.x
        return x, self.chi2(x), sol

    def chi2(self, x):
        c = 0.0
        for k, (f, y, s, th) in enumerate(self.views):
            Xk, tk = view_X(x[None, :], k, self.DL, th)
            G = gamma_ang(Xk, f, self.N, self.back, tk)[0]
            c += float(np.sum(np.abs((G - y) / s) ** 2))
        return c

    def crb_t(self, x):
        """1-sigma error of each solid layer's physical thickness (Fisher, data only)."""
        xf = x[self.m]
        h = 1e-6 * np.maximum(np.abs(xf), 1e-3)
        P = np.vstack([xf + np.diag(h), xf - np.diag(h)])
        R = self.res_batch(self.full(P, x))[:, :-1]
        n = xf.size
        J = ((R[:n] - R[n:]) / (2 * h[:, None])).T
        C = np.linalg.pinv(J.T @ J)
        idx = np.flatnonzero(self.m)
        out = []
        for i in range(self.N):
            jl, ja = list(idx).index(1 + 5 * i), list(idx).index(2 + 5 * i)
            er = 1 + np.exp(x[2 + 5 * i]); t = np.exp(x[1 + 5 * i]) / np.sqrt(er)
            g = np.zeros(n); g[jl] = t; g[ja] = -0.5 * t * (er - 1) / er
            out.append(float(np.sqrt(max(g @ C @ g, 0))))
        return out


def labels_approx(pol, x):
    """Library label per solid layer: nearest class in (ln eps', log10 tan d) by the Mahalanobis
    distance with covariance C_class + C_measurement (Fisher). A loss that the data do not measure
    (thin foam) then does not decide the class."""
    lib = ST.library_stats()
    xf = x[pol.m]
    h = 1e-6 * np.maximum(np.abs(xf), 1e-3)
    P = np.vstack([xf + np.diag(h), xf - np.diag(h)])
    R = pol.res_batch(pol.full(P, x))[:, :-1]
    n = xf.size
    J = ((R[:n] - R[n:]) / (2 * h[:, None])).T
    C = np.linalg.pinv(J.T @ J)
    idx = list(np.flatnonzero(pol.m))
    out = []
    for i in range(pol.N):
        ja, jb = idx.index(2 + 5 * i), idx.index(3 + 5 * i)
        la, lb = x[2 + 5 * i], x[3 + 5 * i]
        er = 1 + np.exp(la)
        v = np.array([np.log(er), lb / np.log(10)])
        A = np.diag([(er - 1) / er, 1 / np.log(10)])
        Cm = A @ C[np.ix_([ja, jb], [ja, jb])] @ A.T
        lp = {}
        for nm, (mu, Cc) in lib.items():
            S = Cc + Cm
            dv = v - mu
            lp[nm] = -0.5 * float(dv @ np.linalg.solve(S, dv)) - 0.5 * float(np.log(np.linalg.det(S)))
        mx = max(lp.values())
        pr = {nm: np.exp(v_ - mx) for nm, v_ in lp.items()}
        Z = sum(pr.values())
        pr = {nm: v_ / Z for nm, v_ in pr.items()}
        out.append(sorted(((p_, nm) for nm, p_ in pr.items()), reverse=True))
    return out


VIS_DCHI2 = 25.0    # layers behind which the data change by less than this are 'not observed'
LABEL_P = 0.10
LABEL_DSCORE = 9.0   # E060d: label set = classes within this evidence-score difference      # every class with posterior >= 10 % is reported (equal priors over classes)


_PSTAT = None


def p_stats(n=400, seed=24680):
    """E060c: mean / sd of the dispersion shape p per library class (generator variability)."""
    global _PSTAT
    if _PSTAT is None:
        rng = np.random.default_rng(seed)
        out = {}
        for nm in ST.library_stats():
            ps = [min(ST.material_law(nm, rng)[1].get("p", 0.0), 0.999) for _ in range(n)]
            out[nm] = (float(np.mean(ps)), max(float(np.std(ps)), 0.03))
        _PSTAT = out
    return _PSTAT


def class_evidence(pol, x, top=3, max_nfev=120):
    """E061c: posterior class probabilities per solid layer by Laplace evidence: refit with that
    layer's prior FORCED to class c (others: nearest class), score_c = chi2_c + Mahalanobis_c +
    ln det C_c; P(c) ~ exp(-score_c / 2). Candidates: the `top` classes nearest in the data-only
    estimate. Replaces reading the label off a fit that one class prior already pulled."""
    lib = ST.library_stats()
    names = list(lib.keys())
    mus, Ls = pol.lib
    logdet = {nm: float(np.log(np.linalg.det(C))) for nm, (mu, C) in lib.items()}
    tab = {nm: (er, td, p) for nm, er, td, p in WS.class_table()}
    out = []
    approx = labels_approx(pol, x)
    for i in range(pol.N):
        cand = [nm for _, nm in approx[i][:top]]
        sc = {}
        for nm in cand:
            c = names.index(nm)
            pol.force = {i: c}
            try:
                # two starts: the current fit, and this class's typical (eps', tan d, p) at the
                # same optical thickness (the dispersion shape p is not in the 2-D prior)
                er0, td0, p0 = tab[nm]
                starts = [x]
                for u0 in sorted({0.0, 3.0, 7.0, float(-np.log(max(1 - min(p0, 0.999), 1e-3)))}):
                    xc = x.copy()
                    xc[2 + 5 * i], xc[3 + 5 * i] = np.log(er0 - 1), np.log(td0)
                    xc[4 + 5 * i] = u0
                    # quick screen: keep the 2 best class-centre starts by chi2
                    starts.append(xc)
                ch = [pol.chi2(z) for z in starts[1:]]
                starts = [x] + [starts[1 + j] for j in np.argsort(ch)[:2]]
                best = None
                for x0 in starts:
                    xp, chi, _ = pol.polish(x0, max_nfev=max_nfev)
                    la, lb = xp[2 + 5 * i], xp[3 + 5 * i]
                    v = np.array([np.log(1 + np.exp(la)), lb / np.log(10)])
                    m = float(np.sum(((v - mus[c]) @ Ls[c]) ** 2))
                    if best is None or chi + m < best:
                        best = chi + m
                sc[nm] = best + logdet[nm]
            finally:
                pol.force = None
        mn = min(sc.values())
        pr = {nm: np.exp(-0.5 * (v_ - mn)) for nm, v_ in sc.items()}
        Z = sum(pr.values())
        # E060d: report every class whose evidence score is within LABEL_DSCORE of the best: score
        # differences of ~10 arise from noise when ~20 correlated parameters are fitted (thin-layer
        # loss vs dispersion shape), so exp(-d/2) probabilities alone overstate the confidence
        out.append(sorted(((p_ / Z, nm) for nm, p_ in pr.items() if sc[nm] - mn <= LABEL_DSCORE), reverse=True))
    return out


LAM2_MM = C0 / (2 * 62e9)          # mmWave half-wavelength: period of a weak interface's phase


def cycle_scan(pol, x, ks=(-3, -2, -1, 1, 2, 3), max_nfev=40):
    """E054f: integer-ambiguity scan of every interface at the mmWave period: move it by k*lambda/2
    (thickness transferred between the two adjacent media, optical sum kept), re-polish, keep the best."""
    N, D = pol.N, x.size
    best_x, best_c = x, pol.chi2(x)
    pairs = []
    for i in range(N - 1):
        g = 1 + 5 * N + i
        if i < pol.NG and x[g] > 0:
            pairs.append(("sg", i, g))                  # solid i <-> gap i
            pairs.append(("gs", i + 1, g))              # gap i <-> solid i+1
        else:
            pairs.append(("ss", i, i + 1))
    if pol.NG > N - 1 and x[1 + 5 * N + N - 1] > 0:
        pairs.append(("sg", N - 1, 1 + 5 * N + N - 1))
    # E062c: per-view standoff offsets (sweeping sensor) also have mmWave cycles: 1 mm = 2.5 rad at 62 GHz
    if x.size > pol.DL:
        V = (x.size - pol.DL) // 2
        pairs += [("off", pol.DL + v, None) for v in range(V)]
    for kind, a, b in pairs:
        for k in ks:
            y = best_x.copy()
            dl = k * LAM2_MM
            if kind == "off":
                y[a] += dl / 2          # air path: two-way 2*dl/2 = one mmWave half-cycle per k
            elif kind == "ss":
                ta, tb = np.exp(y[1 + 5 * a]), np.exp(y[1 + 5 * b])
                if ta + dl <= 1e-4 or tb - dl <= 1e-4:
                    continue
                y[1 + 5 * a], y[1 + 5 * b] = np.log(ta + dl), np.log(tb - dl)
            else:
                ta = np.exp(y[1 + 5 * a])
                if ta + dl <= 1e-4 or y[b] - dl <= 0:
                    continue
                y[1 + 5 * a] = np.log(ta + dl); y[b] -= dl
            if np.any(y < pol.lo) or np.any(y > pol.hi):
                continue
            yp, c, _ = pol.polish(y, max_nfev=max_nfev)
            if c < best_c - 1e-6:
                best_x, best_c = yp, c
    return best_x, best_c


def low_contrast_variants(x, N, pol, er_max=1.3):
    out = []
    for i in range(N):
        er = 1 + np.exp(x[2 + 5 * i])
        if er > er_max:
            continue
        t = np.exp(x[1 + 5 * i]) / np.sqrt(er)
        for gi in (i - 1, i):                      # gap in front of / behind layer i
            if gi < 0 or gi >= pol.NG:
                continue
            j = 1 + 5 * N + gi
            g = x[j]
            if g > 0:                              # merge the gap into the layer
                y = x.copy(); y[j] = -5e-3; y[1 + 5 * i] = np.log((t + g) * np.sqrt(er)); out.append(y)
            else:                                  # split: part of the layer becomes a gap
                for fr in (0.25, 0.5, 0.75):
                    gn = fr * t
                    if gn < 5e-3:
                        continue
                    y = x.copy(); y[j] = gn; y[1 + 5 * i] = np.log((t - gn) * np.sqrt(er)); out.append(y)
    return [np.clip(y, pol.lo, pol.hi) for y in out]


def signature(x, N, back):
    lay, slots = PT.x_to_lay(x, N, back)
    sol = tuple(round(L["t"] * 1e3) for (k, _), L in zip(slots, lay) if k == "solid")
    gp = tuple(round(L["t"] * 1e3) for (k, _), L in zip(slots, lay) if k == "gap")
    return (sol, gp, back)


def solve(views, N, d_guess, steps=4000, restarts=2, n_polish=10, seed=0, verbose=False, fast=None, init=None):
    fast = FAST if fast is None else fast
    if fast:                                           # E058 fast config: early stop, adaptive restart
        return _solve_fast(views, N, d_guess, n_polish, seed, verbose)
    t0 = time.time()
    cands = []
    for back in ("air", "metal"):
        mp = MultiProblem(views, N, back, d_guess)
        for rs in range(restarts):
            best, cand = PT.run(mp, steps=steps, seed=seed + 101 * rs,
                                init=None if init is None else init.get(back))
            cands += [(mp.energy(c[None, :])[0][0], c, back) for c in cand]
            if verbose:
                print(f"  PT {back} r{rs}: chi2n {best[2] / sum(2 * v[0].size for v in mp.views):.2f} {time.time() - t0:.0f}s", flush=True)
    return _finish(cands, views, N, d_guess, n_polish, t0)


def _finish(cands, views, N, d_guess, n_polish, t0):
    cands.sort(key=lambda c: c[0])
    pol = {b: Polisher(views, N, b, d_guess) for b in ("air", "metal")}
    done = []
    for E, x, back in cands[:n_polish]:
        xp, chi, _ = pol[back].polish(x)
        done.append((chi, xp, back))
    done.sort(key=lambda d: d[0])
    # E054f: mmWave integer-ambiguity scan on the two best polished answers
    for j in range(min(2, len(done))):
        c, xx, bk = done[j]
        xs, cs = cycle_scan(pol[bk], xx)
        done[j] = (cs, xs, bk)
    done.sort(key=lambda d: d[0])
    chi, x, back = done[0]
    # E061e: long final polish of the winner (high SNR needs more LS iterations to reach the noise floor)
    xl, cl, _ = pol[back].polish(x, max_nfev=400)
    if cl < chi:
        x, chi = xl, cl
    n_all = sum(2 * v[0].size for v in views)
    # E060b: explicit structural alternatives for low-contrast layers (foam/batt ~ air): merge an
    # adjacent open gap into the layer, or split the layer into layer + gap; a better one replaces the best
    for xa in low_contrast_variants(x, N, pol[back]):
        xp, c, _ = pol[back].polish(xa, max_nfev=120)
        done.append((c, xp, back))
        if c < chi:
            x, chi = xp, c
    alts, seen = [], {signature(x, N, back)}
    for c, xx, bk in done[1:]:
        sg = signature(xx, N, bk)
        if c <= chi + WS.AMBIG_DCHI2 and sg not in seen:
            seen.add(sg); alts.append(dict(chi2=c, solids=list(sg[0]), gaps=list(sg[1]), back=bk))
    crb = pol[back].crb_t(x)
    lay, slots = PT.x_to_lay(x, N, back)
    sol = [L for (k, _), L in zip(slots, lay) if k == "solid"]
    weak = [bool(e > 0.2 * L["t"]) for e, L in zip(crb, sol)]
    unobs = [i for i, L in enumerate(sol) if L["er"] < 1.03]
    labs = class_evidence(pol[back], x)
    # E061b: honesty checks for walls with fewer observable layers than the N the user gave
    #  (1) unresolved interface: adjacent solids (no gap) whose label sets overlap -> one interface unseen
    unresolved = []
    j = 0
    for (k, i), L in zip(slots, lay):
        pass
    sl = [i for (k, i) in slots if k == "solid"]
    for i in range(N - 1):
        gap_between = any(k == "gap" and gi == i for k, gi in slots)
        if not gap_between and set(nm for _, nm in labs[i]) & set(nm for _, nm in labs[i + 1]):
            unresolved.append(i)
    #  (2) visibility depth: everything behind solid i's back face replaced by air (metal backing kept
    #      only if it is the backing) changes chi2 by < VIS_DCHI2 -> layers i+1.. are not observed
    vis_from = None
    for i in range(N - 1):
        xa = x.copy()
        for q in range(i + 1, N):
            xa[2 + 5 * q] = np.log(1e-4); xa[3 + 5 * q] = np.log(1e-4)
        for g in range(i, pol[back].NG):
            xa[1 + 5 * N + g] = -5e-3
        if pol[back].chi2(xa) - chi < VIS_DCHI2:
            vis_from = i + 1
            break
    unobs = sorted(set(unobs) | set(range(vis_from, N) if vis_from is not None else []))
    weak = [w or (i in unresolved) or ((i - 1) in unresolved) for i, w in enumerate(weak)]
    return dict(x=x, back=back, chi2=chi, labels=labs, chi2n=chi / n_all, alarm=bool(chi > ST.chi2_alarm(n_all)),
                lay=lay, slots=slots, crb_t=crb, unresolved=unresolved, visible_layers=vis_from if vis_from is not None else N, weak=weak, unobserved=unobs, alternatives=alts,
                ambiguous=bool(alts), wall_s=time.time() - t0)


def _solve_fast(views, N, d_guess, n_polish, seed, verbose):
    """E058: one PT run per backing with early stopping on a coarser frequency subset; a second
    (independent) run only if the polished, cycle-scanned answer still raises the chi2 alarm."""
    t0 = time.time()
    old = PT.SUB
    PT.SUB = FAST_SUB
    try:
        probs = {b: MultiProblem(views, N, b, d_guess) for b in ("air", "metal")}
    finally:
        PT.SUB = old
    cands = []
    for rs in range(2):
        for b, mp in probs.items():
            best, cand = PT.run(mp, n_rep=FAST_REP, steps=4000, seed=seed + 101 * rs, patience=FAST_PATIENCE)
            cands += [(mp.energy(c[None, :])[0][0], c, b) for c in cand]
        res = _finish(list(cands), views, N, d_guess, n_polish, t0)
        if not res["alarm"]:
            break
    return res


FAST = False
FAST_SUB, FAST_REP, FAST_PATIENCE = 48, 6, 600


def score(res, tr, N, t_tol=(0.3e-3, 0.02), g_tol=(0.3e-3, 0.05)):
    lay, slots = res["lay"], res["slots"]
    sol = [L for (k, _), L in zip(slots, lay) if k == "solid"]
    gaps = [0.0] * len(tr["gaps"])
    for (k, i), L in zip(slots, lay):
        if k == "gap" and i < len(gaps):
            gaps[i] = L["t"]
    t_ok = [abs(L["t"] - T["t"]) <= max(t_tol[0], t_tol[1] * T["t"]) for L, T in zip(sol, tr["layers"])]
    g_ok = [abs(a - b) <= max(g_tol[0], g_tol[1] * b) for a, b in zip(gaps, tr["gaps"])]
    mat_ok = [T["name"] in [nm for _, nm in lab] for lab, T in zip(res["labels"], tr["layers"])]
    mat_top = [lab[0][1] == T["name"] for lab, T in zip(res["labels"], tr["layers"])]
    ok = all(t_ok) and all(g_ok) and res["back"] == tr["back"] and all(mat_ok)
    geo_ok = all(t_ok) and all(g_ok) and res["back"] == tr["back"]
    return dict(ok=bool(ok), geo_ok=bool(geo_ok), mat_top=mat_top, t_ok=t_ok, g_ok=g_ok, mat_ok=mat_ok, back_ok=res["back"] == tr["back"],
                t_err_mm=[1e3 * (L["t"] - T["t"]) for L, T in zip(sol, tr["layers"])])

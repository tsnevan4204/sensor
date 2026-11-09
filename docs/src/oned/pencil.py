"""
Matrix-pencil (shift-invariance) initializer for layered walls (E010). No learning, no starts.

Why it works. For piecewise-constant dry layers (eps independent of frequency), the exact
reflection is a sum over bounce paths,
    Gamma(f_m) = sum_p c_p exp(-j 2 pi f_m tau_p),   tau_p = 2 sum_i k_i Re(n_i) t_i / c  (complex if lossy),
so on the uniform grid f_m = f_0 + m df it is a sum of damped exponentials y_m = sum_p a_p z_p^m with
z_p = exp(-j 2 pi df tau_p) and a_p = c_p exp(-j 2 pi f_0 tau_p).
  1. Hankel matrix H_{ij} = y_{i+j} has rank = number of exponentials (above the noise).
     Forward-backward averaging [H, J conj(H) J] (Le Bastard et al. 2007: best for fully coherent
     echoes, i.e. one scan) doubles the snapshots.
  2. Signal subspace U_s (top M left singular vectors) is shift invariant: U_s[1:] = U_s[:-1] Phi,
     eig(Phi) = z_p (total least squares solve).
  3. Amplitudes a_p by linear least squares on the Vandermonde matrix.
  4. Map echoes to layers with the Fresnel recursion (normal incidence, multiples ignored):
       echo 1 at range d          -> standoff = d,  r_01 = c_1 -> n_1 = (1 - r_01)/(1 + r_01)
       echo 2 at d + n_1 t_1      -> t_1,  c_2 = (1 - r_01^2) r_12 -> n_2
       echo 3 at ... + n_2 t_2    -> t_2
The result seeds Levenberg-Marquardt with the exact model, which removes the bias of step 4
(multiples, merged echoes). Its error only decides how many phase basins must be tried.
"""
from __future__ import annotations
import numpy as np
import wall_toy as W

C0 = W.C0


def pencil(y, M, L=None, fb=True):
    """Return poles z, amplitudes a, normalized singular values."""
    N = y.size
    L = L or N // 2
    H = np.array([y[i:i + L + 1] for i in range(N - L)])          # (N-L) x (L+1)
    if fb:
        Hb = np.conj(H[::-1, ::-1])
        H = np.vstack([H, Hb])
    U, s, Vh = np.linalg.svd(H.T, full_matrices=False)              # columns of H.T are shifts in m
    Us = U[:, :M]
    z = np.linalg.eigvals(np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0])
    Z = np.vander(z, N, increasing=True).T
    a = np.linalg.lstsq(Z, y, rcond=None)[0]
    return z, a, s / s[0]


def echoes(f, y, sigma, r_lo, r_hi, M_max=6):
    """Echo ranges (one-way optical, m) and Fresnel-like amplitudes c_p, strongest-first order
    resolved into range order. Model order from the noise level of the Hankel spectrum."""
    df = f[1] - f[0]
    N = y.size
    L = N // 2
    _, _, s = pencil(y, 1, L)
    # largest singular value of a (2(N-L)) x (L+1) complex Gaussian noise matrix ~ sigma_c (sqrt(m)+sqrt(n))
    s_abs = np.linalg.svd(np.array([y[i:i + L + 1] for i in range(N - L)]), compute_uv=False)
    noise_edge = np.sqrt(2) * sigma * (np.sqrt(2 * (N - L)) + np.sqrt(L + 1))
    M = int(np.clip(np.sum(s_abs * np.sqrt(2) > 1.5 * noise_edge), 1, M_max))
    z, a, _ = pencil(y, M, L)
    tau = np.mod(-np.angle(z) / (2 * np.pi * df), 1 / df)
    r = C0 * tau / 2
    c = a * np.exp(2j * np.pi * f[0] * tau)                          # reflection coefficient at f_0 phase removed
    keep = (r > r_lo) & (r < r_hi)
    o = np.argsort(r[keep])
    return r[keep][o], c[keep][o], M


def wall_estimate(f, y, sigma, d0, n_layers=3, tan_nominal=(0.03, 0.01, 0.03)):
    """Map echoes to the standoff, each layer's optical thickness tau_i = Re(n_i) t_i (from echo
    delays: accurate), and eps_i (from echo amplitudes: rough). Amplitudes are corrected for the
    two-way attenuation exp(-2 k_c Im(n_j) t_j) of the layers above, using nominal loss tangents
    (an assumption; at 79 GHz one pass through 12.7 mm of drywall costs a factor ~e^-1)."""
    r, c, M = echoes(f, y, sigma, d0 - 0.05, d0 + 0.35)
    est = dict(M=M, ranges=r.tolist(), amps=np.abs(c).tolist())
    if r.size == 0:
        return est
    i0 = int(np.argmax(np.abs(c[r < d0 + 0.03]))) if np.any(r < d0 + 0.03) else 0
    r, c = r[i0:], c[i0:]
    kc = 2 * np.pi * f.mean() / C0
    est["standoff"] = float(r[0])
    rho = float(np.clip(-np.abs(c[0]), -0.9, -0.01))                  # air -> denser: r_01 < 0
    n_prev = (1 - rho) / (1 + rho)
    est["eps"] = [float(n_prev**2)]
    est["tau"] = []
    trans, att = 1 - rho**2, 1.0
    for i in range(1, min(len(r), n_layers + 1)):
        tau_i = float(r[i] - r[i - 1])
        est["tau"].append(tau_i)
        if i < len(c) and i < n_layers:
            tl = tan_nominal[i - 1] if i - 1 < len(tan_nominal) else 0.0
            att *= np.exp(-2 * kc * tau_i * tl / 2)                    # Im(n) t ~ Re(n) t tan_d / 2
            ci = c[i] * np.exp(-1j * np.angle(c[0])) * np.sign(rho)
            r_i = float(np.clip(ci.real / (trans * att), -0.9, 0.9))
            n_new = max(n_prev * (1 - r_i) / (1 + r_i), 1.0)
            est["eps"].append(float(n_new**2))
            trans *= (1 - r_i**2)
            n_prev = n_new
    return est


def stage_starts(model, f, y, sigma, d0, stages, width=None):
    """Start models for each stage of search.stud_stages. For every thickness whose optical
    thickness tau the pencil measured, the starts lie on the data's ridge tau = Re(n) t: for each
    eps start value (the pencil's amplitude estimate plus the stage's eps lattice values),
    t = tau / Re n(eps, tan_d), plus half-period offsets within +-width[name] periods. Thicknesses
    the pencil cannot see (the back drywall: its two faces merge) start at the model's value."""
    import itertools
    import wall_inverse as I
    w = dict(PENCIL_WIDTH) if width is None else width
    est = wall_estimate(f, y, sigma, d0)
    # Only layers in PENCIL_LAYERS are seeded from the pencil (E010 calibration: the front
    # drywall's tau is within 1.73 periods on all 50 calibration walls; the insulation echo merges
    # with the back-drywall echo 2 cm behind it, median error 2.1 periods, so it is not used).
    allo = [("drywall_front.t", "drywall_front.eps_r", "drywall_front.tan_d"),
            ("insulation.t", "insulation.eps_r", "insulation.tan_d")]
    tau = dict(zip([o[0] for o in allo], est.get("tau", [])))
    epsg = dict(zip([o[1] for o in allo], est.get("eps", [])))
    order = [o for o in allo if o[0] in PENCIL_LAYERS]

    def make(k):
        st = stages[k]

        def fn(m):
            lo, hi = m.bounds()
            clip = lambda x: np.clip(x, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
            names = m.free_names()
            if not any(tn in names for tn, _, _ in order):
                return None                             # stage untouched by the pencil: own lattice
            axes = []                                   # per thickness: list of (t, eps) pairs
            for tn, en, tdn in order:
                if tn not in names or tn not in st.lattice:
                    continue
                if tn not in tau:                       # pencil did not see it: stage lattice
                    axes.append([(tn, None, None, off) for off in st.lattice[tn]])
                    continue
                # starts on the ridge tau = Re(n) t: eps only moves the start along the ridge,
                # so the pencil's eps alone is used (calibration: front eps error <= 0.21)
                e_vals = [round(epsg[en], 4)] if (en in names and en in epsg) else [None]
                e_vals = [e for e in e_vals if e is None or 1.0 < e < 4.0]
                offs = np.arange(-2 * w[tn], 2 * w[tn] + 1) / 2
                axes.append([(tn, en, e, off) for e in e_vals for off in offs])
            starts = []
            for combo in itertools.product(*axes):
                s = m.copy()
                vals = dict(zip(names, s.get()))
                for tn, en, e, off in combo:            # eps first, then t on the tau ridge
                    if en is not None and e is not None:
                        vals[en] = e
                s.set(clip([vals[n] for n in names]))
                vals = dict(zip(names, s.get()))
                per = I._phase_periods(s, f)
                for tn, en, e, off in combo:
                    if en is None and e is None and tn not in tau:
                        vals[tn] = vals[tn] + off * per[tn]
                    else:
                        L = s.layers[[x.name for x in s.layers].index(tn[:-2])]
                        rn = np.sqrt(L.eps_r.value * (1 - 1j * L.tan_d.value)).real
                        vals[tn] = tau[tn] / rn + off * per[tn]
                s.set(clip([vals[n] for n in names]))
                if "standoff" in names and "standoff" in est:
                    s.standoff.value = float(np.clip(est["standoff"], s.standoff.lo, s.standoff.hi))
                starts.append(s)
            return starts
        return fn

    return {"estimate": est, "stages": [make(k) for k in range(len(stages))]}


# Half-width (in phase periods) of the start window around the pencil thickness. Set in E010 from
# the pencil error distribution on calibration seeds 1001-1050 (never on benchmark seeds 1-20).
PENCIL_WIDTH = {"drywall_front.t": 2}     # covers errors up to 2.25 periods (calibration max 1.73)
PENCIL_LAYERS = ("drywall_front.t",)

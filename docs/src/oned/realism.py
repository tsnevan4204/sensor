"""
Reality-gap generator (doc 12). A more physical "world" than wall_toy, with every effect a
switch, so that each assumption of the inversion model can be tested on its own. With all
switches off it reproduces benchmark.make_case data (gate: tests/test_realism.py).

The inversion model (wall_inverse + search) is NOT changed here: this file only makes data.

Effects (all frequency-domain, e^{+j w t}):
  dispersion    "const"  : eps = eps_r (1 - j tan_d) at every f (today's toy; violates Kramers-Kronig)
                "cq"     : Kjartansson constant-Q, eps(f) = eps_r (f/f_ref)^(-a) e^(-j pi a/2),
                           a = (2/pi) atan(tan_d). Exactly causal (fractional integrator), passive,
                           constant loss angle; eps' drifts as f^-a. Reference f_ref = 79 GHz.
                "itu"    : ITU-R P.2040-style power law (eps' = const, sigma = c f^d), with the
                           material's sigma scaled to match tan_d at 79 GHz; d per material.
  beam_w0       Gaussian antenna beam waist (m). Received co-pol signal = plane-wave spectrum
                integral  S = int W(k_rho) Gamma_co(f, k_rho) exp(-2j k_z d) k_rho dk_rho,
                W = exp(-(k_rho w0)^2/2) normalised, Gamma_co = (Gamma_TE + Gamma_TM)/2.
                None = plane wave (today).
  Hf, Hi        antenna feedback and leakage (Lambot et al. 2004 form):
                y = A [ Hi + S / (1 - Hf * S_local) ], S_local = S exp(+2j k0 d) ... see measure()
  plate_err     true plate distance minus the value used in calibration (m), per band
  rough_sigma   rms height of the front surface (m): front-face reflection times exp(-2 (k0 s)^2)
                (Kirchhoff/Ament specular attenuation)
  paper         drywall facing paper: (thickness m, eps_r, tan_d) on both faces of each sheet
  air_gap       air gap (m) between insulation and the back drywall (batt not filling the cavity)
  drift         (dB, rad) smooth gain/phase change of A(f) between plate scan and wall scan
  band_offset   phase-centre / standoff difference of the second band (separate antenna or a handheld
                unit moving between band sweeps); the model assumes one shared standoff
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import wall_toy as W
import scenarios as S

C0, ETA0 = W.C0, W.ETA0
MU0 = ETA0 / C0
EPS0 = 1 / (MU0 * C0**2)
F_REF = 79e9
ITU_D = dict(drywall=0.9395, fiberglass=1.075)   # sigma ~ f^d: ITU-R P.2040-3 Table 3 plasterboard; ceiling board (verified)


@dataclass
class Realism:
    dispersion: str = "const"
    beam_w0: float | None = None
    Hf: complex = 0.0
    Hi: complex = 0.0
    plate_err: float = 0.0
    rough_sigma: float = 0.0
    paper: tuple | None = None          # (t, eps_r, tan_d)
    air_gap: float = 0.0
    drift: tuple = (0.0, 0.0)
    band_offset: float = 0.0            # second band's phase-centre offset (m), unknown to the model
    fmcw: dict | None = None            # E031: FMCW chain {"S": slope Hz/s, "hpf": (f1, f2) Hz, "nonlin": peak Hz}
    field_drift: tuple | None = None    # E035: (sd gain dB, sd gain slope dB, sd phase rad, sd delay ps) of the
                                        # radar CHAIN between factory calibration and the field scan (per band, random)
    untracked: tuple = (0.0, 0.0)       # E035: (sd gain dB, sd phase rad) of the antenna/lens part (not seen by leakage)
    leak_ref: bool = False              # E035: correct the chain drift with the internal Tx->Rx leakage reference
    n_quad: int = 160                    # 48 was 6e-5 off for w0 = 1 cm (E023 check); 160: 6e-15


# ----------------------------------------------------------------------------- materials
def eps_material(eps_r, tan_d, f, kind, material):
    f = np.asarray(f, float)
    if kind == "const":
        return np.full(f.shape, eps_r * (1 - 1j * tan_d), complex)
    if kind == "cq":
        a = 2 / np.pi * np.arctan(tan_d)
        # eps_r (1 - j tan_d) at f_ref up to the factor 1/cos(pi a/2) ~ 1 + O(tan_d^2): normalise
        mag = eps_r * np.sqrt(1 + tan_d**2)
        return mag * (f / F_REF) ** (-a) * np.exp(-1j * np.pi * a / 2)
    if kind == "itucausal":
        # ITU-R P.2040 loss slope (eps'' ~ f^(d-1)) made causal: Jonscher eps_inf + C (jf/f_ref)^(-a'),
        # a' = max(1 - d, a_cq); eps_inf fixed by matching eps_r (1 - j tan_d) at f_ref
        a_cq = 2 / np.pi * np.arctan(tan_d)
        ap = max(1 - ITU_D[material], a_cq)
        pp = 1 - tan_d / np.tan(np.pi * ap / 2)
        g = (f / F_REF) ** (-ap)
        return eps_r * (pp + (1 - pp - 1j * tan_d) * g)
    if kind == "itu":
        d = ITU_D[material]
        sig_ref = tan_d * eps_r * 2 * np.pi * F_REF * EPS0
        sigma = sig_ref * (f / F_REF) ** d
        return eps_r - 1j * sigma / (2 * np.pi * f * EPS0)
    raise ValueError(kind)


def layers_from_truth(tv, R: Realism):
    """List of (thickness, eps(f)) from benchmark truth values, with paper and air gap."""
    def mat(er, td, name):
        return lambda f: eps_material(er, td, f, R.dispersion, name)
    dw = mat(tv["drywall_front.eps_r"], tv["drywall_front.tan_d"], "drywall")
    ins = mat(tv["insulation.eps_r"], tv["insulation.tan_d"], "fiberglass")

    def sheet(t):
        if R.paper is None:
            return [(t, dw)]
        tp, ep, tdp = R.paper
        pp = mat(ep, tdp, "drywall")
        return [(tp, pp), (t - 2 * tp, dw), (tp, pp)]
    L = sheet(tv["drywall_front.t"])
    L += [(tv["insulation.t"] - R.air_gap, ins)]
    if R.air_gap > 0:
        L += [(R.air_gap, lambda f: np.ones(np.shape(f), complex))]
    L += sheet(tv["drywall_back.t"])
    return L


# ----------------------------------------------------------------------------- oblique TMM
def gamma_layers(f, krho, layers, rough_sigma=0.0):
    """Reflection at the front face (no standoff), TE and TM, for k_rho (F, K) at frequencies f (F,).
    Reflection recursion Gamma_i = (r_i + G e)/(1 + r_i G e); front interface optionally roughened."""
    w = 2 * np.pi * f[:, None]
    k0 = w / C0

    def kz(eps):
        q = np.sqrt(eps * k0**2 - krho**2 + 0j)
        return np.where(q.imag > 0, -q, q)

    def Z(eps, k, pol):
        return w * MU0 / k if pol == "TE" else k / (w * EPS0 * eps)

    out = {}
    for pol in ("TE", "TM"):
        eps_list = [np.ones_like(f, complex)[:, None]] + [lay[1](f)[:, None] for lay in layers] + [np.ones_like(f, complex)[:, None]]
        ks = [kz(e) for e in eps_list]
        Zs = [Z(e, k, pol) for e, k in zip(eps_list, ks)]
        G = np.zeros_like(ks[0])
        for i in range(len(layers), -1, -1):          # interface i between medium i and i+1
            r = (Zs[i + 1] - Zs[i]) / (Zs[i + 1] + Zs[i])
            if i == 0 and rough_sigma > 0:
                r = r * np.exp(-2 * (ks[0] * rough_sigma) ** 2)
            if i == len(layers):
                G = r
            else:
                e = np.exp(-2j * ks[i + 1] * layers[i][0])
                G = (r + G * e) / (1 + r * G * e)
        out[pol] = G
    return out["TE"], out["TM"]


def received(f, layers, d, R: Realism, pec=False):
    """Co-polarised received signal (before A(f)) for a stratified target at distance d."""
    k0 = 2 * np.pi * f / C0
    if R.beam_w0 is None:
        kr = np.zeros((f.size, 1))
        wts = np.ones((f.size, 1))
    else:
        x, wq = np.polynomial.legendre.leggauss(R.n_quad)
        kmax = np.minimum(0.999 * k0, 7.0 / R.beam_w0)
        kr = 0.5 * (x[None, :] + 1) * kmax[:, None]
        wts = 0.5 * wq[None, :] * kmax[:, None] * kr * np.exp(-(kr * R.beam_w0) ** 2 / 2)
        wts = wts / wts.sum(1, keepdims=True)
    kz0 = np.sqrt(k0[:, None] ** 2 - kr**2 + 0j)
    if pec:
        G = -np.ones_like(kz0)
    else:
        GTE, GTM = gamma_layers(f, kr, layers, R.rough_sigma)
        G = 0.5 * (GTE + GTM)
    prop = np.exp(-2j * kz0 * d)
    S = np.sum(wts * G * prop, axis=1)
    S0 = np.sum(wts * G, axis=1)                        # local (no air path) reflection, for Hf
    return S, S0, np.sum(wts * prop, axis=1)


def system(f, R: Realism, S, Sprop):
    """Antenna model (Lambot form): Hi + S/(1 - Hf * S*), where the feedback term uses the wall
    response seen back at the antenna; for a plane wave it is the classical round-trip series."""
    if R.Hf == 0 and R.Hi == 0:
        return S
    return R.Hi + S / (1 - R.Hf * S)


def field_drift(f, R: Realism, seed):
    """E035: multiplicative drift of the wall scan relative to the factory calibration, and its
    leakage-reference correction. Chain drift D_c = exp(g + s x + j(p + 2 pi f delay)) (x across the
    band in [-1/2, 1/2]); antenna part D_a = exp(g_a + j p_a). With leak_ref the radar measures its
    own Tx->Rx leakage L(f) D_c (L known from the factory, |L| = 0.18, i.e. -15 dB, 25 dB SNR per
    sample), fits the same 4-parameter form by linear least squares on log(y_L / L) and divides it out."""
    if R.field_drift is None:
        return np.ones(f.size, complex)
    rng = np.random.default_rng(seed + 77)
    sg, ss, sp, sd = R.field_drift
    x = (f - f.mean()) / (f.max() - f.min())
    ln10 = np.log(10) / 20
    g, sl, ph, dl = rng.normal(0, sg) * ln10, rng.normal(0, ss) * ln10, rng.normal(0, sp), rng.normal(0, sd) * 1e-12
    logDc = g + sl * x + 1j * (ph + 2 * np.pi * (f - f.mean()) * dl)
    Dc = np.exp(logDc)
    Da = np.exp(rng.normal(0, R.untracked[0]) * ln10 + 1j * rng.normal(0, R.untracked[1]))
    if not R.leak_ref:
        return Dc * Da
    tauL = 0.1e-9
    L = 0.18 * np.exp(-2j * np.pi * f * tauL)
    sig = np.sqrt(1.0 / 1e4 / 2)
    yL = L * Dc + sig * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    lr = np.log(yL / L)
    lr = lr.real + 1j * np.unwrap(lr.imag)
    A = np.column_stack([np.ones_like(x), x])
    cr = np.linalg.lstsq(A, lr.real, rcond=None)[0]
    ci = np.linalg.lstsq(A, lr.imag, rcond=None)[0]
    Dhat = np.exp(A @ cr + 1j * (A @ ci))
    return Dc * Da / Dhat


def measure_band(tv, radar: W.Radar, R: Realism, plate_d=0.25, d=None):
    """Calibrated data for one band: wall scan and plate scan through the same physics, then the
    inversion's own calibration (plate at the NOMINAL distance plate_d, Gamma = -1 plane wave)."""
    f = radar.freqs
    layers = layers_from_truth(tv, R)
    S_w, _, _ = received(f, layers, tv["standoff"] if d is None else d, R)
    S_p, _, _ = received(f, layers, plate_d + R.plate_err, R, pec=True)
    yw_clean = system(f, R, S_w, None)
    yp_clean = system(f, R, S_p, None)
    A = W.system_response(radar)
    x = (f - f.mean()) / (f.max() - f.min())
    drift = 10 ** (R.drift[0] / 20 * np.cos(2 * np.pi * x)) * np.exp(1j * R.drift[1] * np.sin(np.pi * x))
    rng = np.random.default_rng(radar.seed)
    sig = np.sqrt(1.0 / 10 ** (radar.snr_db / 10) / 2)
    drift = drift * field_drift(f, R, radar.seed)
    yw = A * drift * yw_clean + sig * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    rng_p = np.random.default_rng(radar.seed + 500)
    yp = A * yp_clean + sig * (rng_p.standard_normal(f.size) + 1j * rng_p.standard_normal(f.size))
    import wall_inverse as I
    yc, _ = I.calibrate(yw, yp, f, plate_d)
    sigma = sig * np.sqrt(1 + np.mean(np.abs(yc) ** 2))
    return f, yc, sigma


def make_case_real(seed, R: Realism, dual=True, plate_d=0.25, band2=None, band1=None):
    """Same truths as benchmark v2/v2dual (same rng sequence), data through the realistic world."""
    rng = np.random.default_rng(seed)
    d_true = 0.30 + 0.002 * rng.random()
    _, tv = S.truth_random_v2(rng, d_true)
    tv["drywall_back.eps_r"] = tv["drywall_front.eps_r"]
    tv["drywall_back.tan_d"] = tv["drywall_front.tan_d"]
    bands = [W.Radar(seed=seed) if band1 is None else W.Radar(f0=band1[0], f1=band1[1], seed=seed)]
    if dual:
        import benchmark as B
        b2 = B.DUAL_BAND if band2 is None else band2
        bands.append(W.Radar(f0=b2[0], f1=b2[1], seed=seed + 1000))
    fs, ys, s2 = [], [], []
    for k, radar in enumerate(bands):
        d = tv["standoff"] + (R.band_offset if k > 0 else 0.0)
        f, yc, sg = measure_band(tv, radar, R, plate_d, d=d)
        fs.append(f); ys.append(yc); s2.append(sg**2)
    r, p = W.range_profile(fs[0], ys[0])
    sel = (r > 0.2) & (r < 0.4)
    d0 = r[sel][np.argmax(p[sel])]
    return np.concatenate(fs), np.concatenate(ys), float(np.sqrt(np.mean(s2))), d0, tv


def plate_positions(f, bases=(0.20, 0.30), steps=3):
    """Calibration plate distances: at each base distance, `steps` positions spaced lambda_c/6
    (60 deg of round-trip phase) so the per-frequency (a, b', c) solve stays well conditioned at
    every frequency of the band (equal spacings of a whole number of half-wavelengths make the
    plate responses identical and the solve singular: E023 first attempt); two separated bases make
    the beam spreading, hence w0, identifiable."""
    lam_c = C0 / np.mean(f)
    return tuple(b + k * lam_c / 6 for b in bases for k in range(steps))


def measure_band_lambot(tv, radar: W.Radar, R: Realism, d_plates=None, d=None):
    """E023: raw (uncalibrated) wall scan and metal-plate scans at several distances, all through
    the same antenna physics (beam, Hf, Hi, A(f)); then the multi-distance Lambot calibration of
    antenna.py. Plate distances are the NOMINAL values used by the calibration; the true plates sit
    at d_p + plate_err. Returns f, calibrated wall beam response, sigma, calibration info."""
    import antenna as AN
    f = radar.freqs
    if d_plates is None:
        d_plates = plate_positions(f)
    layers = layers_from_truth(tv, R)
    A = W.system_response(radar)
    x = (f - f.mean()) / (f.max() - f.min())
    drift = 10 ** (R.drift[0] / 20 * np.cos(2 * np.pi * x)) * np.exp(1j * R.drift[1] * np.sin(np.pi * x))
    sig = np.sqrt(1.0 / 10 ** (radar.snr_db / 10) / 2)

    def noisy(clean, seed):
        rng = np.random.default_rng(seed)
        return clean + sig * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    S_w, _, _ = received(f, layers, tv["standoff"] if d is None else d, R)
    drift = drift * field_drift(f, R, radar.seed)
    yw = noisy(A * drift * system(f, R, S_w, None), radar.seed)
    yps = []
    for k, dp in enumerate(d_plates):
        S_p, _, _ = received(f, layers, dp + R.plate_err, R, pec=True)
        yps.append(noisy(A * system(f, R, S_p, None), radar.seed + 500 + k))
    Sw_cal, info = AN.calibrate(f, yw, yps, d_plates)
    sigma = sig * np.sqrt(np.mean(info["var_per_sigma2"]))     # propagated (wall + calibration) noise
    return f, Sw_cal, float(sigma), info


def make_case_lambot(seed, R: Realism, band2=(57e9, 64e9)):
    """E023: two-band case with multi-distance plate calibration; returns the calibrated beam
    responses, sigma, standoff guess, truth, and the per-band estimated beam waists."""
    rng = np.random.default_rng(seed)
    d_true = 0.30 + 0.002 * rng.random()
    _, tv = S.truth_random_v2(rng, d_true)
    tv["drywall_back.eps_r"] = tv["drywall_front.eps_r"]
    tv["drywall_back.tan_d"] = tv["drywall_front.tan_d"]
    bands = [W.Radar(seed=seed), W.Radar(f0=band2[0], f1=band2[1], seed=seed + 1000)]
    fs, ys, s2, w0s = [], [], [], []
    for k, radar in enumerate(bands):
        d = tv["standoff"] + (R.band_offset if k > 0 else 0.0)
        f, Sw, sg, info = measure_band_lambot(tv, radar, R, d=d)
        fs.append(f); ys.append(Sw); s2.append(sg**2); w0s.append(info["w0"])
    r, p = W.range_profile(fs[0], ys[0])
    sel = (r > 0.2) & (r < 0.4)
    d0 = r[sel][np.argmax(p[sel])]
    return np.concatenate(fs), np.concatenate(ys), float(np.sqrt(np.mean(s2))), d0, tv, tuple(w0s)



# ----------------------------------------------------------------------------- E031: FMCW chain (TI IWR6843-like)
def hpf_matrix(n, fs, corners):
    """Lower-triangular Toeplitz matrix of the IF high-pass chain acting on the IF time series:
    first-order sections H(s) = s/(s + w_c) per corner, discretised by the bilinear transform,
    zero initial state (includes the start-of-chirp transient). y_IF = F x."""
    from scipy.signal import lfilter
    imp = np.zeros(n); imp[0] = 1.0
    h = imp
    for fc in corners:
        K = 2 * fs
        wc = 2 * np.pi * fc
        b = np.array([K, -K]) / (K + wc)
        a = np.array([1.0, (wc - K) / (K + wc)])
        h = lfilter(b, a, h)
    F = np.zeros((n, n))
    for i in range(n):
        F[i, :i + 1] = h[i::-1]
    return F


def fmcw_band(tv, radar: W.Radar, R: Realism, d=None, plate_d=0.25):
    """Raw IF samples of one chirp per scan (wall and plate): the target response at the TRUE
    instantaneous frequencies f_n + eps_n (quadratic nonlinearity, peak R.fmcw["nonlin"]), times
    A(f), filtered by the IF high-pass chain (time domain), plus ADC noise. Returns f (nominal),
    raw wall and plate data, sigma, and the chain matrix F."""
    fm = R.fmcw
    f = radar.freqs
    n = f.size
    Tc = (f[-1] - f[0]) / fm["S"]
    fs = (n - 1) / Tc
    x = np.linspace(-1, 1, n)
    eps = fm.get("nonlin", 0.0) * (x**2 - 1 / 3)
    fa = f + eps
    layers = layers_from_truth(tv, R)
    Sw, _, _ = received(fa, layers, tv["standoff"] if d is None else d, R)
    Sp, _, _ = received(fa, layers, plate_d + R.plate_err, R, pec=True)
    A = W.system_response(radar)
    F = hpf_matrix(n, fs, fm["hpf"])
    sig = np.sqrt(1.0 / 10 ** (radar.snr_db / 10) / 2)
    rng_w, rng_p = np.random.default_rng(radar.seed), np.random.default_rng(radar.seed + 500)
    yw = F @ (A * system(f, R, Sw, None)) + sig * (rng_w.standard_normal(n) + 1j * rng_w.standard_normal(n))
    yp = F @ (A * system(f, R, Sp, None)) + sig * (rng_p.standard_normal(n) + 1j * rng_p.standard_normal(n))
    return f, yw, yp, sig, F


def fmcw_calibrate(f, yp, F, plate_d, deg=10):
    """Estimate the RF response A(f) from a plate scan THROUGH the known IF chain: A is smooth,
    A = sum_k c_k T_k(x) (Chebyshev, x in [-1, 1]), and y_p = F (A * Gamma_p) is linear in c."""
    x = 2 * (f - f[0]) / (f[-1] - f[0]) - 1
    Gp = -np.exp(-2j * 2 * np.pi * f / C0 * plate_d)
    T = np.polynomial.chebyshev.chebvander(x, deg)                    # (n, deg+1)
    Amat = F @ (T * Gp[:, None])
    c, *_ = np.linalg.lstsq(Amat, yp, rcond=None)
    return T @ c


def make_case_fmcw(seed, R: Realism, band2=(60e9, 64e9), model_chain=True):
    """E031: two-band raw IF data (77-81 GHz top band, second band default 60-64 GHz = IWR6843).
    With model_chain: returns raw data and the per-band operators L_b = F_b diag(A_hat_b) for the
    model. Without: the classical calibration y / (plate-derived A) that ignores the IF chain."""
    import wall_inverse as I
    rng = np.random.default_rng(seed)
    d_true = 0.30 + 0.002 * rng.random()
    _, tv = S.truth_random_v2(rng, d_true)
    tv["drywall_back.eps_r"] = tv["drywall_front.eps_r"]
    tv["drywall_back.tan_d"] = tv["drywall_front.tan_d"]
    bands = [W.Radar(seed=seed), W.Radar(f0=band2[0], f1=band2[1], seed=seed + 1000)]
    fs_, ys, s2, ops = [], [], [], []
    for k, radar in enumerate(bands):
        d = tv["standoff"] + (R.band_offset if k > 0 else 0.0)
        f, yw, yp, sig, F = fmcw_band(tv, radar, R, d=d)
        if model_chain:
            Ah = fmcw_calibrate(f, yp, F, 0.25)
            ops.append(F @ np.diag(Ah)); ys.append(yw); s2.append(sig**2)
        else:
            yc, _ = I.calibrate(yw, yp, f, 0.25)
            ys.append(yc); s2.append((sig * np.sqrt(1 + np.mean(np.abs(yc) ** 2))) ** 2)
        fs_.append(f)
    f = np.concatenate(fs_)
    y = np.concatenate(ys)
    if model_chain:     # standoff guess from the chain-equalised top band (only for the guess)
        yg = np.linalg.lstsq(ops[0], ys[0], rcond=None)[0]
    else:
        yg = ys[0]
    r, p = W.range_profile(fs_[0], yg)
    sel = (r > 0.2) & (r < 0.4)
    d0 = r[sel][np.argmax(p[sel])]
    return f, y, float(np.sqrt(np.mean(s2))), d0, tv, (ops if model_chain else None)

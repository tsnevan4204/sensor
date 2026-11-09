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
ITU_D = dict(drywall=0.9395, fiberglass=1.075)   # sigma ~ f^d exponents (plasterboard; ceiling board) - verify


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
    n_quad: int = 48


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
    yw = A * drift * yw_clean + sig * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    rng_p = np.random.default_rng(radar.seed + 500)
    yp = A * yp_clean + sig * (rng_p.standard_normal(f.size) + 1j * rng_p.standard_normal(f.size))
    import wall_inverse as I
    yc, _ = I.calibrate(yw, yp, f, plate_d)
    sigma = sig * np.sqrt(1 + np.mean(np.abs(yc) ** 2))
    return f, yc, sigma


def make_case_real(seed, R: Realism, dual=True, plate_d=0.25):
    """Same truths as benchmark v2/v2dual (same rng sequence), data through the realistic world."""
    rng = np.random.default_rng(seed)
    d_true = 0.30 + 0.002 * rng.random()
    _, tv = S.truth_random_v2(rng, d_true)
    tv["drywall_back.eps_r"] = tv["drywall_front.eps_r"]
    tv["drywall_back.tan_d"] = tv["drywall_front.tan_d"]
    bands = [W.Radar(seed=seed)]
    if dual:
        import benchmark as B
        bands.append(W.Radar(f0=B.DUAL_BAND[0], f1=B.DUAL_BAND[1], seed=seed + 1000))
    fs, ys, s2 = [], [], []
    for k, radar in enumerate(bands):
        d = tv["standoff"] + (R.band_offset if k > 0 else 0.0)
        f, yc, sg = measure_band(tv, radar, R, plate_d, d=d)
        fs.append(f); ys.append(yc); s2.append(sg**2)
    r, p = W.range_profile(fs[0], ys[0])
    sel = (r > 0.2) & (r < 0.4)
    d0 = r[sel][np.argmax(p[sel])]
    return np.concatenate(fs), np.concatenate(ys), float(np.sqrt(np.mean(s2))), d0, tv

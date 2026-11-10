"""
1D layered-wall toy problem: recover layer thickness, permittivity and moisture
from a monostatic radar sweep at normal incidence.

What this file gives you (the "world"):
  * material models: Debye water, CRIM mixing for wet materials
  * an exact reference forward model (transfer-matrix / impedance recursion)
  * a synthetic measurement model (FMCW ~ stepped-frequency S11, calibration error, noise)
  * baseline 1: range-FFT profile + peak picking (what a radar pipeline does today)
  * scenario presets: dry wall, wet insulation, wicked (graded) moisture in drywall

What is left for you (stubs at the bottom):
  * dec_forward(): your DEC / DDEC discretization of 1D Maxwell
  * invert(): your inversion (adjoint gradients, learned closure, priors ...)

Conventions: time dependence e^{+j w t}, so lossy materials have eps = eps' - j eps''.
Units: SI (m, Hz). The radar sits at z = 0; the wall front face is at z = standoff.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

C0 = 299_792_458.0
ETA0 = 376.730313668


# ----------------------------------------------------------------------------- materials
def eps_water_debye(f, T_C=20.0):
    """Single-pole Debye model for liquid water (adequate to ~100 GHz).
    Parameters are textbook nominal values near 20-25 C; tau shrinks as water warms."""
    eps_s = 87.74 - 0.40008 * T_C + 9.398e-4 * T_C**2 - 1.410e-6 * T_C**3
    eps_inf = 5.2
    tau = 1.0 / (2 * np.pi * (20.27e9 + 146.5e6 * T_C + 0.314e6 * T_C**2))  # ~9.4 ps at 20 C
    w = 2 * np.pi * np.asarray(f)
    return eps_inf + (eps_s - eps_inf) / (1 + 1j * w * tau)   # gives eps' - j eps'' with e^{+jwt}? see note
# note: 1/(1 + j w tau) = (1 - j w tau)/(1 + (w tau)^2), so Im < 0: matches eps' - j eps''.


def eps_water_lhm(f, T_C=20.0):
    """Liquid water, double Debye of Liebe, Hufford & Manabe (1991, Int. J. IR mm Waves 12, 659),
    valid to ~1 THz, -10..40 C; converted to e^{+jwt} (eps' - j eps''). theta = 300/T_K."""
    th = 300.0 / (273.15 + T_C)
    e0 = 77.66 + 103.3 * (th - 1)
    e1 = 0.0671 * e0
    e2 = 3.52
    g1 = 20.20 - 146.0 * (th - 1) + 316.0 * (th - 1) ** 2      # GHz
    g2 = 39.8 * g1
    fg = np.asarray(f, float) / 1e9
    return (e0 - e1) / (1 + 1j * fg / g1) + (e1 - e2) / (1 + 1j * fg / g2) + e2


def eps_dielectric(eps_r, tan_d, f):
    """Low-loss dry building material with constant loss tangent."""
    return np.full(np.shape(f), eps_r * (1 - 1j * tan_d), dtype=complex)


def eps_crim(eps_host, theta_w, f, porosity=None, T_C=20.0):
    """Complex Refractive Index Model:  sqrt(eps_mix) = sum_i v_i sqrt(eps_i).
    eps_host: complex permittivity of the dry solid+air matrix (array over f)
    theta_w : volumetric water fraction (0..porosity)
    This is a *closure law* -- a natural candidate for something you learn."""
    ew = eps_water_debye(f, T_C)
    return (( (1 - theta_w) * np.sqrt(eps_host) + theta_w * np.sqrt(ew) ) ** 2)


# ----------------------------------------------------------------------------- scene
@dataclass
class Layer:
    name: str
    thickness: float          # m
    eps: callable             # f -> complex eps array


@dataclass
class Scene:
    layers: list[Layer]
    standoff: float = 0.30    # air gap radar -> wall front face (m)
    backing_eps: callable = field(default=lambda f: np.ones_like(f, dtype=complex))  # half-space behind wall


def nominal_materials():
    """Nominal values -- check against literature for your band before trusting numbers."""
    return dict(
        drywall=dict(eps_r=2.6, tan_d=0.03),
        fiberglass=dict(eps_r=1.15, tan_d=0.01),
        wood_stud=dict(eps_r=2.0, tan_d=0.08),
    )


def scene_dry_wall():
    m = nominal_materials()
    return Scene([
        Layer("drywall", 0.0127, lambda f: eps_dielectric(**m["drywall"], f=f)),
        Layer("insulation", 0.0889, lambda f: eps_dielectric(**m["fiberglass"], f=f)),
        Layer("drywall", 0.0127, lambda f: eps_dielectric(**m["drywall"], f=f)),
    ])


def scene_wet_insulation(theta_w=0.05):
    m = nominal_materials()
    host = lambda f: eps_dielectric(**m["fiberglass"], f=f)
    return Scene([
        Layer("drywall", 0.0127, lambda f: eps_dielectric(**m["drywall"], f=f)),
        Layer("wet insulation", 0.0889, lambda f: eps_crim(host(f), theta_w, f)),
        Layer("drywall", 0.0127, lambda f: eps_dielectric(**m["drywall"], f=f)),
    ])


def scene_wicked_drywall(theta_max=0.15, n_sub=40, decay=0.004):
    """Moisture wicked in from the cavity side of the FRONT drywall:
    theta(z) = theta_max * exp(-(t - z)/decay). A continuous profile -- a piecewise-constant
    layer model with known layer count cannot represent it; a discretized eps(z) can."""
    m = nominal_materials()
    host = lambda f: eps_dielectric(**m["drywall"], f=f)
    t = 0.0127
    dz = t / n_sub
    zc = (np.arange(n_sub) + 0.5) * dz
    layers = [Layer(f"drywall[{i}]", dz,
                    (lambda th: (lambda f: eps_crim(host(f), th, f)))(theta_max * np.exp(-(t - z) / decay)))
              for i, z in enumerate(zc)]
    layers += [
        Layer("insulation", 0.0889, lambda f: eps_dielectric(**m["fiberglass"], f=f)),
        Layer("drywall", 0.0127, lambda f: eps_dielectric(**m["drywall"], f=f)),
    ]
    return Scene(layers)


# ----------------------------------------------------------------------------- exact reference
def _sqrt_branch(eps):
    """Refractive index with the physical branch: Im(n) <= 0 for e^{+jwt} (decaying wave)."""
    n = np.sqrt(eps.astype(complex))
    return np.where(n.imag > 0, -n, n)


def reflection_tmm(f, scene: Scene, include_standoff=True):
    """Exact plane-wave reflection coefficient seen at the radar (z = 0).
    Impedance recursion from the back half-space toward the front."""
    f = np.asarray(f, dtype=float)
    k0 = 2 * np.pi * f / C0
    n_back = _sqrt_branch(scene.backing_eps(f))
    Z = ETA0 / n_back                                   # load impedance
    for L in reversed(scene.layers):
        n = _sqrt_branch(L.eps(f))
        Zc = ETA0 / n
        tl = np.tan(k0 * n * L.thickness)
        Z = Zc * (Z + 1j * Zc * tl) / (Zc + 1j * Z * tl)   # lossless-form TL formula, valid for complex n
    gamma = (Z - ETA0) / (Z + ETA0)
    if include_standoff:
        gamma = gamma * np.exp(-2j * k0 * scene.standoff)
    return gamma


# ----------------------------------------------------------------------------- measurement
@dataclass
class Radar:
    f0: float = 77e9
    f1: float = 81e9
    n_freq: int = 256
    snr_db: float = 40.0     # SNR of a perfect reflector; wall echoes are weaker
    gain_ripple_db: float = 0.5    # unknown smooth amplitude error across band
    phase_ripple_rad: float = 0.1  # unknown smooth phase error across band
    seed: int = 0

    @property
    def freqs(self):
        return np.linspace(self.f0, self.f1, self.n_freq)

    @property
    def range_resolution(self):
        return C0 / (2 * (self.f1 - self.f0))


def system_response(radar: Radar):
    """The radar's unknown smooth gain/phase error A(f) (same for every scan with this radar)."""
    f = radar.freqs
    x = (f - f.mean()) / (f.max() - f.min())
    amp = 10 ** (radar.gain_ripple_db / 20 * (0.6 * x + 0.4 * np.cos(3 * np.pi * x)))
    ph = radar.phase_ripple_rad * (0.7 * x**2 + 0.3 * np.sin(2 * np.pi * x))
    return amp * np.exp(1j * ph)


def measure_gamma(gamma, radar: Radar, seed=None):
    """y = A(f) * gamma(f) + noise. Noise is absolute: radar.snr_db is the SNR a perfect
    reflector (|gamma| = 1) would give, so weak wall echoes get proportionally lower SNR."""
    rng = np.random.default_rng(radar.seed if seed is None else seed)
    A = system_response(radar)
    sig = np.sqrt(1.0 / 10 ** (radar.snr_db / 10) / 2)          # per real component
    y = A * gamma + sig * (rng.standard_normal(gamma.size) + 1j * rng.standard_normal(gamma.size))
    return y, dict(A=A, gamma=gamma, noise_sigma=sig)


def measure(scene: Scene, radar: Radar, seed=None):
    """Synthetic de-chirped FMCW / stepped-frequency measurement of a wall scene.
    (1D plane-wave model: no spherical spreading or antenna pattern -- add later.)"""
    f = radar.freqs
    y, info = measure_gamma(reflection_tmm(f, scene), radar, seed)
    return f, y, info


def measure_metal_plate(radar: Radar, distance, seed=None):
    """Calibration scan: a flat metal plate (gamma = -1 exactly) at a known distance."""
    f = radar.freqs
    g = -np.exp(-2j * 2 * np.pi * f / C0 * distance)
    y, info = measure_gamma(g, radar, seed)
    return f, y, info


# ----------------------------------------------------------------------------- baseline 1
def range_profile(f, y, n_fft=8192, window="hann"):
    """Classical radar processing: windowed IFFT over frequency -> round-trip range profile."""
    w = np.hanning(f.size) if window == "hann" else np.ones(f.size)
    prof = np.fft.ifft(y * w, n_fft)
    df = f[1] - f[0]
    r = np.arange(n_fft) * C0 / (2 * df * n_fft)        # one-way range (m)
    # undo the carrier phase so peaks sit at true range
    return r, np.abs(prof)


def peak_pick_thickness(r, p, standoff_guess, eps_assumed=2.6, rel_height=0.1):
    """Baseline: find the two strongest peaks after the front face and convert the
    apparent (optical) separation to physical thickness with an ASSUMED eps."""
    from scipy.signal import find_peaks
    mask = r > standoff_guess - 0.05
    idx, _ = find_peaks(p * mask, height=rel_height * p[mask].max())
    if len(idx) < 2:
        return None, idx
    top = idx[np.argsort(p[idx])[-2:]]
    top.sort()
    sep = r[top[1]] - r[top[0]]
    return sep / np.sqrt(eps_assumed), top


# ----------------------------------------------------------------------------- YOUR PART
def dec_forward(theta, f, grid, **kw):
    """TODO (you): DEC / DDEC discretization of 1D Maxwell for the transverse fields
    (E_x, H_y) along z, returning the predicted reflection coefficient at z = 0.

    Suggested interface:
      theta : parameters of eps(z), sigma(z) -- or of a learned closure / Hodge star
      f     : frequencies (frequency-domain solve) or a pulse spec (time-domain leapfrog)
      grid  : primal nodes z_i (0-cochains) and primal edges (1-cochains), with d0 = incidence

    Validate against reflection_tmm() on every scene preset before inverting anything."""
    raise NotImplementedError


def invert(f, y, **kw):
    """TODO (you): recover thickness / eps / moisture (and the unknown A(f)) from y."""
    raise NotImplementedError


# ----------------------------------------------------------------------------- demo
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radar = Radar()
    print(f"band {radar.f0/1e9:.0f}-{radar.f1/1e9:.0f} GHz, range resolution {radar.range_resolution*100:.2f} cm")
    for ff in (3e9, 10e9, 77e9):
        e = eps_water_debye(np.array([ff]))[0]
        print(f"water eps at {ff/1e9:5.1f} GHz: {e.real:6.2f} {e.imag:+6.2f}j")
    m = nominal_materials()["drywall"]
    print(f"drywall optical thickness {0.0127*np.sqrt(m['eps_r'])*100:.2f} cm "
          f"vs resolution {radar.range_resolution*100:.2f} cm -> unresolved")

    scenes = {"dry": scene_dry_wall(), "wet insulation 5%": scene_wet_insulation(0.05),
              "wicked drywall": scene_wicked_drywall()}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, sc in scenes.items():
        f, y, info = measure(sc, radar)
        ax[0].plot(f / 1e9, 20 * np.log10(np.abs(info["gamma"])), label=name)
        r, p = range_profile(f, y)
        ax[1].plot(r * 100, 20 * np.log10(p / p.max() + 1e-6), label=name)
        from scipy.signal import find_peaks
        pdb = 20 * np.log10(p / p.max() + 1e-12)
        sel = (r > 0.25) & (r < 0.60)
        idx, _ = find_peaks(np.where(sel, pdb, -200), height=-35)
        print(f"{name:20s} peaks above -35 dB at range (cm): "
              + ", ".join(f"{r[i]*100:.1f} ({pdb[i]:.0f} dB)" for i in idx))
    ax[0].set(xlabel="frequency (GHz)", ylabel="|Γ| (dB)", title="Exact reflection (TMM)")
    ax[1].set(xlabel="range (cm)", ylabel="normalized (dB)", title="Range profile (baseline)",
              xlim=(20, 60), ylim=(-50, 2))
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    # true interface positions in OPTICAL range (dry wall): 30, +n*t for each layer
    nd, ni = np.sqrt(2.6), np.sqrt(1.15)
    for zf in np.cumsum([30, 1.27 * nd, 8.89 * ni, 1.27 * nd]):
        ax[1].axvline(zf, color="k", lw=.6, ls=":")
    ax[1].text(30.3, -48, "dotted: true interfaces\n(optical range, dry wall)", fontsize=8)
    fig.tight_layout()
    fig.savefig("wall_toy_overview.png", dpi=150)
    print("saved wall_toy_overview.png")

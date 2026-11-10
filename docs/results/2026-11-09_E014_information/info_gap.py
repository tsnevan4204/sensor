"""
E014: which extra measurement resolves the back-of-wall cycle ambiguity?  (information analysis,
no inversion pipeline). For random B1-dry-v2 walls and each measurement configuration:
noiseless data y0 = Gamma(truth); for each whole-period shift h of (insulation.t, drywall_back.t)
(period = lambda_c / (2 Re n) at 79 GHz), minimise chi2(p) = sum |Gamma(p) - y0|^2 / sigma^2 over
ALL wall parameters from truth+h (bounded, same box as the model). The minimum is the price the
wrong basin pays in expectation: Delta(h). Two-hypothesis error probability ~ Phi(-sqrt(Delta)/2).
A configuration resolves the cycles if min_h Delta(h) >> 4.
Oblique incidence: plane-wave TMM with k_z = k0 sqrt(eps - sin^2 theta); TE: Z = w mu0 / k_z,
TM: Z = k_z / (w eps0 eps). Gated against wall_toy.reflection_tmm at theta = 0.
Noise: sigma per real component = benchmark calibrated value; 256 samples per band and angle.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, time, itertools
sys.path.insert(0, "/home/user/sensor/docs/src/oned")
import numpy as np
from math import erf, sqrt
from scipy.optimize import least_squares
import wall_toy as W, scenarios as S, benchmark as B

C0, ETA0 = W.C0, W.ETA0
MU0 = ETA0 / C0
EPS0 = 1 / (MU0 * C0**2)
NAMES = ["standoff", "drywall_front.t", "drywall_front.eps_r", "drywall_front.tan_d",
         "insulation.t", "insulation.eps_r", "insulation.tan_d", "drywall_back.t"]
LO = np.array([0.0, 0.009, 1.0, 0.0, 0.080, 1.0, 0.0, 0.009])
HI = np.array([1.0, 0.017, 4.0, 0.15, 0.100, 4.0, 0.15, 0.017])


def gamma_oblique(f, p, theta=0.0, pol="TE"):
    d, tf, ed, tdd, ti, ei, tdi, tb = p
    w = 2 * np.pi * f
    k0 = w / C0
    s2 = np.sin(theta) ** 2
    layers = [(tf, ed * (1 - 1j * tdd)), (ti, ei * (1 - 1j * tdi)), (tb, ed * (1 - 1j * tdd))]

    def kz(eps):
        q = np.sqrt(eps - s2 + 0j)
        q = np.where(q.imag > 0, -q, q)
        return k0 * q

    def Z(eps, k):
        return w * MU0 / k if pol == "TE" else k / (w * EPS0 * eps)

    kb = kz(np.ones_like(f) + 0j)
    Zl = Z(1.0, kb)
    for t, eps in reversed(layers):
        k = kz(eps * np.ones_like(f))
        Zc = Z(eps, k)
        tl = np.tan(k * t)
        Zl = Zc * (Zl + 1j * Zc * tl) / (Zc + 1j * Zl * tl)
    Z0 = Z(1.0, kb)
    g = (Zl - Z0) / (Zl + Z0)
    return g * np.exp(-2j * kb * d)


def gate_normal():
    f = np.linspace(77e9, 81e9, 64)
    _, tv = S.truth_random_v2(np.random.default_rng(5), 0.301)
    p = np.array([tv[n] if n in tv else 0 for n in NAMES])
    sc = S.truth_random_v2(np.random.default_rng(5), 0.301)[0]
    for pol in ("TE", "TM"):
        err = np.abs(gamma_oblique(f, p, 0.0, pol) - W.reflection_tmm(f, sc)).max()   # same Gamma at 0 deg (transverse impedance form)
        assert err < 1e-12, (pol, err)
    return True


CONFIGS = {
    "A 77-81 GHz, normal (today)": [(np.linspace(77e9, 81e9, 256), 0.0, "TE")],
    "B + 57-64 GHz": [(np.linspace(77e9, 81e9, 256), 0.0, "TE"), (np.linspace(57e9, 64e9, 256), 0.0, "TE")],
    "C + 22-26 GHz": [(np.linspace(77e9, 81e9, 256), 0.0, "TE"), (np.linspace(22e9, 26e9, 256), 0.0, "TE")],
    "D + 3.1-10.6 GHz UWB": [(np.linspace(77e9, 81e9, 256), 0.0, "TE"), (np.linspace(3.1e9, 10.6e9, 256), 0.0, "TE")],
    "E + 30 deg TE": [(np.linspace(77e9, 81e9, 256), 0.0, "TE"), (np.linspace(77e9, 81e9, 256), np.radians(30), "TE")],
    "F + 30 deg TM": [(np.linspace(77e9, 81e9, 256), 0.0, "TE"), (np.linspace(77e9, 81e9, 256), np.radians(30), "TM")],
    "G 77-81 GHz twice (same data budget as B-F)": [(np.linspace(77e9, 81e9, 256), 0.0, "TE")] * 2,
}
HOPS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (2, 0), (0, 2), (2, -2), (-2, 2)]


def resid(p, cfg, y0, sigma):
    out = []
    for (f, th, pol), y in zip(cfg, y0):
        d = gamma_oblique(f, p, th, pol) - y
        out += [d.real / sigma, d.imag / sigma]
    return np.concatenate(out)


def analyse(seed, cfg, sigma):
    rng = np.random.default_rng(seed)
    d_true = 0.30 + 0.002 * rng.random()
    _, tv = S.truth_random_v2(rng, d_true)
    p0 = np.array([tv[n] for n in NAMES])
    y0 = [gamma_oblique(f, p0, th, pol) for f, th, pol in cfg]
    lam = C0 / 79e9
    per_i = lam / (2 * np.sqrt(tv["insulation.eps_r"] * (1 - 1j * tv["insulation.tan_d"])).real)
    per_b = lam / (2 * np.sqrt(tv["drywall_front.eps_r"] * (1 - 1j * tv["drywall_front.tan_d"])).real)
    lo, hi = LO.copy(), HI.copy()
    lo[0], hi[0] = d_true - 0.01, d_true + 0.01
    gaps = {}
    for hi_, hb in HOPS:
        x0 = p0.copy()
        x0[4] += hi_ * per_i
        x0[7] += hb * per_b
        if not (lo[4] < x0[4] < hi[4] and lo[7] < x0[7] < hi[7]):
            continue
        sol = least_squares(resid, x0, args=(cfg, y0, sigma), bounds=(lo, hi), x_scale="jac",
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        # keep only solutions that stayed in the hopped basin (else it fell back to the truth = gap 0 trivially)
        moved = abs(sol.x[4] * np.sqrt(sol.x[5]) - p0[4] * np.sqrt(p0[5])) > 0.25 * lam / 2 or \
                abs(sol.x[7] - p0[7]) > 0.25 * per_b
        gaps[f"{hi_:+d},{hb:+d}"] = float(2 * sol.cost) if moved else float("nan")
    return gaps


if __name__ == "__main__":
    gate_normal()
    sigma = 0.0073          # benchmark calibrated sigma per real component (approx.)
    seeds = list(range(1, 9))
    out = {}
    lines = [f"E014 information gap, B1-dry-v2 truths seeds {seeds}, sigma {sigma}; gate: oblique TMM == reference at 0 deg (< 1e-12)"]
    t0 = time.time()
    for name, cfg in CONFIGS.items():
        res = [analyse(s, cfg, sigma) for s in seeds]
        mins = np.array([np.nanmin(list(r.values())) if any(np.isfinite(list(r.values()))) else np.nan for r in res])
        perr = [0.5 * (1 + erf(-sqrt(m) / 2 / sqrt(2))) for m in mins]
        out[name] = dict(per_seed=res, min_gap=mins.tolist())
        lines.append(f"{name:45s} min-gap Delta over hops: median {np.nanmedian(mins):8.1f}  worst {np.nanmin(mins):8.1f}"
                     f"   P(flip to nearest wrong basin) median {np.median(perr):.2g}  worst {np.max(perr):.2g}")
        print(lines[-1], flush=True)
    lines.append(f"time {time.time() - t0:.0f}s")
    open(sys.argv[1], "w").write("\n".join(lines) + "\n")
    json.dump(out, open(sys.argv[1].replace(".txt", ".json"), "w"), indent=1)

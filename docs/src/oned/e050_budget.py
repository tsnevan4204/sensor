"""
E050: information budget. Cramér–Rao bounds (no search) for the known-N problem under different
measurement designs: band sets (current 3 bands; + a 0.1-0.4 GHz band), extra SNR from chirp
averaging (+0 / +20 dB), and multiple incidence angles (0, 15, 30 deg; TE, bistatic geometry).

Parameters per solid layer: t, eps' (10 GHz), tan d (10 GHz), dispersion shape p; every present gap;
the standoff. Exact oblique reflection recursion; Jacobian by central differences.
A layer counts as OBSERVED if CRB(t) <= max(0.3 mm, 2 % t) and CRB(eps')/eps' <= 3 %.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import wallsolve as WS

C0 = WS.C0
F_REF = WS.F_REF
NPTS = 128
BANDS3 = ((0.4e9, 0.96e9, 30.0), (3.1e9, 10.6e9, 40.0), (60e9, 64e9, 40.0))
BAND_VHF = (0.1e9, 0.4e9, 30.0)


def eps_of(er, td, p, f):
    a = 2 / np.pi * np.arctan(td / (1 - p))
    return er * (p + (1 - p - 1j * td) * (f / F_REF) ** (-a))


def gamma_oblique(params, struct, back, f, theta):
    """params: flat vector; struct: list of ('solid'|'gap') in order; TE reflection incl. air path."""
    k0 = 2 * np.pi * f / C0
    s2 = np.sin(theta) ** 2
    d = params[-1]
    eps, ts, j = [], [], 0
    for kind in struct:
        if kind == "solid":
            t, er, td, p = params[j:j + 4]; j += 4
            eps.append(eps_of(er, td, min(p, 0.999), f)); ts.append(t)
        else:
            eps.append(np.ones_like(f, complex)); ts.append(params[j]); j += 1
    kz = [np.sqrt(e - s2 + 0j) for e in eps]
    kz = [np.where(k.imag > 0, -k, k) for k in kz]
    kz0 = np.sqrt(1 - s2) * np.ones_like(f)
    kzb = np.sqrt((WS.ST.METAL_EPS if back == "metal" else 1.0) - s2 + 0j) * np.ones_like(f)
    G = (kz[-1] - kzb) / (kz[-1] + kzb)
    for i in range(len(kz) - 1, -1, -1):
        kp = kz[i - 1] if i > 0 else kz0
        e = np.exp(-2j * k0 * kz[i] * ts[i])
        r = (kp - kz[i]) / (kp + kz[i])
        G = (r + G * e) / (1 + r * G * e)
    return G * np.exp(-2j * k0 * kz0 * d)


def truth_params(tr):
    struct, x = [], []
    for i, L in enumerate(tr["layers"]):
        struct.append("solid"); x += [L["t"], L["eps10"], L["tan10"], L.get("p", 0.0)]
        if i < len(tr["gaps"]) and tr["gaps"][i] > 0:
            struct.append("gap"); x += [tr["gaps"][i]]
    x.append(tr["standoff"])
    return struct, np.array(x, float)


def crb(struct, x, back, bands, angles, snr_boost_db):
    rows = []
    for (a, b, snr) in bands:
        f = np.linspace(a, b, NPTS)
        sig = 10 ** (-(snr + snr_boost_db) / 20) / np.sqrt(2)
        for th in angles:
            J = np.zeros((f.size, x.size), complex)
            for k in range(x.size):
                h = 1e-6 * max(abs(x[k]), 1e-4)
                xp, xm = x.copy(), x.copy(); xp[k] += h; xm[k] -= h
                J[:, k] = (gamma_oblique(xp, struct, back, f, th) - gamma_oblique(xm, struct, back, f, th)) / (2 * h)
            rows.append(np.vstack([J.real, J.imag]) / sig)
    Jt = np.vstack(rows)
    F = Jt.T @ Jt
    cov = np.linalg.pinv(F)
    return np.sqrt(np.abs(np.diag(cov)))


def observed(struct, x, sd):
    out, j = [], 0
    for kind in struct:
        if kind == "solid":
            t, er = x[j], x[j + 1]
            out.append(bool(sd[j] <= max(0.3e-3, 0.02 * t) and sd[j + 1] / er <= 0.03)); j += 4
        else:
            j += 1
    return out


DESIGNS = {
    "3band":            (BANDS3, (0.0,)),
    "3band+VHF":        (BANDS3 + (BAND_VHF,), (0.0,)),
    "3band,3angles":    (BANDS3, (0.0, np.radians(15), np.radians(30))),
    "3band+VHF,3angles": (BANDS3 + (BAND_VHF,), (0.0, np.radians(15), np.radians(30))),
}


def targeted():
    """Hand-made walls for the two hard classes: absorber (thick concrete/brick) and low contrast."""
    rng = np.random.default_rng(0)
    def L(nm, t):
        _, d = WS.ST.material_law(nm, rng)
        return dict(name=nm, t=t, **d)
    return {
        "concrete120|gap20|plasterboard": dict(layers=[L("concrete", 0.12), L("plasterboard", 0.0125)], gaps=[0.02], back="air", standoff=0.3),
        "brick100|batt60|plasterboard":   dict(layers=[L("brick", 0.10), L("batt", 0.06), L("plasterboard", 0.0125)], gaps=[0, 0], back="air", standoff=0.3),
        "foam30|gap10|plasterboard":      dict(layers=[L("foam", 0.03), L("plasterboard", 0.0125)], gaps=[0.01], back="air", standoff=0.3),
        "batt80|plywood12":               dict(layers=[L("batt", 0.08), L("plywood", 0.012)], gaps=[0], back="air", standoff=0.3),
        "concrete150|plasterboard":       dict(layers=[L("concrete", 0.15), L("plasterboard", 0.0125)], gaps=[0], back="air", standoff=0.3),
    }


if __name__ == "__main__":
    out = {}
    cases = dict(targeted())
    for sd in range(30):
        _, _, _, tr = WS.make_problem(1000 + sd, 3)
        cases[f"random{sd}"] = tr
    for name, tr in cases.items():
        struct, x = truth_params(tr)
        row = {}
        for dn, (bands, angles) in DESIGNS.items():
            for boost in (0.0, 20.0):
                s = crb(struct, x, tr["back"], bands, angles, boost)
                row[f"{dn} +{int(boost)}dB"] = observed(struct, x, s)
        out[name] = dict(layers=[L["name"] for L in tr["layers"]], obs=row)
        print(name, out[name]["layers"], {k: sum(v) for k, v in row.items()}, flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
    # summary: fraction of layers observed per design over the 30 random walls
    tot = {}
    for name, r in out.items():
        if not name.startswith("random"):
            continue
        for k, v in r["obs"].items():
            a, b = tot.get(k, (0, 0)); tot[k] = (a + sum(v), b + len(v))
    print("\nlayers observed (30 random 3-layer walls):")
    for k, (a, b) in tot.items():
        print(f"  {k:28s} {a}/{b} = {a / b:.0%}")

"""
E034: arbitrary layered stacks of arbitrary materials (limitation A2 of doc 14).

Two independent halves:

WORLD (truth generator, never used by the inversion)
  MATERIALS   building materials from ITU-R P.2040-3 Table 3 (eps' = a, sigma = c f_GHz^d) made causal
              (Jonscher form matched at 10 GHz, loss slope from d), plus foam, batt insulation, ceramic
              tile, air, liquid water (double-Debye, Liebe-Hufford-Manabe 1991, temperature dependent)
              and a metal backing (sigma = 1e7 S/m, Table 3). Each draw perturbs eps' by +-8 % and
              sigma by x0.6..1.6 (material variability).
  random_stack(rng)    1..NMAX layers of random materials and thicknesses, air or metal behind;
                       rejection-sampled so every interface is visible in the low band
                       (two-way loss to the last interface < VIS_DB at 5 GHz).
  gamma_stack(f, layers, back)   independent normal-incidence transfer-matrix (reflection recursion)
                       reference; the inversion uses the exact-star DEC solve.
  make_case(seed)      data on the prototype band pair (60-64 GHz + UWB 3.1-10.6 GHz), calibrated
                       plane-wave data with the radar's 40 dB SNR, as benchmark v2.

INVERSION (knows no material names; every layer is "a causal dielectric", optionally "water")
  Each layer: thickness t, eps_r and tan_d at F_REF, Jonscher p (dispersion shape), or a water layer
  (CRIM theta = 1). Back medium: air or metal (two hypotheses, chi2 decides).
  invert(f, y, sigma, d0)
     1. frequency continuation (Bunks et al. 1995 multiscale idea): fit the LOW band first,
        starting with its lowest third; low frequencies have few phase cycles per layer, so few
        basins.
     2. greedy model-order growth: from the residual range profile, propose a new interface where
        the largest unexplained echo is (split the layer containing it, or append a layer behind),
        times a few eps values (air, foam, wood, drywall, brick/concrete, glass/tile) and water;
        LM each; accept the new layer only if chi2 drops by more than BIC_PER_LAYER
        (Schwarz criterion: k ln n), stop when chi2 is consistent with noise.
     3. add the high band: LM from the low-band solution (its thickness error is below a quarter
        of the high-band phase period, the GNSS wide-lane condition), then cycle hops and an
        ambiguity scan on all thicknesses.
     4. prune: remove any layer whose removal costs less than BIC_PER_LAYER.
     5. classify each layer against the material library (Gaussian likelihood in
        (log eps'(10 GHz), log tan_d(10 GHz))) and report the class probabilities.
"""
from __future__ import annotations
import itertools
import numpy as np
import wall_toy as W
import wall_inverse as I
import search as SR

C0, ETA0 = W.C0, W.ETA0
EPS0 = 1 / (ETA0 * C0)
F_REF = 10e9
NMAX = 5
VIS_DB = 45.0
METAL_EPS = 1.0 - 1e8j          # |n| ~ 7e3: a perfect conductor to < 1e-4 in reflection

# ============================================================================ world: materials
# ITU-R P.2040-3 Table 3 (a, c, d) and thickness range (m) typical in buildings
ITU = {
    "concrete":     (5.24, 0.0462, 0.7822, (40e-3, 150e-3)),
    "brick":        (3.91, 0.0238, 0.16,   (60e-3, 110e-3)),
    "plasterboard": (2.73, 0.0085, 0.9395, (9.5e-3, 16e-3)),
    "wood":         (1.99, 0.0047, 1.0718, (10e-3, 45e-3)),
    "glass":        (6.31, 0.0036, 1.3394, (3e-3, 10e-3)),
    "ceiling_board": (1.48, 0.0011, 1.0750, (10e-3, 20e-3)),
    "chipboard":    (2.58, 0.0217, 0.7800, (9e-3, 19e-3)),       # OSB / particle board
    "plywood":      (2.71, 0.33,   0.0,    (6e-3, 19e-3)),
    "marble":       (7.074, 0.0055, 0.9262, (8e-3, 20e-3)),
    "floorboard":   (3.66, 0.0044, 1.3515, (10e-3, 22e-3)),
}
# not in Table 3: (eps', tan_d at 10 GHz, thickness range); constant-Q dispersion
OTHER = {
    "foam":       (1.05, 3e-4, (10e-3, 50e-3)),     # XPS / EPS board
    "batt":       (1.12, 5e-3, (30e-3, 90e-3)),     # fibreglass / mineral wool batt
    "tile":       (6.0, 0.03, (6e-3, 12e-3)),       # glazed ceramic tile
}
SPECIAL = {"air": (2e-3, 50e-3), "water": (1e-3, 15e-3)}
ALL_MATERIALS = list(ITU) + list(OTHER) + list(SPECIAL)


water_lhm = W.eps_water_lhm


def material_law(name, rng, T=None):
    """A random instance of material `name`: returns eps(f) callable and its descriptors."""
    if name == "air":
        return (lambda f: np.ones(np.shape(f), complex)), dict(eps10=1.0, tan10=0.0)
    if name == "water":
        T = rng.uniform(5, 35) if T is None else T
        return (lambda f, T=T: water_lhm(f, T)), dict(T=T)
    if name in ITU:
        a, c, d, _ = ITU[name]
        er = a * rng.uniform(0.92, 1.08)
        sig10 = c * 10.0 ** d * rng.uniform(0.6, 1.6)
        tan10 = sig10 / (2 * np.pi * F_REF * EPS0 * er)
        acq = 2 / np.pi * np.arctan(tan10)
        ap = float(np.clip(max(1 - d, acq), acq, 0.98))           # eps'' ~ f^(d-1): Jonscher exponent
        pp = 1 - tan10 / np.tan(np.pi * ap / 2)
    else:
        er0, tan0, _ = OTHER[name]
        er = 1 + (er0 - 1) * rng.uniform(0.6, 1.4) if er0 < 1.3 else er0 * rng.uniform(0.95, 1.05)
        tan10 = tan0 * rng.uniform(0.6, 1.6)
        ap = 2 / np.pi * np.arctan(tan10)
        pp = 0.0

    def eps(f, er=er, t=tan10, ap=ap, pp=pp):
        g = (np.asarray(f, float) / F_REF) ** (-ap)
        return er * (pp + (1 - pp - 1j * t) * g)
    return eps, dict(eps10=er, tan10=tan10, alpha=ap, p=pp)


POROUS = ("plasterboard", "wood", "chipboard", "plywood", "brick", "concrete", "batt")


def wet_law(eps_host, theta, T):
    """E036 world: a porous host with volumetric water fraction theta, mixed with the Looyenga
    (Landau-Lifshitz) cube-root law eps^(1/3) = (1-theta) eps_h^(1/3) + theta eps_w^(1/3). The
    inversion uses CRIM (square-root law), so the mixing law itself is a model error."""
    def eps(f):
        w = W.eps_water_lhm(f, T)
        c = (1 - theta) * np.power(eps_host(f) + 0j, 1 / 3) + theta * np.power(w + 0j, 1 / 3)
        return c ** 3
    return eps


def thickness_range(name):
    if name in ITU:
        return ITU[name][3]
    if name in OTHER:
        return OTHER[name][2]
    return SPECIAL[name]


# ============================================================================ world: stacks + data
def gamma_stack(f, layers, back="air"):
    """Reflection at the front face, normal incidence. layers: list of (t, eps callable)."""
    f = np.asarray(f, float)
    k0 = 2 * np.pi * f / C0
    n = [np.ones_like(f, complex)] + [np.sqrt(e(f) + 0j) for _, e in layers]
    n = [np.where(x.imag > 0, -x, x) for x in n]                    # decaying branch for e^{+jwt}
    nb = np.sqrt(np.full(f.shape, METAL_EPS)) if back == "metal" else np.ones_like(f, complex)
    G = (n[-1] - nb) / (n[-1] + nb)
    for i in range(len(layers), 0, -1):
        e = np.exp(-2j * k0 * n[i] * layers[i - 1][0])
        r = (n[i - 1] - n[i]) / (n[i - 1] + n[i])
        G = (r + G * e) / (1 + r * G * e)
    return G


def _visible(layers, back):
    f = np.array([5e9])
    k0 = 2 * np.pi * f / C0
    loss = 0.0
    for t, e in layers[:-1] if back == "air" else layers:
        n = np.sqrt(e(f) + 0j)
        loss += 2 * k0[0] * abs(n[0].imag) * t * 20 / np.log(10)     # two-way, dB
    return loss < VIS_DB


def random_stack(rng, nmax=NMAX, p_metal=0.3, materials=None, p_wet=0.0, n_exact=None):
    mats = ALL_MATERIALS if materials is None else materials
    while True:
        N = int(rng.integers(1, nmax + 1)) if n_exact is None else int(n_exact)
        names = []
        for _ in range(N):
            choices = [m for m in mats if not names or m != names[-1]]
            if not names:
                choices = [m for m in choices if m != "air"]            # the first layer is a solid
            names.append(str(rng.choice(choices)))
        if names[-1] == "air" and N > 1:
            names[-1] = "plasterboard"          # an air layer must have something behind it
        back = "metal" if rng.random() < p_metal else "air"
        layers, desc = [], []
        Tw = rng.uniform(5, 35)                 # one wall temperature for all liquid water
        for nm in names:
            lo, hi = thickness_range(nm)
            t = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            e, dsc = material_law(nm, rng, Tw)
            if nm in POROUS and rng.random() < p_wet:
                th = float(rng.uniform(0.01, 0.12))
                e = wet_law(e, th, Tw)
                dsc = dict(dsc, theta=th, T=Tw)
            layers.append((t, e))
            desc.append(dict(name=nm, t=t, **dsc))
        if _visible(layers, back):
            return layers, desc, back


BANDS = ((3.1e9, 10.6e9), (60e9, 64e9))          # prototype pair: UWB + IWR6843


def make_case(seed, bands=BANDS, snr_db=40.0, nmax=NMAX, materials=None, p_metal=0.3, p_wet=0.0, n_exact=None):
    rng = np.random.default_rng(seed)
    layers, desc, back = random_stack(rng, nmax, p_metal, materials, p_wet, n_exact)
    d = 0.30 + 0.002 * rng.random()
    fs, ys = [], []
    sig = 10 ** (-snr_db / 20)
    for k, (a, b) in enumerate(bands):
        f = np.linspace(a, b, 256)
        G = gamma_stack(f, layers, back) * np.exp(-2j * 2 * np.pi * f / C0 * d)
        nr = np.random.default_rng(seed + 7919 * (k + 1))
        ys.append(G + sig / np.sqrt(2) * (nr.standard_normal(f.size) + 1j * nr.standard_normal(f.size)))
        fs.append(f)
    f, y = np.concatenate(fs), np.concatenate(ys)
    r, p = W.range_profile(fs[0], ys[0])
    sel = (r > 0.2) & (r < 0.4)
    d0 = float(r[sel][np.argmax(p[sel])])
    return f, y, sig / np.sqrt(2), d0, dict(layers=desc, back=back, standoff=d)   # sigma per real component


# ============================================================================ inversion: model
T_LO, T_HI = 3e-4, 0.25
EPS_LO, EPS_HI = 1.0, 12.0
EPS_STARTS = (1.0, 1.1, 2.0, 2.8, 4.5, 6.5)     # air, foam/batt, wood, drywall/plywood, brick/concrete, glass/tile
WATER_T0 = 20.0


def _layer(i, t, er, td=0.01, water=False, pinf=0.0, theta=0.0, moist=False):
    L = I.LayerSpec(f"L{i}", I.P(float(np.clip(t, T_LO * 1.01, T_HI * 0.99)), T_LO, T_HI, True),
                    I.P(float(np.clip(er, EPS_LO, EPS_HI - 1e-6)), EPS_LO, EPS_HI, not water),
                    I.P(float(np.clip(td, 0.0, 0.49)), 0.0, 0.5, not water),
                    I.P(1.0 if water else float(theta), 0.0, 1.0 if water else 0.4, bool(moist) and not water),
                    I.P(pinf, 0.0, 0.999, False))
    if water:
        L.eps_r.value, L.tan_d.value = 1.0, 0.0
    return L


def build(layers, standoff, d_half=0.01, back="air", water_T=WATER_T0):
    """layers: list of dicts(t, er, td, water). Names L0..L{N-1} are positional."""
    Ls = [_layer(i, **l) for i, l in enumerate(layers)]
    lo, hi = SBOX if SBOX is not None else (standoff - d_half, standoff + d_half)
    standoff = float(np.clip(standoff, lo + 1e-6, hi - 1e-6))
    m = I.WallModel(Ls, I.P(standoff, lo, hi, True),
                    eps_back=(METAL_EPS if back == "metal" else 1.0 + 0j))
    m.law, m.f_ref, m.water_T = "jonscher", F_REF, WATER_T0
    m.water_law = "lhm"
    if any(l.get("water") or l.get("moist") for l in layers):
        m.water_Tp = I.P(water_T, 0.0, 40.0, True)          # liquid water's temperature is fitted
    m.min_cells = True
    m.use_tmm = True          # E041: exact reflection recursion (== exact-star DEC to 1e-14), 5x faster
    m.back = back
    return m


def describe(m):
    out = []
    for L in m.layers:
        w = L.theta.value > 0.5 and not L.theta.free
        out.append(dict(t=L.t.value, er=L.eps_r.value, td=L.tan_d.value, water=w,
                        pinf=L.pinf.value, theta=0.0 if w else L.theta.value, moist=L.theta.free))
    return out


def rebuild(m, layers=None, back=None):
    mm = build(describe(m) if layers is None else layers, m.standoff.value,
               back=m.back if back is None else back,
               water_T=m.water_Tp.value if m.water_Tp is not None else WATER_T0)
    mm.standoff.lo, mm.standoff.hi = m.standoff.lo, m.standoff.hi
    for L, Lo in zip(mm.layers, m.layers if layers is None else []):
        L.pinf.value, L.pinf.free = Lo.pinf.value, Lo.pinf.free
    return mm


def set_dispersion_free(m, on):
    for L in m.layers:
        L.pinf.free = bool(on) and L.theta.value < 0.5
    return m


# ============================================================================ inversion: search
def _fit(m, f, y, sigma, atol=0.0):
    try:
        mo, c = SR.lm(m, f, y, sigma, atol=atol)
    except Exception:
        return m, np.inf
    return mo, (c if np.isfinite(c) else np.inf)


def chi2_alarm(n_res):
    return n_res + 3 * np.sqrt(2 * n_res)


def _residual_peaks(m, f, y, k=3):
    """Ranges (from the radar) of the k largest unexplained echoes in the low band."""
    G = I.forward(m, f, jacobian=False)[0] if m.layers else 0.0
    r, p = W.range_profile(f, y - G)
    sel = (r > m.standoff.value - 0.01) & (r < m.standoff.value + 0.6)
    r, p = r[sel], p[sel]
    pk = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] >= p[i + 1]]
    pk = sorted(pk, key=lambda i: -p[i])
    out = []
    for i in pk:
        if all(abs(r[i] - q) > 0.004 for q in out):
            out.append(r[i])
        if len(out) >= k:
            break
    return out


def _interfaces(m):
    """Optical (radar-range) positions of each interface at F_REF (Re n * t)."""
    pos = [m.standoff.value]
    for L in m.layers:
        eps, _ = I.layer_eps_and_derivs(L, np.array([5e9]), m.water_T, m.law, m.f_ref)
        pos.append(pos[-1] + np.sqrt(eps[0]).real * L.t.value)
    return pos


def _proposals(m, f, y):
    """Candidate (N+1)-layer models: new interface at a residual-echo range."""
    cur = describe(m)
    pos = _interfaces(m)
    props = []
    for rr in _residual_peaks(m, f, y):
        # 1) the echo lies inside layer i: split it there (new layer = back part with other eps)
        for i in range(len(cur)):
            if pos[i] < rr < pos[i + 1]:
                frac = (rr - pos[i]) / (pos[i + 1] - pos[i])
                for er in EPS_STARTS:
                    a = dict(cur[i], t=cur[i]["t"] * frac)
                    b = dict(cur[i], t=cur[i]["t"] * (1 - frac), er=er, water=False, td=0.01)
                    props.append(cur[:i] + [a, b] + cur[i + 1:])
                    a2 = dict(cur[i], t=cur[i]["t"] * (1 - frac))   # or: new layer in FRONT
                    b2 = dict(cur[i], t=cur[i]["t"] * frac, er=er, water=False, td=0.01)
                    props.append(cur[:i] + [b2, a2] + cur[i + 1:])
        # 2) the echo is the back face of a new layer behind the last one
        if rr > pos[-1] + 0.001:
            for er in EPS_STARTS:
                t = (rr - pos[-1]) / np.sqrt(er)
                props.append(cur + [dict(t=t, er=er, td=0.01, water=False)])
            props.append(cur + [dict(t=(rr - pos[-1]) / 8.5, er=1.0, td=0.0, water=True)])
    # 3) thin layers whose echoes merge with a neighbour's: insert at every interface
    for i in range(len(cur) + 1):
        for er in ((2.8, 6.5) if i == 0 else (1.0, 2.8, 6.5)):
            for t in (1.5e-3, 5e-3):
                props.append(cur[:i] + [dict(t=t, er=er, td=0.01, water=False)] + cur[i:])
        props.append(cur[:i] + [dict(t=2e-3, er=1.0, td=0.0, water=True)] + cur[i:])
    return props


def _drop_leading_air(m):
    """A first layer of air is the same as a shorter standoff (exact degeneracy): fold it in."""
    while len(m.layers) > 1 and m.layers[0].theta.value < 0.5 and m.layers[0].eps_r.value < 1.03 \
            and m.layers[0].tan_d.value < 2e-3:
        t = m.layers[0].t.value
        lay = describe(m)[1:]
        mm = rebuild(m, layers=lay)
        mm.standoff.hi = max(mm.standoff.hi, m.standoff.value + t + 1e-4)
        mm.standoff.value = m.standoff.value + t
        for L, Lo in zip(mm.layers, m.layers[1:]):
            L.pinf.value, L.pinf.free = Lo.pinf.value, Lo.pinf.free
        m = mm
    return m


def _drop_trailing_air(m):
    """An air layer in front of an air backing is no layer at all (exact degeneracy)."""
    while len(m.layers) > 1 and m.back == "air" and m.layers[-1].theta.value < 0.5 \
            and m.layers[-1].eps_r.value < 1.03 and m.layers[-1].tan_d.value < 2e-3:
        mm = rebuild(m, layers=describe(m)[:-1])
        for L, Lo in zip(mm.layers, m.layers):
            L.pinf.value, L.pinf.free = Lo.pinf.value, Lo.pinf.free
        m = mm
    return m


def _retype(m):
    """Same layer count, one layer's material type changed (dielectric <-> water / air),
    keeping its optical thickness at 5 GHz."""
    cur = describe(m)
    out = []
    for i, L in enumerate(m.layers):
        eps, _ = I.layer_eps_and_derivs(L, np.array([5e9]), m.water_T if m.water_Tp is None else m.water_Tp.value,
                                        m.law, m.f_ref, m.water_law)
        tau = np.sqrt(eps[0]).real * L.t.value
        if not cur[i]["water"]:
            out.append(cur[:i] + [dict(t=tau / 8.7, er=1.0, td=0.0, water=True)] + cur[i + 1:])
            if L.eps_r.value > 1.15 and not cur[i].get("moist"):     # E036: wet version of the same layer
                for th in (0.04, 0.10):
                    out.append(cur[:i] + [dict(cur[i], moist=True, theta=th, er=max(1.1, L.eps_r.value * (1 - 2 * th)))]
                               + cur[i + 1:])
            if L.eps_r.value > 1.15:
                out.append(cur[:i] + [dict(t=tau, er=1.0, td=0.0, water=False)] + cur[i + 1:])
        else:
            for er in (2.8, 6.5):
                out.append(cur[:i] + [dict(t=tau / np.sqrt(er), er=er, td=0.05, water=False)] + cur[i + 1:])
    return out


def _first_layer_starts(m0, f, y):
    """One-layer models: thickness from the strongest echoes behind the front face."""
    d = m0.standoff.value
    peaks = [p for p in _residual_peaks(m0, f, y, k=4) if p > d + 0.002] or [d + 0.02]
    starts = []
    for rr in peaks:
        for er in EPS_STARTS[1:]:
            starts.append([dict(t=(rr - d) / np.sqrt(er), er=er, td=0.02, water=False)])
        starts.append([dict(t=(rr - d) / 8.5, er=1.0, td=0.0, water=True)])
    for t in (3e-3, 10e-3, 30e-3, 100e-3):                         # echo-free fallbacks (thick lossy)
        for er in (2.8, 5.0):
            starts.append([dict(t=t, er=er, td=0.05, water=False)])
    return starts


def _best_of(models, f, y, sigma, backs=("air", "metal"), atol=0.0, stage_f=None, keep=3, water_T=None):
    """Coarse LM (every 2nd frequency, early stop) of every candidate, then full LM of the `keep`
    best (E009 search/polish split)."""
    fs, ys = f[::2], y[::2]
    coarse = []
    for lay in models:
        for bk in backs:
            m = build(lay, best_d[0], back=bk, water_T=WATER_T0 if water_T is None else water_T)
            set_dispersion_free(m, True)      # E036: a frozen dispersion shape is mimicked by fake layers
            if stage_f is not None:          # continuation: lowest third of the band first
                m, _ = _fit(m, *stage_f, sigma, atol=atol)
            mo, c = _fit(m, fs, ys, sigma, atol=max(atol, 1e-2))
            if np.isfinite(c):
                coarse.append((c, mo))
    coarse.sort(key=lambda t: t[0])
    best = (None, np.inf)
    for _, mo in coarse[:keep]:
        mp, cp = _fit(mo, f, y, sigma)
        if cp < best[1]:
            best = (mp, cp)
    return best


def _safe(fn, model, cost, *a):
    try:
        out = fn(model, *a)
        return out if np.isfinite(out[1]) else (model, cost) + tuple(out[2:])
    except Exception:
        return (model, cost)


best_d = [0.3]
MOISTURE = False     # E036: try a free water fraction per layer
SBOX = None          # standoff box of the current inversion (front face: earliest peak - 30 mm .. + 10 mm)


def _low(f, y):
    sel = f < 20e9
    return f[sel], y[sel]


def front_guess(f, y, lo=0.15, hi=0.6):
    """Front face = EARLIEST range-profile peak (low band) clearly above the noise floor; the
    strongest peak can be a deeper, higher-contrast interface (batt in front of concrete)."""
    sel_f = f < 20e9
    r, p = W.range_profile(f[sel_f], y[sel_f])
    floor = np.median(p[(r > 2.0) & (r < 4.5)])          # noise floor far behind any wall
    sel = (r > lo) & (r < hi)
    r, p = r[sel], p[sel]
    pk = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] > 6 * floor
          and p[i] > 0.05 * p.max()]
    return float(r[pk[0]]) if pk else float(r[np.argmax(p)])


def invert(f, y, sigma, d0=None, nmax=NMAX, bic_per_layer=None, verbose=False):
    import time
    t0 = time.time()
    info = dict(log=[])
    fl, yl = _low(f, y)
    fll, yll = fl[fl < fl.min() + (fl.max() - fl.min()) / 3], yl[fl < fl.min() + (fl.max() - fl.min()) / 3]
    n_low = 2 * fl.size
    kappa = (4 * np.log(n_low)) if bic_per_layer is None else bic_per_layer
    global SBOX
    d0 = front_guess(f, y)
    SBOX = (d0 - 0.030, d0 + 0.010)
    # --- 1-layer, front face at several positions (a thin first layer's echo merges with the next)
    model, cost = None, np.inf
    for dd in d0 + np.array([0.0, -0.005, -0.010, -0.016, -0.023, 0.004]):
        best_d[0] = dd
        m0 = build([], dd)                             # empty model: residual = data
        starts = _first_layer_starts(m0, fl, yl)
        mo, co = _best_of(starts, fl, yl, sigma, stage_f=(fll, yll), atol=1e-3, keep=2)
        if co < cost:
            model, cost = mo, co
    model, cost = _fit(model, fl, yl, sigma)
    info["log"].append((1, cost))
    # --- grow
    while len(model.layers) < nmax and cost > chi2_alarm(n_low):
        best_d[0] = model.standoff.value
        wT = model.water_Tp.value if model.water_Tp is not None else None
        # same-size retyping first (a wrong material type is not fixed by adding layers)
        rt, cr = _best_of(_retype(model), fl, yl, sigma, backs=(model.back,), atol=1e-3, water_T=wT)
        if rt is not None and cr < cost - 1.0:
            model, cost = rt, cr
            if cost <= chi2_alarm(n_low):
                break
        props = _proposals(model, fl, yl)
        if not props:
            break
        cand, cc = _best_of(props, fl, yl, sigma, backs=(model.back,), atol=1e-3, water_T=wT)
        if cand is None:
            break
        cand, cc = _fit(cand, fl, yl, sigma)
        # the new layer can also change which back medium fits best
        alt = rebuild(cand, back="metal" if cand.back == "air" else "air")
        alt, ca = _fit(alt, fl, yl, sigma)
        if ca < cc:
            cand, cc = alt, ca
        info["log"].append((len(cand.layers), cc))
        if verbose:
            print(f"  grow -> {len(cand.layers)} layers chi2 {cc:.1f} (was {cost:.1f})")
        if cost - cc < kappa:
            break
        model, cost = cand, cc
    # --- add the high band (and dispersion shape)
    set_dispersion_free(model, True)
    model, cost = _fit(model, f, y, sigma)
    opts = SR.Opts()
    model, cost = _safe(lambda m, *a: SR.cycle_hop(m, f, y, sigma, opts), model, cost)
    out = _safe(lambda m, *a: SR.ambiguity_scan(m, f, y, sigma, opts), model, cost)
    model, cost = out[0], out[1]
    # retype pass on all data
    wT = model.water_Tp.value if model.water_Tp is not None else None
    rt, cr = _best_of(_retype(model), f, y, sigma, backs=(model.back,), atol=1e-3, water_T=wT)
    if rt is not None and cr < cost:
        set_dispersion_free(rt, True)
        rt, cr = _fit(rt, f, y, sigma)
        if cr < cost:
            model, cost = rt, cr
    n_all = 2 * f.size
    kappa_all = 5 * np.log(n_all)
    # --- one more growth round on all data if still inconsistent with noise
    if cost > chi2_alarm(n_all) and len(model.layers) < nmax:
        best_d[0] = model.standoff.value
        props = _proposals(model, fl, yl)
        cand, cc = _best_of(props, f, y, sigma, backs=(model.back,), atol=1e-3)
        if cand is not None:
            set_dispersion_free(cand, True)
            cand, cc = _fit(cand, f, y, sigma)
            cand, cc = _safe(lambda m, *a: SR.cycle_hop(m, f, y, sigma, opts), cand, cc)
            if cost - cc > kappa_all:
                model, cost = cand, cc
    # --- prune
    m2 = _drop_trailing_air(_drop_leading_air(model))
    if m2 is not model:
        set_dispersion_free(m2, True)
        model, cost = _fit(m2, f, y, sigma)
    changed = True
    while changed and len(model.layers) > 1:
        changed = False
        for i in range(len(model.layers)):
            lay = describe(model)
            if i + 1 < len(lay):                       # merge layer i into its neighbour
                merged = lay[:i] + [dict(lay[i + 1], t=lay[i]["t"] + lay[i + 1]["t"])] + lay[i + 2:]
            else:
                merged = lay[:i]
            mm = rebuild(model, layers=merged)
            set_dispersion_free(mm, True)
            mm, cm = _fit(mm, f, y, sigma)
            if cm - cost < kappa_all:
                model, cost, changed = mm, cm, True
                break
    # --- moisture pass (E036): free the water fraction of one dielectric layer at a time
    if MOISTURE:
        improved = True
        while improved:
            improved = False
            best_m = None
            for i, L in enumerate(model.layers):
                if L.theta.free or L.theta.value > 0.5 or L.eps_r.value < 1.03:
                    continue
                lay = describe(model)
                for th0 in (0.03, 0.10):
                    lay_i = [dict(l) for l in lay]
                    lay_i[i].update(moist=True, theta=th0, er=max(1.05, lay[i]["er"] * (1 - 2 * th0)))
                    mm = rebuild(model, layers=lay_i)
                    for Lm, Lo in zip(mm.layers, model.layers):
                        Lm.pinf.value, Lm.pinf.free = Lo.pinf.value, Lo.pinf.free
                    mm, cm = _fit(mm, f, y, sigma)
                    if cm < cost - np.log(n_all) * 2 and (best_m is None or cm < best_m[1]):
                        best_m = (mm, cm)
            if best_m is not None:
                model, cost = best_m
                improved = True
    # --- back medium check on all data
    alt = rebuild(model, back="metal" if model.back == "air" else "air")
    set_dispersion_free(alt, True)
    alt, ca = _fit(alt, f, y, sigma)
    if ca < cost:
        model, cost = alt, ca
    info["wall_s"] = time.time() - t0
    info["chi2"] = cost
    info["alarm"] = bool(cost > chi2_alarm(n_all))
    info["classes"] = classify(model, f, sigma)
    return model, cost, info


# ============================================================================ classification
_LIB = None


def library_stats(n=400, seed=12345):
    """Mean and covariance of (log eps'(10 GHz), log10 tan_d(10 GHz)) per material class, from the
    world generator's own variability (stands for a measured material database)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    rng = np.random.default_rng(seed)
    out = {}
    for nm in ALL_MATERIALS:
        if nm in ("air", "water"):
            continue
        X = []
        for _ in range(n):
            e, _ = material_law(nm, rng)
            v = e(np.array([F_REF]))[0]
            X.append([np.log(v.real), np.log10(max(-v.imag / v.real, 1e-6))])
        X = np.array(X)
        out[nm] = (X.mean(0), np.cov(X.T) + np.diag([0.02**2, 0.15**2]))
    _LIB = out
    return out


def classify(m, f, sigma):
    lib = library_stats()
    res = []
    try:
        F = I.fisher(m, f, sigma)
        crb = dict(zip(F["names"], F["crb_std"]))
    except Exception:
        crb = {}
    for L in m.layers:
        if L.theta.value > 0.5:
            res.append({"water": 1.0})
            continue
        eps, _ = I.layer_eps_and_derivs(L, np.array([F_REF]), m.water_T, m.law, m.f_ref)
        v = eps[0]
        x = np.array([np.log(v.real), np.log10(max(-v.imag / v.real, 1e-6))])
        ll = {}
        if v.real < 1.03 and -v.imag / v.real < 2e-3:
            ll["air"] = 0.0
        for nm, (mu, C) in lib.items():
            dlt = x - mu
            ll[nm] = -0.5 * dlt @ np.linalg.solve(C, dlt) - 0.5 * np.log(np.linalg.det(C))
        if "air" in ll:
            ll["air"] = max(ll.values()) + 2.0
        mx = max(ll.values())
        p = {k: np.exp(v - mx) for k, v in ll.items()}
        s = sum(p.values())
        res.append({k: v / s for k, v in sorted(p.items(), key=lambda t: -t[1])[:3]})
    return res

"""
E056: exact layer-stripping beam search for the known-N wall problem (replaces echo detection).

Top-down. A state is the wall decided down to the TOP face of medium m (m treated as a half-space).
With the media above known, the measured reflection is an exact Mobius map of one variable,
    Gamma(f) = e^{-2j k0 d} (a z + b)/(c z + d),   [a b; c d] = M_1 ... M_{m-1} [[1, r_m],[r_m, 1]],
    M_i = [[e_i, r_i],[r_i e_i, 1]],  e_i = exp(-2j k0 n_i t_i),  r_i = (n_{i-1}-n_i)/(n_{i-1}+n_i),
    z  = r_{m,n} exp(-2j k0 n_m t_m)           (next interface: thickness t_m, next medium n).
Every (t_m, n) on a fine optical-thickness grid x every library class (+ air gap, + backing) is scored
at once: the residual of the UWB band is transformed to range (Hann, zero-padded FFT) and its energy
up to the new interface (+ margin) in excess of the noise expectation is the score. Echoes from the
unknown deeper part arrive later and are gated out. Local minima along t, per (parent, n), enter the
beam. Complete walls are scored on all bands and the best are refined by the exact LM fit.
"""
from __future__ import annotations
import numpy as np
import wallsolve as WS
import wall_inverse as I
import stack as ST

C0, F_REF = WS.C0, WS.F_REF


def n_of(er, td, p, f):
    a = 2 / np.pi * np.arctan(td / (1 - p))
    h = er * (p + (1 - p - 1j * td) * (f / F_REF) ** (-a))
    s = np.sqrt(h)
    return np.where(s.imag > 0, -s, s)


TRACE = None
EPS_SCALES = (0.93, 1.0, 1.07)
TAN_SCALES = (0.7, 1.4)
T_WIDEN = (0.7, 1.4)        # library thickness ranges, widened (a prior, not the generator's box)


class Grid:
    def __init__(self, f, y, s, d, nfft=2048):
        segs = I.band_segments(f)
        # the widest band above 2 GHz is used for the gated search (UWB)
        a, b = max(((a, b) for a, b in segs if f[a] > 2e9), key=lambda ab: f[ab[1] - 1] - f[ab[0]])
        self.f, self.y, self.s = f[a:b], y[a:b], s[a:b]
        self.k0 = 2 * np.pi * self.f / C0
        self.ph = np.exp(-2j * self.k0 * d)
        self.w = np.hanning(self.f.size)
        df = self.f[1] - self.f[0]
        self.nfft = nfft
        self.rng = np.arange(nfft) * C0 / (2 * df * nfft)            # range of each bin (abs, wraps)
        self.ew = float(np.sum(self.w ** 2))
        self.over = nfft / self.f.size
        self.d = d
        self.lib = [(f"{nm}@{sc}@{tc}", er * sc, td * tc, min(p, 0.998)) for nm, er, td, p in WS.class_table()
                    for sc in (EPS_SCALES if nm not in ("foam", "batt") else (1.0,)) for tc in TAN_SCALES]
        self.nlib = [n_of(er, td, p, self.f) for _, er, td, p in self.lib]

    def range_energy(self, G):
        """G (P, F) model -> cumulative excess energy (P, nfft) vs range, in noise units."""
        r = (self.y[None, :] - G) / self.s[None, :] * self.w[None, :]
        R = np.fft.fft(np.conj(r), n=self.nfft, axis=1)              # conj: +phase convention
        E = np.abs(R) ** 2 / self.ew
        return np.cumsum(E, axis=1) / self.over - (np.arange(self.nfft) + 1)[None, :] / self.over


def mobius(G, media):
    """media: list of (n(f), t) finite layers then the half-space n_m as (n, None). Returns a,b,c,d."""
    one = np.ones_like(G.f, complex)
    A, B, Cc, D = one, 0 * one, 0 * one, one
    nprev = one
    for n, t in media:
        r = (nprev - n) / (nprev + n)
        if t is None:
            M = (one, r, r, one)
        else:
            e = np.exp(-2j * G.k0 * n * t)
            M = (e, r, r * e, one)
        A, B, Cc, D = (A * M[0] + B * M[2], A * M[1] + B * M[3], Cc * M[0] + D * M[2], Cc * M[1] + D * M[3])
        nprev = n
    return A, B, Cc, D


class RangeDFT:
    """Windowed range profile of the UWB residual, bins every `step` from d-3 cm to d+zmax."""
    def __init__(self, G, step=4e-3, zmax=1.3):
        self.z = np.arange(G.d - 0.03, G.d + zmax, step)
        k = 2 * 2 * np.pi * G.f / C0
        self.W = (G.w / G.s)[:, None] * np.exp(1j * k[:, None] * self.z[None, :])     # (F, Z)
        self.norm = G.ew * (C0 / (2 * (G.f[-1] - G.f[0]))) / step                     # noise-unit scaling
        self.step = step

    def energy(self, G, Gm, a=0, b=None):
        """|range profile|^2 of residual y - Gm (P, F) on bins a:b (P, b-a), noise units."""
        r = G.y[None, :] - Gm
        return np.abs(r @ self.W[:, a:b]) ** 2 / self.norm

    def bin(self, z):
        return int(np.clip(np.floor((z - self.z[0]) / self.step), 0, self.z.size - 1))


def _opts(G, st, N):
    kind_m, lab_m, _ = st["m"]
    air = np.ones_like(G.f, complex); metal = np.sqrt(ST.METAL_EPS + 0 * G.f)
    o = []
    if kind_m == "solid":
        if st["ns"] < N:
            o += [("solid", G.lib[i][0], G.nlib[i]) for i in range(len(G.lib)) if G.lib[i][0].split("@")[0] != lab_m.split("@")[0]]
            o.append(("gap", "air", air))
        else:
            o += [("back", "air", air), ("back", "metal", metal), ("gap", "air", air)]
    elif kind_m == "gap":
        if st["ns"] < N:
            o += [("solid", G.lib[i][0], G.nlib[i]) for i in range(len(G.lib)) if G.lib[i][0].split("@")[0] != str(st["prev"]).split("@")[0]]
        else:
            o.append(("back", "metal", metal))
    return o


def _new_state(G, R, media, m, ns, z0, prev, desc, d0, back=None, Gm=None):
    """media: finite layers [(n, t)], m: (kind, label, n) half-space below (None when complete)."""
    if Gm is None:
        A, B, Cc, D = mobius(G, media + [(m[2], None)])
        Gm = G.ph0(d0) * B / D                       # half-space m, nothing below
    else:
        A = B = Cc = D = None
    E = R.energy(G, Gm[None, :])[0]
    return dict(media=media, m=m, ns=ns, z0=z0, prev=prev, desc=desc, d0=d0, back=back,
                abcd=(A, B, Cc, D), Gm=Gm, cumE=np.cumsum(E))


def search(f, y, s, N, d, beam=200, dz=0.01, tstep=0.6e-3, margin=-0.04, win=0.05, d_tol=3e-3,
           verbose=False):
    G = Grid(f, y, s, d)
    G.ph0 = lambda d0: np.exp(-2j * G.k0 * d0)
    R = RangeDFT(G)
    states = []
    for d0 in np.arange(d - d_tol, d + d_tol + 1e-9, 0.25e-3):
        for i, (nm, er, td, p) in enumerate(G.lib):
            states.append(_new_state(G, R, [], ("solid", nm, G.nlib[i]), 1, d0, None, [], d0))
    h = d + dz
    pool, pseen = [], set()                           # complete walls: their own pool, never in the beam
    nre = lambda n: float(np.real(n[np.argmin(np.abs(G.f - 10e9))]))
    while h + margin < R.z[-1]:
        hb = R.bin(h + margin)
        kids = []                                     # (score, parent, child-spec)
        for si, st in enumerate(states):
            kids.append((st["cumE"][hb], si, None))   # stay
            kind_m, lab_m, n_m = st["m"]
            nr = nre(n_m)
            # interfaces whose optical position lies in (h - dz, h]
            zlo = max(h - dz, st["z0"] + (5e-3 if kind_m == "gap" else 1.5e-3) * nr)
            if kind_m == "gap":
                zlo = max(zlo, st["z0"] + 5e-3)
                zhi = min(h, st["z0"] + 40e-3)
            else:
                lo, hi = ST.thickness_range(lab_m.split("@")[0])
                zlo = max(zlo, st["z0"] + T_WIDEN[0] * lo * nr)
                zhi = min(h, st["z0"] + T_WIDEN[1] * hi * nr)
            if zhi <= zlo:
                continue
            zz = np.arange(zlo, zhi, tstep)
            t = (zz - st["z0"]) / (1.0 if kind_m == "gap" else nr)
            A, B, Cc, D = st["abcd"]
            ba = R.bin(zz[0] - win)
            base = st["cumE"][ba - 1] if ba > 0 else 0.0
            for kind, lab, n in _opts(G, st, N):
                r = (n_m - n) / (n_m + n)
                z = r[None, :] * np.exp(-2j * G.k0[None, :] * n_m[None, :] * t[:, None])
                Gm = G.ph0(st["d0"])[None, :] * (A * z + B) / (Cc * z + D)
                sc = base + R.energy(G, Gm, ba, hb + 1).sum(axis=1)
                for j in range(t.size):                # local minima along t only
                    if (j == 0 or sc[j] <= sc[j - 1]) and (j == t.size - 1 or sc[j] <= sc[j + 1]):
                        kids.append((float(sc[j]), si, (kind, lab, n, float(t[j]), float(zz[j]), Gm[j])))
        kids.sort(key=lambda k: k[0])
        new, seen = [], set()
        for sc, si, spec in kids:
            if len(new) >= beam:
                break
            st = states[si]
            if spec is None:
                key = (tuple((k, l, round(tt / 1.5e-3)) for k, l, tt in st["desc"]), st["m"][1] if st["m"] else None, st["back"], round(st["d0"] / 1e-3))
                if key in seen:
                    continue
                seen.add(key); new.append(st); continue
            kind, lab, n, t, zz, Gm = spec
            kind_m, lab_m, n_m = st["m"]
            desc = st["desc"] + [(kind_m, lab_m, t)]
            key = (tuple((k, l, round(tt / 1.5e-3)) for k, l, tt in desc), lab, kind, round(st["d0"] / 1e-3))
            if key in seen:
                continue
            seen.add(key)
            media = st["media"] + [(n_m, t)]
            if kind == "back":
                if key not in pseen:
                    pseen.add(key)
                    c = _new_state(G, R, media, None, st["ns"], zz, lab_m, desc, st["d0"], back=lab, Gm=Gm)
                    c["score"] = float(c["cumE"][-1]); c["cumE"] = None; c["abcd"] = None
                    pool.append(c)
                seen.discard(key)
                continue
            else:
                new.append(_new_state(G, R, media, (kind, lab, n), st["ns"] + (kind == "solid"), zz, lab_m, desc, st["d0"]))
        states = new
        if TRACE is not None:
            TRACE(h, states, kids)
        if verbose and abs((h - d) / 0.1 - round((h - d) / 0.1)) < 1e-6:
            print(f"  h={1e3 * (h - d):.0f} mm  kids {len(kids)}  best {states[0]['cumE'][hb] if states else None}  pool {len(pool)}", flush=True)
        if not states:
            break
        if len(pool) > 2000:
            pool.sort(key=lambda c: c["score"]); pool = pool[:500]
        h += dz
    pool.sort(key=lambda c: c["score"])
    return pool


def to_layers(c):
    """Complete beam state -> (layers for stack.build, slots, back)."""
    tab = {nm: (er, td, p) for nm, er, td, p in WS.class_table()}
    tab.update({f"{nm}@{sc}@{tc}": (er * sc, td * tc, p) for nm, er, td, p in WS.class_table()
                for sc in EPS_SCALES + (1.0,) for tc in TAN_SCALES})
    lay, slots, k = [], [], 0
    for kind, lab, t in c["desc"]:
        if kind == "gap":
            lay.append(dict(t=t, er=1.0001, td=1e-5, water=False)); slots.append(("gap", k - 1))
        else:
            er, td, p = tab[lab]
            lay.append(dict(t=t, er=er, td=td, water=False, pinf=min(p, 0.99))); slots.append(("solid", k)); k += 1
    return lay, slots, c["back"]


def solve(f, y, sigma, N, d_guess, n_fit=12, verbose=False, **kw):
    import time, matlib as ML, search as SR
    t0 = time.time()
    comp = search(f, y, sigma, N, d_guess, verbose=verbose, **kw)
    # rank complete walls on ALL bands (exact batched forward) before the LM fits
    rows = []
    for c in comp:
        lay, slots, back = to_layers(c)
        rows.append((c, lay, slots, back, c["d0"]))
    pri = ML.LibPrior(); old = SR._PRIORS; SR._PRIORS = (pri,)
    best = None
    try:
        for c, lay, slots, back, d0 in rows[:n_fit]:
            mod, cst = WS._fit_start(lay, slots, d0, back, f, y, sigma)
            G, _ = I.forward(mod, f, jacobian=False)
            chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
            if verbose:
                print("   fit", [(k, l, round(t * 1e3, 1)) for k, l, t in c["desc"]], back, round(chi / (2 * f.size), 2), flush=True)
            if best is None or chi < best[0]:
                best = (chi, mod, slots, back)
    finally:
        SR._PRIORS = old
    chi, mod, slots, back = best
    n_all = 2 * f.size
    return dict(model=mod, slots=slots, back=back, chi2=chi, chi2n=chi / n_all, alarm=bool(chi > ST.chi2_alarm(n_all)),
                wall_s=time.time() - t0, labels=pri.labels(mod), weak=[False] * len(mod.layers), unobserved=[],
                route="beam", t_err=[np.nan] * len(mod.layers), alternatives=[], ambiguous=False)

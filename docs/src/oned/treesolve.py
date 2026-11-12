"""
E051: time-marching survivor search over class-discrete wall structures (known N).

Idea (causality + sequence detection). The reflection response at two-way delays earlier than the
round trip to optical depth T depends only on the wall in front of T. So the wall can be decided
front to back - but the data are band limited, so a decision about depth T is only safe once the
gate has moved about one resolution cell past it. This is the situation of sequence detection over
a channel with memory (Viterbi / M-algorithm with decision delay): keep many survivors, extend each
by "no interface in the next cell" or "one interface at tau, material c behind", score every
survivor on the SAME time gate with the EXACT partial-wall response (all multiples, loss,
dispersion), prune, march on.

Pieces
  Library     class-discrete refractive index table n[c, f] (nominal member of every library
              class, water, air, metal) on the search frequencies
  slepian()   time gate as a projection: orthonormal vectors of the band's sample space whose
              energy lies in the delay window [a, b] up to a fraction eps (discrete prolate
              spheroidal sequences of the window). A residual component outside the window leaks
              into the gated misfit by at most eps of its energy  ->  bound in doc 16.
  march()     the survivor search; returns complete candidate walls (class sequence, thicknesses,
              backing), best first
  The exact continuous refinement (materials free, all bands, cycle hops) is in wallsolve2.py.
"""
from __future__ import annotations
import time
import numpy as np
import wall_toy as W
import wall_inverse as I
import stack as ST
import fastbatch as FB

C0 = W.C0
F_NREF = 5e9                    # frequency at which optical thickness is converted to thickness


# ============================================================================ library
class Library:
    """Nominal member of every material class + water + air + metal, as n[c, f]."""

    def __init__(self, f, water_T=20.0, variants=(1.0,), n_draw=300, seed=4321):
        self.f = np.asarray(f, float)
        rng = np.random.default_rng(seed)
        names, laws, base = [], [], []
        for nm in ST.ALL_MATERIALS:
            if nm in ("air", "water"):
                continue
            ds = [ST.material_law(nm, rng)[1] for _ in range(n_draw)]
            er = float(np.median([d["eps10"] for d in ds]))
            td = float(np.median([d["tan10"] for d in ds]))
            pp = float(np.median([d["p"] for d in ds]))
            for v in variants:
                names.append(nm if v == 1.0 else f"{nm}*{v:g}")
                base.append(nm)
                laws.append(self._law(er * v, td, pp))
        self.n_solid_dry = len(names)
        names.append("water"); base.append("water"); laws.append(lambda f, T=water_T: W.eps_water_lhm(f, T))
        self.WATER = len(names) - 1
        names.append("air"); base.append("air"); laws.append(lambda f: np.ones(np.shape(f), complex))
        self.AIR = len(names) - 1
        names.append("metal"); base.append("metal"); laws.append(lambda f: np.full(np.shape(f), ST.METAL_EPS))
        self.METAL = len(names) - 1
        self.names, self.base, self.laws = names, base, laws
        self.params = None
        self.ntab = self.table(self.f)
        self.nref = np.array([max(float(self._n(l(np.array([F_NREF])))[0].real), 1.0) for l in laws])
        self.solid = np.array([b not in ("air", "metal") for b in base])
        self.is_solid_idx = np.where(self.solid)[0]

    @staticmethod
    def _law(er, td, pp):
        ap = 2 / np.pi * np.arctan(td / (1 - pp))

        def eps(f):
            g = (np.asarray(f, float) / ST.F_REF) ** (-ap)
            return er * (pp + (1 - pp - 1j * td) * g)
        eps.er, eps.td, eps.pp = er, td, pp
        return eps

    @staticmethod
    def _n(eps):
        n = np.sqrt(np.asarray(eps) + 0j)
        return np.where(n.imag > 0, -n, n)

    def table(self, f):
        return np.ascontiguousarray(np.array([self._n(l(np.asarray(f, float))) for l in self.laws]))


# ============================================================================ time gates
def slepian(fb, a, b, eps=1e-3, kmax=None):
    """Orthonormal rows Q (K, F): the subspace of sampled spectra (uniform fb) whose range-domain
    energy lies in [a, b] (optical range, metres) up to a fraction eps. y(f) ~ exp(-2j k0 rho)."""
    fb = np.asarray(fb, float)
    df = fb[1] - fb[0]
    Ru = C0 / (2 * df)                                   # unambiguous range
    kap = 4 * np.pi * (fb[:, None] - fb[None, :]) / C0
    A = (b - a) * np.exp(-1j * kap * (a + b) / 2) * np.sinc(kap * (b - a) / (2 * np.pi)) / Ru
    lam, V = np.linalg.eigh(A)
    keep = lam > 1 - eps
    Q = V[:, keep].conj().T
    lam = lam[keep]
    if kmax is not None and Q.shape[0] > kmax:
        Q, lam = Q[-kmax:], lam[-kmax:]
    return np.ascontiguousarray(Q), lam


def search_subset(f, y, s, fmax=12e9, per_band=(48, 128)):
    """Search data: bands below fmax (sub-GHz, UWB), uniformly thinned. Returns f, y, sigma and
    band segments. The thinning keeps the unambiguous range c/(2 df) above 2.5 m."""
    fs, ys, ss, segs = [], [], [], []
    k = 0
    for a, b in I.band_segments(f):
        if f[a] > fmax:
            continue
        n = per_band[min(k, len(per_band) - 1)]
        step = max(1, (b - a) // n)
        idx = np.arange(a, b, step)
        segs.append((sum(len(x) for x in fs), sum(len(x) for x in fs) + len(idx)))
        fs.append(f[idx]); ys.append(y[idx]); ss.append(s[idx] / np.sqrt(1.0))
        k += 1
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ss), segs


class Gates:
    """Whitened gate projections for windows [a, T_j], all search bands stacked into one (K, F)."""

    def __init__(self, f, s, segs, a, Ts, eps=1e-3):
        self.Q = []
        for T in Ts:
            rows = []
            for (i0, i1) in segs:
                Qb, _ = slepian(f[i0:i1], a, T, eps)
                full = np.zeros((Qb.shape[0], f.size), complex)
                full[:, i0:i1] = Qb / s[i0:i1][None, :]
                rows.append(full)
            self.Q.append(np.ascontiguousarray(np.vstack(rows)))


# ============================================================================ survivors
class Beam:
    """Structure-of-arrays survivor list. Layer j: class cls[:, j], thickness t[:, j]; nl layers
    are complete; `cur` is the class of the open half-space behind them; tau = optical range of
    the last placed interface; ns = number of solid layers so far (including `cur` if solid)."""

    def __init__(self, cls, t, nl, cur, tau, ns, d):
        self.cls, self.t, self.nl, self.cur, self.tau, self.ns, self.d = cls, t, nl, cur, tau, ns, d

    def __len__(self):
        return self.cls.shape[0]

    def take(self, idx):
        return Beam(self.cls[idx], self.t[idx], self.nl[idx], self.cur[idx], self.tau[idx], self.ns[idx], self.d[idx])

    @staticmethod
    def cat(bs):
        return Beam(*[np.concatenate([getattr(b, k) for b in bs]) for k in ("cls", "t", "nl", "cur", "tau", "ns", "d")])


def _score(beam, lib, k0, Qy, Q):
    return FB.proj_batch(beam.cls, beam.t, beam.nl, beam.cur, beam.d, lib.ntab, k0, Qy, Q)


def _children(beam, lib, N, taus, tmin):
    """All one-interface extensions of every survivor with the new interface at each tau in `taus`
    (optical range) and every admissible class behind it."""
    out = []
    AIR, MET = lib.AIR, lib.METAL
    M = len(beam)
    for tau in taus:
        dt = tau - beam.tau
        thick = dt / lib.nref[beam.cur]
        ok0 = (thick >= tmin) & (beam.cur != MET)
        for c in list(lib.is_solid_idx) + [AIR, MET]:
            if c == MET:
                ok = ok0 & (beam.ns == N)                       # metal only behind the last solid (or its gap)
            elif c == AIR:
                ok = ok0 & (beam.cur != AIR)                    # gap, or air behind the wall
            else:
                ok = ok0 & (beam.ns < N) & (np.array(lib.base)[beam.cur] != lib.base[c])
            idx = np.where(ok)[0]
            if idx.size == 0:
                continue
            b = beam.take(idx)
            r = np.arange(idx.size)
            b.cls[r, b.nl] = b.cur
            b.t[r, b.nl] = thick[idx]
            b.nl = b.nl + 1
            b.cur = np.full(idx.size, c)
            b.tau = np.full(idx.size, tau)
            b.ns = b.ns + (1 if lib.solid[c] else 0)
            out.append(b)
    return out


def _signature(beam, T, mem, cell):
    """Recent state of a survivor: its open class and the (cell-rounded) position of its last
    interface if that lies within `mem` of the gate end, else -1 (settled)."""
    recent = beam.tau > T - mem
    pos = np.where(recent, np.round(beam.tau / cell).astype(np.int64), -1)
    return beam.cur.astype(np.int64) * 100003 + pos + 7 * beam.ns


def _prune(beam, score, T, M, m_sig, mem, cell):
    order = np.argsort(score, kind="stable")
    sig = _signature(beam, T, mem, cell)[order]
    # rank within signature group
    _, inv = np.unique(sig, return_inverse=True)
    cnt = np.zeros(inv.max() + 1, np.int64)
    rank = np.empty(order.size, np.int64)
    for i, g in enumerate(inv):
        rank[i] = cnt[g]; cnt[g] += 1
    keep = np.zeros(order.size, bool)
    keep[:M] = True                                   # global best M
    keep |= rank < m_sig                              # and the best few of every recent state
    sel = order[keep]
    return beam.take(sel), score[sel]


def march(f, y, s, N, d_guess, lib=None, cell=5e-3, sub=4, M=600, m_sig=3, mem=30e-3, eps=1e-3,
          d_tol=1.5e-3, d_step=0.5e-3, t_end=None, tmin=0.8e-3, trace=None, verbose=False):
    """Survivor search on the data (f, y, s) (already restricted to the search bands, see
    search_subset). Returns a list of complete candidates [(score, classes, thicknesses, back, d)]."""
    t0 = time.time()
    segs = I.band_segments(f)
    lib = Library(f) if lib is None else lib
    k0 = 2 * np.pi * f / C0
    a = d_guess - 0.05
    if t_end is None:
        t_end = _data_extent(f, y, s, segs, d_guess)
    Ts = d_guess + cell * np.arange(1, int(np.ceil(t_end / cell)) + 1)
    gates = Gates(f, s, segs, a, Ts, eps)
    # root: front face at d (grid), every solid class behind it
    ds = d_guess + np.arange(-d_tol, d_tol + 1e-9, d_step)
    R = [(d, c) for d in ds for c in lib.is_solid_idx]
    n0 = len(R)
    beam = Beam(np.zeros((n0, FB.LMAX), np.int64), np.zeros((n0, FB.LMAX)), np.zeros(n0, np.int64),
                np.array([c for _, c in R], np.int64), np.array([d for d, _ in R]),
                np.ones(n0, np.int64), np.array([d for d, _ in R]))
    log = []
    n_eval = 0
    for j, T in enumerate(Ts):
        Q = gates.Q[j]
        if Q.shape[0] == 0:
            continue
        Qy = Q @ y
        taus = T - cell + cell * (np.arange(sub) + 1) / sub
        kids = _children(beam, lib, N, taus, tmin)
        allb = Beam.cat([beam] + kids) if kids else beam
        sc = _score(allb, lib, k0, Qy, Q)
        n_eval += len(allb)
        beam, sc = _prune(allb, sc, T, M, m_sig, mem, cell)
        if trace is not None:
            trace(j, T, beam, sc, lib, Q.shape[0])
        if verbose and j % 10 == 0:
            print(f"   T={1e3 * (T - d_guess):6.1f} mm  K={Q.shape[0]:3d}  beam={len(beam):5d}  best={sc[0]:.1f}", flush=True)
        log.append((T, Q.shape[0], len(beam), float(sc[0])))
    # complete walls only: N solids placed and an air or metal half-space behind
    done = (beam.ns == N) & ((beam.cur == lib.AIR) | (beam.cur == lib.METAL))
    beam, sc = beam.take(np.where(done)[0]), sc[done]
    order = np.argsort(sc)
    out = []
    for i in order:
        nl = int(beam.nl[i])
        out.append((float(sc[i]), [int(c) for c in beam.cls[i, :nl]], [float(x) for x in beam.t[i, :nl]],
                    int(beam.cur[i]), float(beam.d[i])))
    info = dict(n_eval=n_eval, wall_s=time.time() - t0, steps=len(Ts), log=log, t_end=t_end)
    return out, info, lib


def _data_extent(f, y, s, segs, d_guess, margin=0.06, cap=1.2):
    """Optical depth behind the front face beyond which the widest search band shows no echo above
    the noise (Hann-windowed range profile, 4x the far-range level), plus a margin."""
    i0, i1 = max(segs, key=lambda ab: f[ab[1] - 1] - f[ab[0]])
    fb, yb = f[i0:i1], y[i0:i1]
    n = 8 * fb.size
    r = np.arange(n) * C0 / (2 * (fb[1] - fb[0]) * n)
    P = np.abs(np.fft.ifft(yb * np.hanning(fb.size), n))
    far = (r > d_guess + 1.5) & (r < r.max() - 0.3)
    lvl = 4 * np.max(P[far]) if np.any(far) else 0.0
    sel = (r > d_guess) & (r < d_guess + cap) & (P > lvl)
    last = r[sel].max() - d_guess if np.any(sel) else 0.1
    return float(min(cap, last + margin))

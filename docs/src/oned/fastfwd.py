"""
E054b: compiled (numba) batched exact forward, same parameterisation and result as
wallsolve.gamma_batch for walls without water (theta = 0). Verified against gamma_batch to ~1e-13.
"""
from __future__ import annotations
import numpy as np
import numba as nb
import wallsolve as WS
import stack as ST

C0, F_REF = WS.C0, WS.F_REF


@nb.njit(parallel=True, fastmath=False, cache=True)
def _gamma(X, f, N, NG, metal, nmetal):
    P, F = X.shape[0], f.size
    out = np.empty((P, F), np.complex128)
    nl = N + NG
    for p in nb.prange(P):
        ns = np.empty(nl, np.complex128)
        ts = np.empty(nl)
        # per-layer constants
        er_ = np.empty(N); td_ = np.empty(N); pp_ = np.empty(N); a_ = np.empty(N); lt_ = np.empty(N)
        for i in range(N):
            lt, la, lb, u = X[p, 1 + 5 * i], X[p, 2 + 5 * i], X[p, 3 + 5 * i], X[p, 4 + 5 * i]
            er = 1.0 + np.exp(la); td = np.exp(lb); pp = 1.0 - np.exp(-u)
            er_[i] = er; td_[i] = td; pp_[i] = pp; a_[i] = 2.0 / np.pi * np.arctan(td / (1.0 - pp)); lt_[i] = lt
        for k in range(F):
            fk = f[k]
            k0 = 2.0 * np.pi * fk / C0
            lg = -np.log(fk / F_REF)
            j = 0
            for i in range(N):
                h = er_[i] * (pp_[i] + (1.0 - pp_[i] - 1j * td_[i]) * np.exp(a_[i] * lg))
                s = np.sqrt(h)
                if s.imag > 0:
                    s = -s
                ns[j] = s; ts[j] = np.exp(lt_[i]) / np.sqrt(er_[i]); j += 1
                if i < NG:
                    g = X[p, 1 + 5 * N + i]
                    ns[j] = 1.0 + 0j; ts[j] = g if g > 0 else 0.0; j += 1
            nbk = nmetal if metal else 1.0 + 0j
            G = (ns[nl - 1] - nbk) / (ns[nl - 1] + nbk)
            for i in range(nl - 1, -1, -1):
                npv = ns[i - 1] if i > 0 else 1.0 + 0j
                e = np.exp(-2j * k0 * ns[i] * ts[i])
                r = (npv - ns[i]) / (npv + ns[i])
                G = (r + G * e) / (1.0 + r * G * e)
            out[p, k] = G * np.exp(-2j * k0 * X[p, 0])
    return out


def gamma_fast(X, f, N, back):
    NG = N - 1 + (1 if back == "metal" else 0)
    return _gamma(np.ascontiguousarray(X, float), np.ascontiguousarray(f, float), N, NG,
                  back == "metal", complex(np.sqrt(complex(ST.METAL_EPS))))


if __name__ == "__main__":
    import time
    f, y, s, tr = WS.make_problem(404, 3)
    import pt
    for back in ("air", "metal"):
        pb = pt.Problem(f, y, s, 3, back, tr["standoff"])
        X = pb.sample_prior(160, np.random.default_rng(1))
        A = WS.gamma_batch(X, pb.f, 3, back); B = gamma_fast(X, pb.f, 3, back)
        print(back, "max |diff|", float(np.max(np.abs(A - B))))
        t = time.time()
        for _ in range(20):
            gamma_fast(X, pb.f, 3, back)
        tf = (time.time() - t) / 20
        t = time.time()
        for _ in range(5):
            WS.gamma_batch(X, pb.f, 3, back)
        tn = (time.time() - t) / 5
        print(f"  numpy {tn * 1e3:.1f} ms  numba {tf * 1e3:.1f} ms  speedup {tn / tf:.1f}x")

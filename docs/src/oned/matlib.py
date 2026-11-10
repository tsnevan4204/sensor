"""
E043: material-library prior for many-layer inversion.

Why (E042): with every layer free in thickness, eps' and loss, implausible many-layer walls fit the
data as well as the true wall (2 silent failures in 5 held-out walls). Real walls are built from real
materials, so every non-water layer must look like a member of the material library.

Prior (MAP): each non-water layer's x = (log eps'(10 GHz), log10 tan d(10 GHz)) — exactly
(log eps_r, log10 tan_d) of a stack.py layer, because f_ref = 10 GHz — is penalised by the squared
Mahalanobis distance to its NEAREST library class (Gaussian per class, from the generator's material
variability: stack.library_stats). Air is its own class (eps' = 1). Liquid-water layers are physics,
not a building material: no penalty. Slivers thinner than T_MIN get a soft penalty.
The prior enters the LM as extra residual rows (search._PRIORS), so every structural decision
(pruning, backing choice) compares data chi2 + implausibility; the chi2 ALARM uses data chi2 only.

Limit (honest): the truth generator draws from the same library, so this tests the mechanism, not
robustness to materials outside the library (to test separately).
"""
from __future__ import annotations
import numpy as np
import stack as ST
import search as SR

T_MIN, T_SIG = 1.0e-3, 0.4e-3
AIR_SIG = 0.01           # air: log eps' within ~1 %


class LibPrior:
    def __init__(self, weight=1.0):
        lib = ST.library_stats()
        self.cls = [(nm, mu, np.linalg.cholesky(np.linalg.inv(C))) for nm, (mu, C) in lib.items()]
        self.w = np.sqrt(weight)

    def _layer_rows(self, L):
        er, td = L.eps_r.value, max(L.tan_d.value, 1e-7)
        x = np.array([np.log(er), np.log10(td)])
        dx = np.array([[1 / er, 0.0], [0.0, 1 / (td * np.log(10))]])     # d x / d (er, td)
        best = None
        for nm, mu, Lc in self.cls:
            z = Lc.T @ (x - mu)
            if best is None or z @ z < best[0]:
                best = (z @ z, z, Lc.T @ dx, nm)
        za = np.log(er) / AIR_SIG                                         # air class: eps' only
        if za ** 2 < best[0]:
            best = (za ** 2, np.array([za, 0.0]), np.array([[1 / (er * AIR_SIG), 0.0], [0.0, 0.0]]), "air")
        return best

    def rows(self, model):
        free = model.free_names()
        col = {n: j for j, n in enumerate(free)}
        R, J = [], []
        for L in model.layers:
            water = L.theta.value > 0.5 and not L.theta.free
            if not water:
                _, z, dz, _ = self._layer_rows(L)
                for k in range(2):
                    row = np.zeros(len(free))
                    for j, key in enumerate(("eps_r", "tan_d")):
                        o = model.owner(f"{L.name}.{key}")
                        if o is not None:
                            row[col[o]] += dz[k, j]
                    R.append(self.w * z[k]); J.append(self.w * row)
            t = L.t.value                                                 # sliver penalty
            if t < T_MIN:
                row = np.zeros(len(free))
                o = model.owner(f"{L.name}.t")
                if o is not None:
                    row[col[o]] = -1 / T_SIG
                R.append(self.w * (T_MIN - t) / T_SIG); J.append(self.w * row)
        if not R:
            return np.zeros(0), np.zeros((0, len(free)))
        return np.array(R), np.array(J)

    def penalty(self, model):
        r, _ = self.rows(model)
        return float(r @ r)

    def labels(self, model):
        out = []
        for L in model.layers:
            if L.theta.value > 0.5 and not L.theta.free:
                out.append("water")
            else:
                out.append(self._layer_rows(L)[3])
        return out


def invert(f, y, sigma, weight=1.0, verbose=False):
    """E042 global blocky inversion + segmentation + exact polish/pruning, all under the library
    prior. Returns (model, data chi2, info)."""
    import profile as PF
    import wall_inverse as I
    pri = LibPrior(weight)
    old = SR._PRIORS
    SR._PRIORS = (pri,)
    try:
        model, cost, info = PF.invert(f, y, sigma, verbose=verbose)
    finally:
        SR._PRIORS = old
    G, _ = I.forward(model, f, jacobian=False)
    chi = float(np.sum(np.abs((G - y) / sigma) ** 2))
    n_all = 2 * f.size
    info.update(chi2=chi, map_cost=cost, prior=pri.penalty(model), alarm=bool(chi > ST.chi2_alarm(n_all)),
                labels=pri.labels(model))
    return model, chi, info

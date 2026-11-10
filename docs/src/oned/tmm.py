"""
E041: exact transfer-matrix (reflection-recursion) forward with an analytic adjoint Jacobian, for
plain layered models (no beam, facing, roughness, gap, chain, cal or frames).

For planar layers at normal incidence the exact-star DEC solve and the reflection recursion are
the same mathematics (gate: |DEC - TMM| ~ 1e-13); the recursion costs O(F N) with no cells, so
many-layer walls are cheap. Jacobian by reverse accumulation through the recursion:
  G_i = (r_i + G_{i+1} e_i) / (1 + r_i G_{i+1} e_i),  G_{N+1} = r_{N+1},  y = G_1 exp(-2j k0 d)
  dG_i/dG_{i+1} = e (1 - r^2) / D^2,  dG_i/dr_i = (1 - (G e)^2) / D^2,  dG_i/de_i = G (1 - r^2) / D^2,
  D = 1 + r G e;  r_i = (n_{i-1} - n_i)/(n_{i-1} + n_i);  e_i = exp(-2j k0 n_i t_i).
"""
from __future__ import annotations
import numpy as np
import wall_inverse as I

C0 = I.C0


def _n(eps):
    n = np.sqrt(eps + 0j)
    return np.where(n.imag > 0, -n, n)


def applicable(model):
    return (model.beam is None and model.facing is None and model.rough is None and model.gap is None
            and model.chain is None and model.cal is None and model.frame_offsets is None
            and model.band_offset is None)


def forward(model, f, jacobian=True):
    f = np.asarray(f, float)
    model.sync_ties()
    names = model.free_names()
    col = {nm: j for j, nm in enumerate(names)}
    vals = {n: p for n, _, p in model.items()}          # one pass (model.owner() rebuilds this per call)

    def owner(n):
        p = vals.get(n)
        if p is None:
            return None
        src = p.tie if p.tie is not None else n
        return src if (vals[src].free and vals[src].tie is None) else None
    T = model.water_Tp.value if model.water_Tp is not None else model.water_T
    k0 = 2 * np.pi * f / C0
    N = len(model.layers)
    ns, derivs = [np.ones_like(f, complex)], []
    for L in model.layers:
        eps, d = I.layer_eps_and_derivs(L, f, T, model.law, model.f_ref, model.water_law)
        ns.append(_n(eps))
        derivs.append(d)
    nb = _n(np.full(f.shape, model.eps_back, complex))
    ns.append(nb)
    r = [(ns[i] - ns[i + 1]) / (ns[i] + ns[i + 1]) for i in range(N + 1)]      # r[i]: top of layer i (i=N: back)
    e = [np.exp(-2j * k0 * ns[i + 1] * model.layers[i].t.value) for i in range(N)]
    G = [None] * (N + 1)
    G[N] = r[N]
    for i in range(N - 1, -1, -1):
        G[i] = (r[i] + G[i + 1] * e[i]) / (1 + r[i] * G[i + 1] * e[i])
    ph = np.exp(-2j * k0 * model.standoff.value)
    y = G[0] * ph
    if not jacobian:
        return y, None
    J = np.zeros((f.size, len(names)), complex)
    a = ph.copy()                     # dy/dG_i
    dr = [None] * (N + 1)
    de = [None] * N
    for i in range(N):
        D = 1 + r[i] * G[i + 1] * e[i]
        dr[i] = a * (1 - (G[i + 1] * e[i]) ** 2) / D ** 2
        de[i] = a * G[i + 1] * (1 - r[i] ** 2) / D ** 2
        a = a * e[i] * (1 - r[i] ** 2) / D ** 2
    dr[N] = a
    for i, L in enumerate(model.layers):
        nm1, n, nn = ns[i], ns[i + 1], ns[i + 2]
        # n_i enters r_i (as the lower medium), r_{i+1} (as the upper medium) and e_i
        dn = dr[i] * (-2 * nm1 / (nm1 + n) ** 2) + dr[i + 1] * (2 * nn / (n + nn) ** 2) \
            + de[i] * (-2j * k0 * L.t.value * e[i])
        ot = owner(f"{L.name}.t")
        if ot is not None:
            J[:, col[ot]] += de[i] * (-2j * k0 * n * e[i])
        for key in ("eps_r", "tan_d", "theta", "pinf", "water_T"):
            o = owner(f"{L.name}.{key}") if key != "water_T" else (owner("water_T") if model.water_Tp is not None else None)
            if o is None or key not in derivs[i]:
                continue
            J[:, col[o]] += dn * derivs[i][key] / (2 * n)
    if owner("standoff") is not None:
        J[:, col["standoff"]] += y * (-2j * k0)
    return y, J

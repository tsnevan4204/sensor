"""
Exact vs standard Hodge stars for the 1D layered wall (frequency domain).

Both models use the SAME topology: D0 (node-to-edge incidence). They differ only in the
Hodge stars on each primal edge of length l, wavenumber k = w*sqrt(eps*mu):

  standard DEC :  *mu^-1 = 1/(mu l)                 *eps (per half cell) = eps l / 2
  exact        :  *mu^-1 = 1/(mu l) * (kl / sin kl)  *eps (per half cell) = eps l / 2 * tan(kl/2)/(kl/2)

The exact stars come from solving E'' + k^2 E = 0 exactly inside each homogeneous edge.
Both correction factors -> 1 as l -> 0, so standard DEC is their small-cell limit.

System at angular frequency w (after dividing Ampere by jw):
  [ D0^T diag(S_mu/(jw)) D0 + jw * M_eps + boundary ] e = source
where M_eps lumps the two half-cell stars onto each node and S_mu = *mu^-1.
"""
import numpy as np
from wall_toy import C0, ETA0, Scene, reflection_tmm, _sqrt_branch, scene_dry_wall, scene_wet_insulation, scene_wicked_drywall

MU0 = ETA0 / C0   # 2019 SI: mu0 = eta0 / c0 (4*pi*1e-7 is off by ~1e-10)
EPS0 = 1 / (MU0 * C0**2)


def build_mesh(scene: Scene, cells_per_layer):
    """Nodes on every interface; each layer split into n equal edges. Returns edge lengths
    and, per edge, a function f -> complex relative eps."""
    lengths, epsf = [], []
    for L in scene.layers:
        n = max(1, int(cells_per_layer))
        lengths += [L.thickness / n] * n
        epsf += [L.eps] * n
    return np.array(lengths), epsf


def dec_reflection(f, scene: Scene, cells_per_layer=1, exact=True):
    """Reflection at the wall front face, then the analytic air gap (the hybrid)."""
    lengths, epsf = build_mesh(scene, cells_per_layer)
    N = len(lengths)                      # edges; nodes = N + 1
    out = np.empty(len(f), dtype=complex)
    for m, fm in enumerate(f):
        w = 2 * np.pi * fm
        er = np.array([e(np.array([fm]))[0] for e in epsf])
        n = _sqrt_branch(er)
        k = w / C0 * n
        x = k * lengths
        s_mu = 1 / (MU0 * lengths)                       # standard *mu^-1
        m_half = EPS0 * er * lengths / 2                  # standard *eps, half cell
        if exact:
            s_mu = s_mu * (x / np.sin(x))
            m_half = m_half * (np.tan(x / 2) / (x / 2))
        A = np.zeros((N + 1, N + 1), dtype=complex)
        for i in range(N):                                # assemble edge i between nodes i, i+1
            ys = s_mu[i] / (1j * w)
            A[i, i] += ys; A[i + 1, i + 1] += ys; A[i, i + 1] -= ys; A[i + 1, i] -= ys
            A[i, i] += 1j * w * m_half[i]; A[i + 1, i + 1] += 1j * w * m_half[i]
        n_back = _sqrt_branch(scene.backing_eps(np.array([fm])))[0]
        A[0, 0] += 1 / ETA0                               # radar-side port: H(0) = (2a - e0)/eta0
        A[N, N] += n_back / ETA0                          # open back: H = e_N / eta_back
        rhs = np.zeros(N + 1, dtype=complex); rhs[0] = 2 / ETA0   # a = 1
        e = np.linalg.solve(A, rhs)
        out[m] = e[0] - 1.0                               # Gamma at the front face
    return out * np.exp(-2j * 2 * np.pi * f / C0 * scene.standoff)


if __name__ == "__main__":
    f = np.linspace(77e9, 81e9, 64)
    scenes = {"dry": scene_dry_wall(), "wet insulation": scene_wet_insulation(0.05)}
    for name, sc in scenes.items():
        ref = reflection_tmm(f, sc)
        err = np.max(np.abs(dec_reflection(f, sc, 1, exact=True) - ref))
        print(f"{name:15s} EXACT stars, 1 cell per layer (4 nodes): max |error| = {err:.1e}")
        for cpl in (10, 40, 160, 640):
            err = np.max(np.abs(dec_reflection(f, sc, cpl, exact=False) - ref))
            print(f"{name:15s} standard stars, {cpl:4d} cells/layer:            max |error| = {err:.1e}")
    sc = scene_wicked_drywall()
    ref = reflection_tmm(f, sc)   # reference itself uses 40 constant sub-layers
    err = np.max(np.abs(dec_reflection(f, sc, 1, exact=True) - ref))
    print(f"wicked (40 sub-layers as edges) EXACT stars: max |error| = {err:.1e}")

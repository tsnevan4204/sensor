"""S001 part C: why 2D needs learned stars, cell Schur complements, and localization (docs/11, Part C).
Writes to results/2026-11-09_S001_stability/."""
import json, os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # BLAS oversubscription is ~500x slower here
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from dec2d import *
from cells import *
from infsup import beta_dense

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "2026-11-09_S001_stability")
os.makedirs(OUT, exist_ok=True)
R = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def antiherm_min(S):
    H = (S - S.conj().T) / 2j
    return float(np.linalg.eigvalsh((H + H.conj().T) / 2).min())


# ---------------------------------------------------------------- C1 Trefftz rank and star defect
C1 = dict(rank_1d=trefftz_rank_1d(3.0, 0.7)[:2])
khs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
C1["kH"] = khs.tolist()
C1["rank_2d"] = [trefftz_rank_2d(x)[:2] for x in khs]
C1["defect_2d_best_4x4_star"] = [star_defect(*trefftz_family_2d(x)) for x in khs]
# 1D analogue: defect is zero (exact star exists)
v1 = []
for x in khs:
    rng = np.random.default_rng(0)
    cc = rng.standard_normal((2, 50)) + 1j * rng.standard_normal((2, 50))
    val = np.array([[1, 1], [np.exp(-1j * x), np.exp(1j * x)]]) @ cc
    flux = np.vstack([np.array([1j * x, -1j * x]) @ cc, np.array([-1j * x * np.exp(-1j * x), 1j * x * np.exp(1j * x)]) @ cc])
    v1.append(star_defect(val, flux))
C1["defect_1d"] = v1
sl = np.polyfit(np.log(khs[:4]), np.log(C1["defect_2d_best_4x4_star"][:4]), 1)[0]
C1["defect_2d_slope_small_kH"] = float(sl)
R["C1_trefftz"] = C1
log("C1", C1["rank_2d"], C1["defect_2d_best_4x4_star"])

# ---------------------------------------------------------------- C2 cell Schur complement: dense, passive, exact
nf = 16
C2 = {}
for label, epsf in [("lossless", None), ("lossy", lambda x, y: 4.0 - 1.0j + 0 * x)]:
    P = cell_problem(nf, 1.0, epsf)
    k = 3.0
    A = (P.D.T @ sp.diags(P.ratio) @ P.D - k**2 * sp.diags(P.area * P.eps_node)).toarray()
    b = boundary_nodes(P)
    S = schur(A, b)
    C2[label] = dict(n_boundary=int(b.size),
                     fraction_nonzero=float(np.mean(np.abs(S) > 1e-12 * np.abs(S).max())),
                     antiherm_min_eig=antiherm_min(S), antiherm_scale=float(np.abs(S).max()),
                     symmetric_err=float(np.abs(S - S.T).max() / np.abs(S).max()))
# Dirichlet-eigenvalue pole of the lossless cell
P = cell_problem(nf, 1.0)
lam = interior_dirichlet_eigs(P, 2)
kp = np.sqrt(lam[0])
kk = np.linspace(0.5, 7.5, 1401)
smax = []
b = boundary_nodes(P)
K = (P.D.T @ sp.diags(P.ratio) @ P.D).toarray(); Md = np.diag(P.area)
for x in kk:
    smax.append(np.abs(schur(K - x**2 * Md, b)).max())
C2["cell_first_dirichlet_k"] = float(kp)
C2["continuum_first_dirichlet_k"] = float(np.pi * np.sqrt(2))
C2["k_of_max_|S|_on_scan"] = float(kk[int(np.argmax(smax))])

# exact static condensation on a 4x4 array of cells (global Robin BC): skeleton system == fine solution
Nc, nf2, k = 4, 8, 9.0
n = Nc * nf2
G = Problem(Grid(n, n, 1.0 / n))
A = G.assemble(k)
d = (np.cos(0.3), np.sin(0.3))
f = G.robin_rhs_planewave(k, d)
u_fine, _ = solve(A, f)
I = np.rint(G.X * n).astype(int); J = np.rint(G.Y * n).astype(int)
skel = np.where((I % nf2 == 0) | (J % nf2 == 0))[0]
Ad = A.toarray()
Ssk = schur(Ad, skel)
inner = np.setdiff1d(np.arange(G.grid.nn), skel)
# the RHS lives on the boundary only (subset of skeleton), so condensed RHS = f[skel]
u_sk = np.linalg.solve(Ssk, f[skel])
C2["static_condensation_max_rel_err"] = float(np.abs(u_sk - u_fine[skel]).max() / np.abs(u_fine).max())
Ainv = np.linalg.inv(Ad)
C2["Sinv_equals_Ainv_bb_rel_err"] = float(np.abs(np.linalg.inv(Ssk) - Ainv[np.ix_(skel, skel)]).max() / np.abs(Ainv).max())
# assembling per-cell Schur complements (cell-local, natural BC) + global Robin gives the same skeleton matrix
Ssum = np.zeros_like(Ssk)
pos = {p: i for i, p in enumerate(skel)}
cellP = cell_problem(nf2, 1.0 / Nc)
Kc = (cellP.D.T @ sp.diags(cellP.ratio) @ cellP.D - k**2 * sp.diags(cellP.area)).toarray()
bc = boundary_nodes(cellP)
Sc = schur(Kc, bc)
Ic = np.rint(cellP.X[bc] / cellP.grid.h).astype(int); Jc = np.rint(cellP.Y[bc] / cellP.grid.h).astype(int)
for ci in range(Nc):
    for cj in range(Nc):
        gidx = [pos[(cj * nf2 + jj) * (n + 1) + ci * nf2 + ii] for ii, jj in zip(Ic, Jc)]
        Ssum[np.ix_(gidx, gidx)] += Sc
Bsk = (1j * k * G.B_mat()).toarray()[np.ix_(skel, skel)]
C2["cellwise_schur_assembly_rel_err"] = float(np.abs(Ssum + Bsk - Ssk).max() / np.abs(Ssk).max())
R["C2_schur"] = C2
log("C2", C2)

# ---------------------------------------------------------------- C3 compressed cell-local vertex star  Y = P^T S P
C3 = []
for Nc in [4, 8, 16]:
    for kH in [0.25, 0.5, 1.0, 1.5]:
        k = kH * Nc
        nfc = 16
        cP = cell_problem(nfc, 1.0 / Nc)
        Kc = (cP.D.T @ sp.diags(cP.ratio) @ cP.D - k**2 * sp.diags(cP.area)).toarray()
        bnd, Pm = vertex_prolongation(cP)
        Y = Pm.T @ schur(Kc, bnd) @ Pm                       # 4x4, passive by construction (Pm real)
        Gc = Problem(Grid(Nc, Nc, 1.0 / Nc))
        nn = Gc.grid.nn
        Ac = np.zeros((nn, nn), complex)
        for ci in range(Nc):
            for cj in range(Nc):
                idx = [Gc.grid.node(ci, cj), Gc.grid.node(ci + 1, cj), Gc.grid.node(ci + 1, cj + 1), Gc.grid.node(ci, cj + 1)]
                Ac[np.ix_(idx, idx)] += Y
        Ac += 1j * k * Gc.B_mat().toarray()
        d = (np.cos(0.3), np.sin(0.3))
        f = Gc.robin_rhs_planewave(k, d)
        uex = np.exp(-1j * k * (d[0] * Gc.X + d[1] * Gc.Y))
        u_c = np.linalg.solve(Ac, f)
        Astd = Gc.assemble(k).toarray()
        u_s = np.linalg.solve(Astd, f)
        N = Gc.energy_norm(k).toarray()
        C3.append(dict(Nc=Nc, kH=kH, k=k, err_compressed=l2err(Gc, u_c, uex), err_std_DEC=l2err(Gc, u_s, uex),
                       beta_compressed=beta_dense(Ac, N), beta_std=beta_dense(Astd, N),
                       Y_antiherm_min=antiherm_min(Y), Ac_antiherm_min=antiherm_min(Ac)))
R["C3_compressed_vertex_star"] = C3
log("C3 done")

# ---------------------------------------------------------------- C4 localization of the exact coarse operator (LOD-type)
def lod_experiment(Nc, nf, kH, ms=(1, 2, 3, 4)):
    n = Nc * nf
    k = kH * Nc
    G = Problem(Grid(n, n, 1.0 / n))
    A = G.assemble(k).tocsc()
    I = np.rint(G.X * n).astype(int); J = np.rint(G.Y * n).astype(int)
    V = np.where((I % nf == 0) & (J % nf == 0))[0]                   # coarse vertices
    W = np.setdiff1d(np.arange(G.grid.nn), V)
    cv = np.stack([I[V] // nf, J[V] // nf], 1)
    # coarse hats on the fine grid
    Phi = np.zeros((G.grid.nn, V.size))
    for a, (ci, cj) in enumerate(cv):
        Phi[:, a] = np.maximum(0, 1 - np.abs(G.X * Nc - ci)) * np.maximum(0, 1 - np.abs(G.Y * Nc - cj))
    d = (np.cos(0.3), np.sin(0.3))
    f = G.robin_rhs_planewave(k, d)
    u_fine, _ = solve(A, f)
    Gc = Problem(Grid(Nc, Nc, 1.0 / Nc)); Nnorm = Gc.energy_norm(k).toarray()
    # map coarse-grid ordering to V ordering
    order = np.array([np.where((cv[:, 0] == i) & (cv[:, 1] == j))[0][0] for j in range(Nc + 1) for i in range(Nc + 1)])
    out = dict(Nc=Nc, nf=nf, kH=kH, k=k, rows=[])
    Wm = np.zeros(G.grid.nn, bool); Wm[W] = True
    Acsr = A.tocsr()
    for m in list(ms) + [None]:
        Psi = Phi.astype(complex).copy()
        if m is None:     # ideal correctors: one global solve on W for all vertices
            lu = ComplexLU(Acsr[W][:, W])
            rhs = -(Acsr[W] @ Phi)
            Psi[W] += np.stack([lu.solve(rhs[:, a]) for a in range(V.size)], 1)
        else:             # correctors localized to the m-th coarse patch around each vertex
            for a, (ci, cj) in enumerate(cv):
                patch = Wm & (np.abs(G.X * Nc - ci) < m + 1e-9) & (np.abs(G.Y * Nc - cj) < m + 1e-9)
                p = np.where(patch)[0]
                Ap = Acsr[p]
                Psi[p, a] += ComplexLU(Ap[:, p]).solve(-(Ap @ Phi[:, a]))
        # real correctors from the lossless real part (no impedance): Psi real => Psi^T A Psi passive
        if m is not None:
            Ar = (G.D.T @ sp.diags(G.ratio) @ G.D - k**2 * sp.diags(G.area)).tocsr()
            Pr = Phi.copy()
            for a, (ci, cj) in enumerate(cv):
                p = np.where(Wm & (np.abs(G.X * Nc - ci) < m + 1e-9) & (np.abs(G.Y * Nc - cj) < m + 1e-9))[0]
                Ap = Ar[p]
                Pr[p, a] += spla.splu(Ap[:, p].tocsc()).solve(-(Ap @ Phi[:, a]))
            Yr = Pr.T @ (A @ Pr)
            ur = np.linalg.solve(Yr, Pr.T @ f)
            err_R = float(np.linalg.norm(ur - u_fine[V]) / np.linalg.norm(u_fine[V])); ah_R = antiherm_min(Yr) / np.abs(Yr).max()
        else:
            err_R = ah_R = None
        Yt = Psi.T @ (A @ Psi)                 # bilinear Galerkin, A-orthogonal correctors (exact for m=inf)
        Yh = Psi.conj().T @ (A @ Psi)          # sesquilinear Galerkin (passive by construction)
        ft, fh = Psi.T @ f, Psi.conj().T @ f
        ut, uh = np.linalg.solve(Yt, ft), np.linalg.solve(Yh, fh)
        ref = u_fine[V]
        Yt_c = Yt[np.ix_(order, order)]
        out["rows"].append(dict(m="inf" if m is None else m,
                                err_T=float(np.linalg.norm(ut - ref) / np.linalg.norm(ref)),
                                err_H=float(np.linalg.norm(uh - ref) / np.linalg.norm(ref)),
                                antiherm_min_T=antiherm_min(Yt) / np.abs(Yt).max(),
                                antiherm_min_H=antiherm_min(Yh) / np.abs(Yh).max(),
                                beta_T=beta_dense(Yt_c, Nnorm), err_realcorr=err_R, antiherm_min_realcorr=ah_R))
        if m is None:
            # decay of the ideal coarse operator away from the diagonal
            dist = np.max(np.abs(cv[:, None, :] - cv[None, :, :]), axis=2)
            dec = [float(np.abs(Yt[dist == r]).max() / np.abs(Yt).max()) for r in range(0, min(Nc, 9))]
            out["ideal_offdiag_decay_by_coarse_distance"] = dec
    return out


C4 = [lod_experiment(16, 8, 0.5), lod_experiment(16, 8, 1.0), lod_experiment(24, 6, 1.0)]
R["C4_lod_localization"] = C4
log("C4 done")
R["wall_time_s"] = time.time() - T0

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].loglog(khs, C1["defect_2d_best_4x4_star"], "o-", label="2D: best 4x4 vertex star")
ax[0].loglog(khs, np.maximum(np.array(v1), 1e-17), "s-", label="1D: best 2x2 star (exact)")
ax[0].set_xlabel("k H"); ax[0].set_ylabel("relative flux defect on plane waves"); ax[0].legend()
ax[0].set_title("no finite exact star in 2D")
ax[1].semilogy(kk, smax); ax[1].axvline(kp, color="r", lw=0.8, label="first interior Dirichlet eigenvalue")
ax[1].set_xlabel("k H"); ax[1].set_ylabel(r"$\max|S_{ij}(k)|$"); ax[1].legend(); ax[1].set_title("cell Schur complement: pole")
for o in C4:
    r = o["ideal_offdiag_decay_by_coarse_distance"]
    ax[2].semilogy(range(len(r)), r, "o-", label=f"Nc={o['Nc']}, kH={o['kH']}")
ax[2].set_xlabel("coarse distance (cells)"); ax[2].set_ylabel("max |Y_ij| / max |Y|"); ax[2].legend()
ax[2].set_title("ideal coarse operator decays exponentially")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_C_cells.png"), dpi=130)

with open(os.path.join(OUT, "results_cells.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)
log("done")

"""S001 part B: 2D DEC solver verification and discrete inf-sup experiments (docs/11, Part B).
Writes to results/2026-11-09_S001_stability/.  About 10-15 min on one core."""
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
from infsup import beta, passivity_eig
import mie

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "2026-11-09_S001_stability")
os.makedirs(OUT, exist_ok=True)
R = {}
T0 = time.time()
STARS = {"DEC (0,0)": (0.0, 0.0), "blended (1/2,1/2)": (0.5, 0.5), "Q1 FEM (1,1)": (1.0, 1.0)}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def unit_square(n, **kw):
    return Problem(Grid(n, n, 1.0 / n), **kw)


def planewave_error(n, k, a1=0.0, a0=0.0, theta=0.3):
    P = unit_square(n, alpha1=a1, alpha0=a0)
    d = (np.cos(theta), np.sin(theta))
    u, _ = solve(P.assemble(k), P.robin_rhs_planewave(k, d, consistent=bool(a0)))
    return l2err(P, u, np.exp(-1j * k * (d[0] * P.X + d[1] * P.Y)))


# ---------------------------------------------------------------- V1 plane wave convergence
V1 = {}
for name, (a1, a0) in STARS.items():
    ns = [20, 40, 80, 160, 320] + ([640] if a1 == 0 else [])
    e = [planewave_error(n, 10.0, a1, a0) for n in ns]
    V1[name] = dict(n=ns, err=e, rate=list(np.log2(np.array(e[:-1]) / np.array(e[1:]))))
R["V1_planewave_k10"] = V1
log("V1", {k: v["rate"][-1] for k, v in V1.items()})

# discrete plane wave: interior residual (exactness of the assembled operator)
res = {}
for name, (a1, a0) in STARS.items():
    n, k, th = 60, 7.0, 0.4
    P = unit_square(n, alpha1=a1, alpha0=a0)
    q = dispersion_symbol(np.array([th]), k / n, a1, a0)[0] * n
    u = np.exp(-1j * q * (np.cos(th) * P.X + np.sin(th) * P.Y))
    r = (P.D.T @ P.S_mat(k) @ P.D - k**2 * P.M_mat(k)) @ u
    res[name] = dict(interior_residual=float(np.abs(r[~P.on_bnd]).max()), kappa_over_k_minus_1=float(q / k - 1))
R["V1_discrete_planewave"] = res

# dispersion: max over directions of |kappa/k - 1| vs points per wavelength
ppws = np.array([4, 6, 8, 10, 15, 20, 40])
th = np.linspace(0, np.pi / 4, 46)
disp = {}
for name, (a1, a0) in STARS.items():
    disp[name] = [float(np.abs(dispersion_symbol(th, 2 * np.pi / p, a1, a0) / (2 * np.pi / p) - 1).max()) for p in ppws]
disp["Q1 + AW mass (1,1/2)"] = [float(np.abs(dispersion_symbol(th, 2 * np.pi / p, 1.0, 0.5) / (2 * np.pi / p) - 1).max()) for p in ppws]
R["V1_dispersion_max_rel_phase_error"] = dict(ppw=ppws.tolist(), **disp)

# ---------------------------------------------------------------- V2 Mie series (PML, staircase-averaged eps)
k, a, er, L, dp = 4 * np.pi, 0.4, 2.25, 1.0, 0.3
V2 = dict(n=[], h=[], err=[])
for n in [100, 200, 400, 800]:
    h = 2 * (L + dp) / n
    g = Grid(n, n, h, -(L + dp), -(L + dp))
    pml = PML(dp, smax=60.0, xlim=(-(L + dp), L + dp), ylim=(-(L + dp), L + dp))
    P = Problem(g, pml=pml, impedance=np.zeros(g.nn, bool))
    P.eps_node = dual_cell_eps(P, lambda x, y: np.where(x**2 + y**2 < a * a, er, 1.0) + 0j)
    uinc = np.exp(-1j * k * P.X)
    us, _ = solve(P.assemble(k), k**2 * P.area * (P.eps_node - 1) * uinc)
    mask = (np.abs(P.X) <= L) & (np.abs(P.Y) <= L)
    V2["n"].append(n); V2["h"].append(h)
    V2["err"].append(l2err(P, us + uinc, mie.total_field(P.X, P.Y, k, a, er), mask))
V2["rate"] = list(np.log2(np.array(V2["err"][:-1]) / np.array(V2["err"][1:])))
R["V2_mie"] = dict(k=k, a=a, eps_r=er, **V2)
log("V2", V2["err"], V2["rate"])

# ---------------------------------------------------------------- V3 adjoint gradient vs central FD
n = 40; k = 8.0
P = unit_square(n)
P.eps_node = dual_cell_eps(P, lambda x, y: np.where((x - .5)**2 + (y - .5)**2 < .04, 3.0 - 0.2j, 1.0) + 0j)
uinc = np.exp(-1j * k * P.X)
rng = np.random.default_rng(3)
c = rng.standard_normal(P.grid.nn) * P.on_bnd           # "sensor" functional J = c^T u_s


def J_of(P):
    A = P.assemble(k)
    f = k**2 * P.area * (P.eps_node - 1) * uinc
    us, lu = solve(A, f)
    return c @ us, us, lu


J0, us, lu = J_of(P)
lam = lu.solve(c.astype(complex))                     # A^T lam = c, A complex symmetric
grad = k**2 * P.area * lam * (us + uinc)               # dJ/d eps_i  (eps_i complex-analytic)
rel = []
for i in rng.choice(np.where(np.abs(P.eps_node - 1) > 0)[0], 6, replace=False):
    hstep = 1e-6
    Pp = unit_square(n); Pp.eps_node = P.eps_node.copy(); Pp.eps_node[i] += hstep
    Pm = unit_square(n); Pm.eps_node = P.eps_node.copy(); Pm.eps_node[i] -= hstep
    fd = (J_of(Pp)[0] - J_of(Pm)[0]) / (2 * hstep)
    rel.append(abs(fd - grad[i]) / abs(grad[i]))
R["V3_adjoint_vs_fd_max_rel"] = float(max(rel))
log("V3", max(rel))

# ---------------------------------------------------------------- B1 inf-sup vs k (open square, impedance all round)
B1 = {}
ks = [5, 10, 20, 40, 80, 160]
for name, (a1, a0) in STARS.items():
    rows = []
    for kk in ks:
        n = int(np.ceil(10 * kk / (2 * np.pi)))          # 10 points per wavelength
        P = unit_square(n, alpha1=a1, alpha0=a0)
        A = P.assemble(kk)
        bN = beta(A, P.energy_norm(kk))
        rows.append(dict(k=kk, n=n, beta_N=bN, k_beta_N=kk * bN))
    B1[name + " ppw10"] = rows
rows = []
for kk in [5, 10, 20, 40, 80]:
    h = np.sqrt(3.95 / kk**3); n = int(np.ceil(1 / h))  # k^3 h^2 = const (asymptotic regime)
    P = unit_square(n)
    rows.append(dict(k=kk, n=n, beta_N=beta(P.assemble(kk), P.energy_norm(kk)), k3h2=kk**3 / n**2))
    rows[-1]["k_beta_N"] = kk * rows[-1]["beta_N"]
B1["DEC (0,0) k^3h^2 fixed"] = rows
rows = []
for n in [16, 32, 64, 128, 256, 512]:                   # mesh convergence of beta at fixed k
    P = unit_square(n)
    rows.append(dict(n=n, beta_N=beta(P.assemble(20.0), P.energy_norm(20.0))))
B1["DEC k=20 mesh refinement"] = rows
R["B1_infsup_vs_k"] = B1
log("B1 done")

# ---------------------------------------------------------------- B2 pollution at 10 ppw
B2 = {}
for name, (a1, a0) in STARS.items():
    B2[name] = [dict(k=kk, err=planewave_error(int(np.ceil(10 * kk / (2 * np.pi))), float(kk), a1, a0))
                for kk in [5, 10, 20, 40, 80, 160]]
R["B2_pollution_ppw10"] = B2
log("B2 done")

# ---------------------------------------------------------------- B3 cavity: impedance on one side only
n = 80
kgrid = np.linspace(5, 30, 301)
Popen = unit_square(n)
imp = np.zeros(Popen.grid.nn, bool); imp[np.isclose(Popen.X, 0.0)] = True
Pcav = unit_square(n, impedance=imp)
bopen, bcav = [], []
for kk in kgrid:
    bopen.append(beta(Popen.assemble(kk), Popen.energy_norm(kk), tol=1e-6))
    bcav.append(beta(Pcav.assemble(kk), Pcav.energy_norm(kk), tol=1e-6))
R["B3_cavity"] = dict(n=n, k_range=[5, 30], min_beta_open=min(bopen), min_beta_one_side=min(bcav),
                      k_at_min_one_side=float(kgrid[int(np.argmin(bcav))]),
                      min_k_beta_open=float(np.min(kgrid * np.array(bopen))),
                      min_k_beta_one_side=float(np.min(kgrid * np.array(bcav))))
log("B3 done")

# ---------------------------------------------------------------- B4 counterexamples (passive but singular)
n = 40
P = unit_square(n)
imp = np.zeros(P.grid.nn, bool)
imp[P.grid.node(n // 2, 0)] = imp[P.grid.node(n // 2, n)] = True   # absorber only on the nodal line x=1/2
P = unit_square(n, impedance=imp)
k1 = np.sqrt(4 * n**2 * np.sin(np.pi / (2 * n)) ** 2)               # discrete Neumann eigenvalue, mode (1,0)
A = P.assemble(k1)
u_mode = np.cos(np.pi * P.X)
B4 = dict(k=k1, passivity_eig=passivity_eig(A), residual_of_mode=float(np.linalg.norm(A @ u_mode) / np.linalg.norm(u_mode)),
          sigma_min_l2=float(np.linalg.svd(A.toarray(), compute_uv=False).min()))
# pocket: learned edge star ~ 0 on the edges around an interior 5x5 node block (passive, lossless)
P = unit_square(n)
inblk = lambda x, y: (x >= 10) & (x <= 14) & (y >= 10) & (y <= 14)
Dc = P.D.tocoo()
ends = {}
for r_, c_ in zip(Dc.row, Dc.col):
    ends.setdefault(r_, []).append(c_)
gi = lambda p: (p % (n + 1), p // (n + 1))
cut = np.zeros(P.D.shape[0], bool)
for e, (p, q) in ends.items():
    a_, b_ = inblk(*gi(p)), inblk(*gi(q))
    cut[e] = a_ != b_
for sval in [1e-2, 1e-4, 1e-8, 0.0]:
    s = np.ones(P.D.shape[0]); s[cut] = sval
    Pp = unit_square(n, s_theta=s)
    # pocket eigenvalue: Neumann 5x5 block with unit DEC weights -> first nonzero mode (1,0)
    kp = np.sqrt(4 * n**2 * np.sin(np.pi / (2 * 5)) ** 2)
    A = Pp.assemble(kp)
    B4[f"pocket_s={sval:g}_sigma_min_l2"] = float(np.linalg.svd(A.toarray(), compute_uv=False).min())
B4["pocket_k"] = float(kp)
R["B4_counterexamples"] = B4
log("B4", B4)

# ---------------------------------------------------------------- B5 learned perturbations
B5 = {}
for kk in [10, 20, 40]:
    n = int(np.ceil(10 * kk / (2 * np.pi)))
    P0 = unit_square(n)
    N = P0.energy_norm(kk)
    b0 = beta(P0.assemble(kk), N)
    rng = np.random.default_rng(kk)
    out = dict(beta_ref=b0)
    for dlt in [0.1, 0.3, 0.6]:
        rat, thm = [], []
        for trial in range(12):
            ne = P0.D.shape[0]
            s = 1 + dlt * rng.uniform(-1, 1, ne)
            m = (1 + dlt * rng.uniform(-1, 1, P0.grid.nn)) * (1 - 1j * dlt * rng.uniform(0, 0.2, P0.grid.nn))
            Pp = unit_square(n, s_theta=s, m_theta=m)
            bp = beta(Pp.assemble(kk), N, tol=1e-6)
            rat.append(bp / b0)
            dmax = max(np.abs(s - 1).max(), np.abs(m - 1).max())
            thm.append(bp - (b0 - dmax))          # Theorem B3: beta_theta >= beta_ref - delta (meaningful if > 0)
        out[f"delta={dlt}"] = dict(beta_ratio_min=min(rat), beta_ratio_max=max(rat), all_invertible=bool(min(rat) > 0),
                                   theorem_B3_margin_min=min(thm))
    # uniform eps-star scaling m = 1 + tau: singular iff k^2 (1 + tau) is an eigenvalue of (K + jkB, M)
    A0 = P0.assemble(kk); M0 = sp.diags(P0.M_diag(kk)).tocsc()
    lu = ComplexLU(A0)
    op = spla.LinearOperator(A0.shape, matvec=lambda v: lu.solve(M0 @ v), dtype=complex)
    mu, _ = spla.eigs(op, k=6, which="LM", tol=1e-10)    # mu = 1/(lambda - k^2)
    lam_ = kk**2 + 1 / mu
    tau = lam_ / kk**2 - 1
    j = int(np.argmin(np.abs(tau)))
    out["uniform_tau_min_abs"] = float(np.abs(tau[j])); out["uniform_tau_at_min"] = [float(tau[j].real), float(tau[j].imag)]
    out["all_singular_tau_have_Im_gt_0_(active)"] = bool(np.all(tau.imag > 0))
    out["k_times_min_abs_tau"] = float(kk * np.abs(tau[j]))
    B5[f"k={kk}"] = out
R["B5_learned_perturbations"] = B5
log("B5 done")

# ---------------------------------------------------------------- B6 PML stars are not passive
B6 = {}
for kk in [10.0, 20.0]:
    n = 80
    g = Grid(n, n, 1.0 / n)
    P = Problem(g, pml=PML(0.2, 40.0, (0, 1), (0, 1)), impedance=np.zeros(g.nn, bool))
    A = P.assemble(kk)
    B6[f"k={kk}"] = dict(passivity_eig_min=passivity_eig(A), beta_N=beta(A, P.energy_norm(kk)))
R["B6_pml"] = B6

# ---------------------------------------------------------------- B7 strictly lossy medium (Theorem B4)
B7 = []
for kk in [5, 10, 20, 40, 80]:
    for tau in [0.05, 0.2]:
        n = int(np.ceil(10 * kk / (2 * np.pi)))
        P = unit_square(n, eps_node=np.full((n + 1) ** 2, 1 - 1j * tau))
        A = P.assemble(kk)
        M0 = sp.diags(P.area)
        b = beta(A, P.energy_norm(kk), Nout=M0, tol=1e-6)   # 1 / ||A^-1||_{M0^-1 -> N}
        mplus, sminus = abs(1 - 1j * tau), 1.0
        Ct = np.sqrt(((1 + mplus / tau) / sminus + 1 / tau) / tau)
        B7.append(dict(k=kk, tau=tau, norm_Ainv=1 / b, bound=Ct / kk, ok=bool(1 / b <= Ct / kk)))
R["B7_lossy_bound"] = B7
log("B7 done")
R["wall_time_s"] = time.time() - T0

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for name, v in V1.items():
    ax[0].loglog(1 / np.array(v["n"]), v["err"], "o-", label="plane wave, " + name)
ax[0].loglog(V2["h"], V2["err"], "s-", label="Mie cylinder (DEC+PML)")
hh = np.array([1e-3, 5e-2]); ax[0].loglog(hh, 5 * hh**2, "k--", label=r"$h^2$")
ax[0].set_xlabel("h"); ax[0].set_ylabel("rel. L2 error"); ax[0].legend(fontsize=7); ax[0].set_title("2D DEC convergence")
for name, rows in B1.items():
    if "mesh" in name:
        continue
    ax[1].loglog([r["k"] for r in rows], [r["beta_N"] for r in rows], "o-", label=name)
kk = np.array([5, 160]); ax[1].loglog(kk, 1.0 / kk, "k--", label="1/k")
ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$\beta_N$"); ax[1].legend(fontsize=7); ax[1].set_title("discrete inf-sup, unit square, impedance BC")
for name, rows in B2.items():
    ax[2].loglog([r["k"] for r in rows], [r["err"] for r in rows], "o-", label=name)
ax[2].set_xlabel("k (10 points/wavelength)"); ax[2].set_ylabel("rel. L2 error"); ax[2].legend(fontsize=7); ax[2].set_title("pollution")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_B_convergence_infsup.png"), dpi=130)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(kgrid, bopen, label="impedance on all sides")
ax.semilogy(kgrid, bcav, label="impedance on one side (x=0) only")
ax.set_xlabel("k"); ax.set_ylabel(r"$\beta_N$"); ax.legend(); ax.set_title("passive and invertible, but cavity resonances")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_B_cavity_scan.png"), dpi=130)

with open(os.path.join(OUT, "results_2d.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)
log("done")

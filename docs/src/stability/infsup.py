"""
Discrete inf-sup constants of the indefinite system.

  beta_N(A) = inf_u sup_v |v^H A u| / (||u||_N ||v||_N) = sigma_min(N^{-1/2} A N^{-1/2}),
  N = D^T S0 D + k^2 M0  (discrete H^1_k norm with geometric stars).
beta_N^{-2} is the largest eigenvalue of  f -> N A^{-H} N A^{-1} f  (self-adjoint in the N^{-1}
inner product), computed with ARPACK using one LU of A (A^{-H} from the same factorization).
With N = I it is the ordinary smallest singular value.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dec2d import ComplexLU


def beta(A, N=None, lu=None, tol=1e-8, return_vector=False, Nout=None):
    """inf-sup constant. With Nout given: 1 / sup_f ||A^-1 f||_N / ||f||_{Nout^-1}."""
    n = A.shape[0]
    lu = lu or ComplexLU(A)
    if N is None:
        N = sp.identity(n, format="csr")
    if Nout is None:
        Nout = N

    def mv(f):
        return Nout @ lu.solve(N @ lu.solve(f), "H")

    op = spla.LinearOperator((n, n), matvec=mv, dtype=complex)
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    vals, vecs = spla.eigs(op, k=1, which="LM", v0=v0, tol=tol, maxiter=5000, ncv=min(n - 2, 30))
    lam = float(np.real(vals[0]))
    b = 1 / np.sqrt(lam)
    if return_vector:
        f = vecs[:, 0]
        return b, lu.solve(f), f      # u = A^{-1} f is the (near-)singular direction
    return b


def beta_dense(A, N=None):
    """Dense reference: sigma_min(N^{-1/2} A N^{-1/2})."""
    A = A.toarray() if sp.issparse(A) else np.asarray(A)
    if N is None:
        return np.linalg.svd(A, compute_uv=False).min()
    N = N.toarray() if sp.issparse(N) else np.asarray(N)
    w, V = np.linalg.eigh(N)
    Nmh = V @ np.diag(w ** -0.5) @ V.conj().T
    return np.linalg.svd(Nmh @ A @ Nmh, compute_uv=False).min()


def passivity_eig(A, dense_limit=4000):
    """lambda_min of the anti-Hermitian part (A - A^H)/(2j); >= 0 <=> passive (Im u^H A u >= 0)."""
    H = (A - A.conj().T) / 2j
    H = (H + H.conj().T) / 2
    if A.shape[0] <= dense_limit:
        return np.linalg.eigvalsh(H.toarray() if sp.issparse(H) else H).min()
    v = spla.eigsh(H.tocsc(), k=1, which="SA", return_eigenvectors=False, tol=1e-8, maxiter=20000)
    return float(v[0])

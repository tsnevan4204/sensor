"""E026: compiled kernels (numba). Drop-in replacements; results are checked against the numpy
versions in tests/test_search.py. If numba is unavailable the numpy path is used."""
import numpy as np
try:
    import numba
    HAVE_NUMBA = True
except ImportError:      # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:
    @numba.njit(cache=True, fastmath=False)
    def thomas_batch(diag, off, rhs):
        """Symmetric tridiagonal solves, one per row, two right-hand sides each (forward, adjoint).
        diag (R, n), off (R, n-1), rhs (R, n, 2) complex. Returns x (R, n, 2) and the relative
        residual per row (same definition as wall_inverse._tridiag_solve)."""
        R, n = diag.shape
        x = np.zeros_like(rhs)
        res = np.zeros(R)
        cp = np.zeros(n, dtype=diag.dtype)
        dp = np.zeros((n, 2), dtype=diag.dtype)
        for r in range(R):
            den = diag[r, 0]
            if n > 1:
                cp[0] = off[r, 0] / den
            dp[0, 0] = rhs[r, 0, 0] / den
            dp[0, 1] = rhs[r, 0, 1] / den
            for i in range(1, n):
                den = diag[r, i] - off[r, i - 1] * cp[i - 1]
                if i < n - 1:
                    cp[i] = off[r, i] / den
                dp[i, 0] = (rhs[r, i, 0] - off[r, i - 1] * dp[i - 1, 0]) / den
                dp[i, 1] = (rhs[r, i, 1] - off[r, i - 1] * dp[i - 1, 1]) / den
            x[r, n - 1, 0] = dp[n - 1, 0]
            x[r, n - 1, 1] = dp[n - 1, 1]
            for i in range(n - 2, -1, -1):
                x[r, i, 0] = dp[i, 0] - cp[i] * x[r, i + 1, 0]
                x[r, i, 1] = dp[i, 1] - cp[i] * x[r, i + 1, 1]
            # residual check
            mres, mrhs, mdiag, mx = 0.0, 0.0, 0.0, 0.0
            for i in range(n):
                for k in range(2):
                    ax = diag[r, i] * x[r, i, k]
                    if i < n - 1:
                        ax += off[r, i] * x[r, i + 1, k]
                    if i > 0:
                        ax += off[r, i - 1] * x[r, i - 1, k]
                    v = abs(ax - rhs[r, i, k])
                    if v > mres:
                        mres = v
                    if abs(rhs[r, i, k]) > mrhs:
                        mrhs = abs(rhs[r, i, k])
                    if abs(x[r, i, k]) > mx:
                        mx = abs(x[r, i, k])
                if abs(diag[r, i]) > mdiag:
                    mdiag = abs(diag[r, i])
            res[r] = mres / (mrhs + mdiag * mx)
        return x, res


if HAVE_NUMBA:
    @numba.njit(cache=True)
    def chain(b, M, ncell, Y0, Yb, src):
        """E026: exact-star DEC chain with identical cells per sublayer. Row r: sublayer s has
        ncell[s] identical cells with edge star b[r, s] and half-cell star M[r, s]. Ports: Y0[r] at
        node 0 with source src[r], Yb[r] at the last node. Solves A e = src e_0 and A lam = e_0
        (A symmetric) and returns e_0, and per sublayer the adjoint sums
            SQ1[r, s] = sum_cells (lam_i - lam_j)(e_i - e_j),   SQ2[r, s] = sum_cells (lam_i e_i + lam_j e_j),
        plus the relative residual of the forward solve per row."""
        R, Sn = b.shape
        N = 0
        for s in range(Sn):
            N += ncell[s]
        n = N + 1
        e0 = np.zeros(R, dtype=b.dtype)
        SQ1 = np.zeros((R, Sn), dtype=b.dtype)
        SQ2 = np.zeros((R, Sn), dtype=b.dtype)
        res = np.zeros(R)
        dg = np.zeros(n, dtype=b.dtype)
        of = np.zeros(n - 1, dtype=b.dtype)
        cp = np.zeros(n, dtype=b.dtype)
        d1 = np.zeros(n, dtype=b.dtype)
        d2 = np.zeros(n, dtype=b.dtype)
        x1 = np.zeros(n, dtype=b.dtype)
        x2 = np.zeros(n, dtype=b.dtype)
        for r in range(R):
            for i in range(n):
                dg[i] = 0.0
            c = 0
            for s in range(Sn):
                for q in range(ncell[s]):
                    dg[c] += b[r, s] + M[r, s]
                    dg[c + 1] += b[r, s] + M[r, s]
                    of[c] = -b[r, s]
                    c += 1
            dg[0] += Y0[r]
            dg[n - 1] += Yb[r]
            # Thomas, two right-hand sides: src*e_0 (forward) and e_0 (adjoint)
            den = dg[0]
            cp[0] = of[0] / den
            d1[0] = src[r] / den
            d2[0] = 1.0 / den
            for i in range(1, n):
                den = dg[i] - of[i - 1] * cp[i - 1]
                if i < n - 1:
                    cp[i] = of[i] / den
                d1[i] = (0.0 - of[i - 1] * d1[i - 1]) / den
                d2[i] = (0.0 - of[i - 1] * d2[i - 1]) / den
            x1[n - 1] = d1[n - 1]
            x2[n - 1] = d2[n - 1]
            for i in range(n - 2, -1, -1):
                x1[i] = d1[i] - cp[i] * x1[i + 1]
                x2[i] = d2[i] - cp[i] * x2[i + 1]
            # residual (forward system)
            mres, mdg, mx = 0.0, 0.0, 0.0
            for i in range(n):
                ax = dg[i] * x1[i]
                if i < n - 1:
                    ax += of[i] * x1[i + 1]
                if i > 0:
                    ax += of[i - 1] * x1[i - 1]
                if i == 0:
                    ax -= src[r]
                if abs(ax) > mres:
                    mres = abs(ax)
                if abs(dg[i]) > mdg:
                    mdg = abs(dg[i])
                if abs(x1[i]) > mx:
                    mx = abs(x1[i])
            res[r] = mres / (abs(src[r]) + mdg * mx)
            e0[r] = x1[0]
            c = 0
            for s in range(Sn):
                a1 = 0.0 + 0.0j
                a2 = 0.0 + 0.0j
                for q in range(ncell[s]):
                    a1 += (x2[c] - x2[c + 1]) * (x1[c] - x1[c + 1])
                    a2 += x2[c] * x1[c] + x2[c + 1] * x1[c + 1]
                    c += 1
                SQ1[r, s] = a1
                SQ2[r, s] = a2
        return e0, SQ1, SQ2, res

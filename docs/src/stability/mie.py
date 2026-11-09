"""
Mie series for a homogeneous dielectric circular cylinder (radius a, eps_r, mu_r = 1), E_z polarization,
time dependence e^{+jwt}: incident u_i = exp(-j k x) = sum_n (-j)^n J_n(k r) e^{j n phi},
scattered (r > a) u_s = sum_n (-j)^n a_n H_n^(2)(k r) e^{j n phi}   (H^(2): outgoing for e^{+jwt}),
interior (r < a) u   = sum_n (-j)^n c_n J_n(k1 r) e^{j n phi},  k1 = k sqrt(eps_r).
Continuity of u and d_r u at r = a fixes (a_n, c_n).
"""
import numpy as np
from scipy.special import jv, jvp, hankel2, h2vp


def coefficients(k, a, eps_r, nmax):
    k1 = k * np.sqrt(complex(eps_r))
    n = np.arange(-nmax, nmax + 1)
    J, Jp = jv(n, k * a), jvp(n, k * a)
    H, Hp = hankel2(n, k * a), h2vp(n, k * a)
    J1, J1p = jv(n, k1 * a), jvp(n, k1 * a)
    # J + an H = cn J1 ;  k (Jp + an Hp) = cn k1 J1p
    det = -H * k1 * J1p + k * Hp * J1
    an = (J * k1 * J1p - k * Jp * J1) / det
    cn = (k * Hp * J - k * Jp * H) / det
    return n, an, cn, k1


def total_field(x, y, k, a, eps_r, nmax=None):
    if nmax is None:
        nmax = int(np.ceil(k * a * abs(np.sqrt(eps_r)) + 4 * (k * a) ** (1 / 3) + 15))
    n, an, cn, k1 = coefficients(k, a, eps_r, nmax)
    r = np.hypot(x, y); ph = np.arctan2(y, x)
    u = np.zeros(np.shape(x), complex)
    inside = r < a
    ri, po = r[inside], ph[inside]
    ro, pho = r[~inside], ph[~inside]
    ui = np.zeros(ri.shape, complex); us = np.zeros(ro.shape, complex)
    for m, A, Cc in zip(n, an, cn):
        f = (-1j) ** m
        ui += f * Cc * jv(m, k1 * ri) * np.exp(1j * m * po)
        us += f * A * hankel2(m, k * ro) * np.exp(1j * m * pho)
    u[inside] = ui
    u[~inside] = np.exp(-1j * k * x[~inside]) + us
    return u

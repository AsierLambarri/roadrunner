#############################################################################
#
# package:   roadrunner.cosmology
# file:      ishiyama21.py
# brief:     Run-time Ishiyama et al. (2021) median c_vir(M_vir, z) table.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   10 Oct 2026 - Created
#
#############################################################################

"""Run-time Ishiyama et al. (2021) median ``c_vir(M_vir, z)`` table.

Builds, once per :class:`~roadrunner.cosmology.model.Cosmology` instance, a
dense ``(log10 M_vir, ln(1+z))`` table of ``ln c_vir`` using the 'vir',
'all', 'fit' calibration of Ishiyama et al. (2021), a recalibration of the
Diemer & Joyce (2019) ``c(nu, n_eff, alpha_eff)`` model. Every formula below
is checked against ``colossus`` 1.4.0 and references its source:

- ``delta_c = 1.68647``, no Omega correction
  (``utils/constants.py:80``, ``lss/peaks.py:167``).
- Lagrangian radius ``R_L = (3 M h / (4 pi rho_crit Om))^(1/3)`` Mpc/h,
  ``rho_crit = 2.77536627245708e11 Msun h^2 Mpc^-3``
  (``lss/peaks.py:99``, ``utils/constants.py:77``).
- Peak height ``nu = delta_c / (sigma(R_L, 0) D(z))``
  (``lss/peaks.py:225-227``, ``cosmology/cosmology.py:2919-2923``).
- Effective slope ``n_eff = -2 dln(sigma)/dln(R)`` at ``kappa R_L``, minus 3,
  z-independent (``lss/peaks.py:590-597``, ``halo/concentration.py:1416``).
- ``sigma^2(R) = int P(k) W(kR)^2 k^3 dln k``, ``P(k) = T(k)^2 k^n_s``,
  normalised to ``sigma8`` at 8 Mpc/h
  (``cosmology/cosmology.py:2543,2457-2459``, ``cosmology/power_spectrum.py:142``).
- ``T(k)``: Eisenstein & Hu (1998) with baryon wiggles
  (``cosmology/power_spectrum.py:517-607``).
- ``alpha_eff = -(1+z) D'(z)/D(z)`` (``halo/concentration.py:1278``).
- ``A = a0 (1 + a1 (n_eff+3))``, ``B = b0 (1 + b1 (n_eff+3))``,
  ``C = 1 - c_alpha (1 - alpha_eff)``, ``rhs = A/nu (1 + nu^2/B)``
  (``halo/concentration.py:1429-1432``).
- ``c = C G^-1(rhs)``, ``G(x) = x / mu(x)^((5+n_eff)/6)``, ascending branch
  (``halo/concentration.py:1336,1440``); ``mu(x) = ln(1+x) - x/(1+x)``
  (``halo/profile_nfw.py:245``).
- vir/all/fit parameters: ``kappa=1.64, a0=2.67, a1=1.23, b0=3.92, b1=1.30,
  c_alpha=-0.19`` (``halo/concentration.py:1598-1603``); 'vir' is this
  model's native mass definition (``halo/concentration.py:227``).
- Growth factor (``cosmology/cosmology.py:1916-2032``): the Heath/Eisenstein
  & Hu (1999) integral for ``z < 20``, with radiation dropped from its
  ``E(z)`` (negligible there -- the universe is matter-dominated), has the
  closed form ``D_int(a) = a * 2F1(1/3, 1; 11/6; -lam a^3)``,
  ``lam = (1-Om)/Om``; the Gnedin et al. (2011) matter-radiation expansion
  for ``z > 5`` (``cosmology/cosmology.py:2023-2024``), with
  ``a_eq = Or0/Om``; the two are blended linearly in ``ln a`` for
  ``5 < z < 20`` (``cosmology/cosmology.py:1995-1996,2032``), with the
  weight on the integral equal to ``(ln 21 - ln(1+z)) / ln 3.5``, clipped to
  ``[0, 1]``. Radiation is kept in ``a_eq`` because it still shifts ``D`` by
  ~1e-3 at z=10 and ~4e-3 at z=20 (captured by the blend, which is fully on
  the radiation-aware branch by z=20): ``Or0 h^2 = 4.4814665013636476e-7
  Tcmb0^4 (1 + 0.22710731766 Neff)`` (``cosmology/cosmology.py:804-811``),
  ``Tcmb0 = 2.7255``, ``Neff = 3.046`` (``utils/defaults.py:16,19``).
- ``G`` inversion: with ``u = ln x`` and ``p = (5+n_eff)/6``,
  ``g(u) = ln G(e^u) = u - p ln mu(e^u)``, ``g'(u) = 1 - p s(x)``,
  ``s(x) = x^2 / ((1+x)^2 mu(x))`` strictly decreasing, so ``g`` is convex
  in ``u``. Newton's method in ``u``, started at ``x0 = 1e4`` (right of the
  root), descends monotonically onto the ascending branch -- no masks or
  brackets needed. A solution exists over the whole table domain
  (``rhs >= 2A/sqrt(B) > G_min(n_eff)``); roots stay below ``x ~ 300`` even
  at the table's corners.
"""

import warnings

import numpy as np
from numba import njit, prange
from scipy.special import hyp2f1

_TCMB0 = 2.7255
_OMEGA_R_H2 = 4.4814665013636476e-7 * _TCMB0**4 * (1.0 + 0.22710731766 * 3.046)
_RHO_CRIT_H2 = 2.77536627245708e11
_DELTA_C = 1.68647
_KAPPA, _A0, _A1, _B0, _B1, _C_ALPHA = 1.64, 2.67, 1.23, 3.92, 1.30, -0.19
_R8 = 8.0

_LNK = np.linspace(np.log(1e-5), np.log(1e5), 8001)   # E1: 4001 left max|d slope| ~1.5e-6 vs 40001; 8001 clears 1e-6
_K = np.exp(_LNK)
_K_WEIGHTS = (_LNK[1] - _LNK[0]) * np.r_[0.5, np.ones(_LNK.size - 2), 0.5]

_LN_ZP1_MAT, _LN_ZP1_RAD = np.log(6.0), np.log(21.0)   # growth blend window: z = 5 .. 20
_DU = 1e-3                                             # half-step of the alpha_eff central difference
_X_START, _NEWTON_STEPS = 1e4, 10   # E2: converges to |g - ln rhs| < 1e-12 by step 5 on the full table grid

_LOG10_M_LO, _LOG10_M_STEP, _N_M = 3.0, 0.05, 261      # table axis 0: log10(M_vir/Msun), 1e3 .. 1e16
_LN_ZP1_STEP, _N_Z = np.log(31.0) / 172, 173           # table axis 1: ln(1+z), z = 0 .. 30
_LOG10_M = _LOG10_M_LO + _LOG10_M_STEP * np.arange(_N_M)
_LN_ZP1 = _LN_ZP1_STEP * np.arange(_N_Z)


def _transfer_eh98(k, h, omega_m, omega_b):
    """Eisenstein & Hu (1998) transfer function with baryon acoustic wiggles.

    Ported from ``colossus.cosmology.power_spectrum.modelEisenstein98``
    (same names and equation numbers refer to that paper).

    Parameters
    ----------
    k : ndarray
        Wavenumber, comoving h/Mpc.
    h, omega_m, omega_b : float

    Returns
    -------
    Tk : ndarray, shape of ``k``
    """
    Om0, Ob0, Tcmb0 = omega_m, omega_b, _TCMB0

    omc = Om0 - Ob0
    ombom0 = Ob0 / Om0
    h2 = h**2
    om0h2 = Om0 * h2
    ombh2 = Ob0 * h2
    theta2p7 = Tcmb0 / 2.7
    theta2p72 = theta2p7**2
    theta2p74 = theta2p72**2

    kh = k * h   # comoving 1/Mpc

    zeq = 2.50e4 * om0h2 / theta2p74                              # Eq. 2
    keq = 7.46e-2 * om0h2 / theta2p72                              # Eq. 3

    b1d = 0.313 * om0h2**-0.419 * (1.0 + 0.607 * om0h2**0.674)     # Eq. 4
    b2d = 0.238 * om0h2**0.223
    zd = 1291.0 * om0h2**0.251 / (1.0 + 0.659 * om0h2**0.828) * (1.0 + b1d * ombh2**b2d)

    Rd = 31.5 * ombh2 / theta2p74 / (zd / 1e3)                     # Eq. 5
    Req = 31.5 * ombh2 / theta2p74 / (zeq / 1e3)

    s = 2.0 / 3.0 / keq * np.sqrt(6.0 / Req) * np.log((np.sqrt(1.0 + Rd) +   # Eq. 6
        np.sqrt(Rd + Req)) / (1.0 + np.sqrt(Req)))

    ksilk = 1.6 * ombh2**0.52 * om0h2**0.73 * (1.0 + (10.4 * om0h2)**-0.95)  # Eq. 7

    q = kh / 13.41 / keq                                           # Eq. 10

    a1 = (46.9 * om0h2)**0.670 * (1.0 + (32.1 * om0h2)**-0.532)    # Eq. 11
    a2 = (12.0 * om0h2)**0.424 * (1.0 + (45.0 * om0h2)**-0.582)
    ac = a1**(-ombom0) * a2**(-ombom0**3)

    b1 = 0.944 / (1.0 + (458.0 * om0h2)**-0.708)                   # Eq. 12
    b2 = (0.395 * om0h2)**-0.0266
    bc = 1.0 / (1.0 + b1 * ((omc / Om0)**b2 - 1.0))

    y = (1.0 + zeq) / (1.0 + zd)                                   # Eq. 15
    Gy = y * (-6.0 * np.sqrt(1.0 + y) + (2.0 + 3.0 * y) *
        np.log((np.sqrt(1.0 + y) + 1.0) / (np.sqrt(1.0 + y) - 1.0)))

    ab = 2.07 * keq * s * (1.0 + Rd)**(-3.0 / 4.0) * Gy            # Eq. 14

    # CDM part of the transfer function
    f = 1.0 / (1.0 + (kh * s / 5.4)**4)                            # Eq. 18
    C = 14.2 / ac + 386.0 / (1.0 + 69.9 * q**1.08)                 # Eq. 20
    T0t = np.log(np.e + 1.8 * bc * q) / (np.log(np.e + 1.8 * bc * q) + C * q * q)   # Eq. 19
    C1bc = 14.2 + 386.0 / (1.0 + 69.9 * q**1.08)                   # Eq. 17
    T0t1bc = np.log(np.e + 1.8 * bc * q) / (np.log(np.e + 1.8 * bc * q) + C1bc * q * q)
    Tc = f * T0t1bc + (1.0 - f) * T0t

    # Baryon part of the transfer function
    bb = 0.5 + ombom0 + (3.0 - 2.0 * ombom0) * np.sqrt((17.2 * om0h2) * (17.2 * om0h2) + 1.0)   # Eq. 24
    bnode = 8.41 * om0h2**0.435                                    # Eq. 23
    st = s / (1.0 + (bnode / kh / s) * (bnode / kh / s) * (bnode / kh / s))**(1.0 / 3.0)   # Eq. 22
    C11 = 14.2 + 386.0 / (1.0 + 69.9 * q**1.08)                    # Eq. 21
    T0t11 = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C11 * q * q)
    Tb = (T0t11 / (1.0 + (kh * s / 5.2)**2) + ab / (1.0 + (bb / kh / s)**3) * np.exp(-(kh / ksilk)**1.4)) \
        * np.sin(kh * st) / (kh * st)

    return ombom0 * Tb + omc / Om0 * Tc


def _variance(r, cosmo):
    """``sigma^2(r, z=0)`` and ``dln sigma^2 / dln R`` of a flat LCDM cosmology.

    Parameters
    ----------
    r : ndarray
        Radii, comoving Mpc/h.
    cosmo : Cosmology

    Returns
    -------
    sigma2 : ndarray, shape of ``r``
        Normalised to ``cosmo.sigma8`` at 8 Mpc/h.
    slope : ndarray, shape of ``r``
        ``dln sigma^2 / dln R`` at each radius.
    """
    power = _K_WEIGHTS * _K ** (3.0 + cosmo.n_s) * _transfer_eh98(_K, cosmo.h, cosmo.omega_m, cosmo.omega_b) ** 2
    x = np.multiply.outer(np.append(r, _R8), _K)
    sin, cos = np.sin(x), np.cos(x)
    w = 3.0 * (sin - x * cos) / x**3
    dw = (9.0 * x * cos + 3.0 * (x * x - 3.0) * sin) / x**3   # dW/dln x
    s2 = (w * w) @ power
    return cosmo.sigma8**2 * s2[:-1] / s2[-1], 2.0 * ((w * dw) @ power)[:-1] / s2[:-1]


def _growth(ln_zp1, omega_m, a_eq):
    """Linear growth factor ``D(z) / D(0)`` of a flat LCDM cosmology.

    Blends the closed-form matter+Lambda integral (valid at ``z < 20``,
    where radiation is negligible) with the Gnedin et al. (2011)
    matter-radiation expansion (valid at ``z > 5``).

    Parameters
    ----------
    ln_zp1 : ndarray
        ``ln(1 + z)``.
    omega_m : float
    a_eq : float
        Scale factor of matter-radiation equality, ``Omega_r / Omega_m``.

    Returns
    -------
    D : ndarray, shape of ``ln_zp1``
    """
    a = np.exp(-ln_zp1)
    lam = (1.0 - omega_m) / omega_m
    d_int = a * hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -lam * a**3)
    t = np.sqrt(1.0 + a / a_eq)
    d_rad = a + 2.0 / 3.0 * a_eq + a_eq / (2.0 * np.log(2.0) - 3.0) * \
        (2.0 * t + (2.0 / 3.0 + a / a_eq) * np.log((t - 1.0) / (t + 1.0)))
    w = np.clip((_LN_ZP1_RAD - ln_zp1) / (_LN_ZP1_RAD - _LN_ZP1_MAT), 0.0, 1.0)
    return (w * d_int + (1.0 - w) * d_rad) / hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, -lam)


def _inverse_g(ln_g, p):
    """Ascending-branch solution of ``ln G(x) = ln_g``, ``G(x) = x / mu(x)^p``.

    Newton's method in ``u = ln x``, started right of the root so the convex
    ``g(u) = ln G(e^u)`` descends monotonically onto the ascending branch.

    Parameters
    ----------
    ln_g : ndarray
    p : ndarray
        Broadcasts against ``ln_g``.

    Returns
    -------
    u : ndarray
        ``ln x``, shape of ``np.broadcast(ln_g, p)``.
    """
    u = np.full(np.broadcast(ln_g, p).shape, np.log(_X_START))
    for _ in range(_NEWTON_STEPS):
        x = np.exp(u)
        mu = np.log1p(x) - x / (1.0 + x)
        u = u - (u - p * np.log(mu) - ln_g) / (1.0 - p * x * x / ((1.0 + x) ** 2 * mu))
    return u


def ln_concentration(log10_m, ln_zp1, cosmo):
    """``ln c_vir(M_vir, z)`` of a flat LCDM cosmology, on a (mass, redshift) grid.

    Parameters
    ----------
    log10_m : ndarray, shape (n_m,)
        ``log10(M_vir / Msun)``.
    ln_zp1 : ndarray, shape (n_z,)
        ``ln(1 + z)``.
    cosmo : Cosmology

    Returns
    -------
    ln_c : ndarray, shape (n_m, n_z)
    """
    r_l = np.cbrt(3.0 * cosmo.h * 10.0**log10_m / (4.0 * np.pi * _RHO_CRIT_H2 * cosmo.omega_m))
    s2, slope = _variance(np.concatenate([r_l, _KAPPA * r_l]), cosmo)
    s2, slope = s2[:r_l.size, None], slope[r_l.size:, None]   # sigma^2 at R_L; dln sigma^2/dln R at kappa R_L

    a_eq = _OMEGA_R_H2 / (cosmo.h**2 * cosmo.omega_m)
    alpha = np.log(_growth(ln_zp1 - _DU, cosmo.omega_m, a_eq) / _growth(ln_zp1 + _DU, cosmo.omega_m, a_eq)) \
        / (2.0 * _DU)
    nu = _DELTA_C / (np.sqrt(s2) * _growth(ln_zp1, cosmo.omega_m, a_eq))
    rhs = _A0 * (1.0 - _A1 * slope) / nu * (1.0 + nu**2 / (_B0 * (1.0 - _B1 * slope)))   # n_eff + 3 = -slope

    return np.log(1.0 - _C_ALPHA * (1.0 - alpha)) + _inverse_g(np.log(rhs), (2.0 - slope) / 6.0)


def table(cosmo):
    """The full ``ln c_vir`` table of a cosmology, over the fixed (mass, redshift) grid.

    Parameters
    ----------
    cosmo : Cosmology

    Returns
    -------
    ln_c : ndarray, shape (261, 173)
    """
    return ln_concentration(_LOG10_M, _LN_ZP1, cosmo)


@njit(parallel=True, cache=True)
def _bilinear(x, y, table, x0, dx, y0, dy):
    """Bilinear interpolation of ``table`` at flat ``(x, y)``, linearly extrapolated outside it.

    Returns ``(values, n_outside)``; a non-finite coordinate (NaN, or log10 of a zero mass) gives a NaN value and is not counted.
    """
    nx, ny = table.shape
    out = np.empty(x.size)
    n_out = 0
    for i in prange(x.size):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            out[i] = np.nan
            continue
        tx = (x[i] - x0) / dx
        ty = (y[i] - y0) / dy
        ix = min(max(int(np.floor(tx)), 0), nx - 2)
        iy = min(max(int(np.floor(ty)), 0), ny - 2)
        fx = tx - ix
        fy = ty - iy
        out[i] = ((1.0 - fx) * (1.0 - fy) * table[ix, iy] + fx * (1.0 - fy) * table[ix + 1, iy]
                  + (1.0 - fx) * fy * table[ix, iy + 1] + fx * fy * table[ix + 1, iy + 1])
        n_out += int((tx < 0.0) | (tx > nx - 1.0) | (ty < 0.0) | (ty > ny - 1.0))
    return out, n_out


def lookup(ln_c, m_vir, z):
    """Concentration at ``(m_vir, z)``, bilinear in ``(log10 M_vir, ln(1+z))`` (compiled, parallel).

    Parameters
    ----------
    ln_c : ndarray, shape (261, 173)
        A cosmology's table (see :func:`table`).
    m_vir : array_like
        Virial mass, Msun.
    z : array_like
        Redshift. Broadcasts against ``m_vir``.

    Returns
    -------
    c_vir : ndarray
        Shape of ``np.broadcast(m_vir, z)``.
    """
    m = np.asarray(m_vir)
    x, y = np.broadcast_arrays(np.log10(m.astype(np.float64)), np.log1p(np.asarray(z, dtype=np.float64)))
    out, n_out = _bilinear(np.ascontiguousarray(x).ravel(), np.ascontiguousarray(y).ravel(), ln_c,
                           _LOG10_M_LO, _LOG10_M_STEP, 0.0, _LN_ZP1_STEP)
    if n_out:
        warnings.warn(f"{n_out} of {out.size} (m_vir, z) outside the Ishiyama+21 table (1e3-1e16 Msun, z 0-30): "
                      "ln c extrapolated linearly", RuntimeWarning, stacklevel=3)
    return np.exp(out).reshape(x.shape).astype(np.result_type(m.dtype, np.float32))

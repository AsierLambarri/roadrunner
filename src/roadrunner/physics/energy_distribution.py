#############################################################################
#
# package:   roadrunner.physics
# file:      energy_distribution.py
# brief:     Dark-matter energy distribution of an isotropic NFW halo.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
#
#############################################################################

"""Dark-matter energy distribution of an isotropic NFW halo.

Scaled units: ``G = Rs = 1`` and a mass unit such that the potential is
``phi(x) = -ln(1 + x) / x``, so ``phi(0) = -1``. A physical specific energy
``E`` maps to ``E / |Phi_0|``. The binding depth is
``calE = 1 - E / Phi_0`` (0 at the centre, 1 at ``E = 0``), and the
normalised boundness stored on NFW halos is ``eps = E / Phi_0 = 1 - calE``.

The energy distribution of a halo truncated at its virial radius
``x = c`` is ``n(calE | c) ∝ f(E) g(E, c)``, where ``f`` is the Eddington
distribution function of the isotropic NFW profile and ``g`` the density
of states inside ``x <= c``. ``n`` is normalised on ``calE in [0, 1]``.
Tables on a (concentration, calE) grid are built once per process.
"""

import functools
from typing import NamedTuple

import numpy as np

C_GRID = np.geomspace(1.0, 100.0, 48)
CAL_GRID = np.concatenate([
    np.geomspace(1e-4, 0.1, 200, endpoint=False),
    np.linspace(0.1, 1.0 - 1e-4, 200),
])
_LOG_C = np.log(C_GRID)
_X = np.geomspace(1e-7, 1e8, 30000)            # radial grid of the inversion
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(96)
_W, _WW = 0.5 * (_NODES + 1.0), 0.5 * _WEIGHTS  # Gauss-Legendre on [0, 1]


def trapezoid_weights(x):
    """Trapezoid-rule weights: ``integral f dx ≈ trapezoid_weights(x) @ f(x)``.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Increasing abscissae.

    Returns
    -------
    w : ndarray of shape (n,)
    """
    dx = np.diff(x)
    w = np.zeros_like(x, dtype=np.float64)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def nfw_phi(x):
    """Scaled NFW potential ``-ln(1 + x) / x``.

    Parameters
    ----------
    x : ndarray
        Radius in units of the scale radius.

    Returns
    -------
    phi : ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    return -np.log1p(x) / x


def _radius_of_energy(e):
    """Radius where ``phi(x) = e``, for scaled energies ``e`` in ``(-1, 0)``."""
    return np.interp(e, nfw_phi(_X), _X)


def nfw_density_of_states(cal, c):
    """Density of states ``g = 16 pi^2 int_0^{min(c, r_E)} x^2 sqrt(2 (E - phi)) dx``.

    The substitution ``x = x_max (1 - w^2)`` removes the square-root
    endpoint at the radial apocentre ``r_E``.

    Parameters
    ----------
    cal : ndarray
        Binding depths ``calE`` (scaled energy ``E = calE - 1``).
    c : float or ndarray
        Truncation radius (concentration); ``np.inf`` for none.
        Broadcasts against ``cal``.

    Returns
    -------
    g : ndarray
        Broadcast shape of ``cal`` and ``c``.
    """
    e = np.asarray(cal, dtype=np.float64) - 1.0
    x_max = np.minimum(c, _radius_of_energy(e))
    x = x_max[..., None] * (1.0 - _W**2)
    kin = np.sqrt(np.clip(2.0 * (e[..., None] - nfw_phi(x)), 0.0, None))
    return 16.0 * np.pi**2 * np.sum(_WW * 2.0 * x_max[..., None] * _W * x**2 * kin, axis=-1)


@functools.cache
def _eddington():
    """Eddington distribution function of the isotropic NFW profile on ``CAL_GRID``.

    ``f(b) = (1 / sqrt(8) pi^2) int_0^b rho''(psi) dpsi / sqrt(b - psi)``,
    with relative energy ``b = -E = 1 - calE``, and ``psi = b - s^2``.
    """
    psi = -nfw_phi(_X)                                    # decreasing in x
    rho = 1.0 / (4.0 * np.pi * _X * (1.0 + _X) ** 2)
    d2 = np.gradient(np.gradient(rho, psi), psi)
    b = 1.0 - CAL_GRID
    s = np.sqrt(b)[:, None] * _W
    d2_at = np.interp(b[:, None] - s**2, psi[::-1], d2[::-1])
    return np.sqrt(b) * np.sum(_WW * 2.0 * d2_at, axis=-1) / (np.sqrt(8.0) * np.pi**2)


class NFWEnergyTables(NamedTuple):
    """NFW energy-distribution tables on ``(C_GRID, CAL_GRID)``."""

    log_density: np.ndarray   # log n(calE | c), normalised on [0, 1]
    cdf: np.ndarray           # u = F_c(calE), the mass fraction more bound


@functools.cache
def nfw_energy_tables():
    """Build the NFW energy-distribution tables (once per process).

    Returns
    -------
    tables : NFWEnergyTables
        Arrays of shape ``(len(C_GRID), len(CAL_GRID))``.
    """
    n = _eddington()[None, :] * nfw_density_of_states(CAL_GRID[None, :], C_GRID[:, None])
    n /= (n @ trapezoid_weights(CAL_GRID))[:, None]
    seg = 0.5 * (n[:, 1:] + n[:, :-1]) * np.diff(CAL_GRID)
    cdf = np.concatenate([np.zeros((len(C_GRID), 1)), np.cumsum(seg, axis=1)], axis=1)
    cdf /= cdf[:, -1:]
    return NFWEnergyTables(np.log(n), cdf)


def _row(table, c):
    """Row of ``table`` at concentration ``c``: linear in log c, clipped to the grid."""
    lc = np.log(np.clip(c, C_GRID[0], C_GRID[-1]))
    j = int(np.clip(np.searchsorted(_LOG_C, lc), 1, len(C_GRID) - 1))
    w = (lc - _LOG_C[j - 1]) / (_LOG_C[j] - _LOG_C[j - 1])
    return (1.0 - w) * table[j - 1] + w * table[j]


def _depth(eps):
    """Binding depth ``calE = 1 - eps``, clipped to the table range."""
    return np.clip(1.0 - np.asarray(eps, dtype=np.float64), CAL_GRID[0], CAL_GRID[-1])


def nfw_energy_fraction(eps, c):
    """Fraction ``u = F_c(calE)`` of the halo's mass more bound than ``eps``.

    Parameters
    ----------
    eps : ndarray
        Normalised boundness ``E / Phi_0`` (1 at the centre, 0 at ``E = 0``).
    c : float
        Concentration.

    Returns
    -------
    u : ndarray
        In ``[0, 1]``; uniform for the halo's own dark matter.
    """
    return np.interp(_depth(eps), CAL_GRID, _row(nfw_energy_tables().cdf, c))


def nfw_log_energy_density(eps, c):
    """Log dark-matter energy distribution ``log dN/dcalE`` (normalised on [0, 1]).

    Parameters
    ----------
    eps : ndarray
        Normalised boundness ``E / Phi_0``.
    c : float
        Concentration.

    Returns
    -------
    log_n : ndarray
    """
    return np.interp(_depth(eps), CAL_GRID, _row(nfw_energy_tables().log_density, c))

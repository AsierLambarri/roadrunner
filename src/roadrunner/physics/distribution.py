#############################################################################
#
# package:   roadrunner.physics
# file:      distribution.py
# brief:     Energy distribution and phase-space volume of spherical potentials.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
#
#############################################################################

"""Energy distribution and phase-space volume of spherical potentials.

Notation: ``energy`` (``E``) is always the physical specific orbital energy
``Φ + v²/2``, negative for bound particles. Scaled energies have their own
names: ``eps = E / Φ₀`` and the binding depth ``cal = 1 - E / Φ₀`` (0 at the
centre, 1 at ``E = 0``).

For a spherical, isotropic system with potential ``Φ(r)`` and density
``ρ(r)``, both untruncated, counted inside a radius ``R`` (in the pipeline
the boundness search sphere, ``search_factor`` virial radii):

- phase-space volume more bound than ``E``:
  ``q(E) = (16π²/3) ∫_0^{min(R, r_E)} r² [2 (E - Φ)]^{3/2} dr``, with
  ``Φ(r_E) = E`` the apocentre of a radial orbit;
- density of states ``g(E) = dq/dE = 16π² ∫_0^{min(R, r_E)} r² sqrt(2 (E - Φ)) dr``;
- Eddington distribution function of the untruncated profile
  ``f(E) = (1 / (√8 π²)) ∫_0^{ψ_E} (d²ρ/dψ²) dψ / sqrt(ψ_E - ψ)``,
  ``ψ = -Φ``, ``ψ_E = -E`` (no boundary term: ``ρ -> 0`` as ``ψ -> 0``);
- energy distribution of the mass inside ``R``: ``N(E) = f(E) g(E)``,
  normalised to 1 over the bound energies, so ``∫ N dE`` is the fraction of
  that mass more bound than ``E``;
- phase-space fraction ``w = q(E) / q(0)``: uniform on ``(0, 1]`` for
  particles spread uniformly in the bound phase space inside ``R``.

``f`` describes the equilibrium profile, which extends beyond ``R``; ``g``
and ``q`` count only what lies inside ``R``, the volume the pipeline looks
at. :class:`SphericalDistribution` computes all of them for any potential
from its primitives (potential, enclosed mass, density); potentials
override them where exact forms exist.

The NFW tables below are those exact forms for NFW, in scaled units
``G = Rs = 1``, ``phi(x) = -ln(1 + x) / x``. They take ``eps``, their
truncation radius ``c`` is ``R / Rs``, and they are built once per process,
one row per ``c`` of a grid. Each row has its own depth nodes, with the
depth where the radial apocentre reaches ``c`` as a node: ``g`` and ``q``
have a kink there, and rows are blended kink to kink.

Precision: everything here is float64, inputs read as float64 and outputs
returned in it. Finite differences of ``ρ`` and deep quadratures need it,
and so do the plausibility models that exponentiate these quantities: a
float32 cast here shifts their float32 latent prior by tens of ulp. Those
models cast their own output to ``math_dtype()``.
"""

import functools
from typing import NamedTuple

import numpy as np


C_GRID = np.geomspace(1.0, 100.0, 48)
_LOG_C = np.log(C_GRID)
_DEEPEST = 1e-6                                 # deepest node of every table: calE = 1e-6
_KINK_GAP = 1e-3                                # first node past the kink, as a fraction of the way to E = 0
_NFW_NODES = 400, 370                           # nodes deeper and shallower than the kink (see _depth_nodes)
_X = np.geomspace(1e-7, 1e10, 34000)           # radial grid of the inversion
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


def nfw_phase_volume(cal, c):
    """Phase-space volume more bound than ``E`` inside ``x <= c``.

    ``q = (16 pi^2 / 3) int_0^{min(c, r_E)} x^2 [2 (E - phi)]^{3/2} dx``, with
    the same ``x = x_max (1 - w^2)`` substitution as the density of states.

    Parameters
    ----------
    cal : ndarray
        Binding depths ``calE`` (scaled energy ``E = calE - 1``).
    c : float or ndarray
        Truncation radius (concentration). Broadcasts against ``cal``.

    Returns
    -------
    q : ndarray
    """
    e = np.asarray(cal, dtype=np.float64) - 1.0
    x_max = np.minimum(c, _radius_of_energy(e))
    x = x_max[..., None] * (1.0 - _W**2)
    kin = np.clip(2.0 * (e[..., None] - nfw_phi(x)), 0.0, None) ** 1.5
    return 16.0 * np.pi**2 / 3.0 * np.sum(_WW * 2.0 * x_max[..., None] * _W * x**2 * kin, axis=-1)


def _eddington_integral(b, depth, psi, d2):
    """Eddington's ``f(b) = (1 / (√8 π²)) ∫_0^b (d²ρ/dψ²) dψ / sqrt(b - ψ)`` from a tabulated ``d²ρ/dψ²``.

    With ``ψ = b - s²`` and ``s = sqrt(δ) sinh t``, where ``δ = ψ(0) - b``
    is the depth of ``b`` below the bottom of the well, the integral is
    ``∫_0^{asinh(sqrt(b / δ))} 2 (d²ρ/dψ²) sqrt(δ) cosh t dt``, taken by
    Gauss–Legendre in ``t``. Near the centre ``d²ρ/dψ²`` peaks like a power
    of ``δ + s²`` within ``s ~ sqrt(δ)`` of the endpoint; in ``t`` that peak
    is smooth and resolved at any depth.

    Parameters
    ----------
    b : ndarray
        Relative energies ``-E``; ``f = 0`` for ``b <= 0``.
    depth : ndarray
        ``ψ(0) - b``, broadcasting against ``b``; floored at the smallest
        normal float.
    psi, d2 : ndarray
        ``ψ`` (increasing) and ``d²ρ/dψ²`` there.

    Returns
    -------
    f : ndarray
    """
    b = np.clip(b, 0.0, None)
    depth = np.maximum(depth, np.finfo(np.float64).tiny)
    t_max = np.arcsinh(np.sqrt(b / depth))
    t = t_max[..., None] * _W
    root = np.sqrt(depth)[..., None]
    d2_at = np.interp(b[..., None] - (root * np.sinh(t)) ** 2, psi, d2)
    return t_max * np.sum(_WW * 2.0 * d2_at * root * np.cosh(t), axis=-1) / (np.sqrt(8.0) * np.pi**2)


def _eddington(b):
    """Eddington distribution function of the isotropic NFW profile at relative energies ``b = -E = 1 - calE``.

    :func:`_eddington_integral` with ``ψ(0) = 1``. Accurate (to 1e-3) down
    to ``b ~ 1e-8``, set by where ``_X`` ends.
    """
    psi = -nfw_phi(_X)                                    # decreasing in x
    rho = 1.0 / (4.0 * np.pi * _X * (1.0 + _X) ** 2)
    d2 = np.gradient(np.gradient(rho, psi), psi)
    b = np.asarray(b, dtype=np.float64)
    return _eddington_integral(b, 1.0 - b, psi[::-1], d2[::-1])


def _depth_nodes(eps_kink, eps_last, n_deep, n_shallow):
    """Scaled energies ``eps = E / Φ₀`` of a table's nodes, increasing: from ``E = 0`` towards the centre.

    The kink ``eps_kink = Φ(R) / Φ₀``, where the radial apocentre reaches the
    counting radius ``R``, is a node: ``g`` and ``q`` stop growing like the
    untruncated ones there, and ``g`` has a ``(eps_kink - eps)^(3/2)`` term
    just past it.

    - Deeper, ``n_deep`` nodes are uniform in ``logit(calE)``,
      ``calE = 1 - eps``, from the kink to :data:`_DEEPEST`. They are
      geometric in ``calE`` towards the centre, where the distributions are
      power laws of ``calE``, and geometric in ``eps`` below a
      point-mass-like kink (``eps_kink << 1``), where they are power laws of
      ``eps``.
    - Shallower, ``n_shallow`` nodes are uniform in
      ``logit(1 - eps / eps_kink)`` from :data:`_KINK_GAP` to
      ``eps = eps_last``: geometric away from the kink, and towards
      ``E = 0``, where ``N`` is a power law of ``-E``.

    Linear interpolation in ``eps`` is then accurate at both ends and at the
    kink without a logarithm per query, and ``eps``, unlike ``calE``, keeps
    its relative precision as ``E -> 0``.

    Parameters
    ----------
    eps_kink : float or ndarray
        One row of nodes per element.
    eps_last : float or ndarray
        Shallowest node, below ``eps_kink (1 - _KINK_GAP)``; broadcasts
        against ``eps_kink``.
    n_deep, n_shallow : int

    Returns
    -------
    eps : ndarray of shape ``np.shape(eps_kink) + (n_shallow + 1 + n_deep,)``
    """
    k = np.asarray(eps_kink, dtype=np.float64)[..., None]
    r = np.asarray(eps_last, dtype=np.float64)[..., None] / k
    t_last, t_gap = np.log1p(-r) - np.log(r), np.log(_KINK_GAP) - np.log1p(-_KINK_GAP)
    shallow = k / (1.0 + np.exp(t_last + (t_gap - t_last) * np.linspace(0.0, 1.0, n_shallow)))
    t_kink, t_deep = np.log1p(-k) - np.log(k), np.log(_DEEPEST) - np.log1p(-_DEEPEST)   # logit(calE)
    deep = 1.0 / (1.0 + np.exp(t_kink + (t_deep - t_kink) * np.linspace(0.0, 1.0, n_deep + 1)[1:]))
    return np.concatenate([shallow, k, deep], axis=-1)


def _nfw_nodes(c):
    """Node energies ``eps`` of the table row at concentration ``c``: :func:`_depth_nodes` with the kink ``-phi(c)`` and the last node at ``1e-5`` of it.

    Parameters
    ----------
    c : float or ndarray
        Concentration(s); an array gives one row of nodes per element.

    Returns
    -------
    eps : ndarray of shape ``np.shape(c) + (n,)``
    """
    k = -nfw_phi(c)
    return _depth_nodes(k, 1e-5 * k, *_NFW_NODES)


class NFWEnergyTables(NamedTuple):
    """NFW energy-distribution tables: one row per ``C_GRID`` value, on that row's nodes ``_nfw_nodes(c)``."""

    log_density: np.ndarray          # log n(calE | c), normalised on [0, 1]
    cdf: np.ndarray                  # u = F_c(calE), the mass fraction more bound
    log_volume_fraction: np.ndarray  # log w = log q(calE | c) - log q(1 | c)


@functools.cache
def nfw_energy_tables():
    """Build the NFW energy-distribution tables (once per process).

    Returns
    -------
    tables : NFWEnergyTables
        Arrays of shape ``(len(C_GRID), n)``, on the nodes
        ``_nfw_nodes(C_GRID)``.
    """
    eps = _nfw_nodes(C_GRID)
    cal = 1.0 - eps
    n = _eddington(eps) * nfw_density_of_states(cal, C_GRID[:, None])
    seg = 0.5 * (n[:, 1:] + n[:, :-1]) * np.diff(eps, axis=1)
    # mass more bound than each node: summed from the deepest, plus the power-law tail below it
    tail = _deep_tail(n[:, -2:], cal[:, -2:])
    cdf = np.concatenate([np.cumsum(seg[:, ::-1], axis=1)[:, ::-1], np.zeros((len(C_GRID), 1))], axis=1) + tail[:, None]
    n /= cdf[:, :1]
    cdf /= cdf[:, :1]
    log_q = np.log(nfw_phase_volume(cal, C_GRID[:, None]))
    log_q_zero = np.log(nfw_phase_volume(np.ones(1), C_GRID[:, None]))     # E = 0
    return NFWEnergyTables(np.log(n), cdf, log_q - log_q_zero)


def _deep_tail(n, cal):
    """Mass deeper than the deepest node, ``∫_0^{cal_0} n dcal = n_0 cal_0 / (p + 1)``.

    ``n`` is a power law ``cal^p`` there; ``p`` comes from the two deepest
    nodes.

    Parameters
    ----------
    n, cal : ndarray of shape (..., 2)
        ``n`` (per unit ``cal``) and ``cal`` at the second-deepest and the
        deepest node.

    Returns
    -------
    tail : ndarray of shape (...)
    """
    p = np.log(n[..., 1] / n[..., 0]) / np.log(cal[..., 1] / cal[..., 0])
    return n[..., 1] * cal[..., 1] / (p + 1.0)


def _row(table, c):
    """Row of ``table`` at concentration ``c``: linear in log c between rows (node by node, kink to kink), clipped to the grid."""
    lc = np.log(np.clip(c, C_GRID[0], C_GRID[-1]))
    j = int(np.clip(np.searchsorted(_LOG_C, lc), 1, len(C_GRID) - 1))
    w = (lc - _LOG_C[j - 1]) / (_LOG_C[j] - _LOG_C[j - 1])
    return (1.0 - w) * table[j - 1] + w * table[j]


def _lookup(table, eps, c):
    """``table`` at scaled energies ``eps`` and concentration ``c``.

    Linear in ``eps`` between the row's nodes (one ``np.interp``), held
    at the end nodes beyond them; ``c`` is clipped to the grid.
    """
    c = np.clip(c, C_GRID[0], C_GRID[-1])
    return np.interp(np.asarray(eps, dtype=np.float64), _nfw_nodes(c), _row(table, c))


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
    return _lookup(nfw_energy_tables().cdf, eps, c)


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
    return _lookup(nfw_energy_tables().log_density, eps, c)


_CENTRAL_VOLUME_SLOPE = 4.5        # q ∝ calE^(9/2) near the NFW centre (phi ≈ -1 + x/2)


def nfw_log_phase_space_fraction(eps, c):
    """Log fraction ``ln w = ln q(E) - ln q(0)`` of the virial sphere's bound phase space more bound than ``eps``.

    ``w`` is uniform on ``(0, 1]`` for particles spread uniformly in the
    bound phase space of ``x <= c``. Deeper than the table, the exact
    central power law ``q ∝ calE^(9/2)`` extends it; working in logs keeps
    the deepest values (``w`` down to ~1e-20 and below) exact.

    Parameters
    ----------
    eps : ndarray
        Normalised boundness ``E / Phi_0``.
    c : float
        Concentration.

    Returns
    -------
    log_w : ndarray
        ``<= 0``.
    """
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    cal = 1.0 - eps
    log_w = _lookup(nfw_energy_tables().log_volume_fraction, eps, c)
    deep = cal < _DEEPEST
    log_w[deep] += _CENTRAL_VOLUME_SLOPE * np.log(np.maximum(cal[deep], 1e-300) / _DEEPEST)
    return log_w


_SHALLOWEST = 1e-12                            # shallowest node of the generic tables: -E = 1e-12 |Φ(0)|
_GENERIC_NODES = 640, 660                      # nodes deeper and shallower than the kink (see _depth_nodes)
_APO_NODES = 4000                              # radial nodes (0, then log-spaced 1e-6 R .. R) of the apocentre lookup
_EDDINGTON_R = np.geomspace(1e-7, 1e16, 46000)  # physical radii (kpc) of the generic Eddington inversion: f to -E ~ 1e-12 |Φ(0)|


def eddington_distribution_function(potential, energy):
    """Isotropic distribution function ``f(E)`` of a potential's own, untruncated density.

    Eddington's inversion for an isotropic system,
    ``f(E) = (1 / (√8 π²)) ∫_0^{ψ_E} (d²ρ/dψ²) dψ / sqrt(ψ_E - ψ)`` with
    ``ψ = -Φ`` and ``ψ_E = -E``, without the boundary term (``ρ -> 0`` as
    ``ψ -> 0``), taken by :func:`_eddington_integral`, resolved at any
    depth below ``ψ(0) = -Φ(0)``. ``ρ(ψ)`` is tabulated on
    :data:`_EDDINGTON_R`, and its derivatives go through the radius,
    ``dψ/dr = -G M(<r) / r²``, which stays finite in cores where ``ψ``
    barely changes. ``f`` is the phase-space density whose velocity
    integral gives the potential's mass density.

    Parameters
    ----------
    potential : PotentialModel
        Provides ``potential``, ``enclosed_mass``, ``density`` and ``G``.
    energy : ndarray
        Physical specific energies; ``f = 0`` for ``energy >= 0``.

    Returns
    -------
    f : ndarray of float64
    """
    r = _EDDINGTON_R
    psi = -np.asarray(potential.potential(r), dtype=np.float64)
    dpsi = -potential.G * np.asarray(potential.enclosed_mass(r), dtype=np.float64) / r**2
    with np.errstate(divide="ignore", invalid="ignore"):
        drho = np.gradient(np.asarray(potential.density(r), dtype=np.float64), r) / dpsi
        d2 = np.gradient(drho, r) / dpsi
    keep = np.isfinite(d2) & np.concatenate(([True], np.diff(psi) < 0))   # ψ strictly decreasing in r
    psi, d2 = psi[keep][::-1], d2[keep][::-1]                              # increasing ψ
    energy = np.asarray(energy, dtype=np.float64)
    phi0 = float(np.asarray(potential.potential(np.zeros(1)), dtype=np.float64)[0])
    return _eddington_integral(-energy, energy - phi0, psi, d2)


class SphericalDistribution:
    """Energy distribution and phase-space volume of a spherical potential inside ``r_max``.

    It counts the potential's untruncated, isotropic profile inside the
    sphere of radius ``r_max`` (see the module docstring for the
    definitions). It provides three quantities at any physical energy:

    - :meth:`energy_fraction`: the fraction of the mass inside ``r_max``
      that is more bound than ``E``. It is uniform on ``(0, 1)`` for that
      mass itself;
    - :meth:`log_energy_density`: ``ln N(E)``, per unit ``E``;
    - :meth:`log_phase_space_fraction`: ``ln w(E)``. ``w`` is uniform on
      ``(0, 1]`` for particles spread uniformly in the bound phase space
      inside ``r_max``.

    Everything is tabulated over ``eps = E / Φ(0)`` (every potential is
    finite at ``r = 0``) on :func:`_depth_nodes`, from ``calE = 1 - eps =
    1e-6`` to ``eps = 1e-12``, with the kink ``Φ(r_max) / Φ(0)`` as a
    node. That resolves a deep core (``calE -> 0``, where ``q`` is a power
    law of ``calE``), the kink, the approach to ``E = 0``, and a
    point-mass-like well, where most particles sit at ``eps ~ softening / r``.

    The tables are built on first use, in float64:
    - the phase-space fraction needs ``q`` only, so it works for any
      potential;
    - the energy pair also needs ``f`` (the potential's
      :meth:`distribution_function`, which raises for a potential without
      one) and ``g``.

    Instances are cached per potential and ``r_max`` by
    :class:`~roadrunner.physics.potentials.SphericalPotential`. Physical
    energies in; float64 out.

    Parameters
    ----------
    potential : PotentialModel
    r_max : float
        Counting radius (physical).
    """

    def __init__(self, potential, r_max):
        self.potential = potential
        self.r_max = float(r_max)
        self._r = self.r_max * np.concatenate(([0.0], np.geomspace(1e-6, 1.0, _APO_NODES - 1)))
        self._phi = np.asarray(potential.potential(self._r), dtype=np.float64)
        self._eps = _depth_nodes(self._phi[-1] / self._phi[0], _SHALLOWEST, *_GENERIC_NODES)
        self._energies = self._phi[0] * self._eps                 # physical energies at the nodes
        self._log_w = None
        self._log_density = self._cdf = None

    def _scaled(self, energy):
        """``eps = E / Φ(0)``: 0 at ``E = 0``, 1 at the bottom of the well.

        Parameters
        ----------
        energy : ndarray

        Returns
        -------
        eps : ndarray of float64
        """
        return np.asarray(energy, dtype=np.float64) / self._phi[0]

    def _q_and_g(self, energy):
        """Phase-space volume ``q(E)`` and density of states ``g(E)`` inside ``r_max``.

        Both integrals run over ``r`` from 0 to ``x_max = min(r_max, r_E)``,
        with ``r = x_max (1 - w²)``, which removes the square-root endpoint
        at the apocentre, and Gauss–Legendre in ``w``. They share their
        nodes and potential evaluations.

        Parameters
        ----------
        energy : ndarray of shape (n,)
            Physical energies (float64).

        Returns
        -------
        q, g : ndarray of float64, shape (n,)
        """
        x_max = np.interp(energy, self._phi, self._r)       # apocentre, capped at r_max
        x = x_max[:, None] * (1.0 - _W**2)
        kin2 = np.clip(2.0 * (energy[:, None] - np.asarray(self.potential.potential(x), dtype=np.float64)), 0.0, None)
        w = 16.0 * np.pi**2 * _WW * 2.0 * x_max[:, None] * _W * x**2
        return np.sum(w * kin2**1.5, axis=-1) / 3.0, np.sum(w * np.sqrt(kin2), axis=-1)

    def _phase_table(self):
        """``ln q(E) - ln q(0)`` at the depth nodes, built once.

        Returns
        -------
        log_w : ndarray of float64
        """
        if self._log_w is None:
            q, _ = self._q_and_g(self._energies)
            q0, _ = self._q_and_g(np.zeros(1))
            self._log_w = np.log(q) - np.log(q0[0])
        return self._log_w

    def _energy_table(self):
        """``ln N`` per unit ``E`` and its cumulative from the deepest node, built once.

        ``N = f g`` is normalised so that ``∫ N dE = 1``, with
        ``dE = |Φ(0)| d eps``: over the nodes, plus the power-law tail deeper
        than ``calE = 1e-6`` (:func:`_deep_tail`).

        Returns
        -------
        log_n, cdf : ndarray of float64
        """
        if self._log_density is None:
            f = np.asarray(self.potential.distribution_function(self._energies), dtype=np.float64)
            _, g = self._q_and_g(self._energies)
            n = f * g
            seg = -self._phi[0] * 0.5 * (n[1:] + n[:-1]) * np.diff(self._eps)
            tail = -self._phi[0] * _deep_tail(n[-2:], 1.0 - self._eps[-2:])
            cdf = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0])) + tail   # mass more bound than each node
            with np.errstate(divide="ignore"):
                self._log_density = np.log(n / cdf[0])
            self._cdf = cdf / cdf[0]
        return self._log_density, self._cdf

    def energy_fraction(self, energy):
        """Fraction of the mass inside ``r_max`` more bound than ``E``.

        It is uniform on ``(0, 1)`` for that mass itself, which is the null
        hypothesis of the ``energy`` plausibility model. It is held at 0 and
        1 beyond the depth grid.

        Parameters
        ----------
        energy : ndarray
            Physical specific energies.

        Returns
        -------
        u : ndarray of float64
        """
        return np.interp(self._scaled(energy), self._eps, self._energy_table()[1])

    def log_energy_density(self, energy):
        """``ln N(E)``: energy distribution of the mass inside ``r_max``, per unit ``E``.

        It is held at its end values beyond the depth grid.

        Parameters
        ----------
        energy : ndarray
            Physical specific energies.

        Returns
        -------
        log_n : ndarray of float64
        """
        return np.interp(self._scaled(energy), self._eps, self._energy_table()[0])

    def log_phase_space_fraction(self, energy):
        """``ln w = ln q(E) - ln q(0)``: fraction of the bound phase space inside ``r_max`` more bound than ``E``.

        It is uniform on ``(0, 1]`` for particles spread uniformly in that
        phase space. Deeper than the table it follows the local power law of
        ``q`` in ``calE`` through the two deepest nodes; ``calE`` is floored
        at the smallest normal float, so energies at or below ``Φ(0)`` (a
        particle at rest at the centre, or below it by rounding) stay
        finite. It is held at its shallowest node (``≈ 0``) for ``E >= 0``.

        Parameters
        ----------
        energy : ndarray
            Physical specific energies.

        Returns
        -------
        log_w : ndarray of float64
            ``<= 0``.
        """
        eps = np.atleast_1d(self._scaled(energy))
        row = self._phase_table()
        log_w = np.interp(eps, self._eps, row)
        deep = eps > self._eps[-1]
        cal = 1.0 - self._eps[-2:]                                       # the two deepest nodes
        slope = (row[-1] - row[-2]) / np.log(cal[1] / cal[0])
        log_w[deep] = row[-1] + slope * np.log(np.maximum(1.0 - eps[deep], np.finfo(np.float64).tiny) / cal[1])
        return log_w

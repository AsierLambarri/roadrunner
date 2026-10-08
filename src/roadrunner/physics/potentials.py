"""Spherical gravitational potentials: Plummer, Kepler, Hernquist, NFW and particle sets.

:class:`SphericalPotential` implements :class:`~roadrunner._mcf_types.PotentialModel`
for any spherical density: a concrete potential implements the potential and
the density, and overrides the enclosed mass and the central potential (whose
generic forms here are numerical) and any derived quantity only with an exact
closed form. The primitives (potential, enclosed mass, density, central
potential) are linear in mass, so they add over potentials. Provides
:class:`PlummerPotential`, :class:`KeplerPotential` (a Plummer sphere with the
fixed softening :const:`SOFTENING_KEPLER`), :class:`HernquistPotential`,
:class:`NFWPotential`, :class:`ShellPotential` (the spherical potential of a
set of particles) and :func:`get_potential` for dispatch by model name.
Potentials are radial functions of the physical radius; their optional
``centre`` is read by :class:`~roadrunner.physics.halo_model.HaloModel`.
"""

from abc import abstractmethod

import numpy as np
from numba import get_num_threads, njit, prange
from scipy.integrate import cumulative_trapezoid
from scipy.special import betainc

from roadrunner._mcf_types import PotentialModel
from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER, SOFTENING_NFW
from roadrunner.physics.energy_distribution import (
    nfw_energy_fraction,
    nfw_log_energy_density,
    nfw_log_phase_space_fraction,
)

_2PI = 2 * np.pi
_MASS_GRID_NODES = 1024            # generic M(<r): trapezoid nodes, log-spaced in r
_MASS_GRID_INNER = 1e-8            # innermost node / outermost radius (plus r = 0)
_CENTRAL_RADIUS = 1e-6             # generic Φ₀: the potential at this physical radius


class SphericalPotential(PotentialModel):
    """Spherical potential: the protocol's methods from a potential and a density.

    A concrete potential implements :meth:`potential` and :meth:`density`. The
    enclosed mass and the central potential have numerical generic forms (the
    trapezoid rule over the density; the potential at ``_CENTRAL_RADIUS``),
    which a concrete potential overrides with its exact ones. The orbital time
    and the tidal denominator follow from them; a concrete potential overrides
    one only with an exact closed form. The energy distribution and the
    phase-space fraction raise unless a potential defines them.

    Parameters
    ----------
    M : float
        Mass parameter.
    G : float, default=G_KM
        Gravitational constant.
    centre : ndarray of shape (3,), optional
        Centre in the particles' (merger-tree) coordinates, read by
        :class:`HaloModel`; ``None`` is the halo's own centre.
    """

    def __init__(self, M, G=G_KM, centre=None):
        self.M = M
        self.G = G
        self.centre = centre

    @abstractmethod
    def potential(self, r):
        """Potential ``Φ(r)`` at physical radii ``r``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        phi : ndarray
        """

    @abstractmethod
    def density(self, r):
        """Density ``ρ(r)`` at physical radii ``r``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """

    def enclosed_mass(self, r):
        """Mass ``M(<r) = ∫_0^r 4π x² ρ(x) dx`` by the trapezoid rule.

        Nodes: ``r = 0`` and ``_MASS_GRID_NODES`` log-spaced radii from
        ``_MASS_GRID_INNER`` times the largest ``r`` to it; ``x² ρ`` is taken
        as 0 at ``r = 0`` (any density shallower than ``r⁻³``).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        r = np.asarray(r, dtype=np.float64)
        r_max = r.max(initial=0.0)
        if r_max <= 0:
            return np.zeros_like(r)
        x = np.concatenate(([0.0], np.geomspace(_MASS_GRID_INNER * r_max, r_max, _MASS_GRID_NODES)))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 4 * np.pi * x**2 * self.density(x)
        f[0] = 0.0
        return np.interp(r, x, cumulative_trapezoid(f, x, initial=0.0))

    def central_potential(self):
        """Central potential ``Φ₀``: the potential at ``_CENTRAL_RADIUS``.

        Returns
        -------
        phi0 : float
        """
        return float(self.potential(np.array([_CENTRAL_RADIUS]))[0])

    def orbital_time(self, E, r):
        """Orbital timescale: the dynamical time ``2π sqrt(r³ / (G M(<r)))`` at the instantaneous radius.

        ``inf`` where no mass is enclosed.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies (unused).
        r : ndarray

        Returns
        -------
        t : ndarray
        """
        menc = self.enclosed_mass(r)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _2PI * np.sqrt(r**3 / (self.G * menc))
        return np.where(menc > 0, t, np.inf)

    def tidal_denominator(self, r):
        """Tidal denominator ``3 M(<r) - dM/d ln r``, with ``dM/d ln r = 4π r³ ρ(r)`` (0 at ``r = 0``).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        denom : ndarray
        """
        with np.errstate(invalid="ignore"):
            dm_dlnr = 4 * np.pi * r**3 * self.density(r)
        return 3 * self.enclosed_mass(r) - np.where(r > 0, dm_dlnr, 0.0)

    def energy_fraction(self, eps):
        """Not defined for this potential: no dark-matter energy distribution."""
        raise NotImplementedError(f"{type(self).__name__} has no energy distribution.")

    def log_energy_density(self, eps):
        """Not defined for this potential: no dark-matter energy distribution."""
        raise NotImplementedError(f"{type(self).__name__} has no energy distribution.")

    def log_phase_space_fraction(self, boundness):
        """Not defined for this potential: no phase-space volume."""
        raise NotImplementedError(f"{type(self).__name__} has no phase-space volume.")


class PlummerPotential(SphericalPotential):
    """Plummer sphere: ``Φ(r) = -G M / sqrt(r² + a²)``.

    ``M(<r) = M r³ / (r² + a²)^(3/2)``, ``ρ(r) = 3 M / (4π a³) (1 + r²/a²)^(-5/2)``
    and ``Φ₀ = -G M / a``.

    Parameters
    ----------
    M : float
        Total mass.
    a : float
        Plummer radius (physical); the half-mass radius is ``1.305 a``.
    G : float, default=G_KM
        Gravitational constant.
    centre : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, M, a, G=G_KM, centre=None):
        super().__init__(M, G=G, centre=centre)
        self.a = a

    def potential(self, r):
        """``-G M / sqrt(r² + a²)``.

        Parameters
        ----------
        r : ndarray
            Physical radii.

        Returns
        -------
        phi : ndarray
        """
        r_safe = np.sqrt(r**2 + self.a**2)
        return -self.G * self.M / r_safe

    def density(self, r):
        """``3 M / (4π a³) (1 + r²/a²)^(-5/2)``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """
        return 3 * self.M / (4 * np.pi * self.a**3) * (1 + r**2 / self.a**2) ** -2.5

    def enclosed_mass(self, r):
        """Exact: ``M r³ / (r² + a²)^(3/2)``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        return self.M * r**3 / (r**2 + self.a**2) ** 1.5

    def central_potential(self):
        """Exact: ``Φ₀ = -G M / a``.

        Returns
        -------
        phi0 : float
        """
        return -self.G * self.M / self.a

    def orbital_time(self, E, r):
        """Exact dynamical time ``2π sqrt((r² + a²)^(3/2) / (G M))``, finite at ``r = 0``.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies (unused).
        r : ndarray

        Returns
        -------
        t : ndarray
        """
        return _2PI * np.sqrt((r**2 + self.a**2) ** 1.5 / (self.G * self.M))


class KeplerPotential(PlummerPotential):
    """Kepler (point-mass) potential: a Plummer sphere with the fixed softening :const:`SOFTENING_KEPLER`.

    Its central value is set by the softening alone, so it has no meaningful
    ``Φ₀``: that method raises. Its derived quantities are the point mass's:
    the orbital timescale is the period of the orbit, the tidal denominator
    ``3 M``, and its phase-space volume has a closed form.

    Parameters
    ----------
    M : float
        Enclosed mass.
    G : float, default=G_KM
        Gravitational constant.
    centre : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, M, G=G_KM, centre=None):
        super().__init__(M, SOFTENING_KEPLER, G=G, centre=centre)

    def orbital_time(self, E, r):
        """Kepler period ``2π sqrt(s³ / (G M))``, ``s = -G M / (2E)``, of bound particles; 0 when ``E >= 0``.

        Evaluated on bound particles only (I03): unbound ones have no orbit.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies.
        r : ndarray
            Radii (unused).

        Returns
        -------
        t : ndarray
        """
        t = np.zeros(E.shape, dtype=np.result_type(E, self.G * self.M))
        bound = E < 0
        s = -0.5 * self.G * self.M / E[bound]
        t[bound] = _2PI * np.sqrt(s**3 / (self.G * self.M))
        return t

    def tidal_denominator(self, r):
        """Point-mass tidal denominator ``3 M`` at every radius.

        The softened form vanishes at ``r = 0``, which would make the main
        host's own tidal radius (at distance 0) undefined.

        Parameters
        ----------
        r : ndarray (unused)

        Returns
        -------
        denom : float
        """
        return 3 * self.M

    def central_potential(self):
        """Not defined: the softened central value is not physical."""
        raise NotImplementedError("The Kepler potential has no finite central potential.")

    def log_phase_space_fraction(self, boundness):
        """Log fraction ``ln w`` of the virial sphere's bound phase space more bound than ``b = -E / v_vir²``.

        Closed form (softening holds negligible volume):
        ``w = (3/2) B(3/2, 5/2) b^(-3/2) I_x(3/2, 5/2)``, ``x = min(1, b)``,
        ``B(3/2, 5/2) = pi / 16``, so ``w = (3 pi / 32) b^(-3/2)`` once the
        orbit stays inside the virial sphere (``b >= 1``).

        Parameters
        ----------
        boundness : ndarray
            ``b = -E / v_vir²`` (positive for bound particles).

        Returns
        -------
        log_w : ndarray
        """
        b = np.asarray(boundness, dtype=np.float64)
        return np.log(3.0 * np.pi / 32.0) - 1.5 * np.log(b) + np.log(betainc(1.5, 2.5, np.minimum(1.0, b)))


class HernquistPotential(SphericalPotential):
    """Hernquist sphere: ``Φ(r) = -G M / (r + a)``.

    ``M(<r) = M r² / (r + a)²``, ``ρ(r) = M a / (2π r (r + a)³)`` and
    ``Φ₀ = -G M / a``.

    Parameters
    ----------
    M : float
        Total mass.
    a : float
        Scale radius (physical); the half-mass radius is ``(1 + sqrt(2)) a``.
    G : float, default=G_KM
        Gravitational constant.
    centre : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, M, a, G=G_KM, centre=None):
        super().__init__(M, G=G, centre=centre)
        self.a = a

    def potential(self, r):
        """``-G M / (r + a)``.

        Parameters
        ----------
        r : ndarray
            Physical radii.

        Returns
        -------
        phi : ndarray
        """
        return -self.G * self.M / (r + self.a)

    def density(self, r):
        """``M a / (2π r (r + a)³)``; infinite at ``r = 0``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """
        with np.errstate(divide="ignore"):
            return self.M * self.a / (2 * np.pi * r * (r + self.a) ** 3)

    def enclosed_mass(self, r):
        """Exact: ``M r² / (r + a)²``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        return self.M * r**2 / (r + self.a) ** 2

    def central_potential(self):
        """Exact: ``Φ₀ = -G M / a``.

        Returns
        -------
        phi0 : float
        """
        return -self.G * self.M / self.a

    def orbital_time(self, E, r):
        """Exact dynamical time ``2π (r + a) sqrt(r / (G M))``, 0 at ``r = 0``.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies (unused).
        r : ndarray

        Returns
        -------
        t : ndarray
        """
        return _2PI * (r + self.a) * np.sqrt(r / (self.G * self.M))


class NFWPotential(SphericalPotential):
    """Navarro–Frenk–White gravitational potential.

    ``Φ(r) = -G M / r · ln(1 + r/Rs) / A`` where
    ``A = ln(1 + c) - c/(1 + c)``, with a softening length
    :const:`SOFTENING_NFW` to regularise the origin. ``M(<r)`` is truncated at
    ``x = c``; :meth:`density` is its exact derivative.

    Parameters
    ----------
    M : float
        Virial mass.
    Rs : float
        Scale radius.
    c : float
        Concentration ``c = Rvir / Rs``.
    G : float, default=G_KM
        Gravitational constant.
    centre : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, M, Rs, c, G=G_KM, centre=None):
        super().__init__(M, G=G, centre=centre)
        self.Rs = Rs
        self.c = c

    def potential(self, r):
        """Evaluate the NFW potential.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        phi : ndarray
        """
        r_safe = np.sqrt(r**2 + SOFTENING_NFW**2)
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return -self.G * (self.M / r_safe) * np.log(1 + r_safe / self.Rs) / A

    def density(self, r):
        """Density of the softened, truncated ``M(<r)``: ``M / (4π A Rs² r (1 + x)²)``, 0 beyond ``x = c``.

        ``x = sqrt(r² + ε²) / Rs``, so ``∫ 4π r² ρ dr`` is exactly
        :meth:`enclosed_mass`; infinite at ``r = 0``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """
        x = np.sqrt(r**2 + SOFTENING_NFW**2) / self.Rs
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        with np.errstate(divide="ignore"):
            rho = self.M / (4 * np.pi * A * self.Rs**2 * r * (1 + x) ** 2)
        return np.where(x < self.c, rho, 0.0)

    def enclosed_mass(self, r):
        """Enclosed mass at radius ``r``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        x = np.minimum(self.c, np.sqrt(r**2 + SOFTENING_NFW**2) / self.Rs)
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        B = np.log(1 + x) - x / (1 + x)
        return self.M * B / A

    def tidal_denominator(self, r):
        """Tidal denominator for the NFW profile.

        ``(M / A(c)) · (3 A(x) − x²/(1 + x)²)`` where
        ``A(x) = ln(1 + x) − x/(1 + x)`` and ``x = r / Rs``. Inside r_vir it is
        the derivative form with ``x`` softened; beyond, ``x`` is clipped at
        ``c`` (the generic ``3 M(<r) - 4π r³ ρ`` would give ``3 M``).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        denom : ndarray
        """

        def _f(x):
            """``ln(1 + x) - x / (1 + x)``, the NFW profile shape function.

            Parameters
            ----------
            x : ndarray
                ``r / Rs``, the scaled radius.

            Returns
            -------
            f : ndarray
            """
            return np.log(1 + x) - x / (1 + x)

        x = np.minimum(self.c, np.sqrt(r**2 + SOFTENING_NFW**2) / self.Rs)
        return (self.M / _f(self.c)) * (3 * _f(x) - x**2 / (1 + x)**2)

    def central_potential(self):
        """Central potential ``Φ₀ = -G M / (Rs A(c))`` (unsoftened limit).

        The softened potential never reaches it, so ``E / Φ₀ < 1``.

        Returns
        -------
        phi0 : float
        """
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return -self.G * self.M / (self.Rs * A)

    def energy_fraction(self, eps):
        """Mass fraction of the halo more bound than ``eps = E / Φ₀``.

        Parameters
        ----------
        eps : ndarray

        Returns
        -------
        u : ndarray
        """
        return nfw_energy_fraction(eps, self.c)

    def log_energy_density(self, eps):
        """Log dark-matter energy distribution at ``calE = 1 - eps`` (normalised on [0, 1]).

        Parameters
        ----------
        eps : ndarray

        Returns
        -------
        log_n : ndarray
        """
        return nfw_log_energy_density(eps, self.c)

    def log_phase_space_fraction(self, boundness):
        """Log fraction ``ln w`` of the virial sphere's bound phase space more bound than ``eps = E / Φ₀``.

        Parameters
        ----------
        boundness : ndarray
            ``eps = E / Φ₀``.

        Returns
        -------
        log_w : ndarray
        """
        return nfw_log_phase_space_fraction(boundness, self.c)


@njit(cache=True)
def _cell(r, c, inv_d, K):
    """Cell of radius ``r`` in a grid of ``K`` log-radius cells starting at ``ln r = c``.

    Cell 0 lies inside the innermost node, cells ``1..K`` between nodes and
    cell ``K + 1`` beyond the outermost node.

    Parameters
    ----------
    r : float
    c : float
        Log radius of the innermost node.
    inv_d : float
        Inverse log-radius width of a cell.
    K : int
        Number of cells between nodes.

    Returns
    -------
    b : int
    """
    x = (np.log(r) - c) * inv_d
    if not x >= 0.0:
        return 0
    if x >= K:
        return K + 1
    return int(x) + 1


@njit(parallel=True, cache=True)
def _cells(r, c, inv_d, K):
    """:func:`_cell` of every radius (parallel).

    Parameters
    ----------
    r : ndarray of shape (n,)
    c, inv_d, K
        See :func:`_cell`.

    Returns
    -------
    b : ndarray of int64, shape (n,)
    """
    b = np.empty(r.size, dtype=np.int64)
    for i in prange(r.size):
        b[i] = _cell(r[i], c, inv_d, K)
    return b


@njit(parallel=True, cache=True)
def _shell_histogram(r, m, softening, c, inv_d, K, n_chunks):
    """Mass and ``Σ m / r`` of the particles in each of the ``K`` cells (parallel over chunks).

    Parameters
    ----------
    r : ndarray of shape (N,)
        Particle radii.
    m : ndarray of shape (N,)
        Particle masses.
    softening : float
        Floor of the radii.
    c, inv_d, K
        See :func:`_cell`.
    n_chunks : int
        Number of thread-local partial histograms.

    Returns
    -------
    hist : ndarray of shape (2, K)
        Cell masses and cell ``Σ m / r``.
    """
    step = (r.size + n_chunks - 1) // n_chunks
    part = np.zeros((n_chunks, 2, K))
    for t in prange(n_chunks):
        acc = np.zeros((2, K))   # thread-local: no false sharing
        for j in range(t * step, min(r.size, (t + 1) * step)):
            rj = max(r[j], softening)
            b = min(int((np.log(rj) - c) * inv_d), K - 1)
            acc[0, b] += m[j]
            acc[1, b] += m[j] / rj
        part[t] = acc
    return part.sum(axis=0)


@njit(parallel=True, cache=True)
def _shell_potential_kernel(r, coef, c, inv_d, K):
    """``Φ = coef[b, 0] / r + coef[b, 1]`` in the cell ``b`` of every radius (parallel).

    Parameters
    ----------
    r : ndarray of shape (n,)
    coef : ndarray of shape (K + 2, 2)
        Per-cell slope and intercept of ``Φ`` in ``1/r``.
    c, inv_d, K
        See :func:`_cell`.

    Returns
    -------
    phi : ndarray of shape (n,)
    """
    phi = np.empty(r.size, dtype=np.float64)
    for i in prange(r.size):
        ri = np.float64(r[i])
        b = _cell(ri, c, inv_d, K)
        phi[i] = coef[b, 0] / ri + coef[b, 1] if b > 0 else coef[0, 1]
    return phi


class ShellPotential(SphericalPotential):
    """Spherical potential of a particle set, each particle a thin shell at its radius.

    ``Φ(r) = -G (M(<r) / r + Σ_{r_j > r} m_j / r_j)`` (Newton's shell
    theorems): the monopole of the particles' own potential, zero at infinity
    only if the set holds all the mass. It is tabulated exactly at
    ``n_nodes + 1`` nodes uniform in ``ln r`` between the innermost and the
    outermost particle (an O(N) histogram, no sort) and interpolated linearly
    in ``1/r`` between them: continuous, exact in cells holding no particle
    (between shells ``Φ`` is linear in ``1/r``), elsewhere within
    ``Δ² / 8 · (dM/d ln r) / M(<r)`` of the exact value (``Δ`` the node
    spacing in ``ln r``). Inside the innermost node it is the constant ``Φ₀``;
    beyond the outermost, ``-G M / r``. Radii are floored at ``softening`` so a
    particle at the centre keeps ``Φ₀`` finite. The enclosed mass is the one
    of the interpolated potential, ``M(<r) = -(dΦ/d(1/r)) / G``, constant in
    each cell; the density follows from its windowed log slope.

    Parameters
    ----------
    r : ndarray of shape (N,)
        Physical radii of the particles from the centre.
    m : float or ndarray of shape (N,)
        Particle masses.
    G : float, default=G_KM
        Gravitational constant.
    softening : float, default=SOFTENING_KEPLER
        Floor of the particle radii (physical): the simulation's force softening.
    n_nodes : int, default=4096
        Cells between the innermost and the outermost node.
    tidal_dlnr : float, default=0.1
        Half-width in ``ln r`` of the window that estimates ``d ln M / d ln r``
        for :meth:`density` (and so the tidal denominator).
    centre : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, r, m, G=G_KM, softening=SOFTENING_KEPLER, n_nodes=4096, tidal_dlnr=0.1, centre=None):
        r = np.asarray(r, dtype=np.float64)
        m = np.broadcast_to(np.asarray(m, dtype=np.float64), r.shape)
        r_min = max(r.min(), softening)
        d = (np.log(max(r.max(), softening) / r_min) or 1.0) / n_nodes
        self._c, self._inv_d, self._K = np.log(r_min), 1.0 / d, n_nodes
        mass, m_over_r = _shell_histogram(
            r, m, softening, self._c, self._inv_d, n_nodes, max(1, min(get_num_threads(), r.size))
        )
        u = np.exp(-(self._c + d * np.arange(n_nodes + 1)))   # 1 / node radius
        m_in = np.concatenate(([0.0], np.cumsum(mass)))
        # Summed from the outermost (smallest) term inwards.
        phi = -G * (m_in * u + np.concatenate((np.cumsum(m_over_r[::-1])[::-1], [0.0])))
        self._coef = np.empty((n_nodes + 2, 2))
        self._coef[0] = 0.0, phi[0]
        self._coef[1:-1, 0] = np.diff(phi) / np.diff(u)
        self._coef[1:-1, 1] = phi[:-1] - self._coef[1:-1, 0] * u[:-1]
        self._coef[-1] = -G * m_in[-1], 0.0
        self._w = max(1, round(tidal_dlnr * self._inv_d))
        super().__init__(float(m_in[-1]), G=G, centre=centre)

    def potential(self, r):
        """Evaluate the potential.

        Parameters
        ----------
        r : ndarray
            Physical radii.

        Returns
        -------
        phi : ndarray
        """
        r = np.asarray(r)
        return _shell_potential_kernel(r.ravel(), self._coef, self._c, self._inv_d, self._K).reshape(r.shape)

    def density(self, r):
        """``ρ = γ M(<r) / (4π r³)``, ``γ = d ln M / d ln r``; 0 where ``M(<r) = 0``.

        The log slope is the difference of ``ln M(<r)`` across
        ``±tidal_dlnr`` over its span: exact for a power-law ``M(r)``, where a
        difference of ``M`` itself is biased by ``tidal_dlnr² / 6 · d³M/d ln r³``,
        a bias that swamps the tidal denominator ``M(<r) (3 - γ)`` in a core.
        Still noisy there: the particle noise in ``γ`` is amplified by
        ``γ / (3 - γ)``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """
        r = np.asarray(r)
        menc = -self._coef[:, 0] / self.G
        b = _cells(r.ravel(), self._c, self._inv_d, self._K)
        lo = np.clip(b - self._w, 1, self._K)   # cell 0 encloses no mass
        hi = np.clip(b + self._w, lo + 1, self._K + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma = np.log(menc[hi] / menc[lo]) * self._inv_d / (hi - lo)
            rho = gamma * menc[b] / (4 * np.pi * r.ravel() ** 3)
        return np.where(menc[b] > 0, rho, 0.0).reshape(r.shape)

    def enclosed_mass(self, r):
        """Exact for the interpolated potential: ``M(<r) = -(dΦ/d(1/r)) / G``, constant in each cell, 0 inside the innermost node.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        r = np.asarray(r)
        b = _cells(r.ravel(), self._c, self._inv_d, self._K)
        return (-self._coef[b, 0] / self.G).reshape(r.shape)

    def central_potential(self):
        """Central potential ``Φ₀ = -G Σ m_j / r_j``.

        Returns
        -------
        phi0 : float
        """
        return float(self._coef[0, 1])


_POTENTIAL_MODELS: dict[str, type[SphericalPotential]] = {
    "kepler": KeplerPotential,
    "nfw": NFWPotential,
    "plummer": PlummerPotential,
    "hernquist": HernquistPotential,
}


def get_potential(model: str, **kwargs) -> type[SphericalPotential] | SphericalPotential:
    """Return a potential class (no kwargs) or an instantiated object.

    Parameters
    ----------
    model : str
    **kwargs
        Constructor arguments.  If empty the class itself is returned.

    Returns
    -------
    potential : type or instance
    """
    cls = _POTENTIAL_MODELS.get(model.lower())
    if cls is None:
        raise ValueError(
            f"Unknown potential model: {model}. "
            f"Available: {list(_POTENTIAL_MODELS)}"
        )
    if not kwargs:
        return cls
    return cls(**kwargs)

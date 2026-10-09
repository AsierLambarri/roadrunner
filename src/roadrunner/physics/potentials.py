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
set of particles), :class:`CompositeSphericalPotential` (a sum of them) and
:func:`get_potential` for dispatch by model name. Potentials take absolute
physical positions ``xyz`` and measure them from their own ``centre`` (an
absolute physical position; the origin when ``None``); each class implements
its profile in private methods of the physical radius (``_potential(r)``,
``_density(r)``, ...), which the distribution machinery uses on radial
grids.
Radii may be 0: every method evaluates its formula on the whole array and
replaces the 0/0 or 0·∞ it meets at the centre by the limit there. The
energy distribution and the phase-space fraction come from
:class:`~roadrunner.physics.distribution.SphericalDistribution` unless a
potential overrides them. Plummer and Hernquist read one set of tables per
process shared by all their instances (:class:`SelfSimilarPotential`).
"""

import functools
from abc import abstractmethod

import numpy as np
from numba import get_num_threads, njit, prange
from scipy.integrate import cumulative_trapezoid
from scipy.special import betainc

from roadrunner._mcf_types import PotentialModel
from roadrunner.mixture._math import row_squared_norms
from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER
from roadrunner.physics.distribution import (
    PROFILE_HERNQUIST,
    PROFILE_NFW,
    PROFILE_PLUMMER,
    SELF_SIMILAR_C,
    SphericalDistribution,
    eddington_distribution_function,
    nfw_energy_fraction,
    nfw_log_energy_density,
    nfw_log_phase_space_fraction,
    self_similar_lookup,
    self_similar_tables,
)

_2PI = 2 * np.pi
_MASS_GRID_NODES = 1024            # generic M(<r): trapezoid nodes, log-spaced in r
_MASS_GRID_INNER = 1e-8            # innermost node / outermost radius (plus r = 0)


class SphericalPotential(PotentialModel):
    """Spherical potential: the protocol's methods from a potential and a density.

    A concrete potential implements its profile in the physical radius,
    ``_potential(r)`` and ``_density(r)``; the public methods take absolute
    physical positions and measure them from :attr:`centre`. The
    enclosed mass and the central potential have generic forms (the
    trapezoid rule over the density; the potential at ``r = 0``),
    which a concrete potential overrides with its exact ones. The orbital time
    and the tidal denominator follow from them; a concrete potential overrides
    one only with an exact closed form. The energy distribution and the
    phase-space fraction, counted inside a radius ``r_max``, come from
    :class:`SphericalDistribution` (cached per ``r_max``), with ``f`` from
    :meth:`distribution_function`.

    Parameters
    ----------
    M : float
        Mass parameter.
    G : float, default=G_KM
        Gravitational constant.
    centre : ndarray of shape (3,), optional
        Absolute physical position of the potential's centre; the origin
        when ``None``.
    velocity : ndarray of shape (3,), optional
        Absolute velocity of the potential's centre (km/s); at rest when
        ``None``.

    Attributes
    ----------
    has_distribution : bool
        Whether :meth:`distribution_function` is defined (class attribute;
        ``False`` for a point mass and for a particle set).
    """

    has_distribution = True

    def __init__(self, M, G=G_KM, centre=None, velocity=None):
        self.M = M
        self.G = G
        self.centre = np.zeros(3) if centre is None else np.asarray(centre, dtype=np.float64)
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=np.float64)
        self._distributions = {}

    @abstractmethod
    def _potential(self, r):
        """Potential ``Φ(r)`` at physical radii ``r``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        phi : ndarray
        """

    @abstractmethod
    def _density(self, r):
        """Density ``ρ(r)`` at physical radii ``r``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """

    def _enclosed_mass(self, r):
        """Mass ``M(<r) = ∫_0^r 4π x² ρ(x) dx`` by the trapezoid rule.

        Nodes: ``r = 0`` and ``_MASS_GRID_NODES`` log-spaced radii from
        ``_MASS_GRID_INNER`` times the largest ``r`` to it; ``x² ρ`` takes its
        limit 0 at ``r = 0`` (any density shallower than ``r⁻³``).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        r = np.asarray(r, dtype=np.float64)
        x = r.max(initial=0.0) * np.concatenate(([0.0], np.geomspace(_MASS_GRID_INNER, 1.0, _MASS_GRID_NODES)))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.nan_to_num(4 * np.pi * x**2 * self._density(x), nan=0.0)
        return np.interp(r, x, cumulative_trapezoid(f, x, initial=0.0))

    def central_potential(self):
        """Central potential ``Φ₀``: the potential at ``r = 0``.

        Returns
        -------
        phi0 : float
        """
        return float(self._potential(np.zeros(1))[0])

    def profile(self):
        """Compiled description for the q and g quadratures: rows ``(kind, k, a)`` summed by :func:`~roadrunner.physics.distribution.profile_phi`.

        ``None`` (the default) when the potential has none: the quadratures then evaluate
        :meth:`potential`. A composite potential stacks its components' rows.

        Returns
        -------
        profile : ndarray of shape (n, 3) or None
        """
        return None

    def well_depth(self, xyz):
        """Depth ``-Φ₀`` of the potential well, the scale of its binding energies.

        Parameters
        ----------
        xyz : ndarray of shape (1, 3)
            Absolute physical position of the innermost particle; unused by a
            well with a finite centre.

        Returns
        -------
        depth : float
        """
        return -self.central_potential()

    def _orbital_time(self, E, r):
        """Orbital timescale: the dynamical time ``2π sqrt(r³ / (G M(<r)))`` at the instantaneous radius.

        ``inf`` where no mass is enclosed. At ``r = 0`` it is the limit
        ``2π sqrt(3 / (4π G ρ(0)))``, where the mean enclosed density becomes
        ``ρ(0)``: 0 in a cusp, finite in a core.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies (unused).
        r : ndarray

        Returns
        -------
        t : ndarray
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _2PI * np.sqrt(r**3 / (self.G * self._enclosed_mass(r)))
            t0 = _2PI * np.sqrt(3 / (4 * np.pi * self.G * self._density(np.zeros(1))[0]))
        return np.nan_to_num(t, nan=t0, posinf=np.inf)

    def _tidal_denominator(self, r):
        """Tidal denominator ``3 M(<r) - dM/d ln r``, with ``dM/d ln r = 4π r³ ρ(r)`` (0 at ``r = 0``).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        denom : ndarray
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            dm_dlnr = np.nan_to_num(4 * np.pi * r**3 * self._density(r), nan=0.0)
        return 3 * self._enclosed_mass(r) - dm_dlnr

    def potential(self, xyz):
        """Potential at absolute physical positions, measured from :attr:`centre`.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        phi : ndarray of shape (n,)
        """
        return self._potential(np.sqrt(row_squared_norms(np.asarray(xyz, dtype=np.float64) - self.centre)))

    def density(self, xyz):
        """Density at absolute physical positions, measured from :attr:`centre`.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        rho : ndarray of shape (n,)
        """
        return self._density(np.sqrt(row_squared_norms(np.asarray(xyz, dtype=np.float64) - self.centre)))

    def enclosed_mass(self, xyz):
        """Mass inside the sphere about :attr:`centre` through each absolute physical position.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        menc : ndarray of shape (n,)
        """
        return self._enclosed_mass(np.sqrt(row_squared_norms(np.asarray(xyz, dtype=np.float64) - self.centre)))

    def tidal_denominator(self, xyz):
        """Tidal denominator at absolute physical positions, measured from :attr:`centre`.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        denom : ndarray of shape (n,)
        """
        return self._tidal_denominator(np.sqrt(row_squared_norms(np.asarray(xyz, dtype=np.float64) - self.centre)))

    def orbital_time(self, E, xyz):
        """Orbital timescale of energies ``E`` at absolute physical positions, measured from :attr:`centre`.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies.
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        t : ndarray of shape (n,)
        """
        return self._orbital_time(E, np.sqrt(row_squared_norms(np.asarray(xyz, dtype=np.float64) - self.centre)))

    def distribution_function(self, E):
        """Isotropic distribution function ``f(E)`` of the potential's own, untruncated density.

        Generic: Eddington's inversion of :meth:`density` in :meth:`potential`
        (:func:`~roadrunner.physics.distribution.eddington_distribution_function`).
        A potential with a closed form overrides it, and one without a mass
        distribution raises. It feeds the energy distribution ``N = f g``.

        Parameters
        ----------
        E : ndarray
            Physical specific energies; ``f = 0`` for ``E >= 0``.

        Returns
        -------
        f : ndarray of float64
        """
        return eddington_distribution_function(self, E)

    def _distribution(self, r_max):
        """The cached :class:`SphericalDistribution` inside ``r_max``."""
        d = self._distributions.get(r_max)
        if d is None:
            d = self._distributions[r_max] = SphericalDistribution(self, r_max)
        return d

    def energy_fraction(self, E, r_max):
        """Fraction of the mass inside ``r_max`` more bound than ``E``.

        The potential's untruncated, isotropic profile (``f`` from
        :meth:`distribution_function`) counted inside the sphere of radius
        ``r_max``: uniform on ``(0, 1)`` for that mass itself. Generic
        numerical tables (:class:`SphericalDistribution`) unless overridden.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.
        r_max : float
            Counting radius (physical), the boundness search sphere.

        Returns
        -------
        u : ndarray of float64
        """
        return self._distribution(r_max).energy_fraction(E)

    def log_energy_density(self, E, r_max):
        """``ln dN/dE``: energy distribution of the mass inside ``r_max``, per unit physical ``E``.

        ``N = f g``, normalised over the bound energies (see
        :meth:`energy_fraction`). Generic numerical tables unless overridden.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.
        r_max : float
            Counting radius (physical).

        Returns
        -------
        log_n : ndarray of float64
        """
        return self._distribution(r_max).log_energy_density(E)

    def log_phase_space_fraction(self, E, r_max):
        """``ln w``: fraction of the bound phase space inside ``r_max`` more bound than ``E``.

        ``w = q(E) / q(0)``, with ``q`` the phase-space volume inside the
        sphere of radius ``r_max`` more bound than ``E``. It needs only the
        potential, so it is defined for every potential, and it is uniform
        on ``(0, 1]`` for particles spread uniformly in that bound phase
        space. Generic numerical tables unless overridden.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.
        r_max : float
            Counting radius (physical).

        Returns
        -------
        log_w : ndarray of float64
        """
        return self._distribution(r_max).log_phase_space_fraction(E)


@functools.cache
def _self_similar_tables(cls):
    """The shared tables of a self-similar profile class (built once per process): see :func:`self_similar_tables`."""
    return self_similar_tables(cls(1.0, 1.0, G=1.0))


class SelfSimilarPotential(SphericalPotential):
    """A profile fixed up to its mass ``M`` and scale radius ``a``.

    Inside a counting radius ``r_max`` its energy distribution and phase-space fraction depend
    only on ``c = r_max / a`` once energies are scaled by ``Φ₀``, so one set of tables per
    process (:func:`_self_similar_tables`, rows over ``SELF_SIMILAR_C``) serves every
    instance: no per-halo build. Outside that range of ``c`` the generic per-instance route is
    used. A subclass must have ``Φ₀ = -1`` when ``G = M = a = 1`` and the constructor
    signature ``(M, a, G=..., centre=None)``.
    """

    def _lookup(self, name, E, r_max, deep=False):
        """Shared-table value ``name`` at physical energies ``E`` inside ``r_max``."""
        phi0 = self.central_potential()
        eps_kink = float(self._potential(np.array([r_max], dtype=np.float64))[0]) / phi0
        return self_similar_lookup(getattr(_self_similar_tables(type(self)), name),
                                   np.asarray(E, dtype=np.float64) / phi0, eps_kink, r_max / self.a, deep)

    def _tabulated(self, r_max):
        """Whether ``r_max / a`` lies inside the shared tables' rows."""
        return SELF_SIMILAR_C[0] <= r_max / self.a <= SELF_SIMILAR_C[-1]

    def energy_fraction(self, E, r_max):
        """Fraction of the mass inside ``r_max`` more bound than ``E``, from the shared tables (see :meth:`SphericalPotential.energy_fraction`)."""
        if not self._tabulated(r_max):
            return super().energy_fraction(E, r_max)
        return self._lookup("cdf", E, r_max)

    def log_energy_density(self, E, r_max):
        """``ln dN/dE`` inside ``r_max``, per unit physical ``E``, from the shared tables: their density per unit ``eps`` minus ``ln |Φ₀|``."""
        if not self._tabulated(r_max):
            return super().log_energy_density(E, r_max)
        return self._lookup("log_density", E, r_max) - np.log(-self.central_potential())

    def log_phase_space_fraction(self, E, r_max):
        """``ln w`` inside ``r_max``, from the shared tables (see :meth:`SphericalPotential.log_phase_space_fraction`)."""
        if not self._tabulated(r_max):
            return super().log_phase_space_fraction(E, r_max)
        return self._lookup("log_volume_fraction", E, r_max, deep=True)


class PlummerPotential(SelfSimilarPotential):
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
    velocity : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, M, a, G=G_KM, centre=None, velocity=None):
        super().__init__(M, G=G, centre=centre, velocity=velocity)
        self.a = a

    def _potential(self, r):
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

    def _density(self, r):
        """``3 M / (4π a³) (1 + r²/a²)^(-5/2)``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """
        return 3 * self.M / (4 * np.pi * self.a**3) * (1 + r**2 / self.a**2) ** -2.5

    def _enclosed_mass(self, r):
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

    def profile(self):
        """One Plummer row ``(PROFILE_PLUMMER, G M, a)``."""
        return np.array([[PROFILE_PLUMMER, self.G * self.M, self.a]])

    def _orbital_time(self, E, r):
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

    def _tidal_denominator(self, r):
        """Exact: ``3 M(<r) - 4π r³ ρ = 3 M r⁵ / (r² + a²)^(5/2)``, without the generic form's cancellation at small ``r``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        denom : ndarray
        """
        return 3 * self.M * r**5 / (r**2 + self.a**2) ** 2.5

    def distribution_function(self, E):
        """Exact isotropic distribution function of the (untruncated) Plummer sphere.

        ``f(E) = 24√2 / (7π³) · a² / (G⁵ M⁴) · (-E)^(7/2)`` for ``E < 0``, 0
        otherwise (Binney & Tremaine 2008, eq. 4.83): the phase-space density
        whose velocity integral is :meth:`density`. It replaces the generic
        Eddington inversion, which it equals.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.

        Returns
        -------
        f : ndarray of float64
        """
        b = np.clip(-np.asarray(E, dtype=np.float64), 0.0, None)
        return 24 * np.sqrt(2) / (7 * np.pi**3) * self.a**2 / (self.G**5 * self.M**4) * b**3.5


class KeplerPotential(PlummerPotential):
    """Kepler (point-mass) potential: a Plummer sphere with the fixed softening :const:`SOFTENING_KEPLER`.

    Its central value is set by the softening alone, so it has no meaningful
    ``Φ₀``: that method raises, and its well depth is taken at the innermost
    resolved radius. Its derived quantities are the point mass's:
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
    velocity : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    has_distribution = False

    def __init__(self, M, G=G_KM, centre=None, velocity=None):
        super().__init__(M, SOFTENING_KEPLER, G=G, centre=centre, velocity=velocity)

    def _orbital_time(self, E, r):
        """Kepler period ``2π sqrt(s³ / (G M))``, ``s = -G M / (2E)``, of bound particles; 0 when ``E >= 0``.

        Unbound particles have no orbit (I03): their ``s <= 0`` makes the
        square root NaN, which becomes 0.

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
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -0.5 * self.G * self.M / E
            t = _2PI * np.sqrt(s**3 / (self.G * self.M))
        return np.nan_to_num(t, nan=0.0, posinf=np.inf)

    def _tidal_denominator(self, r):
        """Point-mass tidal denominator ``3 M`` at every radius.

        The softened (Plummer) form would vanish inside the softening.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        denom : ndarray
        """
        return np.full(np.shape(r), 3 * self.M)

    def central_potential(self):
        """Not defined: the softened central value is not physical."""
        raise NotImplementedError("The Kepler potential has no finite central potential.")

    def well_depth(self, xyz):
        """Depth ``-Φ`` of the well at the innermost particle.

        A point mass's ``Φ₀`` is set by the softening alone, so its depth is
        taken where the data stop resolving it.

        Parameters
        ----------
        xyz : ndarray of shape (1, 3)
            Absolute physical position of the innermost particle.

        Returns
        -------
        depth : float
        """
        return -float(self.potential(xyz)[0])

    def distribution_function(self, E):
        """Not defined: a point mass has no extended mass distribution, so no energy distribution.

        The energy pair (:meth:`energy_fraction`, :meth:`log_energy_density`)
        therefore raises too; the phase-space fraction has its own closed form.
        """
        raise NotImplementedError("The Kepler potential has no energy distribution.")

    def energy_fraction(self, E, r_max):
        """Not defined: a point mass has no energy distribution (see :meth:`distribution_function`)."""
        raise NotImplementedError("The Kepler potential has no energy distribution.")

    def log_energy_density(self, E, r_max):
        """Not defined: a point mass has no energy distribution (see :meth:`distribution_function`)."""
        raise NotImplementedError("The Kepler potential has no energy distribution.")

    def log_phase_space_fraction(self, E, r_max):
        """Exact ``ln w``: fraction of the bound phase space inside ``r_max`` more bound than ``E``.

        For a point mass, with ``b = -E r_max / (G M)``:
        ``w = (3/2) B(3/2, 5/2) b^(-3/2) I_x(3/2, 5/2)``, ``x = min(1, b)``,
        ``B(3/2, 5/2) = pi / 16``, so ``w = (3 pi / 32) b^(-3/2)`` once the
        orbit stays inside ``r_max`` (``b >= 1``). The softening holds a
        negligible volume, so it is ignored.

        Parameters
        ----------
        E : ndarray
            Physical specific energies (negative for bound particles).
        r_max : float
            Counting radius (physical), the boundness search sphere.

        Returns
        -------
        log_w : ndarray of float64
        """
        b = -np.asarray(E, dtype=np.float64) * r_max / (self.G * self.M)
        return np.log(3.0 * np.pi / 32.0) - 1.5 * np.log(b) + np.log(betainc(1.5, 2.5, np.minimum(1.0, b)))

class HernquistPotential(SelfSimilarPotential):
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
    velocity : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, M, a, G=G_KM, centre=None, velocity=None):
        super().__init__(M, G=G, centre=centre, velocity=velocity)
        self.a = a

    def _potential(self, r):
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

    def _density(self, r):
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

    def _enclosed_mass(self, r):
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

    def profile(self):
        """One Hernquist row ``(PROFILE_HERNQUIST, G M, a)``."""
        return np.array([[PROFILE_HERNQUIST, self.G * self.M, self.a]])

    def _orbital_time(self, E, r):
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

    def distribution_function(self, E):
        """Exact isotropic distribution function of the (untruncated) Hernquist sphere (Hernquist 1990, eq. 17).

        ``f(E) = M / (8√2 π³ a³ v_g³) · (1 - q²)^(-5/2) ·
        [3 asin q + q sqrt(1 - q²) (1 - 2q²) (8q⁴ - 8q² - 3)]``, with
        ``q = sqrt(-E a / (G M))`` and ``v_g = sqrt(G M / a)``; 0 for
        ``E >= 0``. It diverges as ``E -> Φ₀`` (the cusp). It replaces the
        generic Eddington inversion, which it equals. The bracket, which
        cancels to ``(4 asin q)⁵ / 40`` as ``q -> 0``, is evaluated as its
        equal ``(3π/2) I_{q²}(5/2, 5/2)`` (regularised incomplete beta):
        with ``q = sin θ`` it is ``∫_0^{4θ} 2 sin⁴(t/2) dt``.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.

        Returns
        -------
        f : ndarray of float64
        """
        q2 = np.clip(-np.asarray(E, dtype=np.float64) * self.a / (self.G * self.M), 0.0, 1.0)
        vg3 = (self.G * self.M / self.a) ** 1.5
        with np.errstate(divide="ignore"):
            core = 1.5 * np.pi * betainc(2.5, 2.5, q2)
            return self.M / (8 * np.sqrt(2) * np.pi**3 * self.a**3 * vg3) * core / (1 - q2) ** 2.5


class NFWPotential(SphericalPotential):
    """Navarro–Frenk–White gravitational potential.

    ``Φ(r) = -G M / r · ln(1 + r/Rs) / A`` where
    ``A = ln(1 + c) - c/(1 + c)``: finite at ``r = 0`` (``Φ₀``), over the
    density's ``1/r`` cusp. The profile is untruncated: ``M(<R_vir) = M``,
    the virial mass, and ``M(<r)`` keeps growing beyond.

    Parameters
    ----------
    M : float
        Virial mass, ``M(<R_vir)``.
    Rs : float
        Scale radius.
    c : float
        Concentration ``c = Rvir / Rs``.
    G : float, default=G_KM
        Gravitational constant.
    centre : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    velocity : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    def __init__(self, M, Rs, c, G=G_KM, centre=None, velocity=None):
        super().__init__(M, G=G, centre=centre, velocity=velocity)
        self.Rs = Rs
        self.c = c

    def _potential(self, r):
        """``Φ₀ ln(1 + x) / x``, ``x = r / Rs``: ``Φ₀`` at ``r = 0`` (the 0/0 limit).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        phi : ndarray
        """
        x = r / self.Rs
        phi0 = self.central_potential()
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = phi0 * np.log1p(x) / x
        return np.nan_to_num(phi, nan=phi0)

    def _density(self, r):
        """``M / (4π A Rs² r (1 + x)²)``, ``x = r / Rs``; infinite at ``r = 0`` (the cusp).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        rho : ndarray
        """
        x = r / self.Rs
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        with np.errstate(divide="ignore"):
            return self.M / (4 * np.pi * A * self.Rs**2 * r * (1 + x) ** 2)

    def _enclosed_mass(self, r):
        """Exact: ``M A(x) / A(c)``, ``A(x) = ln(1 + x) - x/(1 + x)``, ``x = r / Rs``.

        ``ln(1 + x)`` is ``log1p``: ``A(x) ≈ x²/2`` at small ``x`` stays
        accurate (to ``~ 2e-16 / x`` relative).

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        x = r / self.Rs
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        B = np.log1p(x) - x / (1 + x)
        return self.M * B / A

    def _tidal_denominator(self, r):
        """Exact: ``3 M(<r) - 4π r³ ρ = (M / A(c)) (3 A(x) - x² / (1 + x)²)``.

        ``A(x) = ln(1 + x) - x/(1 + x)``, ``x = r / Rs``: the untruncated
        profile of :meth:`enclosed_mass` and :meth:`density`. ``≈ M x² / (2 A(c))``
        at small ``x``, 0 at ``r = 0``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        denom : ndarray
        """
        x = r / self.Rs
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        B = np.log1p(x) - x / (1 + x)
        return self.M / A * (3 * B - x**2 / (1 + x) ** 2)

    def central_potential(self):
        """Exact: ``Φ₀ = -G M / (Rs A(c))``, the potential at ``r = 0``.

        Returns
        -------
        phi0 : float
        """
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return -self.G * self.M / (self.Rs * A)

    def profile(self):
        """One NFW row ``(PROFILE_NFW, G M / A(c), Rs)``."""
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return np.array([[PROFILE_NFW, self.G * self.M / A, self.Rs]])

    def energy_fraction(self, E, r_max):
        """Exact (tabulated) fraction of the NFW mass inside ``r_max`` more bound than ``E``.

        The per-process NFW tables
        (:func:`~roadrunner.physics.distribution.nfw_energy_fraction`), read at
        the scaled energy ``eps = E / Φ₀`` and the truncation ``c = r_max / Rs``.
        In the tables ``c`` only bounds ``g``; the virial normalisation
        ``A(c)`` is already carried by ``Φ₀``.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.
        r_max : float
            Counting radius (physical).

        Returns
        -------
        u : ndarray of float64
        """
        return nfw_energy_fraction(np.asarray(E, dtype=np.float64) / self.central_potential(), r_max / self.Rs)

    def log_energy_density(self, E, r_max):
        """Exact (tabulated) ``ln dN/dE`` of the NFW mass inside ``r_max``, per unit physical ``E``.

        The tables give ``ln dN/dcal`` (``cal = 1 - E / Φ₀``, normalised on
        ``[0, 1]``), and ``|dcal/dE| = 1 / |Φ₀|`` turns it into a density per
        unit ``E``.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.
        r_max : float
            Counting radius (physical).

        Returns
        -------
        log_n : ndarray of float64
        """
        phi0 = self.central_potential()
        return nfw_log_energy_density(np.asarray(E, dtype=np.float64) / phi0, r_max / self.Rs) - np.log(-phi0)

    def log_phase_space_fraction(self, E, r_max):
        """Exact (tabulated) ``ln w``: fraction of the NFW bound phase space inside ``r_max`` more bound than ``E``.

        The per-process NFW tables at ``eps = E / Φ₀`` and ``c = r_max / Rs``,
        extended deeper than the table by the exact central power law
        ``q ∝ cal^(9/2)``.

        Parameters
        ----------
        E : ndarray
            Physical specific energies.
        r_max : float
            Counting radius (physical).

        Returns
        -------
        log_w : ndarray of float64
        """
        return nfw_log_phase_space_fraction(np.asarray(E, dtype=np.float64) / self.central_potential(), r_max / self.Rs)


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
    each cell; the density follows from its windowed log slope. It has no
    energy distribution yet; its phase-space fraction is the generic one.

    Parameters
    ----------
    xyz : ndarray of shape (N, 3)
        Absolute physical positions of the particles; their radii are
        measured from ``centre``.
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
    velocity : ndarray of shape (3,), optional
        See :class:`SphericalPotential`.
    """

    has_distribution = False

    def __init__(self, xyz, m, G=G_KM, softening=SOFTENING_KEPLER, n_nodes=4096, tidal_dlnr=0.1, centre=None,
                 velocity=None):
        origin = np.zeros(3) if centre is None else np.asarray(centre, dtype=np.float64)
        r = np.sqrt(row_squared_norms(np.asarray(xyz, dtype=np.float64) - origin))
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
        super().__init__(float(m_in[-1]), G=G, centre=centre, velocity=velocity)

    def _potential(self, r):
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

    def _density(self, r):
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

    def _enclosed_mass(self, r):
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

    def distribution_function(self, E):
        """Not defined yet: the density is a windowed log-slope estimate, too noisy for ``d²ρ/dψ²``.

        The energy pair therefore raises; the phase-space fraction, which
        needs only the potential, is the generic one.
        """
        raise NotImplementedError("ShellPotential has no energy distribution yet.")


# Methods a single component answers itself, exactly (its closed forms and tables).
_OWN_METHODS = ("orbital_time", "_orbital_time", "distribution_function", "energy_fraction", "log_energy_density",
                "log_phase_space_fraction")


class CompositeSphericalPotential(SphericalPotential):
    """Sum of spherical potentials: the total potential of a halo built from several components.

    An iterable container of its components (``for p in composite``, ``len``, indexing);
    :meth:`add` adds one on the fly. Each component carries its own ``centre`` and
    ``velocity``; the composite's own are its first component's, and :attr:`com_centre` and
    :attr:`com_velocity` are the components' mass-weighted ones. The primitives and the quantities linear in mass are the sums of the
    components' own methods (potential, density, enclosed mass, central potential, well
    depth, tidal denominator), so every component keeps its exact forms; the profile stacks
    their rows. The public methods take absolute physical positions and measure each
    component from its own ``centre``; the private profile in ``r`` (which the distribution
    machinery uses) treats every component as centred on the composite's centre, its first
    component's.

    The orbital time combines the components' own as ``t^-2 = Σ t_i^-2``, exact for
    dynamical times (``t_i^-2 ∝ M_i(<r)``). The energy distribution and the phase-space
    fraction are those of the whole: the generic
    :class:`~roadrunner.physics.distribution.SphericalDistribution` of the composite, with
    ``f`` Eddington's inversion of the summed density in the summed potential and the q
    and g quadratures compiled through the stacked :meth:`profile`. ``f``, and so the
    energy pair, exists only when every component has one (``has_distribution``); the
    phase-space fraction exists for any components.

    A single component answers those (``_OWN_METHODS``) itself, chosen when components are
    added: its own orbital time, tables and closed forms, exactly.

    Parameters
    ----------
    *components : SphericalPotential
        At least one, all with the same ``G`` (the generic route uses one ``G``).

    Raises
    ------
    ValueError
        If the components' ``G`` differ (also on :meth:`add`).
    """

    def __init__(self, *components):
        super().__init__(0.0, G=components[0].G, centre=components[0].centre, velocity=components[0].velocity)
        self._components = ()
        for p in components:
            self.add(p)

    def add(self, potential):
        """Add a component on the fly (e.g. a halo's stellar counterpart).

        The composite's mass, ``has_distribution`` and single-component
        shortcuts are updated, and its cached distributions are dropped, to be
        rebuilt on next use. Its centre stays its first component's.

        Parameters
        ----------
        potential : SphericalPotential
            With the composite's ``G``.

        Raises
        ------
        ValueError
            If its ``G`` differs from the composite's (the generic route uses one ``G``).
        """
        if potential.G != self.G:
            raise ValueError("The components of a composite potential must share G.")
        self._components += (potential,)
        self.M = sum(p.M for p in self._components)
        self.has_distribution = all(p.has_distribution for p in self._components)
        self._distributions = {}
        for name in _OWN_METHODS:
            self.__dict__.pop(name, None)
        if len(self._components) == 1:
            for name in _OWN_METHODS:
                setattr(self, name, getattr(potential, name))

    def __iter__(self):
        return iter(self._components)

    @property
    def com_centre(self):
        """Mass-weighted centre of the components, ``Σ M_i c_i / Σ M_i`` (absolute physical position)."""
        return sum((p.M / self.M) * p.centre for p in self._components)

    @property
    def com_velocity(self):
        """Mass-weighted velocity of the components, ``Σ M_i v_i / Σ M_i`` (km/s)."""
        return sum((p.M / self.M) * p.velocity for p in self._components)

    def __len__(self):
        return len(self._components)

    def __getitem__(self, i):
        return self._components[i]

    def _sum(self, method, *args):
        """``Σ method(*args)`` over the components, starting from the first (one component's value as is)."""
        first, *rest = self._components
        return sum((getattr(p, method)(*args) for p in rest), getattr(first, method)(*args))

    def _potential(self, r):
        """Co-centred summed potential ``Σ Φ_i(r)`` (the spherical profile the distributions use)."""
        return self._sum("_potential", r)

    def _density(self, r):
        """Co-centred summed density ``Σ ρ_i(r)``."""
        return self._sum("_density", r)

    def _enclosed_mass(self, r):
        """Co-centred summed enclosed mass ``Σ M_i(<r)``."""
        return self._sum("_enclosed_mass", r)

    def _tidal_denominator(self, r):
        """Co-centred summed tidal denominator, each component's exact one."""
        return self._sum("_tidal_denominator", r)

    def _orbital_time(self, E, r):
        """Co-centred orbital timescale ``(Σ t_i^-2)^(-1/2)``."""
        with np.errstate(divide="ignore"):
            return 1.0 / np.sqrt(sum(1.0 / p._orbital_time(E, r) ** 2 for p in self._components))

    def potential(self, xyz):
        """Potential ``Σ Φ_i`` at absolute physical positions, each component from its own centre.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        phi : ndarray of shape (n,)
        """
        return self._sum("potential", xyz)

    def density(self, xyz):
        """Density ``Σ ρ_i`` at absolute physical positions, each component from its own centre."""
        return self._sum("density", xyz)

    def enclosed_mass(self, xyz):
        """``Σ M_i(<r_i)``, each component's mass inside the sphere about its own centre through each position."""
        return self._sum("enclosed_mass", xyz)

    def tidal_denominator(self, xyz):
        """Summed tidal denominator at absolute physical positions, each component from its own centre."""
        return self._sum("tidal_denominator", xyz)

    def orbital_time(self, E, xyz):
        """Orbital timescale ``t = (Σ t_i^-2)^(-1/2)`` over the components' own, each from its own centre.

        Exact for dynamical times ``t_i = 2π sqrt(r_i³ / (G M_i(<r_i)))``, whose ``t_i^-2`` is
        linear in the enclosed mass. A component with ``t_i = 0`` (a cusp, or a point mass's
        unbound particle) gives ``t = 0``; one with ``t_i = ∞`` (no mass enclosed) contributes
        nothing. A point mass's period, taken at the total energy ``E``, enters the same way;
        that combination is not exact.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies (in the total potential).
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        t : ndarray of shape (n,)
        """
        with np.errstate(divide="ignore"):
            return 1.0 / np.sqrt(sum(1.0 / p.orbital_time(E, xyz) ** 2 for p in self._components))

    def central_potential(self):
        """Summed central potential ``Σ Φ_i(0)``; raises if a component has none (Kepler).

        Returns
        -------
        phi0 : float
        """
        return self._sum("central_potential")

    def well_depth(self, xyz):
        """Summed well depth, each component's at the innermost particle's absolute physical position.

        Parameters
        ----------
        xyz : ndarray of shape (1, 3)

        Returns
        -------
        depth : float
        """
        return self._sum("well_depth", xyz)

    def profile(self):
        """The components' profile rows stacked; ``None`` if any component has none (Shell).

        Returns
        -------
        profile : ndarray of shape (n, 3) or None
        """
        rows = [p.profile() for p in self._components]
        return None if any(row is None for row in rows) else np.vstack(rows)

    def distribution_function(self, E):
        """Eddington's ``f`` of the summed density in the summed potential.

        Parameters
        ----------
        E : ndarray

        Returns
        -------
        f : ndarray

        Raises
        ------
        NotImplementedError
            If a component has no distribution function (Kepler, Shell).
        """
        if not self.has_distribution:
            raise NotImplementedError("A component of this composite potential has no energy distribution.")
        return super().distribution_function(E)


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

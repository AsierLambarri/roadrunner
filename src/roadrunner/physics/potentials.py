"""Spherical gravitational potentials: Plummer, Kepler, Hernquist, NFW and particle sets.

:class:`SphericalPotential` implements :class:`~roadrunner._mcf_types.PotentialModel`
for any spherical density: a concrete potential implements the potential and
the density, and overrides the enclosed mass and the central potential (whose
generic forms here are numerical) and any derived quantity only with an exact
closed form. The primitives (potential, enclosed mass, density) are linear in
mass, so they add over potentials; so does the central potential when the
potentials are co-centred (otherwise it is the summed potential at the
composite's own well bottom). Provides
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

from roadrunner._defaults import CONCENTRIC_OFFSET, MONOPOLE_MAX_PULL
from roadrunner._mcf_types import PotentialModel
from roadrunner.mixture._math import row_squared_norms
from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER
from roadrunner.physics.distribution import (
    GL_NODES,
    GL_WEIGHTS,
    PROFILE_HERNQUIST,
    PROFILE_NFW,
    PROFILE_PLUMMER,
    PROFILE_SHELL,
    SELF_SIMILAR_C,
    SphericalDistribution,
    _cell,
    eddington_distribution_function,
    nfw_energy_fraction,
    nfw_log_energy_density,
    nfw_log_phase_space_fraction,
    profile_density,
    profile_potential,
    self_similar_lookup,
    self_similar_tables,
)

_2PI = 2 * np.pi
_MASS_GRID_NODES = 1024            # generic M(<r): trapezoid nodes, log-spaced in r
_MASS_GRID_INNER = 1e-8            # innermost node / outermost radius (plus r = 0)

_RANK_RTOL = 1e-9                  # hull dimension: singular values of the centre offsets above this fraction of the largest
_ZOOM_HALF_POINTS = (256, 16, 6)   # minimum search: lattice points per half-axis for hull dimension k = 1, 2, 3
_ZOOM_MARGIN = 2                   # bracket half-width kept around the best point, in lattice spacings
_ZOOM_RTOL = 1e-3                  # final bracket half-width relative to the hull's extent


def _zoom_plan(k, m):
    """Unit lattice and zoom schedule of the minimum search in a ``k``-dimensional hull.

    Parameters
    ----------
    k : int
        Hull dimension.
    m : int
        Lattice points per half-axis.

    Returns
    -------
    unit : ndarray of shape ((2m + 1)^k, k)
        Offsets ``j / m``, ``j = -m..m``, per axis; the zero offset is exact.
    shrink : float
        Bracket shrink factor per zoom, ``_ZOOM_MARGIN / m``.
    n_zoom : int
        Zooms after the first lattice: the fewest with ``shrink^(n_zoom + 1) <= 2 _ZOOM_RTOL``,
        a bracket of ``_ZOOM_RTOL`` of the hull extent.
    """
    t = np.arange(-m, m + 1) / m
    unit = np.stack(np.meshgrid(*([t] * k), indexing="ij"), axis=-1).reshape(-1, k)
    shrink = _ZOOM_MARGIN / m
    return unit, shrink, int(np.ceil(np.log(2 * _ZOOM_RTOL) / np.log(shrink))) - 1


_ZOOM_PLANS = tuple(_zoom_plan(k, m) for k, m in enumerate(_ZOOM_HALF_POINTS, start=1))


def _locate_minimum(potential, centres):
    """Absolute position of the minimum of a sum of spherical wells centred on ``centres``.

    Each well increases with the distance from its centre, so the minimum lies in the convex
    hull of the centres (outside it every term's gradient points away from the hull); it is
    searched in their affine hull, of dimension ``k`` the rank of the centre offsets
    (orthonormal basis by SVD, singular values above ``_RANK_RTOL`` of the largest).
    Coinciding centres (``k = 0``) are the minimum, returned without evaluating ``potential``.
    Otherwise the search is derivative-free, since a cusp (NFW, Hernquist) puts a kink at its
    centre, where the minimum often sits: one vectorised evaluation of a lattice over the
    hull's bounding box plus the centres themselves (every well's bottom candidate), then
    ``n_zoom`` evaluations of a lattice around the best point so far, each ``_ZOOM_MARGIN``
    spacings of the previous one wide, down to ``_ZOOM_RTOL`` of the hull extent. The zoom
    lattice holds the best point itself (zero offset), so the result is the best point
    evaluated: a centre at the minimum is returned exactly. numpy around the callable: the
    potential classes are Python, which numba cannot compile.

    Parameters
    ----------
    potential : callable
        Potential at absolute physical positions, ``(n, 3) -> (n,)``.
    centres : ndarray of shape (n_c, 3)
        Absolute physical centres of the wells.

    Returns
    -------
    x_min : ndarray of shape (3,)
    """
    d = centres - centres[0]
    if not d.any():
        return centres[0].copy()
    _, s, vt = np.linalg.svd(d, full_matrices=False)
    basis = vt[:np.count_nonzero(s > _RANK_RTOL * s[0])]
    unit, shrink, n_zoom = _ZOOM_PLANS[len(basis) - 1]
    u = d @ basis.T
    lo, hi = u.min(axis=0), u.max(axis=0)
    x = np.concatenate((centres, centres[0] + (0.5 * (lo + hi) + 0.5 * (hi - lo) * unit) @ basis))
    best = x[np.argmin(potential(x))]
    h = 0.5 * shrink * (hi - lo)
    for _ in range(n_zoom):
        x = best + (h * unit) @ basis
        best = x[np.argmin(potential(x))]
        h = shrink * h
    return best.copy()


class SphericalPotential(PotentialModel):
    """Spherical potential: the protocol's methods from a potential and a density.

    A concrete potential implements its profile in the physical radius,
    ``_potential(r)`` and ``_density(r)``; the public methods take absolute
    physical positions and measure them from :attr:`centre`. The
    enclosed mass and the central potential have generic forms (the
    trapezoid rule over the density; the potential at :attr:`x_min`, its centre),
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
    x_min : ndarray of shape (3,)
        Absolute physical position of the bottom of the well (the minimum of the
        potential): the centre of a spherical potential; a composite recomputes it as
        components are added.
    """

    has_distribution = True

    def __init__(self, M, G=G_KM, centre=None, velocity=None):
        self.M = M
        self.G = G
        self.centre = np.zeros(3) if centre is None else np.asarray(centre, dtype=np.float64)
        self.velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=np.float64)
        self._distributions = {}
        self.x_min = _locate_minimum(self.potential, self.centre[None])

    def __repr__(self):
        return f"{type(self).__name__}(M={self.M:.3g})"

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

    def central_potential(self, inner=None):
        """Central potential ``Φ₀``: the potential at the bottom of the well, :attr:`x_min`.

        The scale of the binding energies (``-Φ₀``); a single potential's energy tables start there.

        Parameters
        ----------
        inner : ndarray of shape (3,), optional
            Absolute physical position of the innermost particle; only a point mass, whose
            well the data resolve down to it, reads it.

        Returns
        -------
        phi0 : float
        """
        return float(self.potential(self.x_min[None])[0])

    def _potential_at(self, x, inner):
        """This component's term of a composite's central potential at the absolute position ``x``: its potential there.

        ``inner`` (the innermost particle) is read by a point mass only.
        """
        return float(self.potential(x[None])[0])

    def profile(self):
        """Compiled description for the q and g quadratures: rows ``(kind, k, a, d)`` summed by :func:`~roadrunner.physics.distribution.profile_phi`.

        ``None`` (the default) when the potential has none: the quadratures then evaluate
        :meth:`potential`. Every concrete potential here has rows (a
        :class:`ShellPotential` row is followed by its tabulated payload rows, skipped by the
        kernels). A composite potential stacks its components' rows; ``d`` is the row's offset
        from the profile's centre (0 for a plain potential).

        Returns
        -------
        profile : ndarray of shape (n, 4) or None
        """
        return None

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

    @property
    def _scale_radius(self):
        """The length a composite compares this component's offset with (`CONCENTRIC_OFFSET`)."""
        return self.a

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

    def central_potential(self, inner=None):
        """Exact: ``Φ₀ = -G M / a``.

        Returns
        -------
        phi0 : float
        """
        return -self.G * self.M / self.a

    def profile(self):
        """One Plummer row ``(PROFILE_PLUMMER, G M, a, 0)``."""
        return np.array([[PROFILE_PLUMMER, self.G * self.M, self.a, 0.0]])

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

    def central_potential(self, inner=None):
        """Central potential ``-G M / r_in`` of the point mass at its innermost particle.

        The softened central value is set by the softening alone, so the well is taken as deep
        as the data resolve it: the unsoftened point mass at the innermost particle's distance
        ``r_in``. Without one (``inner`` None) it is ``-inf``.

        Parameters
        ----------
        inner : ndarray of shape (3,), optional
            Absolute physical position of the innermost particle.

        Returns
        -------
        phi0 : float
        """
        return self._potential_at(self.centre, inner)

    def _potential_at(self, x, inner):
        """The unsoftened point mass at ``x``, no deeper than at the innermost particle: ``-G M / max(|x - c|, r_in)``.

        ``r_in = 0`` without an innermost particle, so ``-inf`` at the centre.
        """
        r_in = np.linalg.norm((self.centre if inner is None else np.asarray(inner, dtype=np.float64)) - self.centre)
        with np.errstate(divide="ignore"):
            return float(-self.G * self.M / max(np.linalg.norm(x - self.centre), r_in))

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

    def central_potential(self, inner=None):
        """Exact: ``Φ₀ = -G M / a``.

        Returns
        -------
        phi0 : float
        """
        return -self.G * self.M / self.a

    def profile(self):
        """One Hernquist row ``(PROFILE_HERNQUIST, G M, a, 0)``."""
        return np.array([[PROFILE_HERNQUIST, self.G * self.M, self.a, 0.0]])

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

    @property
    def _scale_radius(self):
        """The length a composite compares this component's offset with (`CONCENTRIC_OFFSET`)."""
        return self.Rs

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

    def central_potential(self, inner=None):
        """Exact: ``Φ₀ = -G M / (Rs A(c))``, the potential at ``r = 0``.

        Returns
        -------
        phi0 : float
        """
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return -self.G * self.M / (self.Rs * A)

    def profile(self):
        """One NFW row ``(PROFILE_NFW, G M / A(c), Rs, 0)``."""
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return np.array([[PROFILE_NFW, self.G * self.M / A, self.Rs, 0.0]])

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
    each cell; the density follows from its windowed log slope. Its table is
    carried as a tabulated profile row (:data:`~roadrunner.physics.distribution.PROFILE_SHELL`,
    :meth:`profile`), so its potential and the q and g quadratures of its
    phase-space fraction are compiled, usable inside a composite; it has no
    energy distribution: :meth:`distribution_function` and the energy pair raise.

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
        # profile row: header (PROFILE_SHELL, n_payload, 0, 0), then (ln r0, 1/d) and the K + 2 cells' (slope, intercept)
        self._rows = np.zeros((n_nodes + 4, 4))
        self._rows[0, :2] = PROFILE_SHELL, n_nodes + 3
        self._rows[1, :2] = self._c, self._inv_d
        self._coef = coef = self._rows[2:, :2]   # per-cell (slope, intercept) of Φ in 1/r: a view into self._rows
        coef[0] = 0.0, phi[0]
        coef[1:-1, 0] = np.diff(phi) / np.diff(u)
        coef[1:-1, 1] = phi[:-1] - coef[1:-1, 0] * u[:-1]
        coef[-1] = -G * m_in[-1], 0.0
        self._rows.flags.writeable = False         # profile() hands it out; _coef is a view into it
        self._w = max(1, round(tidal_dlnr * self._inv_d))
        self._r_half = float(1.0 / u[np.searchsorted(m_in, 0.5 * m_in[-1])])   # first node holding half the mass
        super().__init__(float(m_in[-1]), G=G, centre=centre, velocity=velocity)

    @property
    def _scale_radius(self):
        """The length a composite compares this component's offset with (`CONCENTRIC_OFFSET`): its half-mass radius (to a node)."""
        return self._r_half

    def _potential(self, r):
        """Evaluate the potential: :func:`~roadrunner.physics.distribution.profile_potential` of its own tabulated row.

        Parameters
        ----------
        r : ndarray
            Physical radii.

        Returns
        -------
        phi : ndarray
        """
        r = np.asarray(r)
        return profile_potential(self._rows, r.ravel()).reshape(r.shape)

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

    def central_potential(self, inner=None):
        """Central potential ``Φ₀ = -G Σ m_j / r_j``.

        Returns
        -------
        phi0 : float
        """
        return float(self._coef[0, 1])

    def profile(self):
        """One ``PROFILE_SHELL`` row (``d = 0``) and its tabulated payload: :attr:`_rows`."""
        return self._rows

    def distribution_function(self, E):
        """Not defined: the density is a windowed log-slope estimate, too noisy for ``d²ρ/dψ²``.

        The energy pair therefore raises; the phase-space fraction, which
        needs only the potential, is the generic one, with its q and g
        quadratures compiled through :meth:`profile`'s tabulated row.
        """
        raise NotImplementedError("ShellPotential has no energy distribution yet.")


def _shell_mean(f, r, d):
    """Mean of a radial profile ``f`` over the sphere of radius ``r`` about a point ``d > 0`` from its centre.

    ``½ ∫ f(sqrt(r² + d² - 2 r d μ)) dμ = (1 / (2 r d)) ∫_{|r-d|}^{r+d} s f(s) ds``; with
    ``s = max(r, d) + min(r, d) x`` it is ``Σ_k w_k s_k f(s_k) / (2 max(r, d))`` (Gauss–Legendre on
    [-1, 1]). ``s f(s)`` is finite at a ``1/s`` cusp (``s -> 0`` when ``r = d``); at ``r = 0`` it is
    ``f(d)``; a point mass's potential gives Newton's ``-G M / max(r, d)``; it tends to ``f(r)`` as
    ``d -> 0``.

    Parameters
    ----------
    f : callable
        Radial profile (``_potential`` or ``_density`` of a component).
    r : ndarray
    d : float

    Returns
    -------
    mean : ndarray of float64, shape of ``r``
    """
    r = np.asarray(r, dtype=np.float64)
    mid = np.maximum(r, d)
    s = mid[..., None] + np.minimum(r, d)[..., None] * GL_NODES
    return (s * f(s)) @ GL_WEIGHTS / (2.0 * mid)


def _shell_mass(p, r, d):
    """Mass of ``p`` inside the sphere of radius ``r`` about a point ``d > 0`` from its centre.

    The monopole's enclosed mass, so ``G M = r² dΦ̄/dr`` with ``Φ̄ = _shell_mean(p._potential, r, d)``.
    Shells ``s <= r - d`` lie inside (``p``'s own ``M(<max(r - d, 0))``); one with ``|r - d| < s < r + d``
    has the fraction ``(r + d - s)(r - d + s) / (4 s d)`` inside, so the rest is
    ``(π / d) ∫ s ρ(s) (r + d - s)(r - d + s) ds`` on the nodes of :func:`_shell_mean`, the product
    written without cancellation as ``half (1 - x) (r + max(r - d, 0) + half x)``,
    ``half = min(r, d)``. Both terms are positive; it tends to ``M(<r)`` as ``d -> 0`` and is 0 at ``r = 0``.

    Parameters
    ----------
    p : SphericalPotential
    r : ndarray
    d : float

    Returns
    -------
    menc : ndarray of float64, shape of ``r``
    """
    r = np.asarray(r, dtype=np.float64)
    half = np.minimum(r, d)[..., None]
    inner = np.maximum(r - d, 0.0)
    s = np.maximum(r, d)[..., None] + half * GL_NODES
    chord = half * (1.0 - GL_NODES) * (r[..., None] + inner[..., None] + half * GL_NODES)
    return p._enclosed_mass(inner) + np.pi / d * half[..., 0] * ((s * p._density(s) * chord) @ GL_WEIGHTS)


# a component's radial method about a reference centre d > 0 from its own: its shell average (monopole pathway)
_MONOPOLE = {"_potential": lambda p, r, d: _shell_mean(p._potential, r, d),
             "_density": lambda p, r, d: _shell_mean(p._density, r, d),
             "_enclosed_mass": _shell_mass}


def _about(p, method, r, d):
    """Component ``p``'s radial ``method`` at radii ``r`` about a reference centre ``d`` from its own.

    Its own profile when ``d = 0`` (concentric pathway, today's exact values), its shell average
    otherwise (monopole pathway): the master kernel of the composite's radial profile, like the
    per-row dispatch of :func:`~roadrunner.physics.distribution.profile_phi`.
    """
    return getattr(p, method)(r) if d == 0.0 else _MONOPOLE[method](p, r, d)


# Methods without `density` a single component answers itself, exactly (its closed forms and tables);
# the energy methods go through the density views (`_densities`: the groups, and tracers on first use).
_OWN_METHODS = ("orbital_time", "log_phase_space_fraction", "central_potential")


class _GroupDensity:
    """Some groups of a composite: their density in the composite's whole potential (the two-model route).

    Its radial density is the sum of its components' along the composite's pathways, about
    ``x_min``; :meth:`distribution_function` is Eddington's inversion of it in the composite's
    potential, and the energy pair counts ``N = f g`` (``g`` of the composite) inside ``r_max``,
    normalised to these groups' mass there (a :class:`SphericalDistribution` per ``r_max``, cached).
    Holding every component, it is the composite's own, self-consistent distribution.

    Parameters
    ----------
    composite : CompositeSphericalPotential
    names : tuple of str
        The groups it holds.
    index : tuple of int
        Their components' positions in the composite.
    """

    def __init__(self, composite, names, index):
        self._composite, self._names, self._index = composite, names, index
        self.has_distribution = all(composite[i].has_distribution for i in index)
        # holding every component, its tables are the composite's own (the ones the phase-space fraction reads)
        self._distributions = composite._distributions if len(index) == len(composite) else {}

    def __repr__(self):
        return f"group {'+'.join(self._names)} of {self._composite!r}"

    def _density(self, r):
        """Radial density of its components about the composite's ``x_min``, each along its pathway (compiled, U1).

        Reached only when :attr:`has_distribution` holds: every component then has a ``profile()``
        (Plummer, Hernquist, NFW), so :func:`~roadrunner.physics.distribution.profile_density`
        covers every row.
        """
        comp = self._composite
        rows = np.column_stack((np.vstack([comp[i].profile() for i in self._index])[:, :3],
                                comp._offsets[list(self._index)]))
        return profile_density(rows, np.ascontiguousarray(r, dtype=np.float64)) / comp.G

    def distribution_function(self, E):
        """Eddington's ``f`` of its density in the composite's potential (negative values clipped to 0, with a warning).

        Raises
        ------
        NotImplementedError
            If one of its components has no distribution function (Kepler, Shell).
        """
        if not self.has_distribution:
            raise NotImplementedError(f"A component of the {self!r} has no energy distribution.")
        return eddington_distribution_function(self._composite, E, density=self)

    def _distribution(self, r_max):
        """The cached :class:`SphericalDistribution` of its ``N = f g`` inside ``r_max``."""
        d = self._distributions.get(r_max)
        if d is None:
            d = self._distributions[r_max] = SphericalDistribution(self._composite, r_max, self.distribution_function)
        return d

    def energy_fraction(self, E, r_max):
        """Fraction of its mass inside ``r_max`` more bound than ``E``."""
        return self._distribution(r_max).energy_fraction(E)

    def log_energy_density(self, E, r_max):
        """``ln dN/dE`` of its mass inside ``r_max``, per unit physical ``E``."""
        return self._distribution(r_max).log_energy_density(E)


class _TracerDensity(_GroupDensity):
    """A density model in a composite's whole potential: a tracer, no part of the potential.

    Its radial density is the model's own formula read about the composite's ``x_min`` (as a
    concentric component's); Eddington's inversion of it in the composite's potential and its
    ``N = f g`` inside ``r_max`` are :class:`_GroupDensity`'s, normalised to the tracer's mass there.

    Parameters
    ----------
    composite : CompositeSphericalPotential
    model : SphericalPotential
        Provides ``_density(r)`` (physical radii); e.g. a :class:`HernquistPotential`.
    """

    has_distribution = True

    def __init__(self, composite, model):
        self._composite, self._model, self._distributions = composite, model, {}
        self._density = model._density              # read at once: fails here for a non-model (e.g. an unknown group name)

    def __repr__(self):
        return f"tracer {self._model!r} in {self._composite!r}"


class _DensityViews(dict):
    """A composite's density views: by group name (``None``: the whole composite) and, built on first use, by density model (:class:`_TracerDensity`)."""

    def __init__(self, composite, views):
        super().__init__(views)
        self._composite = composite

    def __missing__(self, model):
        view = self[model] = _TracerDensity(self._composite, model)
        return view


class CompositeSphericalPotential(SphericalPotential):
    """Sum of spherical potentials: the total potential of a halo built from several components.

    An iterable container of its components (``for p in composite``, ``len``, indexing);
    :meth:`add` adds more on the fly, to an existing group or a new one. Each component
    carries its own ``centre`` and ``velocity``; the composite's are :attr:`com_centre` and
    :attr:`com_velocity`, the components' mass-weighted ones ``Σ M_i c_i / Σ M_i`` and
    ``Σ M_i v_i / Σ M_i``, which its ``centre`` and ``velocity`` (the potential protocol's)
    return. The primitives and the quantities linear in mass are the sums of the
    components' own methods (potential, density, enclosed mass, well
    depth, tidal denominator), so every component keeps its exact forms; the profile stacks
    their rows. The public methods take absolute physical positions and measure each
    component from its own ``centre``; the private radial profile (``_potential``, ``_density``,
    ``_enclosed_mass``, :meth:`profile`'s rows) is measured from :attr:`x_min`, each component
    along one pathway fixed in :meth:`add`: monopole (its shell average about ``x_min``,
    :func:`_shell_mean`/:func:`_shell_mass`) when it lies farther from ``x_min`` than
    :data:`~roadrunner._defaults.CONCENTRIC_OFFSET` of its own scale radius and its pull there,
    ``G M(<d) / d²``, is below :data:`~roadrunner._defaults.MONOPOLE_MAX_PULL` of the bottom's
    slope; concentric (its own profile, as if centred on ``x_min``) otherwise: only a component
    far from the bottom for its size and pulling weakly there is shell-averaged, one whose pull
    shapes the bottom keeps the tilt the monopole would drop. Its value at ``r = 0``, the energy
    tables' zero point, is :meth:`central_potential` when every concentric component sits at
    ``x_min``; one off it adds its own ``Φ₀`` there, deeper by its ``Φ(d) - Φ(0)``.
    The central potential is the summed potential at the
    bottom of the well, :attr:`x_min`, located among the components' centres (each component's
    own ``Φ₀`` when they coincide).

    The orbital time combines the components' own as ``t^-2 = Σ t_i^-2``, exact for
    dynamical times (``t_i^-2 ∝ M_i(<r)``). :meth:`distribution_function`,
    :meth:`energy_fraction` and :meth:`log_energy_density` take a ``density`` (a group's name):
    that group's density in the whole potential (the two-model Eddington inversion,
    :class:`_GroupDensity`), or a density model: that tracer's density about :attr:`x_min` in the
    whole potential, no part of it (:class:`_TracerDensity`, built on first use and kept until
    :meth:`add`); the whole composite's when
    ``None`` or when the group holds every component (then its own, self-consistent
    distribution: a single component's exact closed forms and tables, several components the
    generic :class:`~roadrunner.physics.distribution.SphericalDistribution` of the summed
    density in the summed potential). :meth:`log_phase_space_fraction` is always the whole
    composite's (it needs only the potential).

    A single component answers the methods without a ``density`` (``_OWN_METHODS``) itself,
    chosen when components are added: its own orbital time and phase-space fraction, exactly.

    :attr:`groups` maps each group's name to its components, in the order they joined; a
    group accessed as an attribute (``comp.stars``) is a composite of its own, built fresh
    from that group's components on each access (its own ``x_min`` and pathways, unlike
    ``density="stars"``, which reads this composite's).

    Parameters
    ----------
    **groups : SphericalPotential or sequence of SphericalPotential
        At least one component, all with the same ``G`` (the generic route uses one ``G``).

    Raises
    ------
    ValueError
        If the components' ``G`` differ (also on :meth:`add`).
    """

    def __init__(self, **groups):
        groups = {name: (m,) if isinstance(m, SphericalPotential) else tuple(m) for name, m in groups.items()}
        self.G = next(iter(groups.values()))[0].G
        self._components, self.groups, self._index = (), {}, {}
        for name, members in groups.items():
            self.add(name, *members)

    def add(self, group, *potentials):
        """Add components on the fly to a group, existing or new (e.g. a halo's stellar counterpart).

        The composite's mass, :attr:`com_centre`, :attr:`com_velocity`, ``has_distribution``,
        single-component shortcuts, :attr:`x_min`, the pathways (:attr:`_offsets`) and the
        group densities (:attr:`_densities`) are updated, and its cached distributions are
        dropped, to be rebuilt on next use.

        A component takes the monopole pathway only when it is farther from ``x_min`` than
        ``CONCENTRIC_OFFSET`` of its scale radius and its pull there is below ``MONOPOLE_MAX_PULL``
        of the bottom's slope ``K`` (one-sided climbs over ``±s`` on the three axes, averaged, ``s =
        CONCENTRIC_OFFSET`` of the scale radius of the component nearest ``x_min`` for its size); every
        other component is concentric.

        Parameters
        ----------
        group : str
            The group they join: an existing one or a new one.
        *potentials : SphericalPotential
            With the composite's ``G``.

        Raises
        ------
        ValueError
            If one's ``G`` differs from the composite's (the generic route uses one ``G``);
            none is then added.
        """
        if any(p.G != self.G for p in potentials):
            raise ValueError("The components of a composite potential must share G.")
        n = len(self._components)
        self._components += potentials
        self.groups[group] = self.groups.get(group, ()) + potentials
        self._index[group] = self._index.get(group, ()) + tuple(range(n, len(self._components)))
        every = tuple(range(len(self._components)))
        self.M = sum(p.M for p in self._components)
        self.com_centre = sum((p.M / self.M) * p.centre for p in self._components)
        self.com_velocity = sum((p.M / self.M) * p.velocity for p in self._components)
        self.has_distribution = all(p.has_distribution for p in self._components)
        self._distributions = {}
        whole = _GroupDensity(self, tuple(self.groups), every)
        for name in _OWN_METHODS:
            self.__dict__.pop(name, None)
        if len(self._components) == 1:
            for name in _OWN_METHODS:
                setattr(self, name, getattr(self._components[0], name))
            whole = self._components[0]
        centres = np.array([p.centre for p in self._components])
        self.x_min = _locate_minimum(self.potential, centres)
        # one pathway per component, read by every radial kernel: concentric (offset 0) or monopole about x_min.
        # Monopole only far from x_min for its size and pulling weakly there, G M(<d) / d², against the bottom's
        # slope K: the mean climb (Φ(x_min ± s ê) - Φ(x_min)) / s over the three axes (the gradient cancels in it),
        # over s = CONCENTRIC_OFFSET of the scale radius of the component nearest x_min for its size (the bottom's):
        # a cusp's slope, or a core's finite climb over that "near" region (a fixed tiny step would vanish there)
        scale = np.array([p._scale_radius for p in self._components])
        offset = np.sqrt(row_squared_norms(centres - self.x_min))
        s = CONCENTRIC_OFFSET * scale[np.argmin(offset / scale)]
        phi = self.potential(self.x_min + s * np.vstack((np.zeros(3), np.eye(3), -np.eye(3))))
        slope = np.mean(phi[1:] - phi[0]) / s
        with np.errstate(divide="ignore", invalid="ignore"):     # a component at x_min: 0/0 -> inf, concentric
            pull = np.nan_to_num(self.G * np.concatenate([p._enclosed_mass(d) for p, d in zip(self._components, offset[:, None])])
                                 / offset**2, nan=np.inf)
        self._offsets = offset * ((offset > CONCENTRIC_OFFSET * scale) & (pull < MONOPOLE_MAX_PULL * slope))
        self._densities = _DensityViews(self, {None: whole} | {name: whole if len(index) == len(every) else _GroupDensity(self, (name,), index)
                                                                for name, index in self._index.items()})

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{name}={list(members)!r}' for name, members in self.groups.items())})"

    def __iter__(self):
        return iter(self._components)

    def __getattr__(self, name):
        """A group as a new composite of its own (``comp.stars``), built on each access.

        Reached only when ordinary lookup fails: a group named like an attribute (``M``,
        ``groups``, ...) is reachable through :attr:`groups` alone.
        """
        try:
            return CompositeSphericalPotential(**{name: self.__dict__["groups"][name]})
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute or group {name!r}") from None

    @property
    def centre(self):
        """The protocol's ``centre``: :attr:`com_centre`."""
        return self.com_centre

    @property
    def velocity(self):
        """The protocol's ``velocity``: :attr:`com_velocity`."""
        return self.com_velocity

    def __len__(self):
        return len(self._components)

    def __getitem__(self, i):
        return self._components[i]

    def _sum(self, method, *args):
        """``Σ method(*args)`` over the components, starting from the first (one component's value as is)."""
        first, *rest = self._components
        return sum((getattr(p, method)(*args) for p in rest), getattr(first, method)(*args))

    def _pathway_sum(self, method, r, index):
        """``Σ`` over the components at ``index`` of their radial ``method`` about :attr:`x_min`, each along its pathway (:func:`_about`); one component's value as is."""
        first, *rest = index
        return sum((_about(self._components[i], method, r, self._offsets[i]) for i in rest),
                   _about(self._components[first], method, r, self._offsets[first]))

    def _potential(self, r):
        """Summed potential at radii about :attr:`x_min`, each component along its pathway: the spherical profile the distributions use."""
        return self._pathway_sum("_potential", r, range(len(self._components)))

    def _density(self, r):
        """Summed density at radii about :attr:`x_min`, each component along its pathway: the spherical profile the distributions use."""
        return self._pathway_sum("_density", r, range(len(self._components)))

    def _enclosed_mass(self, r):
        """Summed enclosed mass at radii about :attr:`x_min`, each component along its pathway: the spherical profile the distributions use."""
        return self._pathway_sum("_enclosed_mass", r, range(len(self._components)))

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

    def central_potential(self, inner=None):
        """Central potential: the summed potential at the bottom of the well, :attr:`x_min`.

        Each component's term there (:meth:`_potential_at`): its potential, or for a point mass
        the unsoftened ``-G M / max(|x_min - c|, r_in)`` (``-inf`` at its centre without an
        innermost particle). A single component answers itself (its exact ``Φ₀``).

        Parameters
        ----------
        inner : ndarray of shape (3,), optional
            Absolute physical position of the innermost particle (read by point masses).

        Returns
        -------
        phi0 : float
        """
        return sum(p._potential_at(self.x_min, inner) for p in self._components)

    def profile(self):
        """The components' profile rows stacked, with their pathway offsets; ``None`` if any component has none.

        A :class:`ShellPotential` row brings its table rows (payload, skipped by the kernels);
        every one of a component's rows carries its pathway offset, so the composite's row count
        can exceed its component count.

        Returns
        -------
        profile : ndarray of shape (n, 4) or None
            Rows ``(kind, k, a, d)``; ``d`` is the component's pathway offset from
            :attr:`x_min` (0: concentric, its own profile; otherwise its shell average).
        """
        rows = [p.profile() for p in self._components]
        if any(row is None for row in rows):
            return None
        return np.column_stack((np.vstack(rows)[:, :3], np.repeat(self._offsets, [len(row) for row in rows])))

    def release(self):
        """Drop every cached :class:`~roadrunner.physics.distribution.SphericalDistribution` (U6).

        Clears :attr:`_distributions` on the composite and on every density view (groups and tracers)
        (:attr:`_densities`): memory held per ``r_max`` since the last table was read, rebuilt on next use. A single
        component answers for its own group (:attr:`_densities`'s entry is itself, e.g. a bare
        ``NFWPotential``); clearing a potential with no generic tables (the per-process NFW route)
        is a no-op.
        """
        for holder in (self, *self._densities.values()):
            holder._distributions.clear()

    def distribution_function(self, E, density=None):
        """Eddington's ``f`` of a density in the whole potential.

        Parameters
        ----------
        E : ndarray
        density : str or density model, optional
            A group's name: that group's density in the whole potential (Eddington's
            inversion of its radial density). The whole composite's when ``None`` or when
            the group holds every component (a single component's own closed forms and
            tables; several, the generic tables of the summed density in the summed
            potential). Or a density model (``_density(r)``, e.g. a :class:`HernquistPotential`):
            that tracer about :attr:`x_min` in the whole potential.

        Returns
        -------
        f : ndarray

        Raises
        ------
        NotImplementedError
            If a component of that group has no distribution function (Kepler, Shell).
        AttributeError
            If ``density`` is neither a group's name nor a density model.
        """
        return self._densities[density].distribution_function(E)

    def energy_fraction(self, E, r_max, density=None):
        """Fraction of a density's mass inside ``r_max`` more bound than ``E``.

        Parameters
        ----------
        E : ndarray
        r_max : float
        density : str or density model, optional
            A group's name: that group's density in the whole potential, normalised to its
            own mass inside ``r_max``. The whole composite's when ``None`` or when the group
            holds every component. Or a density model (``_density(r)``, e.g. a
            :class:`HernquistPotential`): that tracer about :attr:`x_min` in the whole potential,
            normalised to its own mass inside ``r_max``.

        Returns
        -------
        u : ndarray of float64

        Raises
        ------
        NotImplementedError
            If a component of that group has no distribution function (Kepler, Shell).
        AttributeError
            If ``density`` is neither a group's name nor a density model.
        """
        return self._densities[density].energy_fraction(E, r_max)

    def log_energy_density(self, E, r_max, density=None):
        """``ln dN/dE`` of a density's mass inside ``r_max``, per unit physical ``E``.

        Parameters
        ----------
        E : ndarray
        r_max : float
        density : str or density model, optional
            A group's name: that group's density in the whole potential, normalised to its
            own mass inside ``r_max``. The whole composite's when ``None`` or when the group
            holds every component. Or a density model (``_density(r)``, e.g. a
            :class:`HernquistPotential`): that tracer about :attr:`x_min` in the whole potential,
            normalised to its own mass inside ``r_max``.

        Returns
        -------
        log_n : ndarray of float64

        Raises
        ------
        NotImplementedError
            If a component of that group has no distribution function (Kepler, Shell).
        AttributeError
            If ``density`` is neither a group's name nor a density model.
        """
        return self._densities[density].log_energy_density(E, r_max)


POTENTIAL_MODELS: dict[str, type[SphericalPotential]] = {
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
    cls = POTENTIAL_MODELS.get(model.lower())
    if cls is None:
        raise ValueError(
            f"Unknown potential model: {model}. "
            f"Available: {list(POTENTIAL_MODELS)}"
        )
    if not kwargs:
        return cls
    return cls(**kwargs)

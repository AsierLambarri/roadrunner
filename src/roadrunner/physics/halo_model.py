"""Single-halo model with potential and boundness storage."""

import numpy as np

from roadrunner.mixture._math import row_squared_norms
from roadrunner.physics.constants import G_KM
from roadrunner.physics.potentials import CompositeSphericalPotential, get_potential
from roadrunner._defaults import LOCAL_IDX, math_dtype
from roadrunner._mcf_types import PotentialModel


def physical_length_factor(redshift, comoving):
    """Comoving-to-physical length conversion factor.

    Parameters
    ----------
    redshift : float
    comoving : bool
        If ``True``, lengths are comoving and need converting; if
        ``False``, they're already physical and the factor is a no-op.

    Returns
    -------
    factor : float
    """
    return 1.0 / (1.0 + redshift) if comoving else 1.0


class HaloModel:
    """Merger-tree halo: its record, its potential and its boundness data, in the catalogue's units.

    The halo works in its catalogue's units (lengths comoving when
    ``comoving``, otherwise physical; velocities peculiar, in km/s) and is the
    one place that translates them for the physics. Its potential,
    :attr:`potential_model`, is a
    :class:`~roadrunner.physics.potentials.CompositeSphericalPotential` in
    physical units, each component centred on its own absolute physical
    ``centre`` and moving with its own ``velocity`` (the tree's at
    ``tree_centre * length_scale`` and ``tree_velocity``). The halo exposes the
    potential model's methods as one-line wrappers at catalogue coordinates:
    positions ``xyz`` are made physical (``length_scale``, float64), energies
    ``E`` are physical and pass as they are, and the energy distributions are
    counted inside the halo's search sphere (``search_factor`` virial radii) —
    of the whole halo or, with ``density``, of one group or of a tracer density model in the whole potential.
    Particle-level results are returned in ``math_dtype()``, the energy
    distributions in float64.

    It adds its record (``tree_centre``, ``tree_velocity``, ``tree_mass``:
    the catalogue's centre, bulk velocity and mass, the search sphere being
    centred on ``tree_centre``; ``virial_radius``, ``redshift``,
    ``comoving``, ``length_scale``, ``search_factor``, ``sub_tree_id``;
    ``inner_position``, the innermost particle of the search sphere, set by
    :func:`compute_halo_bound_particles`, the centre until then), the
    composite's ``M``, ``com_centre`` and ``com_velocity`` (:attr:`M`, :attr:`com_centre` in
    catalogue coordinates, :attr:`com_velocity`), the energy of its
    particles (:meth:`compute_energy`, in the centre-of-mass frame), the scale of its binding
    energies (:meth:`binding_energy_scale`) and the bound storage
    ``(indices, energies, tdyns)`` in physical units, the boundness
    ``b = -E / binding_energy_scale()`` on request (:meth:`get_bound`).

    Parameters
    ----------
    potential : PotentialModel
        A plain potential (the tree halo, put in group ``dark_matter``) or a
        composite with a ``dark_matter`` group, copied so the halo's is
        frozen; the tree halo is centred on ``tree_centre * length_scale`` and
        moves with ``tree_velocity``.
    tree_centre : ndarray of shape (3,)
        Halo centre (catalogue coordinates).
    tree_velocity : ndarray of shape (3,)
        Halo bulk velocity (km/s).
    virial_radius : float
        Virial radius (catalogue units).
    sub_tree_id : int
        Halo identifier from the merger tree.
    redshift : float
        Snapshot redshift.
    comoving : bool, default=True
        If ``True``, catalogue lengths are comoving and ``length_scale`` is
        ``1 / (1 + z)``.
    search_factor : float, default=1.0
        Multiple of the virial radius: the boundness search sphere and the
        counting radius of the energy distributions.
    """

    def __init__(
        self,
        potential: PotentialModel,
        tree_centre: np.ndarray,
        tree_velocity: np.ndarray,
        virial_radius: float,
        sub_tree_id: int,
        redshift: float,
        comoving: bool = True,
        search_factor: float = 1.0,
    ):
        md = math_dtype()
        self.tree_centre = np.asarray(tree_centre, dtype=md)
        self.tree_velocity = np.asarray(tree_velocity, dtype=md)
        self.virial_radius = float(virial_radius)
        self.sub_tree_id = sub_tree_id
        self.redshift = float(redshift)
        self.comoving = comoving
        self.search_factor = float(search_factor)
        self.length_scale = physical_length_factor(self.redshift, comoving)
        # the halo's own frozen copy; a plain potential is the tree halo, group dark_matter
        self.potential_model = CompositeSphericalPotential(
            **(potential.groups if isinstance(potential, CompositeSphericalPotential) else {"dark_matter": potential}))
        self.tree_mass = float(sum(p.M for p in self.potential_model.groups["dark_matter"]))
        self.inner_position = self.tree_centre.copy()
        self._bound: tuple | None = None

    def set_bound(
        self, indices: np.ndarray, energies: np.ndarray, tdyns: np.ndarray
    ) -> None:
        """Store the result of a boundness computation.

        Parameters
        ----------
        indices : ndarray of LOCAL_IDX
            Bound particle indices (array positions). Cast on entry so
            all halos share one dtype (mixed uint64/int64 inputs would
            otherwise promote concatenations to float64).
        energies : ndarray
            Physical specific orbital energies ``E`` (negative; ambient math precision).
        tdyns : ndarray
            Dynamical times of the bound particles (physical; ambient math precision).
        """
        self._bound = (np.asarray(indices, dtype=LOCAL_IDX), energies, tdyns)

    def get_bound(self, scaled_energies: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The stored bound particles: ``(indices, energies, tdyns)``.

        Parameters
        ----------
        scaled_energies : bool, default=False
            If ``True`` the energies are the boundness ``b = -E / binding_energy_scale()``
            (dimensionless, in ``(0, 1]`` for a finite well; float64 maths, stored dtype out);
            ``tdyns`` stay physical either way.

        Returns
        -------
        bound : tuple of (indices, energies, tdyns) or None
            ``None`` if boundness has not been computed yet.
        """
        if self._bound is None or not scaled_energies:
            return self._bound
        idx, E, tdyn = self._bound
        return idx, (np.asarray(E, dtype=np.float64) / -self.binding_energy_scale()).astype(E.dtype, copy=False), tdyn

    @property
    def has_bound(self) -> bool:
        """``True`` if boundness data has been stored."""
        return self._bound is not None

    def __iter__(self):
        return iter(self.potential_model)

    @property
    def M(self) -> float:
        """The potential model's ``M``: the mass of all the components, ``Σ M_i``."""
        return float(self.potential_model.M)

    @property
    def com_centre(self) -> np.ndarray:
        """The potential model's ``com_centre`` (mass-weighted) in catalogue coordinates (float64)."""
        return self.potential_model.com_centre / self.length_scale

    @property
    def com_velocity(self) -> np.ndarray:
        """The potential model's ``com_velocity`` (mass-weighted; km/s, float64): the frame of :meth:`compute_energy`."""
        return self.potential_model.com_velocity

    def __len__(self):
        return len(self.potential_model)

    def __getitem__(self, i):
        return self.potential_model[i]

    def _physical(self, xyz: np.ndarray) -> np.ndarray:
        """Catalogue positions as absolute physical positions, in float64."""
        return np.asarray(xyz, dtype=np.float64) * self.length_scale

    def _counting_radius(self) -> float:
        """Physical radius of the boundness search sphere, ``search_factor * r_vir``: where the distributions count."""
        return self.search_factor * self.virial_radius * self.length_scale

    # ── the potential model at catalogue coordinates ─────────────────

    def potential(self, xyz: np.ndarray) -> np.ndarray:
        """Gravitational potential at catalogue positions (each component from its own centre).

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)
            Absolute positions (catalogue coordinates).

        Returns
        -------
        phi : ndarray of math_dtype, shape (n,)
        """
        return self.potential_model.potential(self._physical(xyz)).astype(math_dtype(), copy=False)

    def density(self, xyz: np.ndarray) -> np.ndarray:
        """Density at catalogue positions (each component from its own centre).

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        rho : ndarray of math_dtype, shape (n,)
        """
        return self.potential_model.density(self._physical(xyz)).astype(math_dtype(), copy=False)

    def enclosed_mass(self, xyz: np.ndarray) -> np.ndarray:
        """Each component's mass inside the sphere about its own centre through each catalogue position, summed.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        menc : ndarray of math_dtype, shape (n,)
        """
        return self.potential_model.enclosed_mass(self._physical(xyz)).astype(math_dtype(), copy=False)

    def tidal_denominator(self, xyz: np.ndarray) -> np.ndarray:
        """Tidal denominator at catalogue positions (each component from its own centre).

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        denom : ndarray of math_dtype, shape (n,)
        """
        return self.potential_model.tidal_denominator(self._physical(xyz)).astype(math_dtype(), copy=False)

    def orbital_time(self, E: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Orbital timescale of physical energies ``E`` at catalogue positions.

        A single component gives its own (the Kepler period for a point mass,
        the dynamical time otherwise); several combine theirs as
        ``t^-2 = Σ t_i^-2``, each from its own centre.

        Parameters
        ----------
        E : ndarray
            Specific orbital energies (physical).
        xyz : ndarray of shape (n, 3)

        Returns
        -------
        t : ndarray of math_dtype, shape (n,)
        """
        t = self.potential_model.orbital_time(np.asarray(E, dtype=np.float64), self._physical(xyz))
        return t.astype(math_dtype(), copy=False)

    def central_potential(self) -> float:
        """Central potential ``Φ₀`` of the potential model (physical units), with its innermost particle.

        The bottom of the well (:attr:`inner_position` read by a point mass, whose well the
        data resolve down to it; ``-inf`` for one before :func:`compute_halo_bound_particles`
        sets it away from the centre).

        Returns
        -------
        phi0 : float
        """
        return self.potential_model.central_potential(self._physical(self.inner_position))

    def binding_energy_scale(self) -> float:
        """Scale of the halo's binding energies: the depth ``-Φ₀`` of its well (:meth:`central_potential`).

        The boundness ``b = -E / binding_energy_scale()`` is given on request by :meth:`get_bound`.

        Returns
        -------
        scale : float
        """
        return -self.central_potential()

    def distribution_function(self, E: np.ndarray, density: str | PotentialModel | None = None) -> np.ndarray:
        """Isotropic distribution function ``f(E)`` of the potential model at physical energies.

        Parameters
        ----------
        E : ndarray
        density : str or density model, optional
            A group of the potential model (e.g. ``"dark_matter"``): that group in the whole
            potential. The whole halo when ``None``. Or a density model (e.g. a
            :class:`~roadrunner.physics.potentials.HernquistPotential`): that tracer in the whole
            potential, about the bottom of the well.

        Returns
        -------
        f : ndarray of float64
        """
        return self.potential_model.distribution_function(np.asarray(E, dtype=np.float64), density=density)

    def energy_fraction(self, E: np.ndarray, density: str | PotentialModel | None = None) -> np.ndarray:
        """Fraction of the halo's mass inside the search sphere more bound than the physical energy ``E``.

        Parameters
        ----------
        E : ndarray
        density : str or density model, optional
            A group of the potential model (e.g. ``"dark_matter"``): that group in the whole
            potential. The whole halo when ``None``. Or a density model (e.g. a
            :class:`~roadrunner.physics.potentials.HernquistPotential`): that tracer in the whole
            potential, about the bottom of the well.

        Returns
        -------
        u : ndarray of float64
        """
        return self.potential_model.energy_fraction(np.asarray(E, dtype=np.float64), self._counting_radius(), density=density)

    def log_energy_density(self, E: np.ndarray, density: str | PotentialModel | None = None) -> np.ndarray:
        """``ln dN/dE``: energy distribution of the halo's mass inside the search sphere, per unit physical ``E``.

        Parameters
        ----------
        E : ndarray
        density : str or density model, optional
            A group of the potential model (e.g. ``"dark_matter"``): that group in the whole
            potential. The whole halo when ``None``. Or a density model (e.g. a
            :class:`~roadrunner.physics.potentials.HernquistPotential`): that tracer in the whole
            potential, about the bottom of the well.

        Returns
        -------
        log_n : ndarray of float64
        """
        return self.potential_model.log_energy_density(np.asarray(E, dtype=np.float64), self._counting_radius(), density=density)

    def release_tables(self) -> None:
        """Drop every cached energy/phase-space table of :attr:`potential_model` (U6).

        Memory only: the next call to :meth:`energy_fraction`, :meth:`log_energy_density` or
        :meth:`log_phase_space_fraction` rebuilds what it needs. A no-op for a single NFW
        (the per-process tables carry no per-halo state).
        """
        self.potential_model.release()

    def log_phase_space_fraction(self, E: np.ndarray) -> np.ndarray:
        """``ln w``: fraction of the search sphere's bound phase space more bound than the physical energy ``E``.

        Parameters
        ----------
        E : ndarray

        Returns
        -------
        log_w : ndarray of float64
        """
        return self.potential_model.log_phase_space_fraction(np.asarray(E, dtype=np.float64), self._counting_radius())

    def compute_energy(self, xyz: np.ndarray, vxyz: np.ndarray) -> np.ndarray:
        """Specific orbital energy ``E = Φ + ½|v - v_com|²`` at catalogue coordinates.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)
            Absolute positions (catalogue coordinates); each component is
            measured from its own centre.
        vxyz : ndarray of shape (n, 3)
            Absolute velocities; the components' mass-weighted velocity
            :attr:`com_velocity` is subtracted (the tree velocity for a single
            component).

        Returns
        -------
        E : ndarray of math_dtype, shape (n,)
            Specific orbital energy (negative = bound).
        """
        v = np.asarray(vxyz, dtype=np.float64) - self.com_velocity
        E = self.potential_model.potential(self._physical(xyz)) + 0.5 * row_squared_norms(v)
        return E.astype(math_dtype(), copy=False)


def tree_potential(row, model, length_scale):
    """The tree halo's potential (dark matter): physical, at the row's centre and velocity.

    Parameters
    ----------
    row : mapping
        Merger-tree row (position, velocity, mass, virial and scale radius).
    model : str
        Potential family (``POTENTIAL_MODELS``).
    length_scale : float
        Catalogue-to-physical length factor (:func:`physical_length_factor`).

    Returns
    -------
    potential : SphericalPotential
    """
    scale = row["scale_radius"] * length_scale
    kwargs = {
        "nfw": {"Rs": scale, "c": row["virial_radius"] / row["scale_radius"]},
        "plummer": {"a": scale},
        "hernquist": {"a": scale},
    }.get(model.lower(), {})
    centre = np.array([row["position_x"], row["position_y"], row["position_z"]], dtype=np.float64)
    velocity = np.array([row["velocity_x"], row["velocity_y"], row["velocity_z"]], dtype=np.float64)
    return get_potential(model, M=row["mass"], G=G_KM, centre=centre * length_scale, velocity=velocity, **kwargs)

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
    ``tree_position * length_scale`` and ``tree_velocity``). The halo exposes the
    potential model's methods as one-line wrappers at catalogue coordinates:
    positions ``xyz`` are made physical (``length_scale``, float64), energies
    ``E`` are physical and pass as they are, and the energy distributions are
    counted inside the halo's search sphere (``search_factor`` virial radii).
    Particle-level results are returned in ``math_dtype()``, the energy
    distributions in float64.

    It adds its record (``tree_position``, ``tree_velocity``, ``tree_mass``:
    the catalogue's centre, bulk velocity and mass, the search sphere being
    centred on ``tree_position``; ``virial_radius``, ``redshift``,
    ``comoving``, ``length_scale``, ``search_factor``, ``sub_tree_id``;
    ``inner_position``, the innermost particle of the search sphere, set by
    :func:`compute_halo_bound_particles`, the centre until then), the
    components' totals (:attr:`total_mass`, and the mass-weighted
    :attr:`com_position` and :attr:`com_velocity`), the energy of its
    particles (:meth:`compute_energy`, in the centre-of-mass frame), the scale of its binding
    energies (:meth:`binding_energy_scale`) and the boundness storage
    ``(indices, energies, tdyns)``, the stored boundness being
    ``b = -E / binding_energy_scale()``.

    Parameters
    ----------
    inner : PotentialModel
        The tree's potential (the first component); its ``centre`` must be
        the physical halo centre ``tree_position * length_scale`` and its
        ``velocity`` the halo's bulk velocity ``tree_velocity``.
    tree_position : ndarray of shape (3,)
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
        inner: PotentialModel,
        tree_position: np.ndarray,
        tree_velocity: np.ndarray,
        virial_radius: float,
        sub_tree_id: int,
        redshift: float,
        comoving: bool = True,
        search_factor: float = 1.0,
    ):
        md = math_dtype()
        self.tree_position = np.asarray(tree_position, dtype=md)
        self.tree_velocity = np.asarray(tree_velocity, dtype=md)
        self.virial_radius = float(virial_radius)
        self.sub_tree_id = sub_tree_id
        self.redshift = float(redshift)
        self.comoving = comoving
        self.search_factor = float(search_factor)
        self.length_scale = physical_length_factor(self.redshift, comoving)
        self.potential_model = CompositeSphericalPotential(inner)
        self.tree_mass = float(inner.M)
        self.inner_position = self.tree_position.copy()
        self._boundness: tuple | None = None

    def set_boundness(
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
            Boundness energy values (ambient math precision).
        tdyns : ndarray
            Dynamical times for the bound particles (ambient math precision).
        """
        self._boundness = (
            np.asarray(indices, dtype=LOCAL_IDX), energies, tdyns,
        )

    def get_boundness(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Retrieve the stored boundness data.

        Returns
        -------
        boundness : tuple of (indices, energies, tdyns) or None
            ``None`` if boundness has not been computed yet.
        """
        return self._boundness

    @property
    def has_boundness(self) -> bool:
        """``True`` if boundness data has been stored."""
        return self._boundness is not None

    def __iter__(self):
        return iter(self.potential_model)

    @property
    def total_mass(self) -> float:
        """Mass of all the components, ``Σ M_i``."""
        return float(self.potential_model.M)

    @property
    def com_position(self) -> np.ndarray:
        """Mass-weighted centre of the components (catalogue coordinates, float64)."""
        return self.potential_model.com_centre / self.length_scale

    @property
    def com_velocity(self) -> np.ndarray:
        """Mass-weighted velocity of the components (km/s, float64): the frame of :meth:`compute_energy`."""
        return self.potential_model.com_velocity

    def __len__(self):
        return len(self.potential_model)

    def __getitem__(self, i):
        return self.potential_model[i]

    def add_potential(self, potential: PotentialModel) -> None:
        """Add a component to the halo's potential on the fly.

        Parameters
        ----------
        potential : PotentialModel
            Centred on its own ``centre``, an absolute physical position.
        """
        self.potential_model.add(potential)

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
        """Central potential ``Φ₀`` of the potential model (physical units).

        Returns
        -------
        phi0 : float
        """
        return self.potential_model.central_potential()

    def well_depth(self, xyz: np.ndarray) -> float:
        """Depth of the potential well with the innermost particle at the catalogue position ``xyz``.

        Parameters
        ----------
        xyz : ndarray of shape (1, 3)

        Returns
        -------
        depth : float
        """
        return float(self.potential_model.well_depth(self._physical(xyz)))

    def binding_energy_scale(self) -> float:
        """Scale of the halo's binding energies: the depth of its well at the innermost particle.

        :meth:`well_depth` at :attr:`inner_position`: ``-Φ(0)`` for a well
        with a finite centre, ``-Φ`` there for a point mass, whose ``Φ(0)``
        the softening alone sets. Boundness is stored as
        ``-E / binding_energy_scale()``.

        Returns
        -------
        scale : float
        """
        return self.well_depth(self.inner_position[None])

    def distribution_function(self, E: np.ndarray) -> np.ndarray:
        """Isotropic distribution function ``f(E)`` of the potential model at physical energies.

        Parameters
        ----------
        E : ndarray

        Returns
        -------
        f : ndarray of float64
        """
        return self.potential_model.distribution_function(np.asarray(E, dtype=np.float64))

    def energy_fraction(self, E: np.ndarray) -> np.ndarray:
        """Fraction of the halo's mass inside the search sphere more bound than the physical energy ``E``.

        Parameters
        ----------
        E : ndarray

        Returns
        -------
        u : ndarray of float64
        """
        return self.potential_model.energy_fraction(np.asarray(E, dtype=np.float64), self._counting_radius())

    def log_energy_density(self, E: np.ndarray) -> np.ndarray:
        """``ln dN/dE``: energy distribution of the halo's mass inside the search sphere, per unit physical ``E``.

        Parameters
        ----------
        E : ndarray

        Returns
        -------
        log_n : ndarray of float64
        """
        return self.potential_model.log_energy_density(np.asarray(E, dtype=np.float64), self._counting_radius())

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

    @classmethod
    def from_snapshot_row(cls, row, model="kepler", comoving=True, search_factor=1.0):
        """Construct a HaloModel from a merger-tree row.

        The potential is centred on the row's position, converted to physical
        (``length_scale``), and moves with the row's velocity; its scale radius
        is the row's, converted likewise
        (``Rs`` and ``c = r_vir / r_s`` for NFW, ``a`` for Plummer and
        Hernquist; a Kepler point mass needs neither).

        Parameters
        ----------
        row : pandas.Series
            Row from the merger tree with position, velocity, mass, etc.
        model : str, default='kepler'
            Potential model name.
        comoving : bool, default=True
            Whether coordinates are comoving.
        search_factor : float, default=1.0
            Multiple of the virial radius: the boundness search sphere.

        Returns
        -------
        halo : HaloModel
        """
        tree_position = np.array([
            row["position_x"], row["position_y"], row["position_z"],
        ], dtype=math_dtype())
        tree_velocity = np.array([
            row["velocity_x"], row["velocity_y"], row["velocity_z"],
        ], dtype=math_dtype())
        ls = physical_length_factor(row["Redshift"], comoving)
        scale = row["scale_radius"] * ls
        kwargs = {"M": row["mass"], "G": G_KM, "centre": tree_position.astype(np.float64) * ls,
                  "velocity": tree_velocity.astype(np.float64)}
        kwargs.update({"nfw": {"Rs": scale, "c": row["virial_radius"] / row["scale_radius"]},
                       "plummer": {"a": scale}, "hernquist": {"a": scale}}.get(model.lower(), {}))
        inner = get_potential(model, **kwargs)
        return cls(
            inner=inner,
            tree_position=tree_position,
            tree_velocity=tree_velocity,
            virial_radius=float(row["virial_radius"]),
            sub_tree_id=int(row["Sub_tree_id"]),
            redshift=float(row["Redshift"]),
            comoving=comoving,
            search_factor=search_factor,
        )

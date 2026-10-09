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
    """Merger-tree halo: its record (centre, bulk velocity, virial radius,
    redshift) and one or more gravitational potentials, plus boundness data.

    The potential, :attr:`potential_model`, is a
    :class:`~roadrunner.physics.potentials.CompositeSphericalPotential` of one
    or more components, each centred on its own absolute physical ``centre``
    (the tree's on the halo centre, ``xcen * length_scale``). Positions and
    velocities come in as absolute snapshot coordinates (positions in tree
    units, comoving when ``comoving``; velocities peculiar, in km/s, not
    scaled): :meth:`potential`, :meth:`compute_energy` and
    :meth:`orbital_time` convert positions to physical float64, evaluate the
    potential model there and return ``math_dtype()``. The energy
    distributions are the potential model's, at the stored boundness ``b``.
    ``tree_mass`` is the first component's mass. Boundness results are stored
    as a tuple ``(indices, energies, tdyns)``;
    :func:`compute_halo_bound_particles` also sets ``inner_position``, the
    innermost particle of the search sphere (snapshot coordinates; the centre
    until then), where the energy scale is taken.

    Parameters
    ----------
    inner : PotentialModel
        The tree's potential (the first component); its ``centre`` must be
        the physical halo centre ``xcen * length_scale``.
    xcen : ndarray of shape (3,)
        Halo centre position (snapshot coordinates).
    velocity : ndarray of shape (3,)
        Halo bulk velocity.
    virial_radius : float
        Virial radius in kpc (snapshot units).
    sub_tree_id : int
        Halo identifier from the merger tree.
    redshift : float
        Snapshot redshift.
    comoving : bool, default=True
        If ``True``, snapshot lengths are comoving and ``length_scale`` is
        ``1 / (1 + z)``.
    search_factor : float, default=1.0
        Multiple of the virial radius: the boundness search sphere and the
        counting radius of the energy distributions.
    """

    def __init__(
        self,
        inner: PotentialModel,
        xcen: np.ndarray,
        velocity: np.ndarray,
        virial_radius: float,
        sub_tree_id: int,
        redshift: float,
        comoving: bool = True,
        search_factor: float = 1.0,
    ):
        md = math_dtype()
        self.xcen = np.asarray(xcen, dtype=md)
        self.velocity = np.asarray(velocity, dtype=md)
        self.virial_radius = float(virial_radius)
        self.sub_tree_id = sub_tree_id
        self.redshift = float(redshift)
        self.comoving = comoving
        self.search_factor = float(search_factor)
        self.length_scale = physical_length_factor(self.redshift, comoving)
        self.potential_model = CompositeSphericalPotential(inner)
        self.tree_mass = float(inner.M)
        self.inner_position = self.xcen.copy()
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

    def add_potential(self, potential: PotentialModel) -> None:
        """Add a component to the halo's potential (a new composite).

        Parameters
        ----------
        potential : PotentialModel
            Centred on its own ``centre``, an absolute physical position.
        """
        self.potential_model = CompositeSphericalPotential(*self.potential_model, potential)

    def _physical(self, xyz: np.ndarray) -> np.ndarray:
        """Absolute snapshot positions as absolute physical positions, in float64."""
        return np.asarray(xyz, dtype=np.float64) * self.length_scale

    def potential(self, xyz: np.ndarray) -> np.ndarray:
        """Gravitational potential at absolute snapshot positions.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)
            Absolute positions (snapshot coordinates); each component is
            measured from its own centre.

        Returns
        -------
        phi : ndarray of math_dtype, shape (n,)
        """
        return self.potential_model.potential(self._physical(xyz)).astype(math_dtype(), copy=False)

    def orbital_time(self, E: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """Per-particle orbital timescale for specific energies ``E`` at absolute snapshot positions.

        A single component gives its own (the Kepler period for a point mass,
        the dynamical time otherwise); several combine theirs as
        ``t^-2 = Σ t_i^-2``, each from its own centre.

        Parameters
        ----------
        E : ndarray
        xyz : ndarray of shape (n, 3)
            Absolute positions (snapshot coordinates).

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

    def binding_energy_scale(self) -> float:
        """Energy scale that normalises this halo's binding energies: the depth of its well.

        The potential model's :meth:`well_depth` at the innermost particle,
        :attr:`inner_position`: ``-Φ(0)`` for a well with a finite centre, and
        ``-Φ`` there for a point mass, whose ``Φ(0)`` the softening alone sets.
        Boundness is stored as ``-E / binding_energy_scale()``.

        Returns
        -------
        scale : float
        """
        return float(self.potential_model.well_depth(self._physical(self.inner_position[None])))

    def _counting_radius(self) -> float:
        """Physical radius of the boundness search sphere, ``search_factor * r_vir``: where the distributions count."""
        return self.search_factor * self.virial_radius * self.length_scale

    def energy_fraction(self, b: np.ndarray) -> np.ndarray:
        """Fraction of the halo's mass inside the search sphere more bound than the stored boundness ``b``.

        ``b`` is converted to the physical energy ``E = -b * binding_energy_scale()``
        and passed, with the search sphere's physical radius, to the halo's
        potential; the result is a fraction, so it needs no conversion back.

        Parameters
        ----------
        b : ndarray
            Stored boundness, ``-E / binding_energy_scale()``.

        Returns
        -------
        u : ndarray of float64
        """
        return self.potential_model.energy_fraction(-b * self.binding_energy_scale(), self._counting_radius())

    def log_energy_density(self, b: np.ndarray) -> np.ndarray:
        """``ln dN/db``: energy distribution of the halo's mass inside the search sphere, per unit stored boundness.

        The potential returns ``ln dN/dE`` at ``E = -b s`` (``s`` the energy
        scale); ``|dE/db| = s`` adds ``ln s``.

        Parameters
        ----------
        b : ndarray
            Stored boundness.

        Returns
        -------
        log_n : ndarray of float64
        """
        s = self.binding_energy_scale()
        return self.potential_model.log_energy_density(-b * s, self._counting_radius()) + np.log(s)

    def log_phase_space_fraction(self, b: np.ndarray) -> np.ndarray:
        """``ln w``: fraction of the search sphere's bound phase space more bound than the stored boundness ``b``.

        Converted like :meth:`energy_fraction`; ``w`` is a fraction, so it
        needs no conversion back.

        Parameters
        ----------
        b : ndarray
            Stored boundness.

        Returns
        -------
        log_w : ndarray of float64
        """
        return self.potential_model.log_phase_space_fraction(-b * self.binding_energy_scale(), self._counting_radius())

    def compute_energy(self, xyz, vxyz):
        """Total specific orbital energy ``E = Φ + ½|v - v_halo|²`` at absolute snapshot coordinates.

        Parameters
        ----------
        xyz : ndarray of shape (n, 3)
            Absolute positions (snapshot coordinates); each component is
            measured from its own centre.
        vxyz : ndarray of shape (n, 3)
            Absolute velocities; the halo bulk velocity is subtracted.

        Returns
        -------
        E : ndarray of math_dtype, shape (n,)
            Specific orbital energy (negative = bound).
        """
        v = np.asarray(vxyz, dtype=np.float64) - self.velocity
        E = self.potential_model.potential(self._physical(xyz)) + 0.5 * row_squared_norms(v)
        return E.astype(math_dtype(), copy=False)

    @classmethod
    def from_snapshot_row(cls, row, model="kepler", comoving=True, search_factor=1.0):
        """Construct a HaloModel from a merger-tree row.

        The potential is centred on the row's position, converted to physical
        (``length_scale``); its scale radius is the row's, converted likewise
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
        xcen = np.array([
            row["position_x"], row["position_y"], row["position_z"],
        ], dtype=math_dtype())
        velocity = np.array([
            row["velocity_x"], row["velocity_y"], row["velocity_z"],
        ], dtype=math_dtype())
        ls = physical_length_factor(row["Redshift"], comoving)
        scale = row["scale_radius"] * ls
        kwargs = {"M": row["mass"], "G": G_KM, "centre": xcen.astype(np.float64) * ls}
        kwargs.update({"nfw": {"Rs": scale, "c": row["virial_radius"] / row["scale_radius"]},
                       "plummer": {"a": scale}, "hernquist": {"a": scale}}.get(model.lower(), {}))
        inner = get_potential(model, **kwargs)
        return cls(
            inner=inner,
            xcen=xcen,
            velocity=velocity,
            virial_radius=float(row["virial_radius"]),
            sub_tree_id=int(row["Sub_tree_id"]),
            redshift=float(row["Redshift"]),
            comoving=comoving,
            search_factor=search_factor,
        )

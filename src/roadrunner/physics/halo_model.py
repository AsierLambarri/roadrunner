"""Single-halo model with potential and boundness storage."""

import numpy as np

from roadrunner.physics.constants import G_KM
from roadrunner.physics.potentials import get_potential
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


def _summed(values):
    """Sum of per-potential values, starting from the first (one term is returned as is)."""
    values = iter(values)
    total = next(values)
    for v in values:
        total = total + v
    return total


class HaloModel:
    """Merger-tree halo: its record (centre, bulk velocity, virial radius,
    redshift) and one or more gravitational potentials, plus boundness data.

    The first potential is the tree's. :meth:`potential` and
    :meth:`compute_energy` measure each potential from its own ``centre`` (the
    tree centre when ``None``); the radial summaries (dynamical and orbital
    time, tidal denominator, central potential) and any 1-D radii treat every
    potential as centred on the tree centre. The energy scale, the mass and the
    energy distributions are the first potential's.
    Boundness results are stored as a tuple ``(indices, energies, tdyns)``.

    Parameters
    ----------
    inner : PotentialModel
        The tree's potential (the first).
    xcen : ndarray of shape (3,)
        Halo centre position.
    velocity : ndarray of shape (3,)
        Halo bulk velocity.
    virial_radius : float
        Virial radius in kpc.
    sub_tree_id : int
        Halo identifier from the merger tree.
    redshift : float
        Snapshot redshift.
    comoving : bool, default=True
        If ``True``, ``potential()`` applies the ``(1 + z)`` scaling
        to convert comoving to physical coordinates.
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
    ):
        self._potentials = [inner]
        md = math_dtype()
        self.xcen = np.asarray(xcen, dtype=md)
        self.velocity = np.asarray(velocity, dtype=md)
        self.virial_radius = float(virial_radius)
        self.sub_tree_id = sub_tree_id
        self.redshift = float(redshift)
        self.comoving = comoving
        self._1plusz  = 1 / (1 + self.redshift) if comoving else 1
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
        """Add a potential to the halo's total.

        Parameters
        ----------
        potential : PotentialModel
            Centred on its ``centre`` (the tree centre when ``None``).
        """
        self._potentials.append(potential)

    @property
    def mass(self) -> float:
        """Mass of the tree's (first) potential."""
        return float(self._potentials[0].M)

    def _radius(self, xyz_or_r: np.ndarray) -> np.ndarray:
        """Physical radius from the tree centre (2-D positions) or of 1-D radii."""
        if xyz_or_r.ndim == 2:
            return np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
        return xyz_or_r * self._1plusz

    def _radii(self, xyz_or_r: np.ndarray, relative: bool = False) -> list[np.ndarray]:
        """Physical radius from each potential's centre; 1-D radii are shared.

        ``relative``: 2-D positions are already relative to the tree centre.
        """
        if xyz_or_r.ndim == 1:
            return [xyz_or_r * self._1plusz] * len(self._potentials)
        rel = xyz_or_r if relative else xyz_or_r - self.xcen
        shared, radii = None, []
        for p in self._potentials:
            if p.centre is None:
                if shared is None:
                    shared = np.linalg.norm(rel, axis=1) * self._1plusz
                radii.append(shared)
            else:
                offset = np.asarray(p.centre, dtype=self.xcen.dtype) - self.xcen
                radii.append(np.linalg.norm(rel - offset, axis=1) * self._1plusz)
        return radii

    def _single(self) -> PotentialModel:
        """The halo's only potential; energy distributions are not defined for several."""
        if len(self._potentials) > 1:
            raise NotImplementedError("Energy distributions are defined for a single-potential halo only.")
        return self._potentials[0]

    def potential(self, xyz_or_r: np.ndarray) -> np.ndarray:
        """Evaluate the gravitational potential at given positions or radii.

        Parameters
        ----------
        xyz_or_r : ndarray of shape (n_points, 3) or (n_points,)
            If 2-D each potential is measured from its own centre; if 1-D
            the values are radii shared by every potential.

        Returns
        -------
        phi : ndarray
            Potential energy values.
        """
        radii = self._radii(xyz_or_r)
        return _summed(p.potential(r) for p, r in zip(self._potentials, radii))

    def dynamical_time(self, xyz_or_r: np.ndarray) -> np.ndarray:
        """Compute the dynamical time at given positions.

        With several potentials ``1/t²`` adds up, since every
        ``t = 2π sqrt(r³ / (G M_i(<r)))``.

        Parameters
        ----------
        xyz_or_r : ndarray of shape (n, 3) or (n,)
            If 2-D the norm of differences from ``xcen`` is computed;
            if 1-D the values are treated as radii directly.

        Returns
        -------
        tdyn : ndarray
        """
        r = self._radius(xyz_or_r)
        if len(self._potentials) == 1:
            return self._potentials[0].dynamical_time(r)
        with np.errstate(divide="ignore"):
            return 1.0 / np.sqrt(_summed(1.0 / p.dynamical_time(r) ** 2 for p in self._potentials))

    def orbital_time(self, E: np.ndarray, xyz_or_r: np.ndarray) -> np.ndarray:
        """Per-particle orbital timescale for specific energies ``E``.

        A single potential gives its own (the Kepler period for a point mass,
        the dynamical time otherwise); several give :meth:`dynamical_time`.

        Parameters
        ----------
        E : ndarray
        xyz_or_r : ndarray of shape (n, 3) or (n,)

        Returns
        -------
        t : ndarray
        """
        if len(self._potentials) == 1:
            return self._potentials[0].orbital_time(E, self._radius(xyz_or_r))
        return self.dynamical_time(xyz_or_r)

    def tidal_denominator(self, xyz_or_r: np.ndarray) -> np.ndarray:
        """Compute the tidal denominator for tidal-radius estimation.

        Parameters
        ----------
        xyz_or_r : ndarray of shape (n, 3) or (n,)
            If 2-D the norm of differences from ``xcen`` is computed;
            if 1-D the values are treated as radii directly.

        Returns
        -------
        denom : ndarray
        """
        r = self._radius(xyz_or_r)
        return _summed(p.tidal_denominator(r) for p in self._potentials)

    def central_potential(self) -> float:
        """Central potential ``Φ₀``, summed over the potentials (physical units).

        Returns
        -------
        phi0 : float
        """
        return _summed(p.central_potential() for p in self._potentials)

    def binding_energy_scale(self) -> float:
        """Energy scale that normalises this halo's binding energies (the tree potential's).

        Boundness is stored as ``-E / binding_energy_scale()``: ``-E / v_vir²``
        for Kepler, ``E / Φ₀`` for NFW.

        Returns
        -------
        scale : float
        """
        return self._potentials[0].binding_energy_scale(self.virial_radius * self._1plusz)

    def energy_fraction(self, eps: np.ndarray) -> np.ndarray:
        """Mass fraction of the halo more bound than the normalised boundness ``eps``.

        Parameters
        ----------
        eps : ndarray

        Returns
        -------
        u : ndarray
        """
        return self._single().energy_fraction(eps)

    def log_energy_density(self, eps: np.ndarray) -> np.ndarray:
        """Log dark-matter energy distribution at the normalised boundness ``eps``.

        Parameters
        ----------
        eps : ndarray

        Returns
        -------
        log_n : ndarray
        """
        return self._single().log_energy_density(eps)

    def log_phase_space_fraction(self, boundness: np.ndarray) -> np.ndarray:
        """Log fraction of the virial sphere's bound phase space more bound than ``boundness``.

        Parameters
        ----------
        boundness : ndarray
            Normalised boundness as stored by :func:`compute_halo_bound_particles`.

        Returns
        -------
        log_w : ndarray
        """
        return self._single().log_phase_space_fraction(boundness)

    def compute_energy(self, xyz_or_r, vxyz_or_mag, relative=True):
        """Total specific orbital energy ``E = Φ + ½v²``.

        Parameters
        ----------
        xyz_or_r : ndarray of shape (n, 3) or (n,)
            Positions or radii.  If 2-D and ``relative=True`` they are
            already relative to the halo centre; if ``relative=False``
            ``self.xcen`` is subtracted. Each potential is measured from
            its own centre.
        vxyz_or_mag : ndarray of shape (n, 3) or (n,)
            Velocities or speed magnitudes.  If 2-D and ``relative=True``
            they are already relative to the halo bulk velocity; if
            ``relative=False`` ``self.velocity`` is subtracted.
        relative : bool, default=True
            Whether the 3-D inputs are already relative to the halo.

        Returns
        -------
        E : ndarray
            Specific orbital energy (negative = bound).
        """
        radii = self._radii(xyz_or_r, relative)
        if vxyz_or_mag.ndim == 2:
            vel = vxyz_or_mag if relative else vxyz_or_mag - self.velocity
            v2 = np.sum(vel**2, axis=1)
        else:
            v2 = vxyz_or_mag**2

        return _summed(p.potential(r) for p, r in zip(self._potentials, radii)) + 0.5 * v2

    @classmethod
    def from_snapshot_row(cls, row, model="kepler", comoving=True):
        """Construct a HaloModel from a merger-tree row.

        Parameters
        ----------
        row : pandas.Series
            Row from the merger tree with position, velocity, mass, etc.
        model : str, default='kepler'
            Potential model name.
        comoving : bool, default=True
            Whether coordinates are comoving.

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
        conc = row["virial_radius"] / row["scale_radius"]
        kwargs = {"M": row["mass"], "G": G_KM}
        if model.lower() == "nfw":
            kwargs["Rs"] = row["scale_radius"] * physical_length_factor(
                row["Redshift"], comoving
            )
            kwargs["c"] = conc
        inner = get_potential(model, **kwargs)
        return cls(
            inner=inner,
            xcen=xcen,
            velocity=velocity,
            virial_radius=float(row["virial_radius"]),
            sub_tree_id=int(row["Sub_tree_id"]),
            redshift=float(row["Redshift"]),
            comoving=comoving,
        )

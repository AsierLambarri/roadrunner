"""Single-halo model with potential and boundness storage."""

import numpy as np

from roadrunner.physics.constants import G_KM
from roadrunner.physics.potentials import get_potential
from roadrunner._mcf_types import PotentialModel


class HaloModel:
    """Single halo model containing a gravitational potential and boundness data.

    The halo stores its position, velocity, virial radius, and a
    :class:`PotentialModel` instance for computing energies.
    Boundness results are stored as a tuple ``(indices, energies, tdyns)``.

    Parameters
    ----------
    inner : PotentialModel
        Gravitational potential model.
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
        self._inner = inner
        self.xcen = np.asarray(xcen, dtype=np.float64)
        self.velocity = np.asarray(velocity, dtype=np.float64)
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
        indices : ndarray of uint64
            Bound particle indices.
        energies : ndarray of float32
            Boundness energy values.
        tdyns : ndarray of float32
            Dynamical times for the bound particles.
        """
        self._boundness = (indices, energies, tdyns)

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

    def potential(self, xyz_or_r: np.ndarray) -> np.ndarray:
        """Evaluate the gravitational potential at given positions or radii.

        Parameters
        ----------
        xyz_or_r : ndarray of shape (n_points, 3) or (n_points,)
            If 2-D the norm of the differences from ``xcen`` is computed;
            if 1-D the values are treated as radii directly.

        Returns
        -------
        phi : ndarray
            Potential energy values.
        """
        if xyz_or_r.ndim == 2:
            r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
        else:
            r = xyz_or_r * self._1plusz
        return self._inner.potential(r)

    def dynamical_time(self, xyz_or_r: np.ndarray) -> np.ndarray:
        """Compute the dynamical time at given positions.

        Parameters
        ----------
        xyz_or_r : ndarray of shape (n, 3) or (n,)
            If 2-D the norm of differences from ``xcen`` is computed;
            if 1-D the values are treated as radii directly.

        Returns
        -------
        tdyn : ndarray
        """
        if xyz_or_r.ndim == 2:
            r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
        else:
            r = xyz_or_r * self._1plusz
        return self._inner.dynamical_time(r)

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
        if xyz_or_r.ndim == 2:
            r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
        else:
            r = xyz_or_r * self._1plusz
        return self._inner.tidal_denominator(r)

    def compute_energy(self, xyz_or_r, vxyz_or_mag, relative=True):
        """Total specific orbital energy ``E = Φ + ½v²``.

        Parameters
        ----------
        xyz_or_r : ndarray of shape (n, 3) or (n,)
            Positions or radii.  If 2-D and ``relative=True`` they are
            already relative to the halo centre; if ``relative=False``
            ``self.xcen`` is subtracted.
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
        if xyz_or_r.ndim == 2:
            pos = xyz_or_r if relative else xyz_or_r - self.xcen
            r = np.linalg.norm(pos, axis=1) * self._1plusz
        else:
            r = xyz_or_r * self._1plusz

        if vxyz_or_mag.ndim == 2:
            vel = vxyz_or_mag if relative else vxyz_or_mag - self.velocity
            v2 = np.sum(vel**2, axis=1)
        else:
            v2 = vxyz_or_mag**2

        return self._inner.potential(r) + 0.5 * v2

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
        ], dtype=np.float64)
        velocity = np.array([
            row["velocity_x"], row["velocity_y"], row["velocity_z"],
        ], dtype=np.float64)
        conc = row["virial_radius"] / row["scale_radius"]
        kwargs = {"M": row["mass"], "G": G_KM}
        if model.lower() == "nfw":
            kwargs["Rs"] = row["scale_radius"] / (1 + row["Redshift"])
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

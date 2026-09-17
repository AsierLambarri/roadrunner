"""NPZ-based snapshot reader for mock simulations.

Supports two NPZ formats:
- Mock format (``mock_sim=True``): expects ``indices``, ``masses``,
  ``coords`` keys with 6-D coordinates.
- Generic format: expects a 2-D array with columns
  ``[id, mass, x, y, z, vx, vy, vz, metallicity]``.
"""

from __future__ import annotations

import os

import numpy as np

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner._defaults import SIM_ID, data_dtype, math_dtype


class NPZSnapshotReader:
    """Snapshot reader for NPZ files (mock simulations).

    Parameters
    ----------
    equiv_table : EquivalenceTable
        Table mapping snapshot IDs to file paths, times, and redshifts.
    base_dir : str, default=''
        Base directory for snapshot files.
    mock_sim : bool, default=False
        If ``True``, read the roadrunner mock NPZ format.
    assign_fields : list of str or None, optional
        Attribute names for the assigner input.
    """

    _particle_filter = None

    def __init__(
        self,
        equiv_table: EquivalenceTable,
        base_dir: str = "",
        mock_sim: bool = False,
        assign_fields=None,
    ):
        self._mock_sim = mock_sim
        self._assign_fields = assign_fields
        self._snap_info: dict[str, tuple[float, float]] = {}
        df = equiv_table.dataframe
        for _, row in df.iterrows():
            full = os.path.join(base_dir, row["snapname"])
            self._snap_info[full] = (row["redshift"], row["time"])

    @property
    def particle_filter(self):
        """Currently configured persistent particle-ID filter (or None)."""
        return self._particle_filter

    def set_particle_filter(self, particle_indices) -> None:
        """Set a persistent ID filter applied to every subsequent load()."""
        self._particle_filter = np.asarray(particle_indices, dtype=SIM_ID)

    def erase_particle_filter(self) -> None:
        """Remove the persistent ID filter (back to loading all particles)."""
        self._particle_filter = None

    @staticmethod
    def _region_mask(positions, sphere=None, bbox=None):
        if (sphere is None) == (bbox is None):
            raise ValueError("Provide exactly one of `sphere` or `bbox`.")
        if sphere is not None:
            center = np.asarray(sphere[0], dtype=math_dtype())
            radius = float(sphere[1])
            return np.sum((positions - center) ** 2, axis=1) <= radius ** 2
        lower = np.asarray(bbox[0], dtype=math_dtype())
        upper = np.asarray(bbox[1], dtype=math_dtype())
        return np.all((positions >= lower) & (positions <= upper), axis=1)

    def select_indices(self, file_path: str, sphere=None, bbox=None) -> np.ndarray:
        """Return simulation IDs inside a comoving region.

        Parameters
        ----------
        file_path : str
            Path to the snapshot file.
        sphere : tuple or None, optional
            Sphere selection ``((cx, cy, cz), radius)`` in comoving kpc.
        bbox : tuple or None, optional
            Box selection ``((xlo, ylo, zlo), (xhi, yhi, zhi))`` in comoving kpc.

        Returns
        -------
        indices : ndarray of uint64
            Simulation IDs of the particles inside the region.
        """
        saved = self._particle_filter
        self._particle_filter = None
        try:
            snap = self.load(file_path)
        finally:
            self._particle_filter = saved
        mask = self._region_mask(snap.position, sphere=sphere, bbox=bbox)
        return snap.index[mask]

    def load(self, file_path: str, particle_indices=None) -> SnapshotData:
        """Load a snapshot from an NPZ file.

        Parameters
        ----------
        file_path : str
            Path to the ``.npz`` file.
        particle_indices : ndarray or None, optional
            If given, keep only these simulation IDs for this call only,
            overriding any persistent filter.

        Returns
        -------
        snap_data : SnapshotData
            Loaded particle data (indices, masses, coordinates, metallicity).
        """
        redshift, time = self._snap_info[file_path]
        raw = np.load(file_path, allow_pickle=False)

        extra = {}
        dt = data_dtype()
        if self._mock_sim:
            indices = raw["indices"]
            masses = np.asarray(raw["masses"], dtype=dt)
            coords = np.asarray(raw["coords"], dtype=dt)
            positions = coords[:, :3]
            velocities = coords[:, 3:6]
            if "metallicity" in raw:
                extra["metallicity"] = np.asarray(raw["metallicity"], dtype=dt)
        else:
            arr = raw if isinstance(raw, np.ndarray) else raw[raw.files[0]]
            indices = arr[:, 0].astype(SIM_ID)
            masses = arr[:, 1].astype(dt)
            positions = arr[:, 2:5].astype(dt)
            velocities = arr[:, 5:8].astype(dt)
            if arr.shape[1] >= 9:
                extra["metallicity"] = arr[:, 8].astype(dt)

        snap = SnapshotData(
            index=indices, mass=masses, position=positions,
            velocity=velocities, redshift=redshift, time=time,
            assign_fields=self._assign_fields,
            **extra,
        )
        ids = (particle_indices if particle_indices is not None
               else self._particle_filter)
        if ids is None:
            return snap
        mask = np.isin(snap.index, ids)
        fields = {f: getattr(snap, f)[mask] for f in snap.fields}
        return SnapshotData(
            redshift=snap.redshift, time=snap.time,
            assign_fields=snap._assign_fields, **fields,
        )

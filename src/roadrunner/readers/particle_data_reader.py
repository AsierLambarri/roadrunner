"""Particle-data snapshot reader for custom binary formats.

The HDF5 file must contain:
- ``data/indices``, ``data/masses``, ``data/positions``,
  ``data/velocities``, ``data/scaler/mean``, ``data/scaler/scale``
- ``header/redshift`` and ``header/time`` attributes

Extra datasets (e.g. ``data/metallicity``) are optional and are
loaded as-is (stored in raw physical units).
"""

from __future__ import annotations

import h5py
import numpy as np

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.equivalence import EquivalenceTable


class ParticleDataSnapshotReader:
    """Snapshot reader for pre-processed HDF5 particle data.

    Parameters
    ----------
    equiv_table : EquivalenceTable
        Snapshot equivalence table.
    base_dir : str, default=''
        Base directory for data files.
    assign_fields : list of str or None, optional
        Attribute names for the assigner input.
    extra_fields : list of str or None, optional
        Dataset names under ``data/`` to load as extra particle
        fields (e.g. ``["metallicity"]``).
    """

    _particle_filter = None

    def __init__(self, equiv_table: EquivalenceTable, base_dir: str = "",
                 assign_fields=None, extra_fields=None):
        self._equiv = equiv_table
        self._base_dir = base_dir
        self._assign_fields = assign_fields
        self._extra_fields = extra_fields or []

    @property
    def particle_filter(self):
        """Currently configured persistent particle-ID filter (or None)."""
        return self._particle_filter

    def set_particle_filter(self, particle_indices) -> None:
        """Set a persistent ID filter applied to every subsequent load()."""
        self._particle_filter = np.asarray(particle_indices, dtype=np.uint64)

    def erase_particle_filter(self) -> None:
        """Remove the persistent ID filter (back to loading all particles)."""
        self._particle_filter = None

    @staticmethod
    def _region_mask(positions, sphere=None, bbox=None):
        if (sphere is None) == (bbox is None):
            raise ValueError("Provide exactly one of `sphere` or `bbox`.")
        if sphere is not None:
            center = np.asarray(sphere[0], dtype=np.float64)
            radius = float(sphere[1])
            return np.sum((positions - center) ** 2, axis=1) <= radius ** 2
        lower = np.asarray(bbox[0], dtype=np.float64)
        upper = np.asarray(bbox[1], dtype=np.float64)
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
        """Load a snapshot from an HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the ``.hdf5`` file.
        particle_indices : ndarray or None, optional
            If given, keep only these simulation IDs for this call only,
            overriding any persistent filter.

        Returns
        -------
        snap_data : SnapshotData
            Loaded particle data.
        """
        with h5py.File(file_path, "r") as hf:
            indices = hf["data/indices"][:]
            masses = hf["data/masses"][:]
            pos_scaled = hf["data/positions"][:]
            vel_scaled = hf["data/velocities"][:]
            mean = hf["data/scaler/mean"][:]
            scale = hf["data/scaler/scale"][:]
            redshift = hf["header"].attrs["redshift"]
            time = hf["header"].attrs["time"]

            extra = {}
            for name in self._extra_fields:
                key = f"data/{name}"
                if key in hf:
                    extra[name] = hf[key][:]

        inv_s = 1.0 / scale
        positions = pos_scaled * inv_s[:3] + mean[:3]
        velocities = vel_scaled * inv_s[3:6] + mean[3:6]

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

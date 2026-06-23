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

    def __init__(self, equiv_table: EquivalenceTable, base_dir: str = "",
                 assign_fields=None, extra_fields=None):
        self._equiv = equiv_table
        self._base_dir = base_dir
        self._assign_fields = assign_fields
        self._extra_fields = extra_fields or []

    def load(self, file_path: str) -> SnapshotData:
        """Load a snapshot from an HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the ``.hdf5`` file.

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

        return SnapshotData(
            index=indices, mass=masses, position=positions,
            velocity=velocities, redshift=redshift, time=time,
            assign_fields=self._assign_fields,
            **extra,
        )

"""Particle-data snapshot reader for custom binary formats."""

from __future__ import annotations

import h5py

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.equivalence import EquivalenceTable


class ParticleDataSnapshotReader:
    """Snapshot reader for pre-processed HDF5 particle data.

    The HDF5 file must contain:
    - ``data/indices``, ``data/masses``, ``data/positions``,
      ``data/velocities``, ``data/scaler/mean``, ``data/scaler/scale``
    - ``header/redshift`` and ``header/time`` attributes

    Positions and velocities are stored in scaled coordinates and
    are transformed back to physical units during loading.

    Parameters
    ----------
    equiv_table : EquivalenceTable
        Snapshot equivalence table.
    base_dir : str, default=''
        Base directory for data files.
    """

    def __init__(self, equiv_table: EquivalenceTable, base_dir: str = ""):
        self._equiv = equiv_table
        self._base_dir = base_dir

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

        inv_s = 1.0 / scale
        positions = pos_scaled * inv_s[:3] + mean[:3]
        velocities = vel_scaled * inv_s[3:6] + mean[3:6]

        return SnapshotData(
            indices, masses, positions, velocities,
            redshift, time,
        )

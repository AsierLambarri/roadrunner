from __future__ import annotations

import h5py

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.equivalence import EquivalenceTable


class ParticleDataSnapshotReader:
    def __init__(self, equiv_table: EquivalenceTable, base_dir: str = ""):
        self._equiv = equiv_table
        self._base_dir = base_dir

    def load(self, file_path: str) -> SnapshotData:
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

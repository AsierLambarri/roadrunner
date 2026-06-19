from __future__ import annotations

import os

import numpy as np

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.equivalence import EquivalenceTable


class NPZSnapshotReader:
    def __init__(
        self,
        equiv_table: EquivalenceTable,
        base_dir: str = "",
        mock_sim: bool = False,
    ):
        self._mock_sim = mock_sim
        self._snap_info: dict[str, tuple[float, float]] = {}
        df = equiv_table.dataframe
        for _, row in df.iterrows():
            full = os.path.join(base_dir, row["snapname"])
            self._snap_info[full] = (row["redshift"], row["time"])

    def load(self, file_path: str) -> SnapshotData:
        redshift, time = self._snap_info[file_path]
        raw = np.load(file_path, allow_pickle=False)

        if self._mock_sim:
            indices = raw["indices"]
            masses = raw["masses"]
            coords = raw["coords"]
            positions = coords[:, :3]
            velocities = coords[:, 3:6]
            metallicity = raw.get("metallicity", None)
        else:
            arr = raw if isinstance(raw, np.ndarray) else raw[raw.files[0]]
            indices = arr[:, 0].astype(np.uint64)
            masses = arr[:, 1].astype(np.float64)
            positions = arr[:, 2:5].astype(np.float64)
            velocities = arr[:, 5:8].astype(np.float64)
            metallicity = arr[:, 8].astype(np.float64) if arr.shape[1] >= 9 else None

        return SnapshotData(
            indices, masses, positions, velocities,
            redshift, time, metallicity,
        )

#############################################################################
#
# package:   roadrunner.readers
# file:      npz_reader.py
# brief:     NPZ-based snapshot reader for mock simulations.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 jun 2026 - Created
#            19 jun 2026 - Last edit
#
#############################################################################

from __future__ import annotations

import os

import numpy as np

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.equivalence import EquivalenceTable


class NPZSnapshotReader:
    """Snapshot reader for NPZ files (mock simulations).

    Supports two NPZ formats:
    - Mock format (``mock_sim=True``): expects ``indices``, ``masses``,
      ``coords`` keys with 6-D coordinates.
    - Generic format: expects a 2-D array with columns
      ``[id, mass, x, y, z, vx, vy, vz, metallicity]``.

    Parameters
    ----------
    equiv_table : EquivalenceTable
        Table mapping snapshot IDs to file paths, times, and redshifts.
    base_dir : str, default=''
        Base directory for snapshot files.
    mock_sim : bool, default=False
        If ``True``, read the roadrunner mock NPZ format.
    """

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
        """Load a snapshot from an NPZ file.

        Parameters
        ----------
        file_path : str
            Path to the ``.npz`` file.

        Returns
        -------
        snap_data : SnapshotData
            Loaded particle data (indices, masses, coordinates, metallicity).
        """
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

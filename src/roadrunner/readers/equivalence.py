#############################################################################
#
# package:   roadrunner.readers
# file:      equivalence.py
# brief:     Snapshot-equivalence table (snapshot ID / path / time / redshift).
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 may 2026 - Created
#            13 may 2026 - Last edit
#
#############################################################################

import os

import pandas as pd


class EquivalenceTable:
    """Lookup table mapping snapshot IDs to file paths, cosmic times, and redshifts.

    Parameters
    ----------
    data : str, DataFrame, or tuple
        If str: path to a CSV file with ``snapshot``, ``snapname``,
        ``time``, and ``redshift`` columns.
        If DataFrame: direct data.
        If tuple: ``(snapshots, paths, times, redshifts)``.
    base_dir : str, default=''
        Base directory prepended to ``snapname`` when resolving paths.
    """

    def __init__(self, data, base_dir: str = ""):
        if isinstance(data, str):
            self._df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self._df = data.copy()
        elif isinstance(data, (list, tuple)) and len(data) == 4:
            snapshots, paths, times, redshifts = data
            self._df = pd.DataFrame({
                "snapshot": snapshots,
                "snapname": paths,
                "time": times,
                "redshift": redshifts,
            })
        else:
            raise TypeError(
                "data must be a file path, DataFrame, or "
                "(snapshots, paths, times, redshifts) tuple"
            )
        self._base_dir = base_dir
        self._df.set_index("snapshot", inplace=True, drop=False)

    def snapshot_path(self, snap_id: int) -> str:
        """Full filesystem path to the snapshot file.

        Parameters
        ----------
        snap_id : int
            Snapshot ID.

        Returns
        -------
        path : str
        """
        return os.path.join(self._base_dir, self._df.loc[snap_id, "snapname"])

    def snapshot_time(self, snap_id: int) -> float:
        """Cosmic time for a given snapshot ID.

        Parameters
        ----------
        snap_id : int

        Returns
        -------
        time : float
        """
        return float(self._df.loc[snap_id, "time"])

    def snapshot_redshift(self, snap_id: int) -> float:
        """Redshift for a given snapshot ID.

        Parameters
        ----------
        snap_id : int

        Returns
        -------
        z : float
        """
        return float(self._df.loc[snap_id, "redshift"])

    @property
    def snapshots(self) -> list[int]:
        """List of all snapshot IDs."""
        return sorted(self._df["snapshot"].unique().tolist())

    @property
    def dataframe(self) -> pd.DataFrame:
        """The underlying DataFrame."""
        return self._df

    @property
    def min_snapshot(self) -> int:
        """Earliest (smallest) snapshot ID."""
        return int(self._df["snapshot"].min())

    @property
    def max_snapshot(self) -> int:
        """Latest (largest) snapshot ID."""
        return int(self._df["snapshot"].max())

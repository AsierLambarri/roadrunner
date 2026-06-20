#############################################################################
#
# package:   roadrunner.readers
# file:      merger_tree.py
# brief:     Merger-tree CSV reader.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   12 may 2026 - Created
#            12 may 2026 - Last edit
#
#############################################################################

import pandas as pd


class MergerTreeReaderCSV:
    """Merger-tree reader that loads data from a CSV file or DataFrame.

    The CSV must contain at least ``Snapshot`` and ``Sub_tree_id`` columns.

    Parameters
    ----------
    data : str or DataFrame
        Path to a CSV file or a pre-loaded DataFrame.
    """

    def __init__(self, data: str | pd.DataFrame):
        if isinstance(data, str):
            self._df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self._df = data.copy()
        else:
            raise TypeError("data must be str (file path) or pd.DataFrame")

    @property
    def dataframe(self) -> pd.DataFrame:
        """The underlying DataFrame."""
        return self._df

    @property
    def snapshots(self) -> list[int]:
        """Sorted list of unique snapshot IDs."""
        return sorted(int(x) for x in self._df["Snapshot"].unique())

    @property
    def subtree_ids(self) -> list[int]:
        """Sorted list of unique Sub_tree IDs."""
        return sorted(int(x) for x in self._df["Sub_tree_id"].unique())


    def select_snapshots(self, snap_ids: list[int] | int) -> pd.DataFrame:
        """Filter the tree to specific snapshot IDs.

        Parameters
        ----------
        snap_ids : int or list of int
            Snapshot ID(s) to select.

        Returns
        -------
        df : DataFrame
            Filtered copy.
        """
        if not pd.api.types.is_list_like(snap_ids):
            snap_ids = [snap_ids]
        return self._df[self._df["Snapshot"].isin(snap_ids)].copy()

    def select_subtrees(self, tree_ids: list[int] | int) -> pd.DataFrame:
        """Filter the tree to specific Sub_tree IDs.

        Parameters
        ----------
        tree_ids : int or list of int
            Sub_tree ID(s) to select.

        Returns
        -------
        df : DataFrame
            Filtered copy.
        """
        if not pd.api.types.is_list_like(tree_ids):
            tree_ids = [tree_ids]
        return self._df[self._df["Sub_tree_id"].isin(tree_ids)].copy()

    def select_accretion_host(
        self, snapshot_id: int, criterion: str = "max_mass"
    ) -> pd.DataFrame:
        """Select the accretion host halo from a given snapshot.

        Parameters
        ----------
        snapshot_id : int
            Snapshot to search.
        criterion : str, default='max_mass'
            Selection criterion (``'max_mass'`` is the only option).

        Returns
        -------
        host_df : DataFrame
            Single-row DataFrame with the host halo.
        """
        snap_df = self.select_snapshots([snapshot_id])
        if snap_df.empty:
            raise ValueError("Invalid snapshot selection: empty tree.")
        if criterion == "max_mass":
            idx = snap_df["mass"].idxmax()
            return snap_df.loc[[idx]]
        raise ValueError(f"unknown criterion: {criterion}")

    def to_numeric(self) -> pd.DataFrame:
        """Return a copy of the DataFrame with inferred numeric dtypes.

        Returns
        -------
        df : DataFrame
        """
        return self._df.copy().infer_objects()

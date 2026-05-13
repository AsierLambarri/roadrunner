import pandas as pd


class MergerTreeReaderCSV:
    def __init__(self, data: str | pd.DataFrame):
        if isinstance(data, str):
            self._df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self._df = data.copy()
        else:
            raise TypeError("data must be str (file path) or pd.DataFrame")

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

    @property
    def snapshots(self) -> list[int]:
        return sorted(int(x) for x in self._df["Snapshot"].unique())

    @property
    def subtree_ids(self) -> list[int]:
        return sorted(int(x) for x in self._df["Sub_tree_id"].unique())


    def select_snapshots(self, snap_ids: list[int] | int) -> pd.DataFrame:
        if not pd.api.types.is_list_like(snap_ids):
            snap_ids = [snap_ids]
        return self._df[self._df["Snapshot"].isin(snap_ids)].copy()

    def select_subtrees(self, tree_ids: list[int] | int) -> pd.DataFrame:
        if not pd.api.types.is_list_like(tree_ids):
            tree_ids = [tree_ids]
        return self._df[self._df["Sub_tree_id"].isin(tree_ids)].copy()

    def select_accretion_host(
        self, snapshot_id: int, criterion: str = "max_mass"
    ) -> pd.DataFrame:
        snap_df = self.select_snapshots([snapshot_id])
        if snap_df.empty:
            raise ValueError("Invalid snapshot selection: empty tree.")
        if criterion == "max_mass":
            idx = snap_df["mass"].idxmax()
            return snap_df.loc[[idx]]
        raise ValueError(f"unknown criterion: {criterion}")

    def to_numeric(self) -> pd.DataFrame:
        return self._df.copy().infer_objects()

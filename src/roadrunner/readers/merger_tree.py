import pandas as pd


class MergerTreeReader:
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

    def select_snapshots(self, snap_ids: list[int]) -> pd.DataFrame:
        return self._df[self._df["Snapshot"].isin(snap_ids)].copy()

    def select_subtrees(self, tree_ids: list[int]) -> pd.DataFrame:
        return self._df[self._df["Sub_tree_id"].isin(tree_ids)].copy()

    def select_accretion_host(
        self, snapshot_id: int, criterion: str = "max_mass"
    ) -> pd.DataFrame:
        snap_df = self.select_snapshots([snapshot_id])
        if snap_df.empty:
            return snap_df
        if criterion == "max_mass":
            idx = snap_df["mass"].idxmax()
            return snap_df.loc[[idx]]
        raise ValueError(f"unknown criterion: {criterion}")

    def compute_scale_radii(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        df = snapshot_df.copy()
        if "scale_radius" not in df.columns or df["scale_radius"].isna().all():
            return df
        return df

    def compute_satellites(
        self, snapshot_df: pd.DataFrame, rvir_factor: float = 1.0
    ) -> dict[int, list[int]]:
        satellites: dict[int, list[int]] = {}
        for _, row in snapshot_df.iterrows():
            host_raw = row.get("Sub_tree_id_host")
            if pd.isna(host_raw):
                continue
            host_id = int(host_raw)
            halo_id = int(row["Sub_tree_id"])
            if host_id > 0 and host_id != halo_id:
                satellites.setdefault(host_id, []).append(halo_id)
        return satellites

    def compute_most_bound_satellite(
        self, snapshot_df: pd.DataFrame
    ) -> dict[int, int]:
        most_bound: dict[int, int] = {}
        for _, row in snapshot_df.iterrows():
            host_raw = row.get("Sub_tree_id_host")
            if pd.isna(host_raw):
                continue
            host_id = int(host_raw)
            halo_id = int(row["Sub_tree_id"])
            if host_id > 0 and host_id != halo_id:
                if host_id not in most_bound:
                    most_bound[host_id] = halo_id
                elif not pd.isna(row.get("bound_mass")) and row["bound_mass"] > 0:
                    current = most_bound[host_id]
                    curr_row = snapshot_df[
                        snapshot_df["Sub_tree_id"] == current
                    ]
                    if not curr_row.empty and row["bound_mass"] > curr_row.iloc[0].get("bound_mass", 0):
                        most_bound[host_id] = halo_id
        return most_bound

    def compute_distance_to_host(
        self, snapshot_df: pd.DataFrame, column: str = "host_distance"
    ) -> pd.DataFrame:
        df = snapshot_df.copy()
        if column in df.columns:
            return df
        return df

    def to_numeric(self) -> pd.DataFrame:
        return self._df.copy().infer_objects()

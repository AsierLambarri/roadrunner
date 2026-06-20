#############################################################################
#
# package:   roadrunner.physics
# file:      merger_tree.py
# brief:     <TODO>
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 may 2026 - Created
#            13 may 2026 - Last edit
#
#############################################################################

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from roadrunner.physics.constants import G_KM, DUFFY_A, DUFFY_B, DUFFY_C, DUFFY_PIVOT_MASS, MIN_DISTANCE
from roadrunner.readers.merger_tree import MergerTreeReaderCSV


def nfw_cmz_relation_duffy(M: float | np.ndarray, z: float) -> float | np.ndarray:
    return DUFFY_A * (M / DUFFY_PIVOT_MASS) ** DUFFY_B * (1 + z) ** DUFFY_C


class MergerTreeHandlerCSV(MergerTreeReaderCSV):

    # ── STATE MODIFIERS ──────────────────────────────────────

    def compute_scale_radii(self):
        self._df["scale_radius"] = self._df.apply(self._compute_rs_row, axis=1)

    def set_constant_column(self, name: str, value):
        self._df[name] = value

    def compute_most_bound_satellite(self, rvir_factor: float = 1.0):
        self._df["host_id"] = -1
        for snap in tqdm(
            self._df["Snapshot"].unique(),
            desc="Computing most bound satellite per snapshot",
            leave=False,
        ):
            mask = self._df["Snapshot"] == snap
            snap_df = self._df[mask].copy()
            sat_to_host = self._most_bound_satellite_impl(snap_df, rvir_factor)
            self._df.loc[mask, "host_id"] = (
                self._df.loc[mask, "Sub_tree_id"]
                .map(sat_to_host)
                .fillna(-1)
                .astype(int)
            )

    def compute_distance_to_host(self, column: str = "host_id"):
        target = f"distance_to_{column}"
        self._df[target] = np.nan
        for snap in tqdm(
            self._df["Snapshot"].unique(),
            desc="Computing distance to host per snapshot",
            leave=False,
        ):
            mask = self._df["Snapshot"] == snap
            snap_df = self._df[mask].copy()
            result_df = self._distance_to_host_impl(snap_df, column)
            self._df.loc[mask, target] = result_df[target].values

    # ── STATELESS METHOD ─────────────────────────────────────

    @staticmethod
    def compute_satellites(
        snapshot_df: pd.DataFrame, rvir_factor: float = 1.0
    ) -> dict[int, set[int]]:
        return MergerTreeHandlerCSV._satellites_impl(snapshot_df, rvir_factor)

    # ── PRIVATE HELPERS ──────────────────────────────────────

    @staticmethod
    def _compute_rs_row(row: pd.Series) -> float:
        if np.isnan(row["scale_radius"]):
            conc = nfw_cmz_relation_duffy(row["mass"], row["Redshift"])
            return row["virial_radius"] / conc
        return row["scale_radius"]

    @staticmethod
    def _satellites_impl(
        snap_df: pd.DataFrame, rvir_factor: float
    ) -> dict[int, set[int]]:
        n = len(snap_df)
        if n == 0:
            return {}

        redshift = snap_df["Redshift"].values[0].astype(np.float32)
        positions = (
            snap_df[["position_x", "position_y", "position_z"]].to_numpy(
                dtype=np.float32
            )
            / (1 + redshift)
        )
        velocities = snap_df[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(
            dtype=np.float32
        )
        masses = snap_df["mass"].values.astype(np.float32)
        rvirs = snap_df["virial_radius"].values.astype(np.float32) / (1 + redshift)
        sub_ids = snap_df["Sub_tree_id"].values.astype(int)

        tree = KDTree(positions)
        satellites_map: dict[int, set[int]] = {sid: set() for sid in sub_ids}

        for i in range(n):
            pos_i = positions[i]
            vel_i = velocities[i]
            mass_i = masses[i]
            host_id = sub_ids[i]

            idx_neighbors = np.asarray(
                tree.query_ball_point(pos_i, r=rvir_factor * rvirs[i], workers=10)
            )
            idx_neighbors = idx_neighbors[idx_neighbors != i]
            if idx_neighbors.size == 0:
                continue

            idx_candidates = idx_neighbors[masses[idx_neighbors] < mass_i]
            if idx_candidates.size == 0:
                continue

            r_vec = positions[idx_candidates] - pos_i
            dist = np.linalg.norm(r_vec, axis=1)
            v_rel = velocities[idx_candidates] - vel_i
            v_rel2 = np.sum(v_rel**2, axis=1)
            non_zero = dist > MIN_DISTANCE
            if not np.any(non_zero):
                continue
            idx_candidates = idx_candidates[non_zero]
            v_esc2 = 2 * G_KM * mass_i / dist[non_zero]
            E_bind = v_rel2[non_zero] - v_esc2

            bound_mask = E_bind < 0
            if not np.any(bound_mask):
                continue

            sat_ids = sub_ids[idx_candidates[bound_mask]]
            satellites_map[host_id].update(sat_ids)

        return dict(satellites_map)

    @staticmethod
    def _most_bound_satellite_impl(
        snap_df: pd.DataFrame, rvir_factor: float
    ) -> dict[int, int]:
        n = len(snap_df)
        if n == 0:
            return {}

        redshift = snap_df["Redshift"].values[0]
        positions = (
            snap_df[["position_x", "position_y", "position_z"]].to_numpy(
                dtype=np.float32
            )
            / (1 + redshift)
        )
        velocities = snap_df[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(
            dtype=np.float32
        )
        masses = snap_df["mass"].to_numpy(dtype=np.float32)
        rvirs = snap_df["virial_radius"].to_numpy(dtype=np.float32) / (1 + redshift)
        sub_ids = snap_df["Sub_tree_id"].to_numpy(dtype=int)

        tree = KDTree(positions)

        sat_to_host: dict[int, int] = {}
        min_energy: dict[int, float] = {}

        for i in range(n):
            host_id = sub_ids[i]
            mass_i = masses[i]
            pos_i = positions[i]
            vel_i = velocities[i]

            idx_neighbors = np.asarray(
                tree.query_ball_point(pos_i, r=rvir_factor * rvirs[i], workers=10)
            )

            idx_neighbors = idx_neighbors[idx_neighbors != i]
            if idx_neighbors.size == 0:
                continue

            idx_candidates = idx_neighbors[masses[idx_neighbors] < mass_i]
            if idx_candidates.size == 0:
                continue

            r_vec = positions[idx_candidates] - pos_i
            dist = np.linalg.norm(r_vec, axis=1)
            v_rel = velocities[idx_candidates] - vel_i
            v_rel2 = np.sum(v_rel**2, axis=1)

            non_zero = dist > MIN_DISTANCE
            if not np.any(non_zero):
                continue
            idx_candidates = idx_candidates[non_zero]
            v_esc2 = 2 * G_KM * mass_i / dist[non_zero]
            E_bind = v_rel2[non_zero] - v_esc2

            bound_mask = E_bind < 0
            if not np.any(bound_mask):
                continue

            sat_ids = sub_ids[idx_candidates[bound_mask]]
            energies = E_bind[bound_mask]

            for sid, E in zip(sat_ids, energies):
                if (sid not in min_energy) or (E < min_energy[sid]):
                    min_energy[sid] = E
                    sat_to_host[sid] = host_id

        return sat_to_host

    @staticmethod
    def _distance_to_host_impl(
        snap_df: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        host_positions = snap_df[
            ["Sub_tree_id", "position_x", "position_y", "position_z"]
        ].rename(
            columns={
                "Sub_tree_id": column,
                "position_x": "host_x",
                "position_y": "host_y",
                "position_z": "host_z",
            }
        )
        snap_df = snap_df.merge(host_positions, on=column, how="left")

        dx = snap_df["position_x"] - snap_df["host_x"]
        dy = snap_df["position_y"] - snap_df["host_y"]
        dz = snap_df["position_z"] - snap_df["host_z"]

        snap_df[f"distance_to_{column}"] = np.sqrt(dx**2 + dy**2 + dz**2)

        snap_df = snap_df.drop(columns=["host_x", "host_y", "host_z"])

        return snap_df

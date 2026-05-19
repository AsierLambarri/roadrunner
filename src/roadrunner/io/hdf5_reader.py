import json

import h5py
import numpy as np
import pandas as pd

from roadrunner._exceptions import SnapshotLoadError


class HDF5CatalogueReader:
    def __init__(self, path):
        self._path = path

    def read_header(self):
        with h5py.File(self._path, "r") as hf:
            hdr = hf["header"]
            return {
                "accretion_id": int(hdr["accretion_id"][()]),
                "snapshots": list(hdr["snapshots"][()]),
                "config": json.loads(hdr["config"][()]),
                "last_snapshot": int(hdr["last_snapshot"][()]),
            }

    def read_last_snapshot(self):
        with h5py.File(self._path, "r") as hf:
            val = hf["header"]["last_snapshot"][()]
            return int(val) if val >= 0 else None

    def _read_snapshot_dataset(self, snapshot_id, ds_name):
        snap_key = f"/snapshots/{snapshot_id}/{ds_name}"
        with h5py.File(self._path, "r") as hf:
            if snap_key not in hf:
                raise SnapshotLoadError(
                    f"Snapshot {snapshot_id} has no {ds_name} dataset"
                )
            return pd.DataFrame.from_records(hf[snap_key][()])

    def read_galaxy_properties(self, snapshot_id=None):
        return self._read_snapshot_dataset(
            snapshot_id, "galaxy_properties"
        ) if snapshot_id is not None else self._read_all_snapshots("galaxy_properties")

    def read_riley_criterion(self, snapshot_id=None):
        return self._read_snapshot_dataset(
            snapshot_id, "riley_criterion"
        ) if snapshot_id is not None else self._read_all_snapshots("riley_criterion")

    def _read_all_snapshots(self, ds_name):
        with h5py.File(self._path, "r") as hf:
            frames = []
            for snap_str in hf.get("snapshots", {}):
                snap_key = f"/snapshots/{snap_str}/{ds_name}"
                if snap_key in hf:
                    df = pd.DataFrame.from_records(hf[snap_key][()])
                    df["Snapshot"] = int(snap_str)
                    frames.append(df)
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)

    def read_births(self):
        with h5py.File(self._path, "r") as hf:
            if "final/births" not in hf:
                return pd.DataFrame()
            return pd.DataFrame.from_records(hf["final/births"][()])

    def read_assembly(self):
        with h5py.File(self._path, "r") as hf:
            if "final/assembly" not in hf:
                return pd.DataFrame()
            return pd.DataFrame.from_records(hf["final/assembly"][()])

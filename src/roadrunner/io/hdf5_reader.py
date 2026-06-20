#############################################################################
#
# package:   roadrunner.io
# file:      hdf5_reader.py
# brief:     HDF5 reader for galaxy catalogue data.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 may 2026 - Created
#            19 may 2026 - Last edit
#
#############################################################################

"""HDF5 reader for the galaxy catalogue.

Provides read access to the catalogue header, per-snapshot galaxy
properties and dynamical state, and final birth/assembly tables.
"""

import json

import h5py
import numpy as np
import pandas as pd

from roadrunner._exceptions import SnapshotLoadError


class HDF5CatalogueReader:
    """Reads galaxy catalogue data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the ``catalogue.hdf5`` file.
    """

    def __init__(self, path):
        self._path = path

    def read_header(self):
        """Read the catalogue header (accretion ID, snapshots, config).

        Returns
        -------
        header : dict
            Keys: ``accretion_id``, ``snapshots``, ``config``, ``last_snapshot``.
        """
        with h5py.File(self._path, "r") as hf:
            hdr = hf["header"]
            return {
                "accretion_id": int(hdr["accretion_id"][()]),
                "snapshots": list(hdr["snapshots"][()]),
                "config": json.loads(hdr["config"][()]),
                "last_snapshot": int(hdr["last_snapshot"][()]),
            }

    def read_last_snapshot(self):
        """Read the most recently processed snapshot ID.

        Returns
        -------
        last_snap : int or None
        """
        with h5py.File(self._path, "r") as hf:
            val = hf["header"]["last_snapshot"][()]
            return int(val) if val >= 0 else None

    def _read_snapshot_dataset(self, snapshot_id, ds_name):
        """Read a dataset for a specific snapshot as a DataFrame.

        Parameters
        ----------
        snapshot_id : int
        ds_name : str
            Dataset name (e.g. ``"galaxy_properties"``).

        Returns
        -------
        df : DataFrame

        Raises
        ------
        SnapshotLoadError
            If the dataset does not exist.
        """
        snap_key = f"/snapshots/{snapshot_id}/{ds_name}"
        with h5py.File(self._path, "r") as hf:
            if snap_key not in hf:
                raise SnapshotLoadError(
                    f"Snapshot {snapshot_id} has no {ds_name} dataset"
                )
            return pd.DataFrame.from_records(hf[snap_key][()])

    def read_galaxy_properties(self, snapshot_id=None):
        """Read galaxy properties for one or all snapshots.

        Parameters
        ----------
        snapshot_id : int or None, optional
            If given, read only that snapshot; otherwise concatenate all.

        Returns
        -------
        df : DataFrame
        """
        return self._read_snapshot_dataset(
            snapshot_id, "galaxy_properties"
        ) if snapshot_id is not None else self._read_all_snapshots("galaxy_properties")

    def read_riley_criterion(self, snapshot_id=None):
        """Read the Riley dynamical-state criterion for one or all snapshots.

        Parameters
        ----------
        snapshot_id : int or None, optional
            If given, read only that snapshot; otherwise concatenate all.

        Returns
        -------
        df : DataFrame
        """
        return self._read_snapshot_dataset(
            snapshot_id, "riley_criterion"
        ) if snapshot_id is not None else self._read_all_snapshots("riley_criterion")

    def _read_all_snapshots(self, ds_name):
        """Concatenate a dataset across all snapshots.

        Parameters
        ----------
        ds_name : str

        Returns
        -------
        df : DataFrame
        """
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
        """Read the final birth-tracker data.

        Returns
        -------
        df : DataFrame
            Birth-tracking table, or empty if not present.
        """
        with h5py.File(self._path, "r") as hf:
            if "final/births" not in hf:
                return pd.DataFrame()
            return pd.DataFrame.from_records(hf["final/births"][()])

    def read_assembly(self):
        """Read the final assembly-tracker data.

        Returns
        -------
        df : DataFrame
            Assembly table, or empty if not present.
        """
        with h5py.File(self._path, "r") as hf:
            if "final/assembly" not in hf:
                return pd.DataFrame()
            return pd.DataFrame.from_records(hf["final/assembly"][()])

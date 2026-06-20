#############################################################################
#
# package:   roadrunner.io
# file:      hdf5_catalogue.py
# brief:     HDF5 writer for the galaxy catalogue (snapshot and final).
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 may 2026 - Created
#            19 may 2026 - Last edit
#
#############################################################################

import json
import os

import h5py
import numpy as np
import pandas as pd

from roadrunner.helpers import select_uint_dtype


class HDF5CatalogueWriter:
    def __init__(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, "catalogue.hdf5")

    def _del_existing(self, group, name):
        if name in group:
            del group[name]

    def write_header(self, accretion_id, snapshots, config_dict,
                     merger_tree_df, equivalence_df):
        with h5py.File(self._path, "a") as hf:
            hdr = hf.require_group("header")
            self._del_existing(hdr, "accretion_id")
            hdr.create_dataset("accretion_id", data=accretion_id)
            self._del_existing(hdr, "snapshots")
            snap_arr = np.asarray(snapshots, dtype=select_uint_dtype(max(snapshots)))
            hdr.create_dataset("snapshots", data=snap_arr, compression="gzip")
            self._del_existing(hdr, "config")
            config_json = json.dumps(config_dict, default=str)
            hdr.create_dataset("config", data=config_json,
                               dtype=h5py.string_dtype())
            self._del_existing(hdr, "merger_tree")
            rec_merger = merger_tree_df.to_records(index=False)
            hdr.create_dataset("merger_tree", data=rec_merger,
                               compression="gzip")
            self._del_existing(hdr, "equivalence")
            equiv_json = equivalence_df.to_json(orient="records")
            hdr.create_dataset("equivalence", data=equiv_json,
                               dtype=h5py.string_dtype())
            self._del_existing(hdr, "last_snapshot")
            hdr.create_dataset("last_snapshot", data=-1, dtype=np.int32)

    def write_snapshot(self, snapshot_id, time,
                       properties_df, dynstate_df,
                       satellites_map):
        with h5py.File(self._path, "a") as hf:
            snap_grp = hf.require_group(f"/snapshots/{snapshot_id}")

            snap_grp.attrs["time"] = float(time)

            if not properties_df.empty:
                ds = snap_grp.create_dataset(
                    "galaxy_properties",
                    data=properties_df.to_records(index=False),
                    compression="gzip",
                )
                ds.attrs["Snapshot"] = int(snapshot_id)

            if not dynstate_df.empty:
                ds = snap_grp.create_dataset(
                    "riley_criterion",
                    data=dynstate_df.to_records(index=False),
                    compression="gzip",
                )
                ds.attrs["Snapshot"] = int(snapshot_id)

            sat_grp = snap_grp.require_group("satellite_relations")
            for gal_id, sats in satellites_map.items():
                arr = np.asarray(list(sats), dtype=np.int64)
                sat_grp.create_dataset(str(gal_id), data=arr, compression="gzip")

            # Update last_snapshot
            hf["header"]["last_snapshot"][()] = int(snapshot_id)

    def write_finalize(self, birth_df, assembly_df):
        with h5py.File(self._path, "a") as hf:
            final = hf.require_group("final")
            if "births" in final:
                del final["births"]
            if not birth_df.empty:
                final.create_dataset(
                    "births",
                    data=birth_df.to_records(index=False),
                    compression="gzip",
                )
            if "assembly" in final:
                del final["assembly"]
            if not assembly_df.empty:
                final.create_dataset(
                    "assembly",
                    data=assembly_df.to_records(index=False),
                    compression="gzip",
                )

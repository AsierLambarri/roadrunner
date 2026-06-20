#############################################################################
#
# package:   roadrunner.io
# file:      hdf5_particles.py
# brief:     HDF5 writer for per-snapshot particle data.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 may 2026 - Created
#            19 may 2026 - Last edit
#
#############################################################################

import os

import h5py
import numpy as np

from roadrunner.helpers import select_float_dtype, select_uint_dtype
from roadrunner.physics.scaler import StandardScaler


class HDF5ParticleWriter:
    def __init__(self, output_dir, float_atol=1e-4):
        os.makedirs(output_dir, exist_ok=True)
        self._dir = os.path.join(output_dir, "particle_data")
        self._float_atol = float_atol

    def _snap_path(self, snapshot_id):
        os.makedirs(self._dir, exist_ok=True)
        return os.path.join(self._dir, f"snapshot{snapshot_id:04d}.hdf5")

    def write_snapshot(self, snapshot_id, time, redshift, snapshot_data):
        path = self._snap_path(snapshot_id)
        with h5py.File(path, "w") as hf:
            header = hf.create_group("header")
            header.attrs["snapshot"] = int(snapshot_id)
            header.attrs["time"] = float(time)
            header.attrs["redshift"] = float(redshift)

            data = hf.create_group("data")

            coords = np.column_stack([
                snapshot_data.positions,
                snapshot_data.velocities,
            ])

            scaler = StandardScaler()
            scaled = scaler.fit_transform(coords.astype(np.float64, copy=False))

            scaler_grp = data.require_group("scaler")
            scaler_grp.create_dataset("mean", data=scaler.mean_)
            scaler_grp.create_dataset("scale", data=scaler.scale_)

            float_dtype = select_float_dtype(
                max(abs(scaled).max(), 1e-10), self._float_atol
            )

            data.create_dataset(
                "positions",
                data=scaled[:, :3].astype(float_dtype, copy=False),
                compression="gzip",
            )
            data.create_dataset(
                "velocities",
                data=scaled[:, 3:6].astype(float_dtype, copy=False),
                compression="gzip",
            )

            mass_dtype = select_float_dtype(
                float(snapshot_data.masses.max()), self._float_atol
            )
            data.create_dataset(
                "masses",
                data=snapshot_data.masses.astype(mass_dtype, copy=False),
                compression="gzip",
            )

            idx_dtype = select_uint_dtype(int(snapshot_data.indices.max()))
            data.create_dataset(
                "indices",
                data=snapshot_data.indices.astype(idx_dtype, copy=False),
                compression="gzip",
            )

import os

import h5py
import numpy as np

from roadrunner.helpers import select_float_dtype, select_uint_dtype
from roadrunner.physics.scaler import StandardScaler


class HDF5ParticleWriter:
    def __init__(self, output_dir, float_atol=1e-4):
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, "particles.hdf5")
        self._float_atol = float_atol

    def write_snapshot(self, snapshot_id, time, redshift, snapshot_data):
        with h5py.File(self._path, "a") as hf:
            snap_grp = hf.require_group(f"/snapshots/{snapshot_id}")
            snap_grp.attrs["time"] = float(time)
            snap_grp.attrs["redshift"] = float(redshift)

            coords = np.column_stack([
                snapshot_data.positions,
                snapshot_data.velocities,
            ])

            scaler = StandardScaler()
            scaled = scaler.fit_transform(coords.astype(np.float64, copy=False))

            scaler_grp = snap_grp.require_group("scaler")
            scaler_grp.create_dataset("mean", data=scaler.mean_)
            scaler_grp.create_dataset("scale", data=scaler.scale_)

            float_dtype = select_float_dtype(
                max(abs(scaled).max(), 1e-10), self._float_atol
            )

            snap_grp.create_dataset(
                "positions",
                data=scaled[:, :3].astype(float_dtype, copy=False),
                compression="gzip",
            )
            snap_grp.create_dataset(
                "velocities",
                data=scaled[:, 3:6].astype(float_dtype, copy=False),
                compression="gzip",
            )

            mass_dtype = select_float_dtype(
                float(snapshot_data.masses.max()), self._float_atol
            )
            snap_grp.create_dataset(
                "masses",
                data=snapshot_data.masses.astype(mass_dtype, copy=False),
                compression="gzip",
            )

            idx_dtype = select_uint_dtype(int(snapshot_data.indices.max()))
            snap_grp.create_dataset(
                "indices",
                data=snapshot_data.indices.astype(idx_dtype, copy=False),
                compression="gzip",
            )

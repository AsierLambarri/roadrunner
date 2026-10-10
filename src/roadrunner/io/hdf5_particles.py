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

"""HDF5 writer for per-snapshot particle data.

Writes positions, velocities, masses, indices, and any extra fields
carried by the snapshot data to per-snapshot HDF5 files.  Positions and
velocities are stored in scaled coordinates using :class:`StandardScaler`.
"""

import os
from dataclasses import asdict

import h5py
import numpy as np

from roadrunner.helpers import select_float_dtype, select_uint_dtype
from roadrunner.physics.scaler import StandardScaler
from roadrunner._defaults import COSMOLOGY_GROUP
from roadrunner.cosmology import current_cosmology


class HDF5ParticleWriter:
    """Writes per-snapshot particle data (positions, velocities, masses) to HDF5.

    Data is stored in scaled coordinates using a :class:`StandardScaler`.

    Parameters
    ----------
    output_dir : str
        Root output directory. Files go to ``{output_dir}/particle_data/``.
    float_atol : float, default=1e-4
        Tolerance for selecting float storage dtype.
    """

    def __init__(self, output_dir, float_atol=1e-4):
        os.makedirs(output_dir, exist_ok=True)
        self._dir = os.path.join(output_dir, "particle_data")
        self._float_atol = float_atol

    def _snap_path(self, snapshot_id):
        """Path to the HDF5 file for a given snapshot.

        Parameters
        ----------
        snapshot_id : int

        Returns
        -------
        path : str
        """
        os.makedirs(self._dir, exist_ok=True)
        return os.path.join(self._dir, f"snapshot{snapshot_id:04d}.hdf5")

    def write_snapshot(self, snapshot_id, time, redshift, snapshot_data):
        """Write one snapshot's particle data to an HDF5 file.

        Positions and velocities are stored in scaled coordinates, along
        with the :class:`StandardScaler` mean/scale used to recover the
        unscaled values. Masses, indices, and any fields of
        ``snapshot_data`` other than ``index``, ``mass``, ``position``,
        and ``velocity`` are stored unscaled under their own names. The
        run's cosmology is written as the ``header/cosmology`` group's
        attributes.

        Parameters
        ----------
        snapshot_id : int
            Snapshot identifier.
        time : float
            Cosmic time of the snapshot.
        redshift : float
            Redshift of the snapshot.
        snapshot_data : SnapshotData
            Particle data for this snapshot (positions, velocities,
            masses, indices, and any extra fields).
        """
        path = self._snap_path(snapshot_id)
        with h5py.File(path, "w") as hf:
            header = hf.create_group("header")
            header.attrs["snapshot"] = int(snapshot_id)
            header.attrs["time"] = float(time)
            header.attrs["redshift"] = float(redshift)
            hf.create_group(COSMOLOGY_GROUP).attrs.update(asdict(current_cosmology()))

            data = hf.create_group("data")

            coords = np.column_stack([
                snapshot_data.position,
                snapshot_data.velocity,
            ])

            scaler = StandardScaler()
            scaled = scaler.fit_transform(coords)

            scaler_grp = data.require_group("scaler")
            scaler_grp.create_dataset("mean", data=scaler.mean_)
            scaler_grp.create_dataset("scale", data=scaler.scale_)

            float_dtype = select_float_dtype(
                max(abs(scaled).max(), 1e-10), self._float_atol,
                msg="(scaled positions/velocities)",
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
                float(snapshot_data.mass.max()), self._float_atol,
                msg="(particle masses)",
            )
            data.create_dataset(
                "masses",
                data=snapshot_data.mass.astype(mass_dtype, copy=False),
                compression="gzip",
            )

            idx_dtype = select_uint_dtype(
                int(snapshot_data.index.max()), "(particle indices)")
            data.create_dataset(
                "indices",
                data=snapshot_data.index.astype(idx_dtype, copy=False),
                compression="gzip",
            )

            custom_fields = (
                f for f in snapshot_data.fields
                if f not in ("index", "mass", "position", "velocity")
            )
            for name in custom_fields:
                arr = getattr(snapshot_data, name)
                if np.issubdtype(arr.dtype, np.floating):
                    dtype = select_float_dtype(
                        float(np.abs(arr).max()), self._float_atol,
                        msg=f"(extra field {name!r})",
                    )
                else:
                    dtype = arr.dtype
                data.create_dataset(
                    name, data=np.asarray(arr).astype(dtype, copy=False),
                    compression="gzip",
                )

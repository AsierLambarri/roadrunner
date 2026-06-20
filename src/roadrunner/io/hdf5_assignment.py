#############################################################################
#
# package:   roadrunner.io
# file:      hdf5_assignment.py
# brief:     HDF5 writer for per-snapshot particle assignment data.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 may 2026 - Created
#            19 may 2026 - Last edit
#
#############################################################################

"""HDF5 writer for per-snapshot assignment data.

Writes responsibilities, fitted mixture parameters, hard labels,
boundness values, and particle timescales to per-snapshot HDF5 files.
"""

import os

import h5py
import numpy as np

from roadrunner.helpers import select_float_dtype, select_uint_dtype


class HDF5AssignmentWriter:
    """Writes per-snapshot assignment data (responsibilities, parameters, hard labels) to HDF5.

    Files are written to ``{output_dir}/assignment/snapshot{id:04d}.hdf5``.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    float_atol : float, default=1e-4
        Tolerance for selecting float storage dtype.
    """

    def __init__(self, output_dir, float_atol=1e-4):
        os.makedirs(output_dir, exist_ok=True)
        self._dir = os.path.join(output_dir, "assignment")
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

    def _ts_path(self):
        """Path to the timescales text file.

        Returns
        -------
        path : str
        """
        os.makedirs(self._dir, exist_ok=True)
        return os.path.join(self._dir, "timescales.txt")

    def _gal_uint_dtype(self, gids):
        """Smallest unsigned integer dtype that holds all galaxy IDs.

        Parameters
        ----------
        gids : array-like of int

        Returns
        -------
        dtype : numpy.dtype
        """
        return select_uint_dtype(int(max(gids)) if len(gids) > 0 else 1)

    def _pick_float_dtype(self, arr):
        """Smallest float dtype that preserves the data within ``float_atol``.

        Parameters
        ----------
        arr : ndarray

        Returns
        -------
        dtype : numpy.dtype
        """
        max_abs = float(np.abs(arr).max()) if arr.size > 0 else 1.0
        return select_float_dtype(max_abs, self._float_atol)

    def write_snapshot(self, snapshot_id, time,
                       assignment_result, boundness_csc):
        """Write a snapshot's assignment data to an HDF5 file.

        Parameters
        ----------
        snapshot_id : int
            Snapshot identifier.
        time : float
            Cosmic time of the snapshot.
        assignment_result : AssignmentResult
            Result from the assigner (responsibilities, hard labels, etc.).
        boundness_csc : SparseCSC
            Boundness matrix for this snapshot.
        """
        path = self._snap_path(snapshot_id)
        with h5py.File(path, "w") as hf:
            hf.attrs["time"] = float(time)

            galaxies_grp = hf.require_group("galaxies")

            resp_csc = assignment_result.responsibilities
            all_gids = list(resp_csc.column_id)
            if all_gids:
                all_vals = np.concatenate(resp_csc.column_values)
                float_dtype = self._pick_float_dtype(all_vals)
            else:
                float_dtype = np.float32

            for j, gid in enumerate(all_gids):
                resp_idx = resp_csc.column_indices[j]
                resp_vals = resp_csc.column_values[j]

                col_mask = boundness_csc.column_id == gid
                col_pos = np.where(col_mask)[0]
                bound_vals = (
                    boundness_csc.column_values[col_pos[0]]
                    if len(col_pos) > 0 and col_pos[0] < len(boundness_csc.column_values)
                    else np.array([], dtype=np.float32)
                )

                grp = galaxies_grp.require_group(str(gid))

                pid_dtype = select_uint_dtype(
                    int(resp_idx.max()) if resp_idx.size > 0 else 1
                )
                grp.create_dataset(
                    "indices",
                    data=resp_idx.astype(pid_dtype, copy=False),
                    compression="gzip",
                )
                grp.create_dataset(
                    "log_resp",
                    data=resp_vals.astype(float_dtype, copy=False),
                    compression="gzip",
                )
                grp.create_dataset(
                    "boundness",
                    data=bound_vals.astype(float_dtype, copy=False),
                    compression="gzip",
                )

                params = assignment_result.fitted_parameters.get(int(gid), {})
                mean = params.get("mean")
                if mean is not None:
                    mean_dtype = self._pick_float_dtype(mean)
                    grp.create_dataset(
                        "mean",
                        data=np.asarray(mean).astype(mean_dtype, copy=False),
                        compression="gzip",
                    )
                weight = params.get("weight")
                if weight is not None:
                    w = float(weight)
                    w_dtype = select_float_dtype(w, self._float_atol)
                    grp.create_dataset(
                        "weight",
                        data=np.array([w], dtype=w_dtype),
                        compression="gzip",
                    )
                cov = params.get("covariance")
                if cov is not None:
                    cov_arr = np.asarray(cov)
                    cov_dtype = self._pick_float_dtype(cov_arr)
                    kw = {} if cov_arr.ndim == 0 else {"compression": "gzip"}
                    grp.create_dataset(
                        "covariance",
                        data=cov_arr.astype(cov_dtype, copy=False),
                        **kw,
                    )

            # Hard assignment at root
            hf.create_dataset(
                "hard_assignment",
                data=assignment_result.particle_df.to_records(index=False),
                compression="gzip",
            )

    def write_timescales(self, particle_timescales):
        """Write particle timescales to a text file.

        Parameters
        ----------
        particle_timescales : ndarray of shape (n_particles,)
            Structured array with ``particle_index`` and ``timescale`` fields.
        """
        path = self._ts_path()
        np.savetxt(path, particle_timescales,
                   header="particle_index\ttimescale",
                   fmt="%u\t%.6f")

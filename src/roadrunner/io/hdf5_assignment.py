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


def _aligned_boundness(resp_rows, bound_rows, bound_values, dtype):
    """Join boundness values onto responsibility rows, by particle ID.

    ``resp_rows`` and ``bound_rows`` are two independently-ordered
    particle-index arrays for the same galaxy -- see C06: nothing
    upstream guarantees they share a row order, and their particle
    *sets* can genuinely differ too (a particle can carry nonzero
    responsibility toward a galaxy without ever being one of that
    galaxy's own boundness candidates). Missing entries get NaN,
    flagged via the returned validity mask, rather than a fabricated
    physical value or a value that silently belongs to another particle.

    Parameters
    ----------
    resp_rows : ndarray of int
        Particle indices in the order they'll be written (target order).
    bound_rows : ndarray of int
        Particle indices boundness values are keyed by (any order).
    bound_values : ndarray
        Values parallel to ``bound_rows``.
    dtype : numpy.dtype
        Storage dtype for the returned values array.

    Returns
    -------
    values : ndarray of shape (len(resp_rows),)
        ``NaN`` where no matching boundness row exists.
    valid : ndarray of bool, shape (len(resp_rows),)
    """
    values = np.full(len(resp_rows), np.nan, dtype=dtype)
    valid = np.zeros(len(resp_rows), dtype=bool)
    if len(bound_rows):
        order = np.argsort(bound_rows)
        sorted_rows = bound_rows[order]
        positions = np.searchsorted(sorted_rows, resp_rows)
        possible = np.flatnonzero(positions < len(sorted_rows))
        matched = possible[sorted_rows[positions[possible]] == resp_rows[possible]]
        values[matched] = bound_values[order[positions[matched]]]
        valid[matched] = True
    return values, valid


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
        self._written_timescale_ids = None

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
        """Path to the timescales text file (``{dir}/assignment/timescales.txt``).

        Returns
        -------
        path : str
        """
        os.makedirs(self._dir, exist_ok=True)
        return os.path.join(self._dir, "timescales.txt")

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
        return select_float_dtype(max_abs, self._float_atol, msg="(assignment values)")

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

            # Boundness gets its own storage dtype, chosen from its own
            # finite values -- not reused from the responsibility dtype
            # above, which is picked from an unrelated value range (C06).
            all_bound_vals = (
                np.concatenate(boundness_csc.column_values)
                if len(boundness_csc.column_values) else np.array([])
            )
            finite_bound_vals = all_bound_vals[np.isfinite(all_bound_vals)]
            bound_dtype = self._pick_float_dtype(finite_bound_vals)

            bound_col_by_gid = {
                gid: j for j, gid in enumerate(boundness_csc.column_id)
            }

            for j, gid in enumerate(all_gids):
                resp_idx = resp_csc.column_indices[j]
                resp_vals = resp_csc.column_values[j]

                col = bound_col_by_gid.get(gid)
                bound_rows = boundness_csc.column_indices[col] if col is not None else np.array([], dtype=resp_idx.dtype)
                bound_raw_vals = boundness_csc.column_values[col] if col is not None else np.array([])
                bound_vals, bound_valid = _aligned_boundness(
                    resp_idx, bound_rows, bound_raw_vals, bound_dtype,
                )

                grp = galaxies_grp.require_group(str(gid))

                pid_dtype = select_uint_dtype(
                    int(resp_idx.max()) if resp_idx.size > 0 else 1,
                    "(particle indices)",
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
                    data=bound_vals,  # already bound_dtype from _aligned_boundness
                    compression="gzip",
                )
                grp.create_dataset(
                    "boundness_valid",
                    data=bound_valid,
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
                    w_dtype = select_float_dtype(
                        w, self._float_atol, msg="(component weights)")
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

    def _load_existing_timescale_ids(self, path):
        """Read back which particle IDs already have a recorded timescale.

        Parameters
        ----------
        path : str

        Returns
        -------
        ids : set of int
        """
        if not os.path.exists(path):
            return set()
        existing = np.atleast_1d(np.loadtxt(path, usecols=0, dtype=np.int64))
        return set(existing.tolist())

    def write_timescales(self, particle_timescales):
        """Append newly-seen particle timescales to a text file.

        A particle's timescale is recorded once, the first time it is
        seen, and never rewritten afterward: rows already on disk are
        never touched again, so a later snapshot's batch (which may no
        longer include a particle that merged away) can never erase an
        earlier snapshot's record.

        Parameters
        ----------
        particle_timescales : ndarray of shape (n_particles,)
            Structured array with ``particle_index`` and ``timescale`` fields.
        """
        path = self._ts_path()
        if self._written_timescale_ids is None:
            self._written_timescale_ids = self._load_existing_timescale_ids(path)

        seen = self._written_timescale_ids
        is_new = np.array([pid not in seen for pid in particle_timescales["particle_index"]])
        new_records = particle_timescales[is_new]
        if new_records.size == 0:
            return

        write_header = not os.path.exists(path)
        with open(path, "a") as f:
            np.savetxt(f, new_records, fmt="%u\t%.6f",
                       header="particle_index\ttimescale" if write_header else "")
        seen.update(new_records["particle_index"].tolist())

import os

import h5py
import numpy as np

from roadrunner.helpers import select_float_dtype, select_uint_dtype


class HDF5AssignmentWriter:
    def __init__(self, output_dir, mode="w-", float_atol=1e-4):
        os.makedirs(output_dir, exist_ok=True)
        self._path = os.path.join(output_dir, "assignment.hdf5")
        self._mode = mode
        self._float_atol = float_atol
        self._first = True

    def _mode_for_write(self):
        mode = self._mode if self._first else "a"
        self._first = False
        return mode

    def _gal_uint_dtype(self, gids):
        return select_uint_dtype(int(max(gids)) if len(gids) > 0 else 1)

    def _pick_float_dtype(self, arr):
        max_abs = float(np.abs(arr).max()) if arr.size > 0 else 1.0
        return select_float_dtype(max_abs, self._float_atol)

    def write_snapshot(self, snapshot_id, time,
                       assignment_result, boundness_csc):
        with h5py.File(self._path, self._mode_for_write()) as hf:
            snap_grp = hf.require_group(f"/snapshots/{snapshot_id}")
            snap_grp.attrs["time"] = float(time)

            galaxies_grp = snap_grp.require_group("galaxies")

            resp_csc = assignment_result.responsibilities
            all_gids = list(resp_csc.column_id)
            uint_dtype = self._gal_uint_dtype(all_gids) if all_gids else np.uint32
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

            # Hard assignment
            hard_path = "/".join(["", "snapshots", str(snapshot_id), "hard_assignment"])
            if hard_path in hf:
                del hf[hard_path]
            snap_grp.create_dataset(
                "hard_assignment",
                data=assignment_result.particle_df.to_records(index=False),
                compression="gzip",
            )

    def write_timescales(self, particle_timescales):
        with h5py.File(self._path, "a") as hf:
            hdr = hf.require_group("header")
            ts_dtype = select_float_dtype(
                float(particle_timescales.max()) if particle_timescales.size > 0 else 1.0,
                self._float_atol,
            )
            if "timescales" in hdr:
                del hdr["timescales"]
            hdr.create_dataset(
                "timescales",
                data=particle_timescales.astype(ts_dtype, copy=False),
                compression="gzip",
            )

import numpy as np
from numba import njit, prange


def build_dense_from_csc(matrix_shape, true_indices, column_indices, column_values):
    n_samples, n_components = matrix_shape

    flat_candidates = np.concatenate(column_indices)
    flat_values = np.concatenate(column_values)

    offsets = np.zeros(n_components + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(c) for c in column_indices])

    return _csc_to_dense_kernel(
        matrix_shape, true_indices,
        flat_candidates, flat_values, offsets,
    )


@njit(parallel=True)
def _csc_to_dense_kernel(
    matrix_shape, true_indices, flat_indices, flat_values, offsets
):
    n_samples, n_components = matrix_shape

    imax = true_indices.max()
    idx_map = np.full(int(imax + 1), imax + 2, dtype=np.int64)
    idx_map[true_indices] = np.arange(true_indices.size)

    dense = np.zeros(matrix_shape, dtype=flat_values.dtype)
    for k in prange(n_components):
        start, end = offsets[k], offsets[k + 1]

        indices = flat_indices[start:end]
        values = flat_values[start:end]

        dense[idx_map[indices], k] = values

    return np.asarray(dense, dtype=flat_values.dtype)


@njit(parallel=True)
def stitch_zero_rows(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape, (
        f"Matrices cannot be stitched together: shape {matrix1.shape} "
        f"is incompatible with {matrix2.shape}."
    )
    nrows, ncols = matrix1.shape

    for i in prange(nrows):
        if matrix2[i, :].sum() > 0:
            continue
        matrix2[i, :] = matrix1[i, :]

    return matrix2


def _build_row_id(indices_list):
    all_rows = np.unique(np.concatenate(indices_list))
    return all_rows if all_rows.size > 0 else np.array([], dtype=np.uint64)


class SparseCSC:
    def __init__(self, column_indices, column_values, column_id=None):
        self.column_indices = column_indices
        self.column_values = column_values
        n = len(column_indices)
        self.column_id = (
            np.arange(n) if column_id is None
            else np.asarray(column_id, dtype=np.int64)
        )
        self.row_id = _build_row_id(column_indices)

    def __len__(self):
        return len(self.column_indices)

    def to_dense(self, columns=None, col_func=None):
        if columns is None:
            col_idx_list = self.column_indices
            col_val_list = self.column_values
        else:
            mask = np.isin(self.column_id, columns)
            if not np.any(mask):
                return np.zeros((self.row_id.size, 0), dtype=np.float32)
            idx = np.where(mask)[0]
            col_idx_list = [self.column_indices[i] for i in idx]
            col_val_list = [self.column_values[i] for i in idx]

        if self.row_id.size == 0:
            return np.zeros((0, len(col_idx_list)))

        if col_func is not None:
            col_val_list = [col_func(v) for v in col_val_list]

        return build_dense_from_csc(
            matrix_shape=(self.row_id.size, len(col_idx_list)),
            true_indices=self.row_id,
            column_indices=col_idx_list,
            column_values=col_val_list,
        )

    def align(self, other, how="left"):
        if how == "left":
            new_rows = self.row_id.copy()
            new_cols = self.column_id.copy()
        elif how == "both":
            new_rows = np.union1d(self.row_id, other.row_id).astype(np.int64)
            new_cols = np.union1d(self.column_id, other.column_id).astype(np.int64)
        else:
            raise ValueError(f"how must be 'left' or 'both', got '{how}'")

        if new_rows.size == 0 or new_cols.size == 0:
            empty = SparseCSC([], [], column_id=np.array([], dtype=np.int64))
            return empty, empty

        max_id = max(
            int(self.row_id.max()) if self.row_id.size > 0 else 0,
            int(other.row_id.max()) if other.row_id.size > 0 else 0,
        )
        row_map = np.full(max_id + 1, -1, dtype=np.int64)
        row_map[new_rows] = np.arange(new_rows.size)
        new_cols_arr = new_cols.copy()

        def _align_one(csc):
            col_map = {cid: i for i, cid in enumerate(csc.column_id)}
            new_idx, new_val = [], []
            for cid in new_cols_arr:
                k = col_map.get(cid)
                if k is None:
                    new_idx.append(np.array([], dtype=np.int64))
                    new_val.append(np.array([], dtype=np.float32))
                    continue
                old_idx = csc.column_indices[k]
                old_val = csc.column_values[k]
                mapped = row_map[old_idx]
                valid = mapped >= 0
                new_idx.append(old_idx[valid])
                new_val.append(old_val[valid])

            result = SparseCSC(new_idx, new_val, column_id=new_cols_arr)
            result.row_id = new_rows.copy()
            return result

        return _align_one(self), _align_one(other)

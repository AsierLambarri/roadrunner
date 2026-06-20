#############################################################################
#
# package:   roadrunner.clustering
# file:      sparse.py
# brief:     SparseCSC and SparseCSR matrix classes with label-preserving operations.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 may 2026 - Created
#            13 may 2026 - Last edit
#
#############################################################################

import numpy as np
from numba import njit, prange


def build_dense_from_csc(matrix_shape, true_indices, column_indices, column_values,
                         fill_value=0.0):
    n_samples, n_components = matrix_shape

    flat_candidates = np.concatenate(column_indices)
    flat_values = np.concatenate(column_values)

    offsets = np.zeros(n_components + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(c) for c in column_indices])

    return _csc_to_dense_kernel(
        matrix_shape, true_indices,
        flat_candidates, flat_values, offsets,
        fill_value,
    )


@njit(parallel=True, cache=True)
def _csc_to_dense_kernel(
    matrix_shape, true_indices, flat_indices, flat_values, offsets,
    fill_value,
):
    n_samples, n_components = matrix_shape

    imax = true_indices.max()
    idx_map = np.full(int(imax + 1), imax + 2, dtype=np.int64)
    idx_map[true_indices] = np.arange(true_indices.size)

    dense = np.full(matrix_shape, fill_value, dtype=flat_values.dtype)
    for k in prange(n_components):
        start, end = offsets[k], offsets[k + 1]

        indices = flat_indices[start:end]
        values = flat_values[start:end]

        dense[idx_map[indices], k] = values

    return np.asarray(dense, dtype=flat_values.dtype)


@njit(parallel=True, cache=True)
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
    if len(indices_list) == 0:
        return np.array([], dtype=np.uint64)
    non_empty = [c for c in indices_list if c.size > 0]
    if not non_empty:
        return np.array([], dtype=np.uint64)
    all_rows = np.unique(np.concatenate(non_empty))
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

    def to_dense(self, columns=None, col_func=None, fill_value=0.0):
        if columns is None:
            col_idx_list = self.column_indices
            col_val_list = self.column_values
        else:
            mask = np.isin(self.column_id, columns)
            if not np.any(mask):
                return np.full((self.row_id.size, 0), fill_value, dtype=np.float32)
            idx = np.where(mask)[0]
            col_idx_list = [self.column_indices[i] for i in idx]
            col_val_list = [self.column_values[i] for i in idx]

        if self.row_id.size == 0:
            return np.full((0, len(col_idx_list)), fill_value)

        if col_func is not None:
            col_val_list = [col_func(v) for v in col_val_list]

        return build_dense_from_csc(
            matrix_shape=(self.row_id.size, len(col_idx_list)),
            true_indices=self.row_id,
            column_indices=col_idx_list,
            column_values=col_val_list,
            fill_value=fill_value,
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

    def remap_rows(self, src_ids: np.ndarray, dst_ids: np.ndarray) -> "SparseCSC":
        idx_order = np.argsort(src_ids)
        sorted_src = src_ids[idx_order]
        sorted_dst = dst_ids[idx_order]

        new_indices = []
        new_values = []
        for idx_arr, val_arr in zip(self.column_indices, self.column_values):
            if idx_arr.size == 0:
                new_indices.append(np.array([], dtype=np.int64))
                new_values.append(np.array([], dtype=np.float32))
                continue
            pos = np.searchsorted(sorted_src, idx_arr)
            pos = np.clip(pos, 0, len(sorted_src) - 1)
            found = sorted_src[pos] == idx_arr
            new_indices.append(sorted_dst[pos[found]].astype(np.int64, copy=False))
            new_values.append(val_arr[found].copy())

        return SparseCSC(new_indices, new_values, column_id=self.column_id.copy())

    def to_csr(self) -> "SparseCSR":
        """Convert to SparseCSR for efficient row access."""
        if not self.column_indices or self.row_id.size == 0:
            return SparseCSR([], [], row_id=self.row_id.copy())

        flat_rows = np.concatenate(self.column_indices)
        flat_vals = np.concatenate(self.column_values)
        col_offsets = np.zeros(len(self.column_indices) + 1, dtype=np.int64)
        col_offsets[1:] = np.cumsum([len(c) for c in self.column_indices])

        # Map external row IDs → positional indices for the kernel
        max_row = int(self.row_id.max())
        row_to_pos = np.full(max_row + 1, -1, dtype=np.int64)
        row_to_pos[self.row_id] = np.arange(self.row_id.size)
        flat_rows_pos = row_to_pos[flat_rows]

        n_rows = self.row_id.size
        n_cols = len(self.column_indices)

        row_offsets, csr_cols, csr_vals = _csc_to_csr_kernel(
            flat_rows_pos, flat_vals, col_offsets, n_rows, n_cols,
        )

        # Map positional column indices → external column_id labels
        csr_cols = self.column_id[csr_cols]

        row_indices = [csr_cols[row_offsets[r]:row_offsets[r + 1]].copy()
                       for r in range(n_rows)]
        row_values = [csr_vals[row_offsets[r]:row_offsets[r + 1]].copy()
                      for r in range(n_rows)]

        return SparseCSR(row_indices, row_values, row_id=self.row_id.copy())


# ──────────────────────────────────────────────────────────────────────
# SparseCSR — row-major mirror of SparseCSC
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _csc_to_csr_kernel(flat_rows, flat_vals, col_offsets, n_rows, n_cols):
    """CSC → CSR via three-pass counting sort. O(nnz), no sorting."""
    row_counts = np.zeros(n_rows, dtype=np.int64)
    for i in range(len(flat_rows)):
        row_counts[flat_rows[i]] += 1

    row_offsets = np.zeros(n_rows + 1, dtype=np.int64)
    for i in range(n_rows):
        row_offsets[i + 1] = row_offsets[i] + row_counts[i]

    nnz = len(flat_rows)
    csr_cols = np.empty(nnz, dtype=np.int64)
    csr_vals = np.empty(nnz, dtype=flat_vals.dtype)
    row_counts[:] = 0
    for k in range(n_cols):
        for j in range(col_offsets[k], col_offsets[k + 1]):
            r = flat_rows[j]
            pos = row_offsets[r] + row_counts[r]
            csr_cols[pos] = k
            csr_vals[pos] = flat_vals[j]
            row_counts[r] += 1

    return row_offsets, csr_cols, csr_vals


@njit(cache=True)
def _csr_to_csc_kernel(flat_cols, flat_vals, row_offsets, n_cols, n_rows):
    """CSR → CSC via three-pass counting sort. O(nnz), no sorting."""
    col_counts = np.zeros(n_cols, dtype=np.int64)
    for i in range(len(flat_cols)):
        col_counts[flat_cols[i]] += 1

    col_offsets = np.zeros(n_cols + 1, dtype=np.int64)
    for k in range(n_cols):
        col_offsets[k + 1] = col_offsets[k] + col_counts[k]

    nnz = len(flat_cols)
    csc_rows = np.empty(nnz, dtype=np.int64)
    csc_vals = np.empty(nnz, dtype=flat_vals.dtype)
    col_counts[:] = 0
    for r in range(n_rows):
        for j in range(row_offsets[r], row_offsets[r + 1]):
            k = flat_cols[j]
            pos = col_offsets[k] + col_counts[k]
            csc_rows[pos] = r
            csc_vals[pos] = flat_vals[j]
            col_counts[k] += 1

    return col_offsets, csc_rows, csc_vals

def _build_column_id(row_indices):
    if len(row_indices) == 0:
        return np.array([], dtype=np.uint64)
    non_empty = [r for r in row_indices if r.size > 0]
    if not non_empty:
        return np.array([], dtype=np.uint64)
    all_cols = np.unique(np.concatenate(non_empty))
    return all_cols if all_cols.size > 0 else np.array([], dtype=np.uint64)


def build_dense_from_csr(matrix_shape, true_indices, row_indices, row_values,
                         fill_value=0.0):
    n_rows, n_columns = matrix_shape

    flat_indices = np.concatenate(row_indices)
    flat_values = np.concatenate(row_values)

    offsets = np.zeros(n_rows + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(r) for r in row_indices])

    return _csr_to_dense_kernel(
        matrix_shape, true_indices,
        flat_indices, flat_values, offsets,
        fill_value,
    )


@njit(parallel=True, cache=True)
def _csr_to_dense_kernel(
    matrix_shape, true_indices, flat_indices, flat_values, offsets,
    fill_value,
):
    n_rows, n_columns = matrix_shape

    imax = true_indices.max()
    idx_map = np.full(int(imax + 1), imax + 2, dtype=np.int64)
    idx_map[true_indices] = np.arange(true_indices.size)

    dense = np.full(matrix_shape, fill_value, dtype=flat_values.dtype)
    for i in prange(n_rows):
        start, end = offsets[i], offsets[i + 1]

        indices = flat_indices[start:end]
        values = flat_values[start:end]

        dense[i, idx_map[indices]] = values

    return np.asarray(dense, dtype=flat_values.dtype)


class SparseCSR:
    def __init__(self, row_indices, row_values, row_id=None):
        self.row_indices = row_indices
        self.row_values = row_values
        n = len(row_indices)
        self.row_id = (
            np.arange(n) if row_id is None
            else np.asarray(row_id, dtype=np.int64)
        )
        self.column_id = _build_column_id(row_indices)

    def __len__(self):
        return len(self.row_indices)

    def to_dense(self, rows=None, row_func=None, fill_value=0.0):
        if rows is None:
            row_idx_list = self.row_indices
            row_val_list = self.row_values
        else:
            mask = np.isin(self.row_id, rows)
            if not np.any(mask):
                return np.full((0, self.column_id.size), fill_value, dtype=np.float32)
            idx = np.where(mask)[0]
            row_idx_list = [self.row_indices[i] for i in idx]
            row_val_list = [self.row_values[i] for i in idx]

        if self.column_id.size == 0:
            return np.full((len(row_idx_list), 0), fill_value)

        if row_func is not None:
            row_val_list = [row_func(v) for v in row_val_list]

        return build_dense_from_csr(
            matrix_shape=(len(row_idx_list), self.column_id.size),
            true_indices=self.column_id,
            row_indices=row_idx_list,
            row_values=row_val_list,
            fill_value=fill_value,
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
            empty = SparseCSR([], [], row_id=np.array([], dtype=np.int64))
            return empty, empty

        max_id = max(
            int(self.column_id.max()) if self.column_id.size > 0 else 0,
            int(other.column_id.max()) if other.column_id.size > 0 else 0,
        )
        col_map = np.full(max_id + 1, -1, dtype=np.int64)
        col_map[new_cols] = np.arange(new_cols.size)

        def _align_one(csr):
            row_map = {rid: i for i, rid in enumerate(csr.row_id)}
            new_idx, new_val = [], []
            for rid in new_rows:
                i = row_map.get(rid)
                if i is None:
                    new_idx.append(np.array([], dtype=np.int64))
                    new_val.append(np.array([], dtype=np.float32))
                    continue
                old_idx = csr.row_indices[i]
                old_val = csr.row_values[i]
                mapped = col_map[old_idx]
                valid = mapped >= 0
                new_idx.append(old_idx[valid])
                new_val.append(old_val[valid])

            result = SparseCSR(new_idx, new_val, row_id=new_rows.copy())
            result.column_id = new_cols.copy()
            return result

        return _align_one(self), _align_one(other)

    def remap_rows(self, src_ids: np.ndarray, dst_ids: np.ndarray) -> "SparseCSR":
        idx_order = np.argsort(src_ids)
        sorted_src = src_ids[idx_order]
        sorted_dst = dst_ids[idx_order]

        pos = np.searchsorted(sorted_src, self.row_id)
        pos = np.clip(pos, 0, len(sorted_src) - 1)
        found = sorted_src[pos] == self.row_id
        new_row_id = self.row_id.copy()
        new_row_id[found] = sorted_dst[pos[found]]

        return SparseCSR(self.row_indices, self.row_values, row_id=new_row_id)

    def to_csc(self) -> "SparseCSC":
        """Convert to SparseCSC for efficient column access."""
        if not self.row_indices or self.column_id.size == 0:
            return SparseCSC([], [], column_id=self.column_id.copy())

        flat_cols = np.concatenate(self.row_indices)
        flat_vals = np.concatenate(self.row_values)
        row_offsets = np.zeros(len(self.row_indices) + 1, dtype=np.int64)
        row_offsets[1:] = np.cumsum([len(r) for r in self.row_indices])

        # Map external column IDs → positional indices for the kernel
        max_col = int(self.column_id.max())
        col_to_pos = np.full(max_col + 1, -1, dtype=np.int64)
        col_to_pos[self.column_id] = np.arange(self.column_id.size)
        flat_cols_pos = col_to_pos[flat_cols]

        n_cols = self.column_id.size
        n_rows = len(self.row_indices)

        col_offsets, csc_rows, csc_vals = _csr_to_csc_kernel(
            flat_cols_pos, flat_vals, row_offsets, n_cols, n_rows,
        )

        # Map positional row indices → external row_id labels
        csc_rows = self.row_id[csc_rows]

        column_indices = [csc_rows[col_offsets[k]:col_offsets[k + 1]].copy()
                          for k in range(n_cols)]
        column_values = [csc_vals[col_offsets[k]:col_offsets[k + 1]].copy()
                         for k in range(n_cols)]

        return SparseCSC(column_indices, column_values,
                         column_id=self.column_id.copy())

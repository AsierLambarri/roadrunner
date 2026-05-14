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


class SparseCSC:
    def __init__(self, column_indices, column_values):
        self.column_indices = column_indices
        self.column_values = column_values

    def to_dense(self, columns=None, col_func=None):
        if columns is None:
            col_indices = self.column_indices
            col_values = self.column_values
        else:
            col_indices = [self.column_indices[j] for j in columns]
            col_values = [self.column_values[j] for j in columns]

        true_indices = np.unique(np.concatenate(col_indices))
        if true_indices.size == 0:
            return np.zeros((0, len(col_indices)))

        if col_func is not None:
            col_values = [col_func(v) for v in col_values]

        return build_dense_from_csc(
            matrix_shape=(true_indices.size, len(col_indices)),
            true_indices=true_indices,
            column_indices=col_indices,
            column_values=col_values,
        )

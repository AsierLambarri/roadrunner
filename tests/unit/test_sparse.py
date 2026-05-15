import numpy as np

from roadrunner.clustering.sparse import (
    SparseCSC,
    build_dense_from_csc,
    stitch_zero_rows,
)


class TestBuildDenseFromCSC:
    def test_known_3x2(self):
        shape = (3, 2)
        true_indices = np.array([0, 1, 2])
        col_indices = [np.array([0, 1]), np.array([1, 2])]
        col_values = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = build_dense_from_csc(shape, true_indices, col_indices, col_values)
        expected = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
        assert np.array_equal(result, expected)

    def test_empty_column(self):
        shape = (2, 2)
        true_indices = np.array([0, 1])
        col_indices = [np.array([0]), np.array([], dtype=np.int64)]
        col_values = [np.array([5.0]), np.array([], dtype=np.float64)]
        result = build_dense_from_csc(shape, true_indices, col_indices, col_values)
        expected = np.array([[5.0, 0.0], [0.0, 0.0]])
        assert np.array_equal(result, expected)

    def test_sparse_true_indices(self):
        shape = (2, 1)
        true_indices = np.array([1, 3])
        col_indices = [np.array([1, 3])]
        col_values = [np.array([10.0, 20.0])]
        result = build_dense_from_csc(shape, true_indices, col_indices, col_values)
        expected = np.array([[10.0], [20.0]])
        assert np.array_equal(result, expected)

    def test_numba_matches_python(self):
        shape = (4, 2)
        true_indices = np.array([0, 1, 2, 3])
        col_indices = [np.array([0, 2]), np.array([1, 3])]
        col_values = [np.array([1.0, 3.0]), np.array([2.0, 4.0])]
        numba_result = build_dense_from_csc(shape, true_indices, col_indices, col_values)

        n_samples, n_components = shape
        py_dense = np.zeros(shape)
        for k in range(n_components):
            for idx, val in zip(col_indices[k], col_values[k]):
                row = np.where(true_indices == idx)[0][0]
                py_dense[row, k] = val

        assert np.array_equal(numba_result, py_dense)


class TestSparseCSC:
    def test_to_dense_all_columns(self):
        csc = SparseCSC(
            column_indices=[np.array([0, 1]), np.array([1, 2])],
            column_values=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        )
        result = csc.to_dense()
        expected = np.array([[1.0, 0.0], [2.0, 3.0], [0.0, 4.0]])
        assert np.array_equal(result, expected)

    def test_to_dense_subset(self):
        csc = SparseCSC(
            column_indices=[np.array([0, 1]), np.array([1, 2])],
            column_values=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
        )
        result = csc.to_dense(columns=[0])
        assert result.shape == (3, 1)
        assert list(result[:, 0]) == [1.0, 2.0, 0.0]

    def test_to_dense_subset_by_id(self):
        csc = SparseCSC(
            column_indices=[np.array([0, 1]), np.array([1, 2])],
            column_values=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            column_id=[10, 20],
        )
        result = csc.to_dense(columns=[20])
        assert result.shape == (3, 1)
        assert list(result[:, 0]) == [0.0, 3.0, 4.0]

    def test_to_dense_empty_column(self):
        csc = SparseCSC(
            column_indices=[np.array([], dtype=np.int64)],
            column_values=[np.array([], dtype=np.float64)],
        )
        result = csc.to_dense()
        assert result.shape == (0, 1)


class TestStitchZeroRows:
    def test_basic(self):
        m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        m2 = np.array([[0.0, 0.0], [5.0, 6.0]])
        result = stitch_zero_rows(m1, m2)
        expected = np.array([[1.0, 2.0], [5.0, 6.0]])
        assert np.array_equal(result, expected)

    def test_no_zero_rows(self):
        m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        m2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = stitch_zero_rows(m1, m2)
        assert np.array_equal(result, m2)

    def test_all_zero_rows(self):
        m1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        m2 = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = stitch_zero_rows(m1, m2)
        assert np.array_equal(result, m1)

    def test_numba_matches_python(self):
        m1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        m2 = np.array([[0.0, 0.0], [7.0, 8.0], [0.0, 0.0]])
        numba_result = stitch_zero_rows(m1, m2)

        py_result = m2.copy()
        for i in range(py_result.shape[0]):
            if py_result[i, :].sum() == 0:
                py_result[i, :] = m1[i, :]

        assert np.array_equal(numba_result, py_result)


class TestSparseCSCAlignBoth:
    def test_align_both_same_shape(self):
        a = SparseCSC(
            column_indices=[np.array([0, 1]), np.array([1, 2])],
            column_values=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            column_id=[10, 20],
        )
        b = SparseCSC(
            column_indices=[np.array([0, 2]), np.array([2, 3])],
            column_values=[np.array([5.0, 6.0]), np.array([7.0, 8.0])],
            column_id=[20, 30],
        )
        aa, bb = a.align(b, how="both")
        assert np.array_equal(aa.row_id, np.array([0, 1, 2, 3]))
        assert np.array_equal(bb.row_id, np.array([0, 1, 2, 3]))
        assert np.array_equal(aa.column_id, np.array([10, 20, 30]))
        assert np.array_equal(bb.column_id, np.array([10, 20, 30]))

    def test_align_both_commutative(self):
        a = SparseCSC(
            column_indices=[np.array([0, 1]), np.array([1, 2])],
            column_values=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            column_id=[10, 20],
        )
        b = SparseCSC(
            column_indices=[np.array([0, 2])],
            column_values=[np.array([5.0, 6.0])],
            column_id=[15],
        )
        aa1, bb1 = a.align(b, how="both")
        aa2, bb2 = b.align(a, how="both")
        assert np.array_equal(aa1.row_id, aa2.row_id)
        assert np.array_equal(aa1.column_id, aa2.column_id)
        assert np.array_equal(bb1.row_id, bb2.row_id)
        assert np.array_equal(bb1.column_id, bb2.column_id)

    def test_align_both_dense_values(self):
        a = SparseCSC(
            column_indices=[np.array([0, 1])],
            column_values=[np.array([1.0, 2.0])],
            column_id=[10],
        )
        b = SparseCSC(
            column_indices=[np.array([1, 2])],
            column_values=[np.array([3.0, 4.0])],
            column_id=[20],
        )
        aa, bb = a.align(b, how="both")
        aa_dense = aa.to_dense()
        bb_dense = bb.to_dense()
        assert aa_dense.shape == bb_dense.shape
        assert aa_dense.shape[0] == 3
        assert aa_dense.shape[1] == 2
        assert list(aa_dense[:, 0]) == [1.0, 2.0, 0.0]
        assert list(bb_dense[:, 1]) == [0.0, 3.0, 4.0]

    def test_align_both_empty_columns(self):
        a = SparseCSC(
            column_indices=[np.array([], dtype=np.uint64)],
            column_values=[np.array([], dtype=np.float32)],
            column_id=[10],
        )
        b = SparseCSC(
            column_indices=[np.array([0])],
            column_values=[np.array([5.0])],
            column_id=[20],
        )
        aa, bb = a.align(b, how="both")
        assert aa.row_id.size == 1
        assert bb.row_id.size == 1
        assert len(aa.column_id) == 2


class TestSparseCSCAlignLeft:
    def test_align_left_self_unchanged(self):
        a = SparseCSC(
            column_indices=[np.array([0, 1]), np.array([1, 2])],
            column_values=[np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            column_id=[10, 20],
        )
        b = SparseCSC(
            column_indices=[np.array([0, 2]), np.array([2, 3])],
            column_values=[np.array([5.0, 6.0]), np.array([7.0, 8.0])],
            column_id=[20, 30],
        )
        aa, bb = a.align(b, how="left")
        assert np.array_equal(aa.row_id, a.row_id)
        assert np.array_equal(aa.column_id, a.column_id)
        assert np.array_equal(bb.row_id, a.row_id)
        assert np.array_equal(bb.column_id, a.column_id)

    def test_align_left_default(self):
        a = SparseCSC(
            column_indices=[np.array([0, 1])],
            column_values=[np.array([1.0, 2.0])],
            column_id=[10],
        )
        # b has matching column 10 but with particle 2 not in a's rows
        b = SparseCSC(
            column_indices=[np.array([1, 2])],
            column_values=[np.array([3.0, 4.0])],
            column_id=[10],
        )
        aa, bb = a.align(b)  # default how="left"
        assert np.array_equal(aa.row_id, [0, 1])
        assert np.array_equal(aa.column_id, [10])
        assert np.array_equal(bb.row_id, [0, 1])
        assert np.array_equal(bb.column_id, [10])
        da = aa.to_dense()
        db = bb.to_dense()
        assert da.shape == (2, 1)
        assert db.shape == (2, 1)
        assert list(da[:, 0]) == [1.0, 2.0]
        # b has particle 1 (value 3) and particle 2 (dropped from row space).
        # After left alignment to a's row space [0,1]:
        #   row 0 = 0 (no particle 0 in b col 10)
        #   row 1 = 3.0 (particle 1)
        assert list(db[:, 0]) == [0.0, 3.0]

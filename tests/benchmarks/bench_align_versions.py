"""Benchmark three align() implementations."""

import time

import numpy as np

from roadrunner.clustering.sparse import SparseCSC

RNG = np.random.default_rng(42)

CONFIGS = [
    (10_000,   5,   0.3, "10kx5"),
    (50_000,   5,   0.3, "50kx5"),
    (100_000,  5,   0.3, "100kx5"),
    (10_000,   25,  0.3, "10kx25"),
    (50_000,   25,  0.3, "50kx25"),
    (100_000,  25,  0.3, "100kx25"),
    (10_000,   100, 0.3, "10kx100"),
    (50_000,   100, 0.3, "50kx100"),
]


def make_csc(N, K, fill=0.3, cid_offset=0):
    candidates, values = [], []
    cids = np.arange(K, dtype=np.int64) + cid_offset
    for k in range(K):
        nnz = max(1, int(N * fill))
        rows = RNG.choice(N, nnz, replace=False).astype(np.int64)
        vals = RNG.uniform(0, 1, nnz).astype(np.float32)
        candidates.append(np.sort(rows))
        values.append(vals)
    return SparseCSC(candidates, values, column_id=cids)


def align_current(self, other):
    new_rows = np.union1d(self.row_id, other.row_id).astype(np.int64)
    new_cols = np.union1d(self.column_id, other.column_id).astype(np.int64)
    def _align_one(csc):
        row_map = np.full(int(new_rows.max()) + 1 if new_rows.size > 0 else 1, -1, dtype=np.int64)
        row_map[new_rows] = np.arange(new_rows.size)
        new_idx, new_val = [], []
        for cid in new_cols:
            col_pos = np.where(csc.column_id == cid)[0]
            if col_pos.size > 0:
                k = col_pos[0]
                old_idx = csc.column_indices[k]
                old_val = csc.column_values[k]
                mapped = row_map[old_idx]
                valid = mapped >= 0
                new_idx.append(old_idx[valid])
                new_val.append(old_val[valid])
            else:
                new_idx.append(np.array([], dtype=np.int64))
                new_val.append(np.array([], dtype=np.float32))
        result = SparseCSC(new_idx, new_val, column_id=new_cols.copy())
        result.row_id = new_rows.copy()
        return result
    return _align_one(self), _align_one(other)


def align_searchsorted(self, other):
    new_rows = np.union1d(self.row_id, other.row_id).astype(np.int64)
    new_cols = np.union1d(self.column_id, other.column_id).astype(np.int64)
    if new_rows.size == 0:
        empty = SparseCSC([], [], column_id=np.array([], dtype=np.int64))
        return empty, empty
    row_map = np.full(int(new_rows.max()) + 1, -1, dtype=np.int64)
    row_map[new_rows] = np.arange(new_rows.size)
    new_cols_arr = new_cols.copy()
    def _align_one(csc):
        col_pos = np.searchsorted(new_cols_arr, csc.column_id)
        new_idx = [np.array([], dtype=np.int64) for _ in range(len(new_cols_arr))]
        new_val = [np.array([], dtype=np.float32) for _ in range(len(new_cols_arr))]
        for i, k in enumerate(col_pos):
            old_idx = csc.column_indices[i]
            old_val = csc.column_values[i]
            mapped = row_map[old_idx]
            valid = mapped >= 0
            new_idx[k] = old_idx[valid]
            new_val[k] = old_val[valid]
        result = SparseCSC(new_idx, new_val, column_id=new_cols_arr)
        result.row_id = new_rows.copy()
        return result
    return _align_one(self), _align_one(other)


def align_dict_lc(self, other):
    new_rows = np.union1d(self.row_id, other.row_id).astype(np.int64)
    new_cols = np.union1d(self.column_id, other.column_id).astype(np.int64)
    if new_rows.size == 0:
        empty = SparseCSC([], [], column_id=np.array([], dtype=np.int64))
        return empty, empty
    row_map = np.full(int(new_rows.max()) + 1, -1, dtype=np.int64)
    row_map[new_rows] = np.arange(new_rows.size)
    new_cols_arr = new_cols.copy()
    def _align_one(csc):
        col_map = {cid: i for i, cid in enumerate(csc.column_id)}
        def _col(cid):
            k = col_map.get(cid)
            if k is None:
                return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
            idx = csc.column_indices[k]
            val = csc.column_values[k]
            m = row_map[idx]
            v = m >= 0
            return (idx[v], val[v])
        pairs = [_col(cid) for cid in new_cols_arr]
        result = SparseCSC([p[0] for p in pairs], [p[1] for p in pairs], column_id=new_cols_arr)
        result.row_id = new_rows.copy()
        return result
    return _align_one(self), _align_one(other)


VER = [
    ("current (np.where)", align_current),
    ("searchsorted", align_searchsorted),
    ("dict+listcomp", align_dict_lc),
]

# Verify all produce the same result
N, K = 1000, 5
a = make_csc(N, K, 0.3, 0)
b = make_csc(N, K, 0.2, K // 2)
results = [fn(a, b) for _, fn in VER]
for i in range(1, len(results)):
    for j in range(2):
        r0 = results[0][j]
        ri = results[i][j]
        assert np.array_equal(r0.row_id, ri.row_id), f"row_id mismatch {VER[i][0]}"
        assert np.array_equal(r0.column_id, ri.column_id), f"col_id mismatch {VER[i][0]}"
        for k in range(len(r0.column_indices)):
            assert np.array_equal(r0.column_indices[k], ri.column_indices[k]), f"col_indices {k}"
            assert np.allclose(r0.column_values[k], ri.column_values[k]), f"col_values {k}"
print("All versions produce identical results.")

for N, K, fill, label in CONFIGS:
    a = make_csc(N, K, fill, 0)
    b = make_csc(N, K, fill * 0.6, K // 2)

    mem = 4 * N * K * 2 * 2 / 1024 / 1024
    if mem > 2000:
        print(f"\n{label:>15s}  SKIP (est. {mem:.0f} MB)")
        continue

    best_name = ""
    best_time = float("inf")
    times = []

    print(f"\n{label:>15s}  N={N}  K={K}:")
    for name, fn in VER:
        # warmup
        fn(a, b)
        ts = []
        for _ in range(15):
            t0 = time.perf_counter()
            fn(a, b)
            t1 = time.perf_counter()
            ts.append(t1 - t0)
        avg = np.mean(ts) * 1000
        std = np.std(ts) * 1000
        times.append((avg, name))
        print(f"    {name:>25s}:  {avg:8.2f} ms  ± {std:.2f}")

    best = min(times, key=lambda x: x[0])
    print(f"    {'winner':>25s}:  {best[1]}")

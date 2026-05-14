"""Benchmark SparseCSC.to_dense() and align() for various NxK sizes with fill < 0.5."""

import time

import numpy as np

from roadrunner.clustering.sparse import SparseCSC

RNG = np.random.default_rng(42)

CONFIGS = [
    (10_000,   5,   0.3, "10kx5"),
    (50_000,   5,   0.3, "50kx5"),
    (100_000,  5,   0.3, "100kx5"),
    (500_000,  5,   0.3, "500kx5"),
    (10_000,   25,  0.3, "10kx25"),
    (50_000,   25,  0.3, "50kx25"),
    (100_000,  25,  0.3, "100kx25"),
    (500_000,  25,  0.3, "500kx25"),
    (10_000,   100, 0.3, "10kx100"),
    (50_000,   100, 0.3, "50kx100"),
    (5_000,    500, 0.3, "5kx500"),
]


def make_csc(N, K, fill=0.3, cid_offset=0):
    candidates = []
    values = []
    cids = np.arange(K, dtype=np.int64) + cid_offset
    for k in range(K):
        nnz = max(1, int(N * fill))
        rows = RNG.choice(N, nnz, replace=False).astype(np.int64)
        vals = RNG.uniform(0, 1, nnz).astype(np.float32)
        candidates.append(np.sort(rows))
        values.append(vals)
    return SparseCSC(candidates, values, column_id=cids)


def bench(label, fn, rounds=20):
    # warmup
    fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    return avg, std


if __name__ == "__main__":
    print(f"{'Config':>15s}  {'N':>8s} {'K':>6s}  "
          f"{'to_dense ms':>11s} {'align ms':>11s}  "
          f"{'peak MB':>8s}")
    print("-" * 75)

    for N, K, fill, label in CONFIGS:
        a = make_csc(N, K, fill, cid_offset=0)
        b = make_csc(N, K, fill * 0.6, cid_offset=K // 2)

        mem_est = 4 * N * K * 2 * 2 / 1024 / 1024  # bound + prev + temp
        if mem_est > 3000:
            print(f"  {label:>15s}  SKIP (est. {mem_est:.0f} MB > 3 GB)")
            continue

        t_dense, s_dense = bench("to_dense", lambda: a.to_dense())

        def do_align():
            aa, bb = a.align(b, fill=True)
            return aa.to_dense(), bb.to_dense()

        t_align, s_align = bench("align", do_align)

        nrows = len(np.union1d(a.row_id, b.row_id))
        ncols = len(np.union1d(a.column_id, b.column_id))
        peak = 4 * nrows * ncols * 3 / 1024 / 1024  # 3 dense matrices

        print(f"  {label:>15s}  {N:>8d} {K:>6d}  "
              f"{t_dense:>10.2f} ±{s_dense:>5.2f}  "
              f"{t_align:>10.2f} ±{s_align:>5.2f}  "
              f"{peak:>7.0f}")

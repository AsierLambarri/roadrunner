"""Benchmark _merge_resp_kernel (Numba) vs 6-line numpy equivalent.

Configurations: N x K where peak memory < 4 GB (safe on 8 GB machine).
Each array is float32: 4 * N * K bytes.
Peak with kernel: ~8 * N * K (bound_dense + prev_dense).
Peak with numpy: ~17 * N * K (bound_dense + prev_dense + 2x where temps + masks).
"""

import time

import numpy as np
from numba import njit, prange

RNG = np.random.default_rng(42)

CONFIGS = [
    # (N, K, label)   — N >> K regime
    (100_000,   2,    "100kx2"),
    (500_000,   2,    "500kx2"),
    (1_000_000, 2,    "1Mx2"),
    (2_000_000, 2,    "2Mx2"),
    (500_000,   5,    "500kx5"),
    (1_000_000, 5,    "1Mx5"),
    (100_000,   10,   "100kx10"),
    (500_000,   10,   "500kx10"),
    (1_000_000, 10,   "1Mx10"),
    (100_000,   50,   "100kx50"),
    (500_000,   50,   "500kx50"),
    (100_000,   100,  "100kx100"),
    (50_000,    200,  "50kx200"),
]


@njit(parallel=True)
def _merge_resp_kernel(raw, bound, n_components, newborn_1d):
    n = raw.shape[0]
    inv_ncomp = 1.0 / n_components
    for n_idx in prange(n):
        is_newb = newborn_1d[n_idx]
        for k_idx in range(n_components):
            b = bound[n_idx, k_idx]
            p = raw[n_idx, k_idx]
            if b > 0:
                if p > 0:
                    raw[n_idx, k_idx] = p
                elif not is_newb:
                    raw[n_idx, k_idx] = inv_ncomp
                else:
                    raw[n_idx, k_idx] = 0.0
            else:
                raw[n_idx, k_idx] = 0.0


def numpy_dense_merge(prev_dense, bound_dense, newborn_1d, n_components):
    bound_mask = bound_dense > 0
    prev_mask = prev_dense > 0
    newborn_mask = newborn_1d[:, None]
    raw = np.where(bound_mask & prev_mask, prev_dense, 0.0).astype(np.float32)
    raw += np.where(
        bound_mask & ~prev_mask & ~newborn_mask,
        1.0 / n_components, 0.0,
    ).astype(np.float32)
    return raw


def bench_kernel(bound, prev, newborn, n_comp, rounds=20):
    # warmup
    p = prev.copy()
    _merge_resp_kernel(p, bound, n_comp, newborn)

    times = []
    for _ in range(rounds):
        p = prev.copy()
        t0 = time.perf_counter()
        _merge_resp_kernel(p, bound, n_comp, newborn)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_numpy(bound, prev, newborn, n_comp, rounds=20):
    # warmup
    _ = numpy_dense_merge(prev, bound, newborn, n_comp)

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        _ = numpy_dense_merge(prev, bound, newborn, n_comp)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


if __name__ == "__main__":
    print(f"{'Config':>15s}  {'N':>8s} {'K':>6s}  "
          f"{'kernel ms':>9s} {'kernel std':>10s}  "
          f"{'numpy ms':>9s} {'numpy std':>10s}  "
          f"{'speedup':>7s}")
    print("-" * 95)

    for N, K, label in CONFIGS:
        mem_numpy_mb = 4 * N * K * 17 / 1024 / 1024
        mem_kernel_mb = 4 * N * K * 8 / 1024 / 1024

        # Generate data
        bound = RNG.uniform(0, 1, (N, K)).astype(np.float32)
        prev = RNG.uniform(0, 1, (N, K)).astype(np.float32)
        newborn = RNG.random(N) < 0.1
        bound[RNG.random((N, K)) < 0.3] = 0.0
        prev[RNG.random((N, K)) < 0.5] = 0.0

        # Always run kernel
        mk, sk = bench_kernel(bound, prev, newborn, K)

        # Run numpy only if safe (< 4 GB peak)
        if mem_numpy_mb < 4000:
            mn, sn = bench_numpy(bound, prev, newborn, K)
            speedup = mn / mk if mk > 0 else float("inf")
            print(f"  {label:>15s}  {N:>8d} {K:>6d}  "
                  f"{mk:>9.3f} {sk:>10.3f}  "
                  f"{mn:>9.3f} {sn:>10.3f}  "
                  f"{speedup:>6.1f}x")
        else:
            print(f"  {label:>15s}  {N:>8d} {K:>6d}  "
                  f"{mk:>9.3f} {sk:>10.3f}  "
                  f"{'OOM SKIP':>9s} {'':>11s}  "
                  f"{'—':>7s}")

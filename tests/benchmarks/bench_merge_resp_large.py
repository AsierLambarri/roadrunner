"""Kernel-only benchmark at large regimes, respecting 8 GB RAM limit.

Peak memory ~ 8 * N * K (bound_dense + prev_dense).
Generation adds ~50% overhead, so target 8 * N * K < 4 GB → N*K < 500M.
"""

import time

import numpy as np
from numba import njit, prange

RNG = np.random.default_rng(42)

CONFIGS = [
    # (N, K, label)   — keep N*K < 120M to stay in 8 GB RAM
    (1_000_000, 100,   "1Mx100"),
    (1_500_000, 100,   "1.5Mx100"),
    (1_800_000, 100,   "1.8Mx100"),
    (2_000_000, 100,   "2Mx100"),
    (500_000,   200,   "500kx200"),
    (750_000,   200,   "750kx200"),
    (1_000_000, 200,   "1Mx200"),
    (100_000,   500,   "100kx500"),
    (150_000,   500,   "150kx500"),
    (200_000,   500,   "200kx500"),
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


def bench_kernel(bound, prev, newborn, n_comp, rounds=5):
    times = []
    for _ in range(rounds):
        p = prev.copy()
        t0 = time.perf_counter()
        _merge_resp_kernel(p, bound, n_comp, newborn)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del p
    return np.mean(times) * 1000, np.std(times) * 1000


if __name__ == "__main__":
    print(f"{'Config':>15s}  {'N':>10s} {'K':>6s} {'cells':>10s} "
          f"{'est.GB':>7s}  {'kernel ms':>9s} {'std':>7s}")
    print("-" * 75)

    for N, K, label in CONFIGS:
        cells = N * K
        est_gb = 8 * cells / 1024**3

        print(f"  {label:>15s}  {N:>10,d} {K:>6d} {cells:>10,d} "
              f"{est_gb:>6.2f}GB  ", end="", flush=True)

        bound = RNG.random((N, K), dtype=np.float32)
        prev = RNG.random((N, K), dtype=np.float32)
        newborn = RNG.random(N) < 0.1
        bound *= (RNG.random((N, K), dtype=np.float32) < 0.7).astype(np.float32)
        prev *= (RNG.random((N, K), dtype=np.float32) < 0.5).astype(np.float32)

        mk, sk = bench_kernel(bound, prev, newborn, K, rounds=10)

        print(f"{mk:>9.3f} {sk:>7.3f}")

        del bound, prev, newborn

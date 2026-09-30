"""Memory pressure test for _estimate_initial_params.

Traces peak RSS at key allocation points to find the true peak.
"""

import gc
import os

import numpy as np

from roadrunner.clustering.sparse import SparseCSC
from roadrunner.mixture._math import row_l1_normalize, row_squared_norms
from roadrunner.mixture.weighted_gmm import (
    _estimate_gaussian_parameters,
)
from roadrunner.clustering.assignment.gmm import _drop_unbound_kernel, _rows_to_pad_kernel


def rss_mb():
    """Return current RSS in MB."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


def make_csc(N, K, fill=0.3):
    candidates, values = [], []
    col_ids = np.arange(K, dtype=np.int64)
    for k in range(K):
        nnz = max(1, int(N * fill))
        rows = np.sort(np.random.choice(N, nnz, replace=False).astype(np.int64))
        vals = np.random.uniform(0, 1, nnz).astype(np.float32)
        candidates.append(rows)
        values.append(vals)
    return SparseCSC(candidates, values, column_id=col_ids)


CONFIGS = [
    ("1Mx10",   1_000_000, 10),
    ("500kx25",   500_000, 25),
    ("500kx50",   500_000, 50),
]

N_TRIALS = 3


def measure(label, N, K):
    print(f"\n{'='*60}")
    print(f"  {label}:  N={N:,}  K={K}")
    print(f"  Theoretical 8nk = {8 * 4 * N * K / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")

    # Build data
    csc_b = make_csc(N, K, fill=0.3)
    csc_prev = make_csc(N, K, fill=0.2)

    gc.collect()

    for trial in range(N_TRIALS):
        print(f"\n  Trial {trial + 1}:")
        gc.collect()

        # 1. baseline
        base = rss_mb()
        print(f"    baseline:                {base:>8.1f} MB")

        # 2. prior = row_l1_normalize(csc_b.to_dense(col_func=_rank_transform))
        from roadrunner.clustering.assignment.gmm import _rank_transform
        dense_raw = csc_b.to_dense(col_func=_rank_transform)
        r1 = rss_mb()
        print(f"    after to_dense(rank):    {r1 - base:>8.1f} MB  (peak: {r1:.1f})")

        prior = row_l1_normalize(dense_raw).astype(np.float32, copy=False)
        del dense_raw
        r2 = rss_mb()
        print(f"    after row_l1_norm + del: {r2 - base:>8.1f} MB")

        # 3. align
        _, aligned_p = csc_b.align(csc_prev, how="left")
        r3 = rss_mb()
        print(f"    after align:             {r3 - base:>8.1f} MB  (no N×K alloc)")

        # 4. prev_dense = aligned_p.to_dense()
        prev_dense = aligned_p.to_dense()
        r4 = rss_mb()
        print(f"    after prev_dense alloc:  {r4 - base:>8.1f} MB  "
              f"(peak: {r4:.1f})")
        print(f"      prior.shape={prior.shape}, prev_dense.shape={prev_dense.shape}")
        print(f"      csc_b.row_id.size={csc_b.row_id.size}")

        del aligned_p
        del csc_prev

        # 5. kernel (in-place on prev_dense)
        pad = _rows_to_pad_kernel(prev_dense, prior)
        del pad
        r5 = rss_mb()
        print(f"    after pad mask:          {r5 - base:>8.1f} MB")

        # 6. drop unbound entries (in-place on prev_dense)
        _drop_unbound_kernel(prev_dense, prior)
        resp = prev_dense
        r6 = rss_mb()
        print(f"    after drop (in-place):   {r6 - base:>8.1f} MB")

        # 7. row_l1_normalize (in-place on resp/prev_dense)
        resp = row_l1_normalize(resp)
        r7 = rss_mb()
        print(f"    after norm (in-place):   {r7 - base:>8.1f} MB")

        # 8. _estimate_gaussian_parameters
        coords = np.random.randn(prior.shape[0], 6).astype(np.float32)
        nk, means, covs = _estimate_gaussian_parameters(
            coords, resp, np.ones(prior.shape[0], dtype=np.float32),
            "full", 1e-6,
        )
        r8 = rss_mb()
        print(f"    after GMM param estim:   {r8 - base:>8.1f} MB")

        peak = max(r2, r4, r5, r6, r7, r8)
        theoretical = 8 * 4 * N * K / 1024 / 1024
        print(f"\n    >>> Measured peak:  {peak - base:.1f} MB "
              f"(theoretical 8nk: {theoretical:.1f} MB)")
        del prior, prev_dense, resp, coords, nk, means, covs
        gc.collect()

    del csc_b
    gc.collect()


if __name__ == "__main__":
    for label, N, K in CONFIGS:
        measure(label, N, K)

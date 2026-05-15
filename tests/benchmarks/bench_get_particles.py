"""Multi-run benchmark for HaloEnsemble.get_particles() three variants.

10 runs with varying (n_halos, n_starry, n_total_particles).
Each run: build dataset, time all 3 versions (10 rounds each).
"""

import time

import numpy as np

from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential

RNG = np.random.default_rng(42)

CONFIGS = [
    # (n_halos, n_starry, n_total_particles, label)
    (1000,  100,  1_000_000,   "1000h/100s/1Mp"),
    (1000,  500,  5_000_000,   "1000h/500s/5Mp"),
    (1000,  1000, 10_000_000,  "1000h/1ks/10Mp"),
    (2000,  200,  2_000_000,   "2000h/200s/2Mp"),
    (2000,  1000, 5_000_000,   "2000h/1ks/5Mp"),
    (2000,  2000, 10_000_000,  "2000h/2ks/10Mp"),
    (5000,  500,  5_000_000,   "5000h/500s/5Mp"),
    (5000,  2500, 10_000_000,  "5000h/2.5ks/10Mp"),
    (10000, 1000, 10_000_000,  "10kh/1ks/10Mp"),
    (10000, 5000, 20_000_000,  "10kh/5ks/20Mp"),
]


def _make_halo(n_particles, sid):
    inner = KeplerPotential(M=1e12, G=4.3e-6)
    h = HaloModel(
        inner,
        RNG.uniform(-100, 100, 3).astype(np.float64),
        np.zeros(3, dtype=np.float64),
        100.0,
        sub_tree_id=sid,
        redshift=0.0,
    )
    if n_particles > 0:
        h.set_boundness(
            np.arange(n_particles, dtype=np.uint64),
            RNG.uniform(0, 1, n_particles).astype(np.float32),
            RNG.uniform(0, 0.5, n_particles).astype(np.float32),
        )
    return h


def build_dataset(n_halos, n_starry, n_total):
    halos = []
    remaining = n_total
    for i in range(n_halos):
        if i < n_starry:
            n = remaining // (n_starry - i)
            halos.append(_make_halo(n, sid=i))
            remaining -= n
        else:
            halos.append(_make_halo(0, sid=i))
    return halos


def v1_for_loop(halos):
    candidates, boundness, tdyns = [], [], []
    for h in halos:
        if h.has_boundness:
            inds, ener, td_arr = h.get_boundness()
            candidates.append(inds)
            boundness.append(ener)
            tdyns.append(td_arr)
        else:
            e_i = np.array([], dtype=np.uint64)
            e_v = np.array([], dtype=np.float32)
            candidates.append(e_i)
            boundness.append(e_v)
            tdyns.append(e_v)
    return SparseCSC(candidates, boundness), SparseCSC(candidates, tdyns)


def v2_comp_3unpack(halos):
    data = [
        h.get_boundness() if h.has_boundness
        else (np.array([], dtype=np.uint64),
              np.array([], dtype=np.float32),
              np.array([], dtype=np.float32))
        for h in halos
    ]
    candidates = [d[0] for d in data]
    boundness = [d[1] for d in data]
    tdyns = [d[2] for d in data]
    return SparseCSC(candidates, boundness), SparseCSC(candidates, tdyns)


def v3_comp_zip(halos):
    raw = [
        h.get_boundness() if h.has_boundness
        else (np.array([], dtype=np.uint64),
              np.array([], dtype=np.float32),
              np.array([], dtype=np.float32))
        for h in halos
    ]
    candidates, boundness, tdyns = map(list, zip(*raw))
    return SparseCSC(candidates, boundness), SparseCSC(candidates, tdyns)


def bench_one(fn, halos, rounds=10):
    fn(halos)  # warmup
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn(halos)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


if __name__ == "__main__":
    print(f"{'Config':>30s}  {'n_halos':>7s} {'n_starry':>9s} {'n_part':>12s}  "
          f"{'v1 mean':>8s} {'v1 std':>7s}  {'v2 mean':>8s} {'v2 std':>7s}  "
          f"{'v3 mean':>8s} {'v3 std':>7s}  {'fastest':>8s}")
    print("-" * 145)

    all_results = []
    for n_halos, n_starry, n_total, label in CONFIGS:
        halos = build_dataset(n_halos, n_starry, n_total)

        m1, s1 = bench_one(v1_for_loop, halos)
        m2, s2 = bench_one(v2_comp_3unpack, halos)
        m3, s3 = bench_one(v3_comp_zip, halos)

        fastest = "v1" if m1 <= min(m2, m3) else ("v2" if m2 <= min(m1, m3) else "v3")

        print(f"{label:>30s}  {n_halos:>7d} {n_starry:>9d} {n_total:>12,d}  "
              f"{m1:>8.2f} {s1:>7.2f}  {m2:>8.2f} {s2:>7.2f}  "
              f"{m3:>8.2f} {s3:>7.2f}  {fastest:>8s}")

        all_results.append((label, m1, m2, m3))

    print("\nSummary: fastest version per config:")
    v1_wins = sum(1 for _, m1, m2, m3 in all_results if m1 <= min(m2, m3))
    v2_wins = sum(1 for _, m1, m2, m3 in all_results if m2 <= min(m1, m3))
    v3_wins = sum(1 for _, m1, m2, m3 in all_results if m3 <= min(m1, m2))
    print(f"  v1 (for-loop):      {v1_wins}/10")
    print(f"  v2 (comp+unpack):   {v2_wins}/10")
    print(f"  v3 (comp+zip):      {v3_wins}/10")

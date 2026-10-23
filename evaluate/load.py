"""Read finished catalogues, align them by particle ID, and spread work over processes.

A catalogue is either a roadrunner run (catalogue, merger tree, hard labels, stored pairs, Gaussians,
particles) or an ACCME result (hard labels, galaxy properties, satellite relations, births). Both use
the tree's ``Sub_tree_id``.

Particle IDs are ``particle_index`` in the outputs. Hard labels are ``Sub_tree_id``, ``UNASSIGNED``
for particles bound to no halo; ``ABSENT`` marks an ID missing from a snapshot after alignment.
"""
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property, lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

UNASSIGNED, ABSENT = -1, -2
RESOLVED, MAJOR = 100, 1000           # particles: a resolved / a major galaxy
OLD_GYR = 1.0                         # a star is old when it already existed this long before
DUPLICATE_GYR = 1e-3                  # an output closer than this to the previous one is a duplicate
BLAS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class Catalogue:
    """What every catalogue offers: ``name``, ``snaps``, ``time``, ``host``, ``labels(s)``,
    ``galaxies(s)``, ``satellites(s)``, ``births()``; ``hosts(s)`` is derived from them."""

    def hosts(self, s):
        """Satellite -> the set of hosts listing it at ``s``."""
        out = {}
        for h, sats in self.satellites(s).items():
            for g in sats:
                out.setdefault(g, set()).add(h)
        return out


class Run(Catalogue):
    """One roadrunner run's outputs.

    Parameters
    ----------
    path : str or Path
        Run directory (``catalogue.hdf5``, ``assignment/``, ``run.log``, ...).
    particles : str or Path, optional
        Directory whose ``particle_data/`` holds this run's particles (the same simulation);
        default ``path`` when it has one.
    """

    def __init__(self, path, particles=None):
        self.path = Path(path)
        self.name = self.path.name
        self.cat = h5py.File(self.path / "catalogue.hdf5", "r")
        self.snaps = np.sort(self.cat["header/snapshots"][:].astype(int))
        self.config = json.loads(self.cat["header/config"][()])
        self.host = int(self.cat["header/accretion_id"][()])
        eq = pd.DataFrame(json.loads(self.cat["header/equivalence"][()]))
        self.time = dict(zip(eq["snapshot"].astype(int), eq["time"].astype(float)))
        self.redshift = dict(zip(eq["snapshot"].astype(int), eq["redshift"].astype(float)))
        self.cosmology = dict(self.cat["header/cosmology"].attrs)
        pdata = Path(particles or path) / "particle_data"
        self.pdata = pdata if pdata.is_dir() else None

    @cached_property
    def tree(self):
        """The merger tree as the pipeline used it (read on first use: workers that skip it save memory)."""
        tree = pd.DataFrame(self.cat["header/merger_tree"][:])
        tree["Snapshot"] = tree["Snapshot"].astype(int)
        return tree

    @property
    def source(self):
        """The directory holding ``particle_data/`` (None without one): what a worker reopens."""
        return self.pdata.parent if self.pdata is not None else None

    def assignment_file(self, s):
        return self.path / "assignment" / f"snapshot{s:04d}.hdf5"

    def galaxies(self, s):
        """Catalogue galaxy properties at ``s``, indexed by ``Sub_tree_id``."""
        return pd.DataFrame(self.cat[f"snapshots/{s}/galaxy_properties"][:]).set_index("Sub_tree_id")

    def satellites(self, s):
        """Host -> its satellites at ``s``. A satellite is any less massive halo bound to the host inside
        its R_vir (the tree's relations), so it can be listed under several hosts."""
        rel = self.cat[f"snapshots/{s}/satellite_relations"]
        return {int(h): set(rel[h][:].astype(np.int64).tolist()) for h in rel}

    def row_labels(self, s):
        """``(ids, labels)`` in the snapshot's row order (the order of the stored pairs' rows)."""
        with h5py.File(self.assignment_file(s), "r") as f:
            h = f["hard_assignment"][:]
        return h["particle_index"].astype(np.int64), h["Sub_tree_id"].astype(np.int64)

    @lru_cache(maxsize=6)
    def labels(self, s):
        """``(ids, labels)`` sorted by ID."""
        ids, lab = self.row_labels(s)
        o = np.argsort(ids)
        return ids[o], lab[o]

    @lru_cache(maxsize=6)
    def masses(self, s):
        """Particle masses parallel to ``labels(s)`` (NaN where missing); ones (counts) without particle data."""
        ids = self.labels(s)[0]
        if self.pdata is None:
            return np.ones(ids.size)
        with h5py.File(self.pdata / f"snapshot{s:04d}.hdf5", "r") as f:
            pid = f["data/indices"][:].astype(np.int64)
            m = f["data/masses"][:].astype(np.float64)
        o = np.argsort(pid)
        return on(ids, pid[o], m[o], fill=np.nan)

    def pairs(self, s):
        """Every stored (particle, galaxy) pair at ``s``, concatenated (float64):

        ``row``, ``galaxy``, ``log_resp``, ``energy`` (physical, ``-boundness x energy_scale``; NaN where
        not valid) and ``log_alpha`` (NaN when the run did not store it).
        """
        keys = ("row", "galaxy", "log_resp", "energy", "log_alpha")
        parts = []
        with h5py.File(self.assignment_file(s), "r") as f:
            for k, g in f["galaxies"].items():
                n = g["indices"].size
                b = np.where(g["boundness_valid"][:], g["boundness"][:].astype(np.float64), np.nan)
                parts.append((
                    g["indices"][:].astype(np.int64),
                    np.full(n, int(k), np.int64),
                    g["log_resp"][:].astype(np.float64),
                    -b * g.attrs.get("energy_scale", np.nan),
                    g["log_alpha"][:].astype(np.float64) if "log_alpha" in g else np.full(n, np.nan),
                ))
        if not parts:
            return {key: np.array([]) for key in keys}
        return {key: np.concatenate(col) for key, col in zip(keys, zip(*parts))}

    def gaussians(self, s):
        """``{galaxy: (mean, covariance, weight, count)}`` for galaxies with stored parameters
        (catalogue units; ``count`` the fitted effective count, NaN when not stored)."""
        with h5py.File(self.assignment_file(s), "r") as f:
            g = f["galaxies"]
            return {
                int(k): (
                    g[k]["mean"][:].astype(np.float64),
                    g[k]["covariance"][()].astype(np.float64),
                    float(g[k]["weight"][0]),
                    float(g[k]["count"][0]) if "count" in g[k] else np.nan,
                )
                for k in g
                if "mean" in g[k] and "covariance" in g[k] and "weight" in g[k]
            }

    def births(self):
        """``(ids, birth galaxy)`` sorted by ID, from the final birth table (negative: unknown)."""
        b = self.cat["final/births"][:]
        o = np.argsort(b["particle_index"])
        return b["particle_index"].astype(np.int64)[o], b["birth_id"].astype(np.int64)[o]

    @lru_cache(maxsize=2)
    def particles_by_id(self, s):
        """``(ids, X, m)`` of the particle data at ``s``, sorted by ID (catalogue units, km/s)."""
        with h5py.File(self.pdata / f"snapshot{s:04d}.hdf5", "r") as f:
            ids = f["data/indices"][:].astype(np.int64)
            X = np.hstack([f["data/positions"][:], f["data/velocities"][:]]).astype(np.float64)
            X = X / f["data/scaler/scale"][:] + f["data/scaler/mean"][:]
            m = f["data/masses"][:].astype(np.float64)
        o = np.argsort(ids)
        return ids[o], X[o], m[o]

    def fitted(self, s):
        """Galaxies with stored pairs at ``s``: they had bound stars and were fitted (sizes only, no data read)."""
        with h5py.File(self.assignment_file(s), "r") as f:
            return {int(k) for k, g in f["galaxies"].items() if g["indices"].size}

    def ids(self, s):
        """Particle IDs at ``s`` (from the particle data when there is one: lighter than the labels)."""
        if self.pdata is None:
            return self.row_labels(s)[0]
        with h5py.File(self.pdata / f"snapshot{s:04d}.hdf5", "r") as f:
            return f["data/indices"][:].astype(np.int64)

    def particles(self, s):
        """Phase-space coordinates (n, 6; catalogue units, km/s) and masses, in this run's row order."""
        with h5py.File(self.pdata / f"snapshot{s:04d}.hdf5", "r") as f:
            pid = f["data/indices"][:].astype(np.int64)
            X = np.hstack([f["data/positions"][:], f["data/velocities"][:]]).astype(np.float64)
            X = X / f["data/scaler/scale"][:] + f["data/scaler/mean"][:]
            m = f["data/masses"][:].astype(np.float64)
        ids = self.row_labels(s)[0]
        if not np.array_equal(ids, pid):
            o = np.argsort(pid)
            perm = o[np.searchsorted(pid, ids, sorter=o)]
            X, m = X[perm], m[perm]
        return X, m


class Accme(Catalogue):
    """An ACCME result directory (``catalogues.hdf5``): per-snapshot hard labels, galaxy properties,
    satellite relations and the final birth table."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = f"accme-{self.path.name}"
        self.cat = h5py.File(self.path / "catalogues.hdf5", "r")
        self.snaps = np.array(sorted(int(k) for k in self.cat["catalogue"]))
        self.host = int(self.cat["header/id"][()])
        eq = self.cat["header/equiv"][:]
        self.time = dict(zip(eq["snapshot"].astype(int), eq["time"].astype(float)))
        props = pd.DataFrame(self.cat["properties/galaxy_properties"][:])
        props["Sub_tree_id"] = props["Sub_tree_id"].astype(np.int64)
        self._props = {
            int(s): g.drop(columns=["index", "Snapshot", "Time"]).set_index("Sub_tree_id")
            for s, g in props.groupby("Snapshot")
        }

    @lru_cache(maxsize=6)
    def labels(self, s):
        """``(ids, labels)`` sorted by ID."""
        c = self.cat[f"catalogue/{s}"][:]
        o = np.argsort(c["particle_index"])
        return c["particle_index"].astype(np.int64)[o], c["Sub_tree_id"].astype(np.int64)[o]

    def galaxies(self, s):
        return self._props.get(int(s), pd.DataFrame(columns=["Mtot"]))

    def satellites(self, s):
        rel = self.cat[f"satellite_relations/{s}"]
        return {int(h): set(rel[h][:].astype(np.int64).tolist()) for h in rel}

    def births(self):
        b = self.cat["reduced_catalogue/births"][:]
        o = np.argsort(b["particle_index"])
        return b["particle_index"].astype(np.int64)[o], b["birth_id"].astype(np.int64)[o]


def open_catalogue(path, particles=None):
    """A :class:`Run`, or an :class:`Accme` when ``path`` holds ``catalogues.hdf5``."""
    if (Path(path) / "catalogues.hdf5").exists():
        return Accme(path)
    return Run(path, particles=particles)


def on(ids_ref, ids_sorted, x_sorted, fill=ABSENT):
    """``x`` (parallel to the sorted ``ids_sorted``) read at ``ids_ref``; ``fill`` where the ID is absent.

    Fast when ``ids_ref`` is sorted too (the ``Run.labels`` order).
    """
    pos = np.minimum(np.searchsorted(ids_sorted, ids_ref), ids_sorted.size - 1)
    return np.where(ids_sorted[pos] == ids_ref, x_sorted[pos], fill)


def label_mass(L, m):
    """Hard-label mass of every label."""
    g, inv = np.unique(L, return_inverse=True)
    return pd.Series(np.bincount(inv.ravel(), weights=m), index=g)


def log_ratio(a, b):
    """``log10(a / b)``, NaN where either is not positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a > 0) & (b > 0), np.log10(a / b), np.nan)


def wmedian(x, w):
    """The ``w``-weighted median of ``x``."""
    if not x.size:
        return np.nan
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def common_snaps(cats, particles):
    """Snapshots in every catalogue and in the particle data, sorted, without near-duplicate outputs: a
    snapshot less than DUPLICATE_GYR after the previous one kept is left out (halo685x095's 178 comes
    0.6 Myr after 177, and the merger tree loses 340 of its halos there, live satellites among them)."""
    snaps = set(particles.snaps.tolist())
    for c in cats:
        snaps &= set(np.asarray(c.snaps).tolist())
    kept = []
    for s in sorted(snaps):
        if not kept or particles.time[s] - particles.time[kept[-1]] >= DUPLICATE_GYR:
            kept.append(s)
    return kept


def aligned(cats, particles, s):
    """Particles at ``s`` that every catalogue holds: IDs (sorted), X, m, each catalogue's labels
    (n_catalogues, n), and their share of the particle data."""
    ids, X, m = particles.particles_by_id(s)
    L = np.stack([on(ids, *c.labels(s)) for c in cats])
    keep = (L != ABSENT).all(axis=0)
    return ids[keep], X[keep], m[keep], L[:, keep], float(keep.mean())


def member(keys, q):
    """Whether each of ``q`` is in the sorted ``keys``."""
    if keys.size == 0:
        return np.zeros(np.shape(q), bool)
    return keys[np.minimum(np.searchsorted(keys, q), keys.size - 1)] == q


def seen_before(run, s, ids, gyr):
    """Whether each of ``ids`` already existed ``gyr`` before ``s``: present at the last snapshot at
    least ``gyr`` earlier. Stars persist once formed, so this is "first seen by then", with no table
    indexed by particle ID."""
    t = np.array([run.time[x] for x in run.snaps])
    earlier = run.snaps[t <= run.time[s] - gyr]
    if not earlier.size:
        return np.zeros(np.shape(ids), bool)
    return member(np.sort(run.ids(earlier[-1])), ids)


def newborn(run, s, ids):
    """Whether each of ``ids`` is first seen at ``s``: absent from the previous snapshot (all of them at
    the first snapshot)."""
    k = int(np.searchsorted(run.snaps, s))
    if not k:
        return np.ones(np.shape(ids), bool)
    return ~member(np.sort(run.ids(run.snaps[k - 1])), ids)


def pool(fn, items, workers, init, initargs=(), desc=None):
    """``[fn(x) for x in items]``, in order, over ``workers`` processes (here when ``workers <= 1``).

    Processes, not threads: h5py holds the GIL while it reads. They are spawned, and each runs
    ``init(*initargs)`` once to open its own files (HDF5 handles cannot cross processes); their numpy
    runs single-threaded, so ``workers`` processes use ``workers`` cores. With ``desc``, a progress bar
    when stderr is a terminal, and one line when the step is done.
    """
    items = list(items)
    start = time.time()
    bar = dict(total=len(items), desc=desc, disable=desc is None or not sys.stderr.isatty())
    if workers <= 1 or len(items) <= 1:
        init(*initargs)
        results = [fn(x) for x in tqdm(items, **bar)]
    else:
        saved = {k: os.environ.get(k) for k in BLAS}
        os.environ.update(dict.fromkeys(BLAS, "1"))
        try:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(min(workers, len(items)), mp_context=context,
                                     initializer=init, initargs=initargs) as ex:
                results = list(tqdm(ex.map(fn, items), **bar))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k)
                else:
                    os.environ[k] = v
    if desc is not None:
        print(f"{desc}: {len(items)} done in {time.time() - start:.0f} s", flush=True)
    return results


def mergers(tree):
    """Tree mergers as rows ``(progenitor, into, snapshot)``: a branch whose descendant is another
    branch's row. Rows with NaN ``uid`` / ``desc_uid`` cannot be followed and give no merger."""
    t = tree.dropna(subset=["uid", "desc_uid"])[["uid", "desc_uid", "Sub_tree_id", "Snapshot"]]
    d = t.merge(t, left_on="desc_uid", right_on="uid", suffixes=("", "_d"))
    d = d[d["Sub_tree_id"] != d["Sub_tree_id_d"]]
    return pd.DataFrame({
        "progenitor": d["Sub_tree_id"].to_numpy(np.int64),
        "into": d["Sub_tree_id_d"].to_numpy(np.int64),
        "snapshot": d["Snapshot_d"].to_numpy(np.int64),
    })

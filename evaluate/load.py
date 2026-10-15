"""Read finished catalogues: a roadrunner run (catalogue, merger tree, hard labels, stored pairs,
Gaussians, particles) or an ACCME result (hard labels, galaxy properties, relations, births).

Particle IDs are ``particle_index`` in the outputs. Hard labels are ``Sub_tree_id``,
``UNASSIGNED`` for particles bound to no halo; ``ABSENT`` marks an ID missing
from a snapshot after alignment. Both kinds share the tree's ``Sub_tree_id``.
"""
import json
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

UNASSIGNED, ABSENT = -1, -2
RESOLVED, MAJOR = 100, 1000           # particles: resolved / major galaxy


class Catalogue:
    """What every catalogue offers: ``name``, ``snaps``, ``time``, ``host``, ``labels(s)``,
    ``galaxies(s)``, ``satellites(s)``, ``births()``; ``hosts(s)`` is derived."""

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
        Run directory whose ``particle_data/`` holds this run's particles
        (the same simulation); default ``path`` when it has one.
    """

    def __init__(self, path, particles=None):
        self.path = Path(path)
        self.name = self.path.name
        self.cat = h5py.File(self.path / "catalogue.hdf5", "r")
        self.snaps = np.sort(self.cat["header/snapshots"][:].astype(int))
        self.config = json.loads(self.cat["header/config"][()])
        self.host = int(self.cat["header/accretion_id"][()])
        self.tree = pd.DataFrame(self.cat["header/merger_tree"][:])
        self.tree["Snapshot"] = self.tree["Snapshot"].astype(int)
        eq = pd.DataFrame(json.loads(self.cat["header/equivalence"][()]))
        self.time = dict(zip(eq["snapshot"].astype(int), eq["time"].astype(float)))
        self.redshift = dict(zip(eq["snapshot"].astype(int), eq["redshift"].astype(float)))
        self.cosmology = dict(self.cat["header/cosmology"].attrs)
        pdata = Path(particles or path) / "particle_data"
        self.pdata = pdata if pdata.is_dir() else None

    def assignment_file(self, s):
        return self.path / "assignment" / f"snapshot{s:04d}.hdf5"

    def galaxies(self, s):
        """Catalogue galaxy properties at ``s``, indexed by ``Sub_tree_id``."""
        return pd.DataFrame(self.cat[f"snapshots/{s}/galaxy_properties"][:]).set_index("Sub_tree_id")

    def satellites(self, s):
        """Host -> its satellites at ``s``: every alive halo is a key; a satellite is any less massive
        halo bound to the host inside its R_vir, so it can be listed under several hosts."""
        rel = self.cat[f"snapshots/{s}/satellite_relations"]
        return {int(h): set(rel[h][:].astype(np.int64).tolist()) for h in rel}

    def row_labels(self, s):
        """``(ids, labels)`` in the snapshot's row order (the order of the responsibilities' rows)."""
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
        """Particle masses parallel to ``labels(s)`` (NaN if missing); ones (counts) without particle data."""
        ids = self.labels(s)[0]
        if self.pdata is None:
            return np.ones(ids.size)
        with h5py.File(self.pdata / f"snapshot{s:04d}.hdf5", "r") as f:
            pid, m = f["data/indices"][:].astype(np.int64), f["data/masses"][:].astype(np.float64)
        o = np.argsort(pid)
        return on(ids, pid[o], m[o], fill=np.nan)

    def pairs(self, s):
        """Every stored (particle, galaxy) pair at ``s``, concatenated (float64): ``row``, ``galaxy``,
        ``log_resp``, ``energy`` (physical, ``-boundness x energy_scale``; NaN where not valid) and
        ``log_alpha`` (NaN when the run did not store it)."""
        keys = ("row", "galaxy", "log_resp", "energy", "log_alpha")
        parts = []
        with h5py.File(self.assignment_file(s), "r") as f:
            for k, g in f["galaxies"].items():
                n = g["indices"].size
                b = np.where(g["boundness_valid"][:], g["boundness"][:].astype(np.float64), np.nan)
                parts.append((g["indices"][:].astype(np.int64), np.full(n, int(k), np.int64),
                              g["log_resp"][:].astype(np.float64), -b * g.attrs.get("energy_scale", np.nan),
                              g["log_alpha"][:].astype(np.float64) if "log_alpha" in g else np.full(n, np.nan)))
        return {key: np.concatenate(col) for key, col in zip(keys, zip(*parts))} if parts else \
            {key: np.array([]) for key in keys}

    def gaussians(self, s):
        """``{galaxy: (mean, covariance, weight)}`` for galaxies with stored parameters (physical units)."""
        with h5py.File(self.assignment_file(s), "r") as f:
            g = f["galaxies"]
            return {int(k): (g[k]["mean"][:].astype(np.float64), g[k]["covariance"][()].astype(np.float64),
                             float(g[k]["weight"][0]))
                    for k in g if "mean" in g[k] and "covariance" in g[k] and "weight" in g[k]}

    def births(self):
        """``(ids, birth galaxy)`` sorted by ID, from the final birth table (negative: unknown)."""
        b = self.cat["final/births"][:]
        o = np.argsort(b["particle_index"])
        return b["particle_index"].astype(np.int64)[o], b["birth_id"].astype(np.int64)[o]

    @lru_cache(maxsize=2)
    def particles_by_id(self, s):
        """``(ids, X, m)`` of the particle data at ``s`` sorted by ID: catalogue units, km/s."""
        with h5py.File(self.pdata / f"snapshot{s:04d}.hdf5", "r") as f:
            ids = f["data/indices"][:].astype(np.int64)
            X = np.hstack([f["data/positions"][:], f["data/velocities"][:]]).astype(np.float64)
            X = X / f["data/scaler/scale"][:] + f["data/scaler/mean"][:]
            m = f["data/masses"][:].astype(np.float64)
        o = np.argsort(ids)
        return ids[o], X[o], m[o]

    def ids(self, s):
        """Particle IDs at ``s`` (particle data when available: lighter than the labels)."""
        if self.pdata is None:
            return self.row_labels(s)[0]
        with h5py.File(self.pdata / f"snapshot{s:04d}.hdf5", "r") as f:
            return f["data/indices"][:].astype(np.int64)

    def particles(self, s):
        """Phase-space coordinates (n, 6; catalogue units, km/s) and masses in this run's row order."""
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
        self._props = {int(s): g.drop(columns=["index", "Snapshot", "Time"]).set_index("Sub_tree_id")
                       for s, g in props.groupby("Snapshot")}

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
    return Accme(path) if (Path(path) / "catalogues.hdf5").exists() else Run(path, particles=particles)


def on(ids_ref, ids_sorted, x_sorted, fill=ABSENT):
    """``x`` (parallel to the sorted ``ids_sorted``) read at ``ids_ref``; ``fill`` where the ID is absent.

    Fast when ``ids_ref`` is sorted too (``Run.labels`` order).
    """
    pos = np.minimum(np.searchsorted(ids_sorted, ids_ref), ids_sorted.size - 1)
    return np.where(ids_sorted[pos] == ids_ref, x_sorted[pos], fill)


def common_snaps(cats, particles):
    """Snapshots in every catalogue and in the particle data, sorted."""
    snaps = set(particles.snaps.tolist())
    for c in cats:
        snaps &= set(np.asarray(c.snaps).tolist())
    return sorted(snaps)


def aligned(cats, particles, s):
    """Particles at ``s`` that every catalogue holds: IDs (sorted), X, m, each catalogue's labels
    (n_catalogues, n), and their share of the particle data."""
    ids, X, m = particles.particles_by_id(s)
    L = np.stack([on(ids, *c.labels(s)) for c in cats])
    keep = (L != ABSENT).all(axis=0)
    return ids[keep], X[keep], m[keep], L[:, keep], float(keep.mean())


def first_seen(run, last):
    """First snapshot in which each particle ID appears, up to ``last`` (-1 for IDs not seen)."""
    first = np.full(0, -1, np.int64)
    for s in run.snaps[run.snaps <= last]:
        ids = run.ids(s)
        if ids.max() >= first.size:
            first = np.concatenate([first, np.full(int(ids.max()) + 1 - first.size, -1, np.int64)])
        first[ids] = np.where(first[ids] < 0, s, first[ids])
    return first


def mergers(tree):
    """Tree mergers as rows ``(progenitor, into, snapshot)``: a branch whose descendant is another branch's row.

    Rows with NaN ``uid`` / ``desc_uid`` cannot be followed and give no merger.
    """
    t = tree.dropna(subset=["uid", "desc_uid"])[["uid", "desc_uid", "Sub_tree_id", "Snapshot"]]
    d = t.merge(t, left_on="desc_uid", right_on="uid", suffixes=("", "_d"))
    d = d[d["Sub_tree_id"] != d["Sub_tree_id_d"]]
    return pd.DataFrame({"progenitor": d["Sub_tree_id"].to_numpy(np.int64),
                         "into": d["Sub_tree_id_d"].to_numpy(np.int64),
                         "snapshot": d["Snapshot_d"].to_numpy(np.int64)})


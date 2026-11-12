"""Per-star energies, owners and host-frame positions of one snapshot,
computed in memory from the roadrunner output (runs where it lives). Feeds
the simulation maps (plots/sim/s01_*, s02_* through replay/simmap.py).

    from replay.starmap import StarMaps
    sm = StarMaps(rr_out, particle_dir)   # lazy: nothing read yet
    d = sm.at(87)                         # dict of per-star arrays (cached)

Per star (stars within r_keep host R_vir), from roadrunner's stored
energies (assignment/snapshotNNNN.hdf5, E = -boundness * energy_scale, used
where boundness_valid, i.e. the star is in that halo's bound set; where the
scale was stored as inf the energies are recomputed from particle_data,
with a warning: replay.energies):
  E_H        energy relative to the central host (NaN if not bound/stored)
  E_S        lowest stored energy relative to any other halo (NaN if none)
  best_id    that halo's Sub_tree_id (-1 if none)
  owner_id   where the star should be (replay.ownership: birth map plus the
             hierarchical accretion of roadrunner's infall lists, replayed
             to this snapshot; -1 none)
  E_O        energy relative to the owner (= E_H when the owner is the
             host; NaN if the owner does not bind it or has no energies)
  flags      bit 0: bound to the host; bit 1: bound to another halo;
             bit 2: owner is the host; bit 3: owner (not the host) binds it
  ids        particle ids
  pos, mass  host-centred position in the host-disc frame (physical kpc;
             z = angular momentum of the host's own stars within r_disc
             kpc), stellar mass
  vel        velocity relative to the host, in the same frame (km/s)
  mem        the infall-list view (replay.ownership.in_list): one entry per
             (star, galaxy other than the host) with the star in that
             galaxy's list and bound to it: row, gid, E (energy to it)
  lists      per galaxy other than the host, on the whole snapshot (no
             r_keep cut): gid, n and owned (its list's stars present, their
             mass), bound (those it binds: roadrunner's M*, catalogue Mtot),
             stolen (of those, E_H < E_g)
plus snapshot, redshift, rvir_kpc, host, basis, n_groups. Only galaxies
have stored energies, and each only for its candidate stars: dark subhalos
never compete, and a star missing from a halo's candidate list counts as
unbound to it.
"""
import os

import numpy as np

from . import energies as EN
from . import particles as PT

FLAG_HOST, FLAG_OTHER, FLAG_OWNER_HOST, FLAG_OWNER_BOUND = 1, 2, 4, 8
RUN = '/home/asier/roadrunner_test/halo685x095_v6'
DEFAULT_RR_OUT = f'{RUN}/vanilla/phase'
DEFAULT_PARTICLE_DIR = f'{RUN}/particle_data'


def disc_basis(xh, vh, m):
    """(e_x, e_y, e_z): e_z along the stars' angular momentum, e_x the
    catalogue x axis projected on the disc plane."""
    L = np.sum(m[:, None] * np.cross(xh, vh), axis=0)
    ez = L / np.linalg.norm(L)
    ref = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ex = ref - (ref @ ez) * ez
    ex /= np.linalg.norm(ex)
    return ex, np.cross(ez, ex), ez


class StarMaps:
    """Lazy, cached per-snapshot star tables of one roadrunner run."""

    def __init__(self, rr_out=DEFAULT_RR_OUT, particle_dir=None, host=None, r_disc=20.0, r_keep=2.0,
                 owners=True):
        self.rr_out = rr_out
        self.particle_dir = particle_dir or os.path.join(rr_out, 'particle_data')
        self.r_disc, self.r_keep = r_disc, r_keep
        self._host = host
        self._tree = None
        self._cache = {}
        self.ownership = None
        if owners:
            from .ownership import Ownership
            self.ownership = Ownership(rr_out)

    def _catalogue(self):
        if self._tree is None:
            import h5py
            with h5py.File(os.path.join(self.rr_out, 'catalogue.hdf5'), 'r') as f:
                self._tree = f['header/merger_tree'][()]
                if self._host is None:
                    self._host = int(f['header/accretion_id'][()])
        return self._tree, self._host

    def at(self, s, cache=True):
        """Star table of snapshot s (cached unless cache=False)."""
        s = int(s)
        if s in self._cache:
            return self._cache[s]
        d = self._compute(s)
        if cache:
            self._cache[s] = d
        return d

    def _compute(self, s):
        import h5py
        tree, host = self._catalogue()
        trs = tree[tree['Snapshot'] == s]
        hrow = trs[trs['Sub_tree_id'] == host][0]
        z = float(hrow['Redshift'])
        a = 1.0 / (1.0 + z)
        rvir = float(hrow['virial_radius']) * a
        ids, Xs, Vs, mean, scale, masses, _ = PT.load_particles(
            os.path.join(self.particle_dir, f'snapshot{s:04d}.hdf5'))
        pos = (Xs.astype(np.float64) / scale[:3] + mean[:3] - PT.xyz(hrow)) * a
        vel = Vs.astype(np.float64) / scale[3:] + mean[3:] - PT.vxyz(hrow)
        n = ids.size
        owner = (self.ownership.owners(ids, s) if self.ownership is not None
                 else np.full(n, -1, np.int64))
        owner = np.where(owner == -2, -1, owner)

        E_O = np.full(n, np.nan)
        E_H = np.full(n, np.nan)
        E_S = np.full(n, np.inf)
        best = np.full(n, -1, dtype=np.int64)
        groups = []                                  # (galaxy, rows, energies) of each other galaxy's bound stars
        # lost energies (energy_scale inf) are recomputed from this snapshot's particles, already in memory
        fallback = EN.Fallback(self.rr_out, self.particle_dir, tree, particles=lambda _s: (
            Xs.astype(np.float64) / scale[:3] + mean[:3], Vs.astype(np.float64) / scale[3:] + mean[3:], masses))
        with h5py.File(os.path.join(self.rr_out, 'assignment', f'snapshot{s:04d}.hdf5'), 'r') as fa:
            ha = fa['hard_assignment'][()]
            n_groups = len(fa['galaxies'])
            for k, (idx, E) in EN.stored_energies(fa, s, fallback).items():
                mine = owner[idx] == k
                E_O[idx[mine]] = E[mine]
                if k == host:
                    E_H[idx] = E
                else:
                    better = E < E_S[idx]            # indices are unique within a group
                    E_S[idx[better]] = E[better]
                    best[idx[better]] = k
                    groups.append((k, idx, E))
        E_S[~np.isfinite(E_S)] = np.nan
        flags = (np.isfinite(E_H) & (E_H < 0)).astype(np.uint8) * FLAG_HOST
        flags |= np.isfinite(E_S).astype(np.uint8) * FLAG_OTHER
        owner_host = owner == host
        flags |= owner_host.astype(np.uint8) * FLAG_OWNER_HOST
        flags |= (~owner_host & np.isfinite(E_O)).astype(np.uint8) * FLAG_OWNER_BOUND
        E_O[owner_host] = E_H[owner_host]

        host_rows = PT.row_index_of_ids(ids, ha['particle_index'][ha['Sub_tree_id'] == host].astype(np.int64))
        inner = host_rows[np.linalg.norm(pos[host_rows], axis=1) < self.r_disc]
        ex, ey, ez = disc_basis(pos[inner], vel[inner], masses[inner])
        P = np.stack([pos @ ex, pos @ ey, pos @ ez], axis=1)
        V = np.stack([vel @ ex, vel @ ey, vel @ ez], axis=1)
        keep = np.linalg.norm(P, axis=1) < self.r_keep * rvir

        lists, mem = None, None
        if self.ownership is not None:
            lists, mem = self._lists(s, host, ids, masses, E_H, groups)
            row = np.cumsum(keep) - 1                # full-array row -> row among the kept stars
            on = keep[mem['row']]
            mem = dict(row=row[mem['row'][on]], gid=mem['gid'][on], E=mem['E'][on])
        return dict(ids=ids[keep], pos=P[keep], vel=V[keep], mass=masses[keep], E_H=E_H[keep], E_S=E_S[keep], best_id=best[keep],
                    flags=flags[keep], owner_id=owner[keep], E_O=E_O[keep], snapshot=s, redshift=z,
                    rvir_kpc=rvir, host=host, basis=np.stack([ex, ey, ez]), n_groups=n_groups,
                    lists=lists, mem=mem)

    def _lists(self, s, host, ids, masses, E_H, groups):
        """The infall-list view of snapshot s, on the full snapshot (no r_keep cut).
        lists: per galaxy other than the host, dict of arrays gid, n and owned (the
        stars of its infall list present at s, their mass), bound (those it also
        binds: roadrunner's own stellar mass, the catalogue's Mtot) and stolen (of
        those, the ones the host binds more, E_H < E_g). mem: one entry per (star,
        galaxy) pair with the star in that galaxy's list and bound to it: row
        (full arrays), gid, E (its energy relative to that galaxy)."""
        own = self.ownership
        host_bound = np.isfinite(E_H) & (E_H < 0)
        g_b, m_b, m_s = [], [], []
        rows, gals, ens = [], [], []
        for k, idx, E in groups:
            inl = own.in_list(ids[idx], np.full(idx.size, k, np.int64), s)
            i, e = idx[inl], E[inl]
            taken = host_bound[i] & (E_H[i] < e)
            g_b.append(k)
            m_b.append(masses[i].sum())
            m_s.append(masses[i[taken]].sum())
            rows.append(i)
            gals.append(np.full(i.size, k, np.int64))
            ens.append(e)
        g_b = np.asarray(g_b, np.int64)

        g_l, n_l, m_l = own.list_sums(ids, masses, s)
        other = g_l != host
        g_l, n_l, m_l = g_l[other], n_l[other], m_l[other]

        gid = np.union1d(g_l, g_b)
        lists = {key: np.zeros(gid.size) for key in ('n', 'owned', 'bound', 'stolen')}
        lists['gid'] = gid
        j = np.searchsorted(gid, g_l)
        lists['n'][j], lists['owned'][j] = n_l, m_l
        j = np.searchsorted(gid, g_b)
        lists['bound'][j], lists['stolen'][j] = m_b, m_s

        cat = lambda parts, dtype: np.concatenate(parts) if parts else np.empty(0, dtype)
        mem = dict(row=cat(rows, np.int64), gid=cat(gals, np.int64), E=cat(ens, np.float64))
        return lists, mem

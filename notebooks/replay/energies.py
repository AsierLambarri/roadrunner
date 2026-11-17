"""The energies roadrunner stores (assignment/snapshotNNNN.hdf5), recomputed where they were lost.

roadrunner keeps each galaxy's bound stars with E = -boundness * energy_scale (boundness float32,
energy_scale the depth of the galaxy's well, HaloModel.binding_energy_scale). With halo_model
'kepler' (a point mass: the v7 runs) that depth is G M / r_in, r_in the distance of the innermost
star from the centre; when a star sits exactly on the centre r_in = 0, the scale is written as inf
and every boundness as 0, and the energies cannot be read back (0 * inf). A huge but finite scale
loses nothing: float32 keeps ~7 significant digits however small the boundness.

For a galaxy whose stored scale is not finite and > 0, stored_energies() recomputes its energies
from the run's particle_data/snapshotNNNN.hdf5 (it must exist: no fallback to the raw
simulation) with roadrunner's own recipe, roadrunner.physics.boundness.stellar_composite, copied
here in numpy (replay does not need roadrunner):

  the tree halo as a point mass of the tree mass, softened as a Plummer sphere of PLUMMER_SOFTENING
  kpc (default 0: a pure point mass, whose star exactly on the centre, if any, takes the radius of
  the nearest other star instead of r = 0; roadrunner's own KeplerPotential is a Plummer sphere of
  SOFTENING_KEPLER = 1e-3 kpc: set PLUMMER_SOFTENING = 1e-3 to reproduce its energies, to ~1e-6),
  (halo_model 'nfw', or 'errani_navarro' (EN21; satellites: infall NFW cut at x_cut Rs, centrals NFW):
  that halo's own potential, finite at the centre), plus, 'pinned' stars, STARS_ITERATIONS times, a ShellPotential of
  the stars bound so far (each star a thin shell about the tree centre, tabulated on N_NODES
  log-radius cells, linear in 1/r between nodes); E = Phi + 1/2 |v - v_tree|^2 for every star
  within search_factor * R_vir of the tree centre (catalogue units; physical for the potential).

roadrunner's run settings (halo_model, stars, search_factor, comoving) come from the catalogue
header and run.log. Each snapshot that needs it raises an EnergyFallbackWarning naming the
galaxies. The recomputed energies keep roadrunner's stored membership (only E is replaced).

PLUMMER_SOFTENING is read when the energies are recomputed: set it (replay.energies.PLUMMER_SOFTENING
= ...) before the data are computed; data already in memory keep the value they were computed with.
"""
import json
import os
import warnings

import numpy as np

# roadrunner's constants (roadrunner.physics.constants, roadrunner._defaults, ShellPotential)
G_KM = 4.300917270038E-6          # kpc (km/s)^2 / Msun
SOFTENING_KEPLER = 1E-3           # kpc: roadrunner's point-mass softening and the shells' radius floor
STARS_ITERATIONS = 3              # shell rebuilds of the stars' own potential
N_NODES = 4096                    # ShellPotential cells

PLUMMER_SOFTENING = 0.0           # kpc: the recomputed point mass's Plummer softening (0: pure point mass;
                                  # roadrunner's own: SOFTENING_KEPLER)


class EnergyFallbackWarning(UserWarning):
    """Stored energies lost (energy_scale not finite) and recomputed from particle_data."""


def run_settings(rr_out):
    """roadrunner's boundness settings of a run: halo_model, stars, search_factor (catalogue
    header/config) and comoving (run.log '-comoving:', else roadrunner's default True)."""
    import h5py
    with h5py.File(os.path.join(rr_out, 'catalogue.hdf5'), 'r') as f:
        c = f['header/config'][()]
    c = json.loads(c.decode() if isinstance(c, bytes) else str(c))
    comoving = True
    log = os.path.join(rr_out, 'run.log')
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if line.startswith('-comoving:'):
                    comoving = line.split(':', 1)[1].strip().lower() == 'true'
                    break
    return dict(halo_model=str(c.get('halo_model', 'kepler')).lower(), stars=str(c.get('stars', 'free')).lower(),
                search_factor=float(c.get('search_factor', 1.0)), comoving=comoving)


def _shell_potential(x, m, centre, G=G_KM, softening=SOFTENING_KEPLER, n_nodes=N_NODES):
    """roadrunner's ShellPotential of the particles at physical positions x (masses m) about centre:
    the function Phi(r) of physical radius."""
    r = np.sqrt(np.sum((x - centre) ** 2, axis=1))
    r_min = max(r.min(), softening)
    d = (np.log(max(r.max(), softening) / r_min) or 1.0) / n_nodes
    c, inv_d, K = np.log(r_min), 1.0 / d, n_nodes
    rj = np.maximum(r, softening)
    b = np.minimum(((np.log(rj) - c) * inv_d).astype(np.int64), K - 1)
    mass = np.bincount(b, weights=m, minlength=K)
    m_over_r = np.bincount(b, weights=m / rj, minlength=K)
    u = np.exp(-(c + d * np.arange(K + 1)))                      # 1 / node radius
    m_in = np.concatenate(([0.0], np.cumsum(mass)))
    phi = -G * (m_in * u + np.concatenate((np.cumsum(m_over_r[::-1])[::-1], [0.0])))
    slope, icpt = np.empty(K + 2), np.empty(K + 2)              # per cell: Phi = slope / r + icpt
    slope[0], icpt[0] = 0.0, phi[0]                              # inside the innermost node: constant
    slope[1:-1] = np.diff(phi) / np.diff(u)
    icpt[1:-1] = phi[:-1] - slope[1:-1] * u[:-1]
    slope[-1], icpt[-1] = -G * m_in[-1], 0.0                     # beyond the outermost: -G M / r

    def potential(rr):
        # cell 0 inside the innermost node (r = 0 included), 1..K between nodes, K + 1 beyond the outermost
        with np.errstate(divide='ignore'):
            xx = (np.log(rr) - c) * inv_d
        cell = np.zeros(rr.shape, np.int64)
        inside = xx >= 0
        cell[inside] = np.minimum(xx[inside], K).astype(np.int64) + 1
        out = np.full(rr.shape, icpt[0])
        out[inside] = slope[cell[inside]] / rr[inside] + icpt[cell[inside]]
        return out
    return potential


ERRANI_NAVARRO_KAPPA = 0.3        # EN21 eq. 7: the central depletion (1 + Rs / r_cut)^-kappa


def _nfw_family(row, halo_model, comoving):
    """(M, Rs, c, b) of a tree row's NFW-type halo, physical; b = Rs / r_cut for a stripped satellite
    (halo_model 'errani_navarro' with a host: its infall NFW and cut, as roadrunner.physics.halo_model.
    tree_potential), else 0 (plain NFW: the tree halo, or a central of an 'errani_navarro' run)."""
    ls = 1.0 / (1.0 + float(row['Redshift'])) if comoving else 1.0
    if halo_model == 'errani_navarro' and int(row['host_id']) != -1:
        Rs = float(row['r_s_infall']) / (1.0 + float(row['Redshift_infall'])) if comoving else float(row['r_s_infall'])
        return (float(row['M_vir_infall']), Rs, float(row['virial_radius_infall']) / float(row['r_s_infall']),
                1.0 / float(row['x_cut']))
    return float(row['mass']), float(row['scale_radius']) * ls, float(row['virial_radius']) / float(row['scale_radius']), 0.0


def _nfw_family_potential(M, Rs, c, b, r):
    """Phi(r) of an NFW (b = 0) or of EN21's stripped NFW (b = Rs / r_cut > 0; Errani & Navarro 2021, as
    roadrunner.physics.potentials.ErraniNavarroPotential): finite at r = 0 (Phi_0)."""
    from scipy.special import exp1
    x = r / Rs
    A = np.log(1 + c) - c / (1 + c)
    if b == 0:
        phi0 = -G_KM * M / (Rs * A)
        with np.errstate(divide='ignore', invalid='ignore'):
            phi = phi0 * np.log1p(x) / x
        return np.nan_to_num(phi, nan=phi0)
    k = (1 + b) ** -ERRANI_NAVARRO_KAPPA
    closed = (1 + b) * np.exp(b) * (exp1(b) - exp1(b * (1 + x))) - 1 + np.exp(-b * x) / (1 + x)
    c1, c2, c3 = -(2 + b), 3 + 2 * b + b ** 2 / 2, -(4 + 3 * b + b ** 2 + b ** 3 / 6)
    series = x ** 2 * (1 / 2 + x * (c1 / 3 + x * (c2 / 4 + x * c3 / 5)))      # the closed form cancels at small x
    menc = M * k / A * np.where((1 + b) * x < 1e-2, series, closed)
    outer = np.exp(-b * x) / (1 + x) - b * np.exp(b) * exp1(b * (1 + x))
    phi0 = -G_KM * M * k * (1 - b * np.exp(b) * exp1(b)) / (A * Rs)
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = -G_KM * (menc / r + M * k * outer / (A * Rs))
    return np.nan_to_num(phi, nan=phi0)


def point_mass_energies(row, pos, vel, mass, halo_model='kepler', stars='pinned', search_factor=1.0, comoving=True,
                        softening=None):
    """roadrunner's energies of the stars within search_factor R_vir of a tree halo (stellar_composite
    with a point-mass, NFW or Errani-Navarro halo). row: its merger-tree row; pos (catalogue units), vel (km/s), mass: every
    star of the snapshot; softening: the point mass's Plummer softening, kpc (None: PLUMMER_SOFTENING).
    (rows of those stars, their E in (km/s)^2)."""
    if halo_model not in ('kepler', 'nfw', 'errani_navarro') or stars not in ('pinned', 'none'):
        raise NotImplementedError(f"energy fallback: only halo_model 'kepler', 'nfw' or 'errani_navarro' with "
                                  f"'pinned' or 'none' stars is implemented (roadrunner's recipe for this run: "
                                  f"{halo_model!r}, {stars!r})")
    a = 1.0 / (1.0 + float(row['Redshift'])) if comoving else 1.0
    centre = np.array([row['position_x'], row['position_y'], row['position_z']], dtype=np.float64)
    v_tree = np.array([row['velocity_x'], row['velocity_y'], row['velocity_z']], dtype=np.float64)
    R = search_factor * float(row['virial_radius'])
    local = np.flatnonzero(np.sum((pos - centre) ** 2, axis=1) <= R * R)
    x = np.asarray(pos[local], dtype=np.float64) * a
    c = centre * a
    r = np.sqrt(np.sum((x - c) ** 2, axis=1))
    kin = 0.5 * np.sum((np.asarray(vel[local], dtype=np.float64) - v_tree) ** 2, axis=1)  # pinned: v_com = v_tree
    eps = PLUMMER_SOFTENING if softening is None else float(softening)
    if halo_model != 'kepler':                   # NFW / EN21: finite at the centre, no softening
        phi_dm = _nfw_family_potential(*_nfw_family(row, halo_model, comoving), r)
    elif eps > 0:
        phi_dm = -G_KM * float(row['mass']) / np.sqrt(r * r + eps * eps)
    else:                                        # pure point mass: a star on the centre takes the nearest other's radius
        on_centre = r == 0
        r_dm = np.where(on_centre, r[~on_centre].min() if (~on_centre).any() else SOFTENING_KEPLER, r)
        phi_dm = -G_KM * float(row['mass']) / r_dm
    E = phi_dm + kin
    for _ in range(STARS_ITERATIONS * (stars != 'none')):
        b = E < 0
        if not b.any():
            break
        E = phi_dm + _shell_potential(x[b], np.asarray(mass[local][b], dtype=np.float64), c)(r) + kin
    return local, E


class Fallback:
    """Recomputes the lost energies of one run (stored_energies), one snapshot's particles at a time.
    particles(s) -> (pos catalogue units, vel km/s, mass) may be given (e.g. the arrays a caller has
    already loaded); by default read from particle_dir/snapshotNNNN.hdf5, which must exist."""

    def __init__(self, rr_out, particle_dir, tree, particles=None):
        self.rr_out, self.particle_dir, self.tree = rr_out, particle_dir, tree
        self._particles = particles
        self._settings = None
        self._cache = (None, None)                     # (snapshot, its particles)

    def settings(self):
        if self._settings is None:
            self._settings = run_settings(self.rr_out)
        return self._settings

    def particles(self, s):
        if self._cache[0] != s:
            if self._particles is not None:
                self._cache = (s, self._particles(s))
            else:
                from . import particles as PT
                path = os.path.join(self.particle_dir, f'snapshot{s:04d}.hdf5')
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f'energy fallback: {path} not found. Its stored energies are lost (energy_scale not finite) '
                        f'and are recomputed from the run\'s particle_data; provide it, or the original simulation '
                        f'data, for this snapshot.')
                _, Xs, Vs, mean, scale, masses, _ = PT.load_particles(path)
                pos = Xs.astype(np.float64) / scale[:3] + mean[:3]
                vel = Vs.astype(np.float64) / scale[3:] + mean[3:]
                self._cache = (s, (pos, vel, masses))
        return self._cache[1]

    def energies(self, s, gid, idx):
        """E of galaxy gid's stars at rows idx of snapshot s, recomputed."""
        row = self.tree[(self.tree['Snapshot'] == s) & (self.tree['Sub_tree_id'] == gid)]
        if row.size == 0:
            raise KeyError(f'energy fallback: galaxy {gid} has no merger-tree row at snapshot {s}')
        pos, vel, mass = self.particles(s)
        local, E = point_mass_energies(row[0], pos, vel, mass, **self.settings())
        E_all = np.full(pos.shape[0], np.nan)
        E_all[local] = E
        return E_all[idx]


def stored_energies(fa, s, fallback=None):
    """{galaxy id: (rows of its bound stars, their energies)} of an open assignment file (snapshot s):
    E = -boundness * energy_scale over boundness_valid; a galaxy whose energy_scale is not finite and
    > 0 gets its energies recomputed by fallback (a Fallback), with one EnergyFallbackWarning for the
    snapshot naming them (without fallback: ValueError)."""
    out, lost = {}, []
    for key in fa['galaxies']:
        g = fa['galaxies'][key]
        k = int(key)
        idx = g['indices'][()].astype(np.int64)
        ok = g['boundness_valid'][()].astype(bool)
        idx = idx[ok]
        scale = float(g.attrs['energy_scale'])
        if np.isfinite(scale) and scale > 0:
            out[k] = (idx, -g['boundness'][()][ok].astype(np.float64) * scale)
        else:
            lost.append((k, scale, idx))
    if lost:
        if fallback is None:
            raise ValueError(f'snapshot {s}: stored energies lost (energy_scale not finite) for galaxies '
                             f'{[k for k, _, _ in lost]}, and no fallback to recompute them')
        st = fallback.settings()
        warnings.warn(
            f"snapshot {s}: roadrunner stored energy_scale = {', '.join(f'{sc:g}' for _, sc, _ in lost[:3])}"
            f"{', ...' if len(lost) > 3 else ''} for galaxies {[k for k, _, _ in lost]} (halo_model "
            f"{st['halo_model']!r}: a star sits on the centre of the point mass, so its depth G M / r_in is "
            f"infinite), so their stored boundness is all 0 and their energies are lost. Recomputed from "
            f"{fallback.particle_dir}/snapshot{s:04d}.hdf5 with roadrunner's recipe (point mass, Plummer softening "
            f"{PLUMMER_SOFTENING:g} kpc, + the stars' shells, stars {st['stars']!r}; replay/energies.py); "
            f"roadrunner's membership is kept.",
            EnergyFallbackWarning, stacklevel=2)
        for k, _, idx in lost:
            out[k] = (idx, fallback.energies(s, k, idx))
    return out

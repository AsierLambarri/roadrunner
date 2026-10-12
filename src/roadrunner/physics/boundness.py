#############################################################################
#
# package:   roadrunner.physics
# file:      boundness.py
# brief:     Gravitational boundness computation for halo particles.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 may 2026 - Created
#            13 may 2026 - Last edit
#
#############################################################################

"""Gravitational boundness computation for halo particles."""

import numpy as np
from scipy.spatial import KDTree

from roadrunner._defaults import LOCAL_IDX, STARS_ITERATIONS, math_dtype
from roadrunner.mixture._math import row_squared_norms
from roadrunner.physics.halo_model import HaloModel, physical_length_factor, tree_potential
from roadrunner.physics.potentials import CompositeSphericalPotential, ShellPotential
from roadrunner.postprocessing.centering import ssc_center

from roadrunner.threads import tree_workers


def stellar_composite(dark_matter, x, v, m, stars):
    """Dark matter plus the stars it binds, self-consistently, and the neighbours' energies in it.

    ``STARS_ITERATIONS`` times, a :class:`ShellPotential` of the particles bound so far joins the
    dark matter and the neighbours are re-bound: ``"free"`` stars at their shrinking-sphere frame
    (:func:`ssc_center`), ``"pinned"`` at the tree's; ``"none"``, or nothing bound, keeps the
    dark matter alone.

    Parameters
    ----------
    dark_matter : SphericalPotential
        The tree halo's potential (physical; its centre and velocity are the tree's).
    x, v : ndarray of shape (n, 3)
        Neighbours: absolute physical positions and velocities (km/s).
    m : ndarray of shape (n,)
        Neighbours' masses.
    stars : {"none", "pinned", "free"}

    Returns
    -------
    potential : CompositeSphericalPotential
    E : ndarray of float64, shape (n,)
        ``Φ + ½ |v - v_com|²``, as :meth:`HaloModel.compute_energy`.
    """
    energy = lambda p: p.potential(x) + 0.5 * row_squared_norms(v - p.com_velocity)
    potential = CompositeSphericalPotential(dark_matter=dark_matter)
    E = energy(potential)
    for _ in range(STARS_ITERATIONS * (stars != "none")):
        b = E < 0
        if not b.any():
            break
        c, w = ssc_center(x[b], v[b], m[b]) if stars == "free" else (dark_matter.centre, dark_matter.velocity)
        stellar = ShellPotential(x[b], m[b], G=dark_matter.G, centre=c, velocity=w)
        potential = CompositeSphericalPotential(dark_matter=dark_matter, stars=stellar)
        E = energy(potential)
    return potential, E


def compute_halo_bound_particles(rows, snap_data, config):
    """Build every halo once, with the stars it binds, and store its bound particles.

    The neighbours of each tree row (a KD-tree query inside ``search_factor`` virial radii)
    are bound by :func:`stellar_composite`; the halo is built with that final potential and
    stores its bound particles (physical energies and dynamical times), sorted by particle
    index (C06).

    Parameters
    ----------
    rows : list of mapping
        The snapshot's merger-tree rows.
    snap_data : SnapshotData
        Particle positions and velocities (``data``) and masses (``mass``).
    config : ProcessingConfig
        ``halo_model``, ``comoving``, ``search_factor``, ``stars``.

    Returns
    -------
    halos : list of HaloModel
    """
    positions, velocities = snap_data.data[:, :3], snap_data.data[:, 3:6]
    tree = KDTree(positions)
    halos = []
    for row in rows:
        ls = physical_length_factor(row["Redshift"], config.comoving)
        centre = np.array([row["position_x"], row["position_y"], row["position_z"]])
        local = np.asarray(
            tree.query_ball_point(centre, r=config.search_factor * row["virial_radius"], workers=tree_workers()),
            dtype=np.intp,
        )
        pos, vel = positions[local], velocities[local]

        potential, E = stellar_composite(
            tree_potential(row, config.halo_model, ls),
            np.asarray(pos, dtype=np.float64) * ls,
            vel,
            snap_data.mass[local],
            config.stars,
        )
        halo = HaloModel(
            potential,
            centre,
            [row["velocity_x"], row["velocity_y"], row["velocity_z"]],
            row["virial_radius"],
            int(row["Sub_tree_id"]),
            row["Redshift"],
            config.comoving,
            config.search_factor,
        )
        if local.size:  # a point mass's depth is taken at the innermost neighbour
            halo.inner_position = pos[np.argmin(row_squared_norms(np.asarray(pos, dtype=np.float64) - centre))]

        bound = E < 0
        order = np.argsort(local[bound])  # C06: sorted by particle index
        halo.set_bound(
            local[bound][order].astype(LOCAL_IDX),
            E[bound][order].astype(math_dtype(), copy=False),
            halo.orbital_time(E[bound], pos[bound])[order],
        )
        halos.append(halo)
    return halos

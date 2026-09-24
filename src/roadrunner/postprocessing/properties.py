"""Galaxy property computation: mass, half-mass radii, velocity dispersion.

Provides functions for structural (r20, rh, r80), kinematic (sigma),
and projected (Rhp, sigma_los) galaxy properties, as well as the
top-level :func:`compute_galaxy_properties` driver.
"""

import warnings

import numpy as np
import pandas as pd

from roadrunner._defaults import (
    MIN_PARTICLES_STRUCTURAL, SSC_NMIN, SSC_ALPHA, UNBOUND, math_dtype,
)
from roadrunner.physics.potentials import get_potential
from roadrunner.physics.timescales import compute_tidal_radius


def random_lines_of_sight(N, half_sphere=True, seed=None):
    """Generate random unit vectors on a sphere.

    Parameters
    ----------
    N : int
        Number of lines of sight.
    half_sphere : bool, default=True
        If ``True``, generate vectors only in the upper hemisphere.
    seed : int or None, optional
        Random seed.

    Returns
    -------
    los : ndarray of shape (N, 3)
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, size=N)
    if half_sphere:
        cos_theta = rng.uniform(0, 1, size=N)
    else:
        cos_theta = rng.uniform(-1, 1, size=N)
    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = cos_theta
    return np.column_stack((x, y, z)).astype(math_dtype(), copy=False)


def rotation_matrix_from_los(los):
    """Construct a rotation matrix that aligns the z-axis with *los*.

    Uses Gram–Schmidt orthogonalisation starting from a random vector
    that is not parallel to *los*.

    Parameters
    ----------
    los : ndarray of shape (3,)
        Line-of-sight direction.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Orthonormal rotation matrix.
    """
    los = np.asarray(los, dtype=math_dtype())
    if los.shape != (3,):
        raise ValueError("los must be a 3-element vector")
    ez = los / np.linalg.norm(los)

    for i in range(10):
        randvecs = random_lines_of_sight(100, half_sphere=False, seed=42 + i)[0]
        dots = np.abs(randvecs @ ez)
        mask = dots < 0.5
        if np.any(mask):
            idx = np.argmin(dots[mask])
            randvec = randvecs[mask][idx]
            break
    else:
        warnings.warn("Failed to draw a suitable random vector")
        if abs(np.dot([0, 0, 2], ez)) < 0.9:
            randvec = np.array([0, 0, 1]) - ez
        else:
            randvec = np.array([0, 1, 0]) - ez

    a = ez + randvec
    ex = a - np.dot(a, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return np.vstack((ex, ey, ez))


def find_center(positions, velocities, masses):
    """Find a galaxy centre using a median-based shrink.

    Starts from the median position, selects particles within the
    median radius, then returns the mass-weighted mean of both
    position and velocity.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    velocities : ndarray of shape (n, 3)
    masses : ndarray of shape (n,)

    Returns
    -------
    center_pos : ndarray of shape (3,)
    center_vel : ndarray of shape (3,)
    """
    center = np.median(positions, axis=0)
    radii = np.sum((positions - center)**2, axis=1)
    mask = radii <= np.median(radii)
    center_pos = np.average(positions[mask], weights=masses[mask], axis=0)
    center_vel = np.average(velocities[mask], weights=masses[mask], axis=0)
    return center_pos, center_vel


def enclosed_mass_radius(radii, masses, mass_fraction):
    """Radius that encloses a given fraction of the total mass.

    Uses linear interpolation between sorted cumulative-mass bins.

    Parameters
    ----------
    radii : ndarray of shape (n,)
        Per-particle radii.
    masses : ndarray of shape (n,)
        Per-particle masses.
    mass_fraction : float
        Fraction in [0, 1].

    Returns
    -------
    r : float
    """
    if mass_fraction > 1 or mass_fraction < 0:
        raise ValueError(f"mass_fraction must be in [0, 1], got {mass_fraction}")
    tot_mass = masses.sum()
    if tot_mass <= 0.0:
        return 0.0

    idx_sort = np.argsort(radii)
    radii_sorted = radii[idx_sort]
    mass_sorted = masses[idx_sort]
    cum_mass = np.cumsum(mass_sorted) / tot_mass

    i = np.searchsorted(cum_mass, mass_fraction)
    if i == 0:
        return radii_sorted[0]
    elif i >= len(radii_sorted):
        return radii_sorted[-1]
    else:
        r_low, r_high = radii_sorted[i - 1], radii_sorted[i]
        f_low, f_high = cum_mass[i - 1], cum_mass[i]
        return r_low + (mass_fraction - f_low) / (f_high - f_low) * (r_high - r_low)


def half_mass_radius(positions, masses, center, mass_fraction=0.5):
    """Radius enclosing *mass_fraction* of the total mass from *center*.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    masses : ndarray of shape (n,)
    center : ndarray of shape (3,)
    mass_fraction : float, default=0.5

    Returns
    -------
    r : float
    """
    radii = np.sqrt(np.sum((positions - center) ** 2, axis=1))
    return enclosed_mass_radius(radii, masses, mass_fraction)


def projected_half_mass_radius(positions, masses, center, los_matrices, mass_fraction=0.5):
    """Projected half-mass radius along multiple lines of sight.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    masses : ndarray of shape (n,)
    center : ndarray of shape (3,)
    los_matrices : ndarray of shape (n_los, 3, 3)
        Rotation matrices from :func:`rotation_matrix_from_los`.
    mass_fraction : float, default=0.5

    Returns
    -------
    Rhp : ndarray of shape (n_los,)
    """
    Nlos = los_matrices.shape[0]
    centered = positions - center
    result = np.empty(Nlos, dtype=math_dtype())
    for i in range(Nlos):
        pos_rot = centered @ los_matrices[i].T
        radii = np.sqrt(pos_rot[:, 0] ** 2 + pos_rot[:, 1] ** 2)
        result[i] = enclosed_mass_radius(radii, masses, mass_fraction)
    return result


def velocity_dispersion(velocities):
    """Total 3-D velocity dispersion.

    ``sigma = sqrt(sum(sigma_i^2))`` where ``sigma_i`` is the
    sample standard deviation along each axis.

    Parameters
    ----------
    velocities : ndarray of shape (n, 3)

    Returns
    -------
    sigma : float
    """
    sigma_per_axis = np.std(velocities, axis=0, ddof=1)
    return np.sqrt(np.sum(sigma_per_axis ** 2))


def line_of_sight_velocity_dispersion(positions, velocities, los_matrices, apertures):
    """Velocity dispersion along one or more lines of sight.

    For each line of sight, projects positions and velocities using
    the corresponding rotation matrix and computes the 1-D velocity
    dispersion of particles within the given aperture.

    Parameters
    ----------
    positions : ndarray of shape (n, 3)
    velocities : ndarray of shape (n, 3)
    los_matrices : ndarray of shape (n_los, 3, 3)
    apertures : ndarray of shape (n_los,)

    Returns
    -------
    sigma_los : ndarray of shape (n_los,)
    """
    los_matrices = np.asarray(los_matrices, dtype=math_dtype())
    apertures = np.atleast_1d(np.asarray(apertures, dtype=math_dtype()))
    if los_matrices.shape[0] != apertures.shape[0]:
        raise ValueError("Number of los_matrices must match number of apertures")

    Nlos = los_matrices.shape[0]
    sigma_los = np.full(Nlos, np.nan, dtype=math_dtype())

    for i in range(Nlos):
        R = los_matrices[i]
        pos_rot = positions @ R.T
        vel_rot = velocities @ R.T
        r_proj = np.sqrt(pos_rot[:, 0] ** 2 + pos_rot[:, 1] ** 2)
        mask = r_proj <= apertures[i]
        if np.any(mask):
            v_los = vel_rot[mask, 2]
            sigma_los[i] = np.std(v_los, ddof=1)
    return sigma_los


def compute_galaxy_properties(
    accretion_id,
    particle_masses,
    particle_coords,
    galaxy_bound,
    galaxy_table,
    host_props,
    halo_model,
    n_los=15,
    galaxy_centers=None,
    use_gmm_centers=True,
    min_particles_structural=MIN_PARTICLES_STRUCTURAL,
    ssc_nmin=SSC_NMIN,
    ssc_alpha=SSC_ALPHA,
):
    """Compute structural and kinematic properties for all galaxies.

    For each galaxy this computes: centre (SSC, GMM, or median-based),
    stellar mass, enclosed-mass radii (r20, rh, r80), 3-D velocity
    dispersion, projected half-mass radius (Rhp), line-of-sight
    velocity dispersion, and tidal radius.

    Parameters
    ----------
    accretion_id : int
        Host galaxy ID.
    particle_masses : ndarray of shape (n,)
    particle_coords : ndarray of shape (n, 6)
    galaxy_bound : dict of {int: ndarray}
        Bound particle array indices per galaxy.
    galaxy_table : DataFrame
        Merger-tree rows indexed by ``Sub_tree_id``.
    host_props : dict
        Host galaxy properties (``mass``, ``scale_radius``, ...).
    halo_model : str
        ``"kepler"`` or ``"nfw"``.
    n_los : int, default=15
        Number of lines of sight.
    galaxy_centers : dict or None, optional
        Pre-computed centres (e.g. from GMM).
    use_gmm_centers : bool, default=True
        If ``True``, prefer GMM centres over SSC.
    min_particles_structural : int, default=MIN_PARTICLES_STRUCTURAL
    ssc_nmin : int, default=SSC_NMIN
    ssc_alpha : float, default=SSC_ALPHA

    Returns
    -------
    props : DataFrame
        Columns: ``Sub_tree_id``, ``mb_host_id``, position, velocity,
        ``Mtot``, ``r20``, ``rh``, ``r80``, ``Rhp``, ``sigma``,
        ``sigma_los``, ``r_t``.
    """
    columns = [
        "Sub_tree_id", "mb_host_id",
        "position_x", "position_y", "position_z",
        "velocity_x", "velocity_y", "velocity_z",
        "Mtot",
        "r20", "rh", "r80",
        "Rhp",
        "sigma",
        "sigma_los",
        "r_t",
    ]

    los_vectors = random_lines_of_sight(n_los)
    los_matrices = np.array([rotation_matrix_from_los(los) for los in los_vectors],
                            dtype=math_dtype())

    host_mass = host_props["mass"]
    if halo_model.lower() == "kepler":
        host_potential = get_potential(halo_model, M=host_mass)
    else:
        host_Rs = host_props["scale_radius"] / (1 + host_props["Redshift"])
        host_c = host_props["virial_radius"] / host_props["scale_radius"]
        host_potential = get_potential(halo_model, M=host_mass, Rs=host_Rs, c=host_c)

    if UNBOUND in galaxy_bound:
        del galaxy_bound[UNBOUND]

    records = []
    for sid, indices in galaxy_bound.items():
        if indices.size == 0:
            continue

        host_id = galaxy_table.at[sid, "host_id"]
        sat_mass = galaxy_table.at[sid, "mass"]
        distance = galaxy_table.at[sid, "distance_to_acc_id"]

        r_t = compute_tidal_radius(host_potential, sat_mass, distance)

        indices_int = np.asarray(indices, dtype=np.intp)
        gal_pos = particle_coords[indices_int, :3]
        gal_vel = particle_coords[indices_int, 3:6]
        gal_masses = particle_masses[indices_int]
        Mtot = gal_masses.sum()
        npart = indices.size

        if npart <= 2:
            records.append(dict(
                Sub_tree_id=sid, mb_host_id=host_id,
                position_x=np.nan, position_y=np.nan, position_z=np.nan,
                velocity_x=np.nan, velocity_y=np.nan, velocity_z=np.nan,
                Mtot=Mtot, r20=np.nan, rh=np.nan, r80=np.nan,
                Rhp=np.nan, sigma=np.nan, sigma_los=np.nan, r_t=r_t,
            ))
            continue

        fitted = None if galaxy_centers is None else galaxy_centers.get(sid)
        if use_gmm_centers and fitted is not None:
            center_pos = fitted[:3]
            center_vel = fitted[3:6]
        elif npart >= ssc_nmin:
            from roadrunner.postprocessing.centering import ssc_center
            center_pos, center_vel = ssc_center(
                gal_pos, gal_vel, gal_masses, alpha=ssc_alpha, nmin=ssc_nmin,
            )
        else:
            center_pos, center_vel = find_center(gal_pos, gal_vel, gal_masses)

        if npart < min_particles_structural:
            records.append(dict(
                Sub_tree_id=sid, mb_host_id=host_id,
                position_x=center_pos[0], position_y=center_pos[1],
                position_z=center_pos[2],
                velocity_x=center_vel[0], velocity_y=center_vel[1],
                velocity_z=center_vel[2],
                Mtot=Mtot, r20=np.nan, rh=np.nan, r80=np.nan,
                Rhp=np.nan, sigma=np.nan, sigma_los=np.nan, r_t=r_t,
            ))
            continue

        centered_pos = gal_pos - center_pos
        centered_vel = gal_vel - center_vel

        r20 = half_mass_radius(centered_pos, gal_masses, np.zeros(3, dtype=centered_pos.dtype), mass_fraction=0.2)
        rh = half_mass_radius(centered_pos, gal_masses, np.zeros(3, dtype=centered_pos.dtype), mass_fraction=0.5)
        r80 = half_mass_radius(centered_pos, gal_masses, np.zeros(3, dtype=centered_pos.dtype), mass_fraction=0.8)
        sigma = velocity_dispersion(centered_vel)

        Rhp = projected_half_mass_radius(
            centered_pos, gal_masses, np.zeros(3, dtype=centered_pos.dtype), los_matrices, mass_fraction=0.5,
        )
        sigma_los_arr = line_of_sight_velocity_dispersion(
            centered_pos, centered_vel, los_matrices, Rhp,
        )

        records.append(dict(
            Sub_tree_id=sid, mb_host_id=host_id,
            position_x=center_pos[0], position_y=center_pos[1],
            position_z=center_pos[2],
            velocity_x=center_vel[0], velocity_y=center_vel[1],
            velocity_z=center_vel[2],
            Mtot=Mtot,
            r20=r20, rh=rh, r80=r80,
            Rhp=np.nanmedian(Rhp),
            sigma=sigma,
            sigma_los=np.nanmedian(sigma_los_arr),
            r_t=r_t,
        ))

    if not records:
        return pd.DataFrame(columns=columns, dtype=np.float32)
    return pd.DataFrame.from_records(records)[columns]

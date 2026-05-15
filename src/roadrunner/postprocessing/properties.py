import warnings

import numpy as np
import pandas as pd

from roadrunner.physics.potentials import get_potential
from roadrunner.physics.timescales import compute_tidal_radius


def random_lines_of_sight(N, half_sphere=True, seed=None):
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

    return np.column_stack((x, y, z))


def rotation_matrix_from_los(los):
    los = np.asarray(los)
    if los.shape != (3,):
        raise ValueError("los must be a 3-element vector")
    ez = los / np.linalg.norm(los)

    for i in range(10):
        randvecs = random_lines_of_sight(
            100, half_sphere=False, seed=42 + i
        )[0]

        dots = np.abs(randvecs @ ez)
        mask = dots < 0.5
        if np.any(mask):
            idx = np.argmin(dots[mask])
            randvec = randvecs[mask][idx]
            break
    else:
        warnings.warn(
            "Failed to draw a suitable random vector"
        )
        if abs(np.dot([0, 0, 2], ez)) < 0.9:
            randvec = np.array([0, 0, 1]) - ez
        else:
            randvec = np.array([0, 1, 0]) - ez

    a = ez + randvec
    ex = a - np.dot(a, ez) * ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)

    return np.vstack((ex, ey, ez))


def find_center(particle_positions, particle_velocities, particle_masses):
    radii = np.linalg.norm(
        particle_positions - np.median(particle_positions, axis=0),
        axis=1,
    )
    quantile = np.quantile(radii, 0.95)
    rc_scale = 0.5
    inner = radii <= rc_scale * quantile
    center_pos = np.average(
        particle_positions[inner], axis=0, weights=particle_masses[inner],
    )
    center_vel = np.average(
        particle_velocities[inner], axis=0, weights=particle_masses[inner],
    )
    return center_pos, center_vel


def enclosed_mass_radius(radii, masses, mass_fraction):
    if mass_fraction > 1 or mass_fraction < 0:
        raise ValueError(
            f"mass_fraction must be in [0, 1], got {mass_fraction}"
        )
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
    radii = np.sqrt(np.sum((positions - center) ** 2, axis=1))
    return enclosed_mass_radius(radii, masses, mass_fraction)


def projected_half_mass_radius(positions, masses, center, los_matrices, mass_fraction=0.5):
    Nlos = los_matrices.shape[0]
    centered = positions - center
    result = np.empty(Nlos, dtype=np.float64)
    for i in range(Nlos):
        pos_rot = centered @ los_matrices[i].T
        radii = np.sqrt(pos_rot[:, 0] ** 2 + pos_rot[:, 1] ** 2)
        result[i] = enclosed_mass_radius(radii, masses, mass_fraction)
    return result if Nlos > 1 else result[0]


def velocity_dispersion(velocities):
    sigma_per_axis = np.std(velocities, axis=0, ddof=1)
    return np.sqrt(np.sum(sigma_per_axis ** 2))


def line_of_sight_velocity_dispersion(positions, velocities, los_matrices, apertures):
    los_matrices = np.asarray(los_matrices)
    apertures = np.asarray(apertures)

    if los_matrices.shape[0] != apertures.shape[0]:
        raise ValueError(
            "Number of los_matrices must match number of apertures"
        )

    Nlos = los_matrices.shape[0]
    sigma_los = np.full(Nlos, np.nan)

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
    galaxy_particles,
    galaxy_table,
    host_props,
    n_los=11,
    halo_model="kepler",
):
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
    los_matrices = np.array([rotation_matrix_from_los(los) for los in los_vectors])

    host_mass = host_props["mass"]
    if halo_model.lower() == "kepler":
        host_potential = get_potential(halo_model, M=host_mass)
    else:
        host_Rs = host_props["scale_radius"] / (1 + host_props["Redshift"])
        host_c = host_props["virial_radius"] / host_props["scale_radius"]
        host_potential = get_potential(halo_model, M=host_mass, Rs=host_Rs, c=host_c)

    if -1 in galaxy_particles:
        del galaxy_particles[-1]

    records = []
    for sid, indices in galaxy_particles.items():
        if indices.size == 0:
            continue

        row = galaxy_table.loc[sid]
        host_id = row["host_id"]
        sat_mass = row["mass"]
        distance = row["distance_to_acc_id"]

        r_t = compute_tidal_radius(host_potential, sat_mass, distance)

        gal_pos = particle_coords[indices, :3]
        gal_vel = particle_coords[indices, 3:6]
        gal_masses = particle_masses[indices]
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

        center_pos, center_vel = find_center(gal_pos, gal_vel, gal_masses)

        if npart < 30:
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

        r20 = half_mass_radius(centered_pos, gal_masses, np.zeros(3), mass_fraction=0.2)
        rh = half_mass_radius(centered_pos, gal_masses, np.zeros(3), mass_fraction=0.5)
        r80 = half_mass_radius(centered_pos, gal_masses, np.zeros(3), mass_fraction=0.8)
        sigma = velocity_dispersion(centered_vel)

        Rhp = projected_half_mass_radius(
            centered_pos, gal_masses, np.zeros(3), los_matrices, mass_fraction=0.5,
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
            Rhp=np.nanmedian(Rhp) if isinstance(Rhp, np.ndarray) else Rhp,
            sigma=sigma,
            sigma_los=np.nanmedian(sigma_los_arr),
            r_t=r_t,
        ))

    if not records:
        return pd.DataFrame(columns=columns, dtype=np.float32)
    return pd.DataFrame.from_records(records)[columns]

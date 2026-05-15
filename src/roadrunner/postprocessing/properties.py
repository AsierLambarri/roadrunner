import warnings

import numpy as np


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

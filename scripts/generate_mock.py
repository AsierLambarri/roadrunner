#!/usr/bin/env python3
"""Generate a mock galaxy snapshot for testing the roadrunner pipeline.

Produces:
  test_data/mock_snap/merger_tree.csv
  test_data/mock_snap/equivalence.csv
  test_data/mock_snap/particles.hdf5  (or .npy)

Usage:
  python scripts/generate_mock.py --n-galaxies 30 --n-groups 3 \\
      --max-particles 2000 --mm-max 10 --seed 42
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["LIMEY_SUPPRESS_WARNINGS"] = "1"

import limepy


def default_radial_cdf(u, group_radius):
    """Linear CDF: P(r < R) = (R / R_max)^3 (uniform in volume)."""
    return group_radius * u ** (1.0 / 3.0)


def sample_galaxy_masses(n_galaxies, mm_max, rng):
    """Assign masses following a power law, with mm_max constraint."""
    masses = rng.uniform(0.1, 1.0, n_galaxies)
    masses = masses / masses.sum()
    return masses


def create_mock_snapshot(
    n_particles=100,
    n_galaxies=20,
    n_groups=3,
    radial_cdf=None,
    mm_max=10,
    max_particles=1000,
    box_size=500.0,
    group_radius=80.0,
    phi0=6.0,
    g=1.5,
    seed=42,
    output_dir="test_data/mock_snap",
    limepy_mass_scale=1e5,
):
    rng = np.random.default_rng(seed)
    if radial_cdf is None:
        radial_cdf = default_radial_cdf

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Generate galaxy masses ──────────────────────────────────
    galaxy_masses = sample_galaxy_masses(n_galaxies, mm_max, rng)

    # Assign to groups
    group_assignments = rng.choice(n_groups, size=n_galaxies, replace=True)
    group_ids = np.arange(n_groups)

    # Assign position within each group
    galaxy_positions = np.zeros((n_galaxies, 3))
    galaxy_velocities = np.zeros((n_galaxies, 3))
    group_centers = rng.uniform(-box_size, box_size, (n_groups, 3))

    for g in range(n_groups):
        mask = group_assignments == g
        idx = np.where(mask)[0]
        n_in_group = len(idx)
        if n_in_group == 0:
            continue

        # Choose the most massive galaxy in this group
        most_massive = idx[np.argmax(galaxy_masses[idx])]

        for i in idx:
            if i == most_massive:
                galaxy_positions[i] = group_centers[g]
            else:
                u = rng.random()
                frac_distance = radial_cdf(u, group_radius)
                direction = rng.uniform(-1, 1, 3)
                direction /= np.linalg.norm(direction)
                galaxy_positions[i] = group_centers[g] + direction * frac_distance

        galaxy_velocities[idx] = rng.normal(0, 10, (n_in_group, 3))

    # ── Step 2: Build limepy galaxy for each galaxy ─────────────────────
    n_particles_per_galaxy = np.zeros(n_galaxies, dtype=int)
    max_mass_idx = np.argmax(galaxy_masses)
    for i in range(n_galaxies):
        fraction = galaxy_masses[i] / galaxy_masses[max_mass_idx]
        n_particles_per_galaxy[i] = max(3, int(max_particles * fraction))

    total_particles = n_particles_per_galaxy.sum()

    # Pre-allocate particle arrays
    all_pos = np.empty((total_particles, 3), dtype=np.float64)
    all_vel = np.empty((total_particles, 3), dtype=np.float64)
    all_masses = np.empty(total_particles, dtype=np.float64)
    all_galaxy_ids = np.empty(total_particles, dtype=np.int64)

    offset = 0
    limepy_models = []
    limepy_samples = []

    for i in range(n_galaxies):
        ni = n_particles_per_galaxy[i]
        # Scale the limepy mass to get the desired number of particles
        model_mass = limepy_mass_scale
        model = limepy.limepy(phi0=phi0, g=g, M=model_mass)
        sample = limepy.sample(model, N=ni * 10, seed=seed + i)
        # Pick a random subset
        pick = rng.choice(sample.N, size=ni, replace=False)
        # Offset to galaxy position/velocity
        all_pos[offset : offset + ni, 0] = sample.x[pick] + galaxy_positions[i, 0]
        all_pos[offset : offset + ni, 1] = sample.y[pick] + galaxy_positions[i, 1]
        all_pos[offset : offset + ni, 2] = sample.z[pick] + galaxy_positions[i, 2]
        all_vel[offset : offset + ni, 0] = sample.vx[pick] + galaxy_velocities[i, 0]
        all_vel[offset : offset + ni, 1] = sample.vy[pick] + galaxy_velocities[i, 1]
        all_vel[offset : offset + ni, 2] = sample.vz[pick] + galaxy_velocities[i, 2]
        # Each particle has exactly mass = 1e4 Msun
        all_masses[offset : offset + ni] = 1e4
        all_galaxy_ids[offset : offset + ni] = i + 1  # 1-indexed Sub_tree_id

        limepy_models.append(model)
        limepy_samples.append(sample)
        offset += ni

    assert offset == total_particles

    # ── Step 3: Build merger tree CSV ────────────────────────────────────
    # Using 1-indexed Sub_tree_id for readability
    subtree_ids = np.arange(1, n_galaxies + 1)

    # Estimate virial radius from limepy model
    virial_radii = np.array([m.rt for m in limepy_models])
    scale_radii = np.array([m.rh for m in limepy_models])

    # No host_id or distance for now (isolated groups)
    merger_tree = pd.DataFrame({
        "Snapshot": np.full(n_galaxies, 0, dtype=int),
        "Sub_tree_id": subtree_ids,
        "mass": galaxy_masses * 1e11,
        "virial_radius": virial_radii,
        "scale_radius": np.where(scale_radii > 0, scale_radii, virial_radii / 10.0),
        "position_x": galaxy_positions[:, 0],
        "position_y": galaxy_positions[:, 1],
        "position_z": galaxy_positions[:, 2],
        "velocity_x": galaxy_velocities[:, 0],
        "velocity_y": galaxy_velocities[:, 1],
        "velocity_z": galaxy_velocities[:, 2],
        "Redshift": np.full(n_galaxies, 0.0),
    })

    merger_tree_path = os.path.join(output_dir, "merger_tree.csv")
    merger_tree.to_csv(merger_tree_path, index=False)
    print(f"Wrote {merger_tree_path}")

    # ── Step 4: Build equivalence CSV ────────────────────────────────────
    equivalence = pd.DataFrame({
        "snapshot": [0],
        "snapname": ["particles.npy"],
        "time": [13.8],
        "redshift": [0.0],
    })
    equiv_path = os.path.join(output_dir, "equivalence.csv")
    equivalence.to_csv(equiv_path, index=False)
    print(f"Wrote {equiv_path}")

    # ── Step 5: Save particle data ────────────────────────────────────────
    particle_coords = np.column_stack([
        all_pos,
        all_vel,
    ])
    particle_data = {
        "indices": np.arange(total_particles, dtype=np.uint64),
        "masses": all_masses,
        "coords": particle_coords,
        "galaxy_id": all_galaxy_ids,
    }
    npz_path = os.path.join(output_dir, "particles.npz")
    np.savez_compressed(npz_path, **particle_data)
    print(f"Wrote {npz_path}")
    print(f"  Total particles: {total_particles}")
    print(f"  Box size: {box_size} kpc")
    print(f"  Groups: {n_groups}")
    print(f"  Galaxies: {n_galaxies}")
    print(f"  Max particles/galaxy: {max_particles}")

    return {
        "merger_tree": merger_tree,
        "equivalence": equivalence,
        "particle_data": particle_data,
        "n_particles": total_particles,
        "n_galaxies": n_galaxies,
        "n_groups": n_groups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a mock galaxy snapshot for testing"
    )
    parser.add_argument("--n-particles", type=int, default=100,
                        help="Total number of particles (approximate)")
    parser.add_argument("--n-galaxies", type=int, default=20,
                        help="Number of galaxies")
    parser.add_argument("--n-groups", type=int, default=3,
                        help="Number of overlapping groups")
    parser.add_argument("--radial-cdf", type=str, default=None,
                        help="Not implemented: pass a callable for radial distribution")
    parser.add_argument("--mm-max", type=float, default=10,
                        help="Max mass ratio between most and 2nd most massive in group")
    parser.add_argument("--max-particles", type=int, default=1000,
                        help="Particles for the most massive galaxy")
    parser.add_argument("--box-size", type=float, default=500.0,
                        help="Simulation box size (kpc)")
    parser.add_argument("--group-radius", type=float, default=80.0,
                        help="Group virial-like radius (kpc)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="test_data/mock_snap",
                        help="Output directory")
    args = parser.parse_args()

    create_mock_snapshot(
        n_particles=args.n_particles,
        n_galaxies=args.n_galaxies,
        n_groups=args.n_groups,
        mm_max=args.mm_max,
        max_particles=args.max_particles,
        box_size=args.box_size,
        group_radius=args.group_radius,
        seed=args.seed,
        output_dir=args.output_dir,
    )

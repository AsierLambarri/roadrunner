#!/usr/bin/env python3
"""Generate a multi-snapshot mock simulation from a base mock snapshot.

Takes a base mock_snap (single snapshot with particles + merger tree)
and produces n_snap cumulative snapshots with:
  - New particles added each snapshot (N_newborn, distributed round-robin across galaxies)
  - Systematic bulk displacement applied to all particles at each step
  - Variable time spacing

Usage:
  python test_scripts/generate_mock_simulation.py
      --input-dir  test_data/mock_snap
      --output-dir test_data/mock_simulation
      --n-snap 11 --n-newborn 1000
      --v-sys 1000 1000 1000 --dt0 0.3 --dt1 0.3
"""

import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Generate mock simulation data")
    parser.add_argument("--input-dir", default="test_data/mock_snap",
                        help="Base mock snapshot directory")
    parser.add_argument("--output-dir", default="test_data/mock_simulation",
                        help="Output directory for generated data")
    parser.add_argument("--n-snap", type=int, default=11,
                        help="Number of snapshots to generate")
    parser.add_argument("--n-newborn", type=int, default=1000,
                        help="New particles per snapshot")
    parser.add_argument("--v-sys", type=float, nargs=3, default=[1000, 1000, 1000],
                        help="Systematic velocity [vx, vy, vz] kpc/Gyr")
    parser.add_argument("--dt0", type=float, default=0.3,
                        help="Time spacing parameter dt0 (Gyr)")
    parser.add_argument("--dt1", type=float, default=0.3,
                        help="Time spacing parameter dt1 (Gyr)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_snap = args.n_snap
    n_newborn = args.n_newborn
    v_sys = np.asarray(args.v_sys, dtype=np.float64)
    dt0 = args.dt0
    dt1 = args.dt1

    os.makedirs(output_dir, exist_ok=True)

    # ── Load base data ────────────────────────────────────────────
    print(f"Loading base data from {input_dir}")
    base_p = np.load(os.path.join(input_dir, "particles.npz"))
    base_coords = base_p["coords"]
    base_masses = base_p["masses"]
    base_galaxy_id = base_p["galaxy_id"]

    base_tree = pd.read_csv(os.path.join(input_dir, "merger_tree.csv"))

    n_initial = len(base_masses)
    halos = base_tree["Sub_tree_id"].values
    n_halos = len(halos)

    # ── Per-galaxy statistics (for newborn generation) ────────────
    print("Computing per-galaxy statistics...")
    gal_centers = base_tree[["position_x", "position_y", "position_z"]].values.astype(np.float64)
    gal_velocities = base_tree[["velocity_x", "velocity_y", "velocity_z"]].values.astype(np.float64)

    gal_pos_std = np.zeros((n_halos, 3), dtype=np.float64)
    gal_vel_std = np.zeros((n_halos, 3), dtype=np.float64)
    for i, hid in enumerate(halos):
        mask = base_galaxy_id == hid
        gal_pos_std[i] = base_coords[mask, :3].std(axis=0)
        gal_vel_std[i] = base_coords[mask, 3:6].std(axis=0)

    median_mass = np.median(base_masses)

    # ── Time model ────────────────────────────────────────────────
    print("Computing time model...")
    dt = np.array([dt0 - dt1 / np.sqrt(k + 1) for k in range(n_snap)], dtype=np.float64)
    cumulative_time = np.cumsum(dt)

    # ── Generate snapshots cumulatively ──────────────────────────
    print("Generating snapshots...")
    redshifts = np.linspace(0.1, 0.0, n_snap)

    # Track cumulative particles
    cum_coords = base_coords.copy()
    cum_masses = base_masses.copy()
    cum_galaxy_id = base_galaxy_id.copy()
    cum_born = np.zeros(n_initial, dtype=np.int32)

    for snap_k in range(n_snap):
        t_k = cumulative_time[snap_k]

        if snap_k > 0:
            # ---- Add newborn particles ----
            # Round-robin across halos
            halo_assignments = np.arange(n_newborn) % n_halos
            newborn_gid = halos[halo_assignments]
            newborn_pos = gal_centers[halo_assignments] + rng.normal(0, 1, (n_newborn, 3)) * gal_pos_std[halo_assignments]
            newborn_vel = gal_velocities[halo_assignments] + rng.normal(0, 1, (n_newborn, 3)) * gal_vel_std[halo_assignments]

            # Ensure no NaN if pos_std is 0 (edge case)
            newborn_pos = np.nan_to_num(newborn_pos, nan=0.0)
            newborn_vel = np.nan_to_num(newborn_vel, nan=0.0)

            newborn_coords = np.column_stack([newborn_pos, newborn_vel])
            newborn_masses = np.full(n_newborn, median_mass)
            newborn_born = np.full(n_newborn, snap_k, dtype=np.int32)

            # ── Apply displacement to ALL existing particles ─────
            cum_coords[:, :3] += v_sys * dt[snap_k]

            # ── Displace newborn particles to current snapshot ────
            newborn_coords[:, :3] += v_sys * t_k

            # ── Append newborn particles ──────────────────────────
            cum_coords = np.vstack([cum_coords, newborn_coords])
            cum_masses = np.hstack([cum_masses, newborn_masses])
            cum_galaxy_id = np.hstack([cum_galaxy_id, newborn_gid])
            cum_born = np.hstack([cum_born, newborn_born])

            # ── Save particles for this snapshot ──
        n_curr = len(cum_masses)
        snap_indices = np.arange(n_curr, dtype=np.uint64)

        snap_path = os.path.join(output_dir, f"particles_{snap_k:03d}.npz")
        np.savez_compressed(
            snap_path,
            indices=snap_indices,
            masses=cum_masses,
            coords=cum_coords,
            galaxy_id=cum_galaxy_id,
            born_snap=cum_born,
        )

        print(f"  Snap {snap_k}: t={t_k:.3f} Gyr, dt={dt[snap_k]:.4f} Gyr, "
              f"{n_curr} particles, {n_newborn if snap_k > 0 else 0} newborn, "
              f"displacement={v_sys * t_k}")

    # ── Write merger tree ─────────────────────────────────────────
    print("Writing merger tree...")
    frames = []
    for snap_k in range(n_snap):
        dup = base_tree.copy()
        dup["Snapshot"] = snap_k
        frames.append(dup)
    tree_all = pd.concat(frames, ignore_index=True)
    tree_all["host_id"] = -1
    tree_all["distance_to_acc_id"] = 0.0
    tree_all.to_csv(os.path.join(output_dir, "merger_tree.csv"), index=False)

    # ── Write equivalence table ──────────────────────────────────
    print("Writing equivalence table...")
    equiv = pd.DataFrame({
        "snapshot": list(range(n_snap)),
        "snapname": [f"particles_{k:03d}.npz" for k in range(n_snap)],
        "time": cumulative_time,
        "redshift": redshifts,
    })
    equiv.to_csv(os.path.join(output_dir, "equivalence.csv"), index=False)

    # ── Write cumulative assignment ──────────────────────────────
    print("Writing cumulative assignment...")
    assign = pd.DataFrame({
        "array_index": np.arange(len(cum_galaxy_id)),
        "Sub_tree_id": cum_galaxy_id,
    })
    assign.to_csv(os.path.join(output_dir, "assignment.csv"), index=False)

    total_particles = len(cum_galaxy_id)
    total_halos = n_halos * n_snap
    print(f"\nDone. Output in {output_dir}")
    print(f"  {n_snap} snapshots, {total_particles} total particles, {total_halos} halo rows")
    print(f"  Memory: ~{total_particles * 8 * 8 / 1e6:.0f} MB in coords + masses")


if __name__ == "__main__":
    main()

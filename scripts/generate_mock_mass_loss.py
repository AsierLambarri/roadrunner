#!/usr/bin/env python3
"""Generate a mock simulation with controlled mass loss for 9 galaxies.

Based on generate_mock_simulation.py, adds mass loss tracks to 9
galaxies distributed across 3 patterns (3 galaxies per pattern).

Pattern i  (IDs 2,5,8):   rapid 90% loss snap 0-2, then stable
Pattern ii (IDs 3,6,9):   linear loss to 20% by snap 8
Pattern iii(IDs 4,7,10):  dip to 10% at snap 5, recover to 50% at snap 8

Usage:
  python test_scripts/generate_mock_mass_loss.py
      --input-dir  test_data/mock_snap
      --output-dir test_data/mock_mass_loss
      --n-snap 9 --n-newborn 1000
      --v-sys 1000 1000 1000 --dt0 0.3 --dt1 0.3
      --t-end 13.82 --H0 70.2 --omega-m 0.272
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Mass loss tracks (9 snapshots) ─────────────────────────────
PATTERN_I  = np.array([1.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
PATTERN_II = np.linspace(1.0, 0.2, 9)
PATTERN_III = np.array([1.0, 0.82, 0.64, 0.46, 0.28, 0.1, 0.23, 0.37, 0.5])

MASS_LOSS_DICT = {
    2: PATTERN_I,   5: PATTERN_I,   8: PATTERN_I,
    3: PATTERN_II,  6: PATTERN_II,  9: PATTERN_II,
    4: PATTERN_III, 7: PATTERN_III, 10: PATTERN_III,
}
MASS_LOSS_IDS = set(MASS_LOSS_DICT.keys())


def _age_to_redshift(t, H0_Gyr, omega_m, omega_lambda):
    arg = np.sinh(1.5 * H0_Gyr * np.sqrt(omega_lambda) * t)
    return (arg * np.sqrt(omega_m / omega_lambda)) ** (-2.0 / 3.0) - 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="test_data/mock_snap")
    parser.add_argument("--output-dir", default="test_data/mock_mass_loss")
    parser.add_argument("--n-snap", type=int, default=9)
    parser.add_argument("--n-newborn", type=int, default=1000)
    parser.add_argument("--v-sys", type=float, nargs=3, default=[1000, 1000, 1000])
    parser.add_argument("--dt0", type=float, default=0.3)
    parser.add_argument("--dt1", type=float, default=0.3)
    parser.add_argument("--t-end", type=float, default=13.82)
    parser.add_argument("--H0", type=float, default=70.2)
    parser.add_argument("--omega-m", type=float, default=0.272)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    input_dir = args.input_dir
    output_dir = args.output_dir
    n_snap = args.n_snap
    n_newborn = args.n_newborn
    v_sys = np.asarray(args.v_sys, dtype=np.float64)
    dt0 = args.dt0
    dt1 = args.dt1
    t_end = args.t_end
    H0_km = args.H0
    omega_m = args.omega_m
    omega_lambda = 1.0 - omega_m
    H0_Gyr = H0_km * 1.0227e-3
    os.makedirs(output_dir, exist_ok=True)

    # ── Load base data ─────────────────────────────────────────
    print(f"Loading base data from {input_dir}")
    base_p = np.load(os.path.join(input_dir, "particles.npz"))
    base_coords = base_p["coords"]
    base_masses = base_p["masses"]
    base_galaxy_id = base_p["galaxy_id"]

    base_tree = pd.read_csv(os.path.join(input_dir, "merger_tree.csv"))
    halos = base_tree["Sub_tree_id"].values
    n_halos = len(halos)
    base_halo_pos = base_tree[["position_x", "position_y", "position_z"]].values.astype(np.float64)
    base_halo_vel = base_tree[["velocity_x", "velocity_y", "velocity_z"]].values.astype(np.float64)

    base_virial = base_tree["virial_radius"].values.astype(np.float64) * 1.5
    base_scale = base_tree["scale_radius"].values.astype(np.float64) * 1.5

    # ── KD-trees for newborns ──────────────────────────────────
    print("Building per-galaxy KD-trees...")
    gal_centers = base_halo_pos.copy()
    gal_trees = []
    gal_base_vels = []
    for i, hid in enumerate(halos):
        mask = base_galaxy_id == hid
        gal_trees.append(cKDTree(base_coords[mask, :3]))
        gal_base_vels.append(base_coords[mask, 3:6])
    median_mass = np.median(base_masses)

    # ── Time model ─────────────────────────────────────────────
    print(f"Computing time model...")
    dt = np.array([dt0 - dt1 / np.sqrt(k + 1) for k in range(n_snap)], dtype=np.float64)
    raw_cumulative = np.cumsum(dt)
    total_span = raw_cumulative[-1]
    if total_span > t_end:
        raise ValueError(f"Total span ({total_span:.3f}) > t_end ({t_end})")
    cosmic_time = t_end - total_span + raw_cumulative
    z = np.maximum(_age_to_redshift(cosmic_time, H0_Gyr, omega_m, omega_lambda), 0.0)
    print(f"  Times: [{cosmic_time[0]:.3f}, ..., {cosmic_time[-1]:.3f}] Gyr")
    print(f"  Redshifts: [z={z[0]:.4f}, ..., z={z[-1]:.4f}]")

    # ── Validate tracks ────────────────────────────────────────
    for gid in sorted(MASS_LOSS_IDS):
        assert len(MASS_LOSS_DICT[gid]) == n_snap, f"Gal {gid}: track length mismatch"
    print(f"  Mass-losing galaxies: IDs={sorted(MASS_LOSS_IDS)}")

    # ── Cumulative particle storage ────────────────────────────
    cum_coords = base_coords.copy()
    cum_masses = base_masses.copy()
    cum_galaxy_id = base_galaxy_id.copy()
    cum_born = np.zeros(len(base_masses), dtype=np.int32)
    alive_mask = np.ones(len(base_masses), dtype=bool)  # true = particle is alive

    # ── Snapshot 0 ─────────────────────────────────────────────
    save_coords = cum_coords.copy()
    save_coords[:, :3] *= (1.0 + z[0])
    save_coords[:, 3:6] += v_sys
    np.savez_compressed(
        os.path.join(output_dir, "particles_000.npz"),
        indices=np.arange(len(save_coords), dtype=np.uint64),
        masses=cum_masses, coords=save_coords,
        galaxy_id=cum_galaxy_id, born_snap=cum_born,
    )
    print(f"  Snap 0: {len(save_coords)} particles")

    # ── Snapshots 1..N-1 ──────────────────────────────────────
    for snap_k in range(1, n_snap):
        t_k = raw_cumulative[snap_k]

        # Displace existing alive particles
        cum_coords[alive_mask, :3] += v_sys * dt[snap_k]

        # ── Newborn particles ──────────────────────────────────
        halo_assignments = np.arange(n_newborn) % n_halos
        newborn_gid = halos[halo_assignments]
        newborn_pos = np.zeros((n_newborn, 3), dtype=np.float64)
        newborn_vel = np.zeros((n_newborn, 3), dtype=np.float64)
        for j in range(n_newborn):
            hi = int(halo_assignments[j])
            newborn_pos[j] = gal_centers[hi] + rng.normal(0, 2, 3)
            _, idxs = gal_trees[hi].query(newborn_pos[j], k=min(4, gal_trees[hi].data.shape[0]))
            newborn_vel[j] = gal_base_vels[hi][idxs].mean(axis=0)
        newborn_pos = np.nan_to_num(newborn_pos, nan=0.0)
        newborn_vel = np.nan_to_num(newborn_vel, nan=0.0)
        newborn_coords = np.column_stack([newborn_pos, newborn_vel])
        newborn_mases = np.full(n_newborn, median_mass)
        newborn_born = np.full(n_newborn, snap_k, dtype=np.int32)
        # Displace newborns
        newborn_coords[:, :3] += v_sys * t_k

        # Append to cumulative arrays
        cum_coords = np.vstack([cum_coords, newborn_coords])
        cum_masses = np.hstack([cum_masses, newborn_mases])
        cum_galaxy_id = np.hstack([cum_galaxy_id, newborn_gid])
        cum_born = np.hstack([cum_born, newborn_born])
        alive_mask = np.hstack([alive_mask, np.ones(n_newborn, dtype=bool)])

        # ── Mass loss: kill particles ──────────────────────────
        for gid in MASS_LOSS_IDS:
            frac = MASS_LOSS_DICT[gid][snap_k]
            gmask = (cum_galaxy_id == gid) & alive_mask
            n_old = gmask.sum()
            if n_old == 0:
                continue
            n_keep = max(30, int(round(n_old * frac)))
            # Select which indices of gmask to keep alive
            gmask_idx = np.where(gmask)[0]
            keep = rng.choice(gmask_idx, n_keep, replace=False)
            # Kill the rest
            kill = np.setdiff1d(gmask_idx, keep)
            alive_mask[kill] = False

        # ── Assemble snapshot from alive particles ─────────────
        alive_idx = np.where(alive_mask)[0]
        snap_coords = cum_coords[alive_idx].copy()
        snap_masses = cum_masses[alive_idx]
        snap_gid = cum_galaxy_id[alive_idx]
        snap_born = cum_born[alive_idx]

        # Physical → comoving + bulk velocity
        snap_coords[:, :3] *= (1.0 + z[snap_k])
        snap_coords[:, 3:6] += v_sys

        snap_path = os.path.join(output_dir, f"particles_{snap_k:03d}.npz")
        np.savez_compressed(
            snap_path,
            indices=np.arange(len(snap_coords), dtype=np.uint64),
            masses=snap_masses, coords=snap_coords,
            galaxy_id=snap_gid, born_snap=snap_born,
        )
        n_alive = alive_mask.sum()
        newborn_this = n_newborn
        print(f"  Snap {snap_k}: {n_alive} alive, {newborn_this} newborn")

    # ── Write merger tree ─────────────────────────────────────
    print("Writing merger tree...")
    frames = []
    for snap_k in range(n_snap):
        dup = base_tree.copy()
        dup["Snapshot"] = snap_k
        dup["Redshift"] = z[snap_k]
        factor = 1.0 + z[snap_k]
        displ = v_sys * raw_cumulative[snap_k]
        for i, col in enumerate(["position_x", "position_y", "position_z"]):
            dup[col] = (base_halo_pos[:, i] + displ[i]) * factor
        for i, col in enumerate(["velocity_x", "velocity_y", "velocity_z"]):
            dup[col] = base_halo_vel[:, i] + v_sys[i]
        dup["virial_radius"] = base_virial * factor
        dup["scale_radius"] = base_scale * factor
        for gid in MASS_LOSS_IDS:
            m = dup["Sub_tree_id"] == gid
            dup.loc[m, "mass"] = (dup.loc[m, "mass"] * MASS_LOSS_DICT[gid][snap_k]).values
        frames.append(dup)
    tree_all = pd.concat(frames, ignore_index=True)
    tree_all["host_id"] = -1
    tree_all["distance_to_acc_id"] = 0.0
    tree_all.to_csv(os.path.join(output_dir, "merger_tree.csv"), index=False)

    # ── Equivalence table ──────────────────────────────────────
    equiv = pd.DataFrame({
        "snapshot": list(range(n_snap)),
        "snapname": [f"particles_{k:03d}.npz" for k in range(n_snap)],
        "time": cosmic_time, "redshift": z,
    })
    equiv.to_csv(os.path.join(output_dir, "equivalence.csv"), index=False)

    # ── Assignment ─────────────────────────────────────────────
    alive_idx = np.where(alive_mask)[0] if n_snap > 0 else np.arange(len(cum_coords))
    assign = pd.DataFrame({
        "array_index": np.arange(len(alive_idx)),
        "Sub_tree_id": cum_galaxy_id[alive_idx],
    })
    assign.to_csv(os.path.join(output_dir, "assignment.csv"), index=False)

    print(f"\nDone. Output in {output_dir}")
    print(f"  {n_snap} snapshots, {len(halos)} galaxies, {len(MASS_LOSS_IDS)} mass-losing")
    for gid in sorted(MASS_LOSS_IDS):
        gmask = (cum_galaxy_id == gid) & alive_mask
        print(f"  Gal {gid:2d} final count: {gmask.sum()}")


if __name__ == "__main__":
    main()

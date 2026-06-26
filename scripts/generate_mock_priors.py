"""Generate mock simulation with controlled mass loss/gain for prior testing.

5 galaxies, 8 snapshots:
  - Galaxy 1: stable mass (host)
  - Galaxy 2: loses 30% over time (stripping)
  - Galaxy 3: gains 50% over time (accretion)
  - Galaxy 4: slowly loses (mild stripping)
  - Galaxy 5: slowly gains
"""
import os, sys, argparse
import numpy as np, pandas as pd

sys.path.insert(0, "src")
from roadrunner._defaults import KMEANS_PP_MAX_ITER

# Cosmology
H0 = 70.2
OM = 0.272
OL = 0.728
H0_Gyr = H0 / 1000.0 / 0.9778  # km/s/Mpc → 1/Gyr

def age_to_z(t):
    arg = np.sinh(1.5 * H0_Gyr * np.sqrt(OL) * t)
    return (arg * np.sqrt(OM / OL)) ** (-2.0 / 3.0) - 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="test_data/mock_priors")
    parser.add_argument("--n-snap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    n_snap = args.n_snap
    n_gal = 5
    n_per_gal_base = 5000  # particles per galaxy at peak

    # Mass evolution tracks (fraction of peak mass)
    mass_tracks = np.array([
        [1.0] * n_snap,               # gal 1: stable
        np.linspace(1.0, 0.7, n_snap),  # gal 2: loses 30%
        np.linspace(1.0, 1.5, n_snap),  # gal 3: gains 50%
        np.linspace(1.0, 0.85, n_snap), # gal 4: mild loss
        np.linspace(1.0, 1.25, n_snap), # gal 5: mild gain
    ])

    # Galaxy positions — clustered for group overlap
    gal_pos = np.array([[0, 0, 0], [80, 0, 0], [-60, 60, 0],
                        [0, -70, 0], [40, 40, 0]], dtype=np.float64)
    gal_vel = rng.normal(0, 20, (n_gal, 3))

    # Halo masses (Msun) — realistic values for the potential model
    halo_masses = np.array([5e11, 3e11, 2e11, 1e11, 8e10], dtype=np.float64)
    virial_radii = np.array([200, 160, 140, 110, 100], dtype=np.float64)
    scale_radii = virial_radii / np.array([10, 8, 7, 6, 5], dtype=np.float64)

    # Host is galaxy 1
    host_id_arr = np.array([1, 1, 1, 1, 1])

    # Bulk velocity across snapshots
    v_bulk = np.array([500.0, 500.0, 500.0])

    # Time/redshift: linearly spaced
    t_arr = np.linspace(0.5, 13.0, n_snap)  # Gyr
    z_arr = np.array([age_to_z(t) for t in t_arr])

    # Build merger tree rows and particle files per snapshot
    all_rows = []
    for snap_k in range(n_snap):
        z = z_arr[snap_k]
        dt = t_arr[snap_k] - (t_arr[snap_k - 1] if snap_k > 0 else 0.0)
        frac_comoving = 1.0 / (1.0 + z)  # physical / comoving

        particles_list = []
        for g in range(n_gal):
            sid = g + 1
            mass_frac = mass_tracks[g, snap_k]
            n_particles = max(int(n_per_gal_base * mass_frac), 50)

            pos = gal_pos[g] + gal_vel[g] * t_arr[snap_k] * 0.001
            vel = gal_vel[g]

            # Generate particles in comoving kpc
            r_phys = virial_radii[g] * mass_frac**(1/3)
            r_com = r_phys / frac_comoving

            coords = np.zeros((n_particles, 6))
            coords[:, :3] = rng.normal(pos, r_com * 0.15, (n_particles, 3))
            sigma_v = 15.0 * np.sqrt(halo_masses[g] / 1e11)
            coords[:, 3:] = rng.normal(vel, sigma_v, (n_particles, 3))

            particles_list.append(coords)

            # Merger tree row — mass in Msun for potential model
            all_rows.append({
                "Snapshot": snap_k,
                "Sub_tree_id": sid,
                "mass": float(halo_masses[g]),
                "virial_radius": float(virial_radii[g]),
                "scale_radius": float(scale_radii[g]),
                "position_x": pos[0],
                "position_y": pos[1],
                "position_z": pos[2],
                "velocity_x": vel[0],
                "velocity_y": vel[1],
                "velocity_z": vel[2],
                "Redshift": float(z),
                "host_id": host_id_arr[g],
                "distance_to_acc_id": float(np.linalg.norm(pos - gal_pos[0])),
            })

        # Concatenate particles for this snapshot
        X = np.vstack(particles_list)
        galaxy_id = np.concatenate([
            np.full(particles_list[g].shape[0], g + 1, dtype=np.int32)
            for g in range(n_gal)
        ])
        indices = np.arange(snap_k * n_gal * n_per_gal_base,
                            snap_k * n_gal * n_per_gal_base + len(X), dtype=np.int64)

        masses = np.ones(len(X), dtype=np.float32)
        born_snap = np.full(len(X), 0, dtype=np.int32)

        np.savez_compressed(
            os.path.join(out, f"particles_{snap_k:03d}.npz"),
            indices=indices, masses=masses, coords=X,
            galaxy_id=galaxy_id, born_snap=born_snap,
        )
        print(f"Snap {snap_k}: {len(X)} particles, {n_gal} galaxies")

    # Write merger tree
    mt = pd.DataFrame(all_rows)
    mt.to_csv(os.path.join(out, "merger_tree.csv"), index=False)

    # Write equivalence table
    eq_rows = []
    for snap_k in range(n_snap):
        eq_rows.append({
            "snapshot": snap_k,
            "snapname": f"particles_{snap_k:03d}.npz",
            "redshift": float(z_arr[snap_k]),
            "time": float(t_arr[snap_k]),
        })
    eq = pd.DataFrame(eq_rows)
    eq.to_csv(os.path.join(out, "equivalence.csv"), index=False)

    print(f"\nSaved to {out}/")
    print(f"Merger tree: {len(mt)} rows, {n_gal} galaxies × {n_snap} snaps")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot particle assignment for a single snapshot.

Usage:
  python scripts/analyze_snapshot.py --catalogue ./output --snapshot 0
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.io.hdf5_reader import HDF5CatalogueReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory (contains catalogue.hdf5)")
    parser.add_argument("--snapshot", type=int, default=0,
                        help="Snapshot number to plot")
    parser.add_argument("--output", "-o", type=str,
                        default="figures/snapshot.png",
                        help="Output image path")
    parser.add_argument("--max-particles", type=int, default=50000,
                        help="Limit plotted particles")
    args = parser.parse_args()

    cat_path = os.path.join(args.output_dir, "catalogue.hdf5")
    assign_path = os.path.join(args.output_dir, "assignment.hdf5")
    parts_path = os.path.join(args.output_dir, "particles.hdf5")

    reader = HDF5CatalogueReader(cat_path)
    hdr = reader.read_header()
    snapshots = hdr["snapshots"]
    print(f"Available snapshots: {snapshots}")

    if args.snapshot not in snapshots:
        print(f"Snapshot {args.snapshot} not found")
        return

    import h5py
    with h5py.File(assign_path, "r") as hf:
        hard_key = f"/snapshots/{args.snapshot}/hard_assignment"
        if hard_key not in hf:
            print(f"No hard assignment for snapshot {args.snapshot}")
            return
        assign = pd.DataFrame.from_records(hf[hard_key][()])
        pid = assign["Sub_tree_id"].values

    with h5py.File(parts_path, "r") as hf:
        snap = hf[f"/snapshots/{args.snapshot}"]
        mean = snap["scaler/mean"][:6]
        scale = snap["scaler/scale"][:6]
        pos_scaled = snap["positions"][:]
        vel_scaled = snap["velocities"][:]
        pos = pos_scaled / scale[:3] + mean[:3]
        vel = vel_scaled / scale[3:6] + mean[3:6]

    N = len(pid)
    if args.max_particles and N > args.max_particles:
        idx = np.random.default_rng(0).choice(N, args.max_particles, replace=False)
        pos, vel, pid = pos[idx], vel[idx], pid[idx]

    unique_ids = np.unique(pid)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, gid in enumerate(unique_ids):
        mask = pid == gid
        axes[0].scatter(pos[mask, 0], pos[mask, 1],
                        s=1, c=[colors[i % 10]], alpha=0.5)
        axes[1].scatter(vel[mask, 0], vel[mask, 1],
                        s=1, c=[colors[i % 10]], alpha=0.5)

    axes[0].set_xlabel("x [kpc]")
    axes[0].set_ylabel("y [kpc]")
    axes[0].set_title(f"Snapshot {args.snapshot} — XY positions")
    axes[0].set_aspect("equal")

    axes[1].set_xlabel("vx [km/s]")
    axes[1].set_ylabel("vy [km/s]")
    axes[1].set_title(f"Snapshot {args.snapshot} — VX/VY velocities")
    axes[1].set_aspect("equal")

    fig.suptitle(f"{len(unique_ids)} galaxies, {N} particles")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

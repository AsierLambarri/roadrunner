#!/usr/bin/env python3
"""Plot mock simulation comparison: ground truth vs pipeline output.

For each snapshot, produces a 2x2 figure:
  Left: ground truth (galaxy_id from mock data)
  Right: pipeline assignment (Sub_tree_id from hard_assignment)
  On the right: galaxy centers + Rh (XY) / σ (VX/VY) circles from catalogue.

Saves as comparison{snap:03d}.png inside the output_dir.

Usage:
  python test_scripts/plot_mock_comparison_simulation.py
      --data-dir test_data/mock_simulation
      --output-dir test_output/mock_sim
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot mock simulation comparison")
    parser.add_argument("--data-dir", default="test_data/mock_simulation",
                        help="Mock simulation data directory")
    parser.add_argument("--output-dir", default="test_output/mock_sim",
                        help="Pipeline output directory (also where pngs are saved)")
    parser.add_argument("--n-snap", type=int, default=None,
                        help="Only process first N snapshots (default: all)")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    equiv_path = os.path.join(data_dir, "equivalence.csv")
    equiv = np.genfromtxt(equiv_path, delimiter=",", names=True)
    n_snap = args.n_snap or len(equiv)

    for snap_k in range(n_snap):
        z_k = equiv[snap_k]["redshift"]
        t_k = equiv[snap_k]["time"]

        # ── Load ground truth ──────────────────────────────────────
        snap_file = os.path.join(data_dir, f"particles_{snap_k:03d}.npz")
        if not os.path.exists(snap_file):
            print(f"Snap {snap_k}: missing {snap_file}, skipping")
            continue
        p = np.load(snap_file)
        coords = p["coords"]
        true_id = p["galaxy_id"]

        # ── Load pipeline output ───────────────────────────────────
        assign_path = os.path.join(output_dir, "assignment.hdf5")
        if not os.path.exists(assign_path):
            print(f"Snap {snap_k}: no assignment.hdf5, skipping")
            continue
        with h5py.File(assign_path, "r") as f:
            group = f.get(f"snapshots/{snap_k}")
            if group is None:
                print(f"Snap {snap_k}: no snapshot group in assignment.hdf5, skipping")
                continue
            hard_assign = group["hard_assignment"][:]
        pred_id = hard_assign["Sub_tree_id"]

        # ── Load galaxy properties ─────────────────────────────────
        cat_path = os.path.join(output_dir, "catalogue.hdf5")
        if not os.path.exists(cat_path):
            print(f"Snap {snap_k}: no catalogue.hdf5, skipping")
            continue
        with h5py.File(cat_path, "r") as f:
            gp_group = f.get(f"snapshots/{snap_k}/galaxy_properties")
            if gp_group is None:
                print(f"Snap {snap_k}: no galaxy_properties, skipping")
                continue
            gp = gp_group[:]

        # ── Build lookup dicts ─────────────────────────────────────
        id_to_pos_x = {int(r["Sub_tree_id"]): r["position_x"] for r in gp}
        id_to_pos_y = {int(r["Sub_tree_id"]): r["position_y"] for r in gp}
        id_to_vel_x = {int(r["Sub_tree_id"]): r["velocity_x"] for r in gp}
        id_to_vel_y = {int(r["Sub_tree_id"]): r["velocity_y"] for r in gp}
        id_to_rh = {int(r["Sub_tree_id"]): r["rh"] for r in gp}
        id_to_sigma = {int(r["Sub_tree_id"]): r["sigma"] for r in gp}

        # ── Plot ───────────────────────────────────────────────────
        unique_true = np.unique(true_id)
        unique_pred = np.unique(pred_id)
        n_colors = max(len(unique_true), len(unique_pred))
        colors = plt.cm.tab20(np.linspace(0, 1, n_colors))

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        datasets = [
            (coords[:, 0], coords[:, 1], "x [kpccm]", "y [kpccm]"),
            (coords[:, 3], coords[:, 4], "vx [km/s]", "vy [km/s]"),
        ]

        for row in range(2):
            x_data, y_data, xlabel, ylabel = datasets[row]

            true_ax = axes[row, 0]
            pred_ax = axes[row, 1]

            # Ground truth
            for i, gid in enumerate(unique_true):
                mask = true_id == gid
                true_ax.scatter(
                    x_data[mask], y_data[mask],
                    s=0.5, c=[colors[i % 20]], alpha=0.6,
                )
            true_ax.set_xlabel(xlabel)
            true_ax.set_ylabel(ylabel)
            true_ax.set_title(["XY \u2014 Ground truth", "VX/VY \u2014 Ground truth"][row])
            true_ax.set_aspect("equal")

            # Pipeline assignment
            for i, pid in enumerate(unique_pred):
                mask = pred_id == pid
                pred_ax.scatter(
                    x_data[mask], y_data[mask],
                    s=0.5, c=[colors[i % 20]], alpha=0.6,
                )

            # Overlays: centers, Rh / sigma circles
            for sid in unique_pred:
                sid_int = int(sid)
                if sid_int not in id_to_pos_x:
                    continue

                if row == 0:
                    cx = id_to_pos_x[sid_int]
                    cy = id_to_pos_y[sid_int]
                else:
                    cx = id_to_vel_x[sid_int]
                    cy = id_to_vel_y[sid_int]

                pred_ax.plot(cx, cy, "k+", markersize=8, markeredgewidth=1.5, zorder=5)

                if row == 0:
                    rh = id_to_rh.get(sid_int, 0)
                    if rh > 0:
                        circle = mpatches.Circle(
                            (cx, cy), rh,
                            fill=False, edgecolor="black", linewidth=1.0,
                            linestyle="--", alpha=0.7, zorder=4,
                        )
                        pred_ax.add_patch(circle)
                else:
                    sigma = id_to_sigma.get(sid_int, 0)
                    if sigma > 0:
                        circle = mpatches.Circle(
                            (cx, cy), sigma,
                            fill=False, edgecolor="black", linewidth=1.0,
                            linestyle="--", alpha=0.7, zorder=4,
                        )
                        pred_ax.add_patch(circle)

            pred_ax.set_xlabel(xlabel)
            pred_ax.set_ylabel(ylabel)
            title_parts = ["XY", "VX/VY"]
            ann_parts = ["centers + Rh circles", "centers + \u03c3 circles"]
            pred_ax.set_title(f"{title_parts[row]} \u2014 Orchestrator ({ann_parts[row]})")
            pred_ax.set_aspect("equal")

        fig.suptitle(
            f"Mock simulation \u2014 {coords.shape[0]:,} particles, "
            f"{len(unique_pred)} predicted galaxies, "
            f"snapshot {snap_k} (z={z_k:.4f}, t={t_k:.3f} Gyr)",
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = os.path.join(output_dir, f"comparison{snap_k:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Snap {snap_k}: saved {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()

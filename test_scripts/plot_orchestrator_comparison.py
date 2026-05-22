#!/usr/bin/env python3
"""Plot mock snapshot: ground truth vs orchestrator output side by side.

Left column: ground truth (galaxy_id from mock data).
Right column: orchestrator assignment (Sub_tree_id from hard_assignment).
On the right column, overlay galaxy centers, Rh (half-mass radius) circles
on XY, and sigma circles on VX/VY.

Usage:
    python test_scripts/plot_orchestrator_comparison.py
"""

import os
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "test_data", "mock_snap_tight")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "mock_tight_comparison")


def main():
    particles = np.load(os.path.join(DATA_DIR, "particles.npz"))
    coords = particles["coords"]
    true_id = particles["galaxy_id"]

    with h5py.File(os.path.join(OUTPUT_DIR, "assignment.hdf5"), "r") as f:
        hard_assign = f["snapshots"]["0"]["hard_assignment"][:]
    pred_id = hard_assign["Sub_tree_id"]

    with h5py.File(os.path.join(OUTPUT_DIR, "catalogue.hdf5"), "r") as f:
        gp = f["snapshots"]["0"]["galaxy_properties"][:]

    unique_true = np.unique(true_id)
    unique_pred = np.unique(pred_id)
    n_colors = max(len(unique_true), len(unique_pred))
    colors = plt.cm.tab20(np.linspace(0, 1, n_colors))

    id_to_center_x = {int(row["Sub_tree_id"]): row["position_x"] for row in gp}
    id_to_center_y = {int(row["Sub_tree_id"]): row["position_y"] for row in gp}
    id_to_center_vx = {int(row["Sub_tree_id"]): row["velocity_x"] for row in gp}
    id_to_center_vy = {int(row["Sub_tree_id"]): row["velocity_y"] for row in gp}
    id_to_rh = {int(row["Sub_tree_id"]): row["rh"] for row in gp}
    id_to_sigma = {int(row["Sub_tree_id"]): row["sigma"] for row in gp}

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    datasets = [
        (coords[:, 0], coords[:, 1], "x [kpc]", "y [kpc]"),
        (coords[:, 3], coords[:, 4], "vx [km/s]", "vy [km/s]"),
    ]

    for row in range(2):
        x_data, y_data, xlabel, ylabel = datasets[row]

        true_ax = axes[row, 0]
        pred_ax = axes[row, 1]

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

        for i, pid in enumerate(unique_pred):
            mask = pred_id == pid
            pred_ax.scatter(
                x_data[mask], y_data[mask],
                s=0.5, c=[colors[i % 20]], alpha=0.6,
            )

        for sid in unique_pred:
            sid_int = int(sid)
            if sid_int not in id_to_center_x:
                continue
            cx = id_to_center_x[sid_int]
            cy = id_to_center_y[sid_int]
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
        f"Mock snapshot tight \u2014 {coords.shape[0]:,} particles, "
        f"{len(unique_true)} galaxies, snapshot 0",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUTPUT_DIR, "orchestrator_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
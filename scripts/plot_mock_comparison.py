#!/usr/bin/env python3
"""Plot mock snapshot: ground truth vs pipeline assignment side by side.

Usage:
  python scripts/plot_mock_comparison.py test_data/mock_snap -o figures/mock_comparison.png
  python scripts/plot_mock_comparison.py test_data/mock_snap_tight -o figures/mock_comparison_tight.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_comparison(data_dir, output="figures/mock_comparison.png"):
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    assignment = pd.read_csv(os.path.join(data_dir, "assignment.csv"))

    coords = particles["coords"]
    true_id = particles["galaxy_id"]
    pred_id = assignment["Sub_tree_id"].values
    unique_ids = np.unique(true_id)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    datasets = [
        (coords[:, 0], coords[:, 1], "x [kpc]", "y [kpc]"),
        (coords[:, 3], coords[:, 4], "vx [km/s]", "vy [km/s]"),
    ]

    for row, (true_ax, pred_ax) in enumerate([
        (axes[0, 0], axes[0, 1]),
        (axes[1, 0], axes[1, 1]),
    ]):
        x_data, y_data, xlabel, ylabel = datasets[row]

        for i, gid in enumerate(np.unique(true_id)):
            mask = true_id == gid
            true_ax.scatter(
                x_data[mask], y_data[mask],
                s=0.5, c=[colors[i % 10]], alpha=0.6,
            )
        true_ax.set_xlabel(xlabel)
        true_ax.set_ylabel(ylabel)
        true_ax.set_title(["XY — Ground truth", "VX/VY — Ground truth"][row])
        true_ax.set_aspect("equal")

        for i, pid in enumerate(np.unique(pred_id)):
            mask = pred_id == pid
            pred_ax.scatter(
                x_data[mask], y_data[mask],
                s=0.5, c=[colors[i % 10]], alpha=0.6,
            )
        pred_ax.set_xlabel(xlabel)
        pred_ax.set_ylabel(ylabel)
        pred_ax.set_title(["XY — Pipeline assignment", "VX/VY — Pipeline assignment"][row])
        pred_ax.set_aspect("equal")

    fig.suptitle(
        f"Mock snapshot — {coords.shape[0]:,} particles, "
        f"{len(unique_ids)} galaxies",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="test_data/mock_snap")
    parser.add_argument("--output", "-o", default="figures/mock_comparison.png")
    args = parser.parse_args()
    plot_comparison(args.data_dir, args.output)

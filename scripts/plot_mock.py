#!/usr/bin/env python3
"""Plot XY positions of a mock snapshot dataset.

Usage:
  python scripts/plot_mock.py test_data/mock_snap
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mock_dataset(data_dir, output="mock_xy.png"):
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))

    coords = particles["coords"]
    galaxy_id = particles["galaxy_id"]
    unique_ids = np.unique(galaxy_id)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot particles colored by galaxy
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))
    for i, gid in enumerate(unique_ids):
        mask = galaxy_id == gid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=0.5, c=[colors[i]], alpha=0.6, label=f"G{gid}",
        )

    # Overplot galaxy centres from merger tree
    ax.scatter(
        tree["position_x"], tree["position_y"],
        c="black", marker="x", s=80, linewidths=2, label="Centres",
    )

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"Mock snapshot — {len(particles['masses']):,} particles, "
                 f"{len(tree)} galaxies")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=6, ncol=2)

    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mock snapshot")
    parser.add_argument("data_dir", nargs="?", default="test_data/mock_snap",
                        help="Path to mock snapshot directory")
    parser.add_argument("--output", "-o", default="mock_xy.png",
                        help="Output filename")
    args = parser.parse_args()
    plot_mock_dataset(args.data_dir, args.output)

#!/usr/bin/env python3
"""Plot XY positions of a mock snapshot dataset.

Usage:
  python scripts/plot_mock.py test_data/mock_snap -o figures/mock_xy.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mock_dataset(data_dir, output="figures/mock_xy.png"):
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))

    coords = particles["coords"]
    galaxy_id = particles["galaxy_id"]
    unique_ids = np.unique(galaxy_id)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, gid in enumerate(unique_ids):
        mask = galaxy_id == gid
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=0.5, c=[colors[i % 10]], alpha=0.6,
        )

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"Mock snapshot — {len(particles['masses']):,} particles, "
                 f"{len(tree)} galaxies, {tree['mass'].iloc[0]:.0e} M$_{{\odot}}$")
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mock snapshot")
    parser.add_argument("data_dir", nargs="?", default="test_data/mock_snap",
                        help="Path to mock snapshot directory")
    parser.add_argument("--output", "-o", default="figures/mock_xy.png",
                        help="Output filename")
    args = parser.parse_args()
    plot_mock_dataset(args.data_dir, args.output)

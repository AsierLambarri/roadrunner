#!/usr/bin/env python3
"""Plot vx, vy of a mock snapshot dataset.

Usage:
  python test_scripts/plot_mock_vel.py test_data/mock_snap -o figures/mock_vel.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mock_velocity(data_dir, output="figures/mock_vel.png"):
    particles = np.load(os.path.join(data_dir, "particles.npz"))
    tree = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))

    coords = particles["coords"]
    galaxy_id = particles["galaxy_id"]
    unique_ids = np.unique(galaxy_id)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, gid in enumerate(unique_ids):
        mask = galaxy_id == gid
        ax.scatter(
            coords[mask, 3], coords[mask, 4],
            s=0.5, c=[colors[i % 10]], alpha=0.6,
        )

    ax.scatter(
        tree["velocity_x"], tree["velocity_y"],
        marker="x", c="black", s=60, linewidths=1.5,
        label="Centres",
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors[i % 10], markersize=4,
                   label=f"G{gid}")
        for i, gid in enumerate(unique_ids)
    ]
    handles.append(
        plt.Line2D([0], [0], marker="x", color="black",
                   markersize=6, linewidth=1.5, label="Centre")
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=5, ncol=2, frameon=True)

    ax.set_xlabel("vx [km/s]")
    ax.set_ylabel("vy [km/s]")
    ax.set_title(f"Mock snapshot — {len(particles['masses']):,} particles, "
                 f"{len(tree)} galaxies")
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mock snapshot velocity")
    parser.add_argument("data_dir", nargs="?", default="test_data/mock_snap",
                        help="Path to mock snapshot directory")
    parser.add_argument("--output", "-o", default="figures/mock_vel.png",
                        help="Output filename")
    args = parser.parse_args()
    plot_mock_velocity(args.data_dir, args.output)

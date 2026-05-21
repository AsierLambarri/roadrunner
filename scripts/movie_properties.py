#!/usr/bin/env python3
"""Generate per-snapshot property plots for a movie.

Usage:
  python scripts/movie_properties.py --output-dir ./output \
      --subtree 1 --output-dir-frames ./frames/
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
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--subtree", type=int, default=None,
                        help="Sub_tree_id to highlight (default: all)")
    parser.add_argument("--output-dir-frames", type=str,
                        default="./frames",
                        help="Output directory for frame images")
    args = parser.parse_args()

    cat_path = os.path.join(args.output_dir, "catalogue.hdf5")
    reader = HDF5CatalogueReader(cat_path)
    hdr = reader.read_header()
    snapshots = hdr["snapshots"]

    if not snapshots:
        print("No snapshots found")
        return

    os.makedirs(args.output_dir_frames, exist_ok=True)

    for snap in snapshots:
        props = reader.read_galaxy_properties(snapshot_id=snap)
        if props.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].scatter(props["rh"], props["Mtot"], s=10, alpha=0.6)
        axes[0, 0].set_xlabel("rh [kpc]")
        axes[0, 0].set_ylabel("Mtot [Msun]")
        axes[0, 0].set_yscale("log")

        axes[0, 1].scatter(props["sigma"], props["r_t"], s=10, alpha=0.6)
        axes[0, 1].set_xlabel("sigma [km/s]")
        axes[0, 1].set_ylabel("r_t [kpc]")

        axes[1, 0].scatter(props["sigma"], props["rh"], s=10, alpha=0.6,
                           c=props["dynstate"] if "dynstate" in props.columns
                           else None)
        axes[1, 0].set_xlabel("sigma [km/s]")
        axes[1, 0].set_ylabel("rh [kpc]")

        axes[1, 1].hist(props["dynstate"] if "dynstate" in props.columns
                        else [0], bins=3, range=(-0.5, 2.5), alpha=0.7)
        axes[1, 1].set_xlabel("Dynamical state")
        axes[1, 1].set_ylabel("Count")

        fig.suptitle(f"Snapshot {snap} — {len(props)} galaxies")
        plt.tight_layout()

        out_path = os.path.join(args.output_dir_frames,
                                f"snap_{snap:04d}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Generated {len(snapshots)} frames in {args.output_dir_frames}")


if __name__ == "__main__":
    main()

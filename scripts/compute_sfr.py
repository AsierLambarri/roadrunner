#!/usr/bin/env python3
"""Compute star formation rate from birth catalogue.

Usage:
  python scripts/compute_sfr.py --catalogue ./output --output figures/sfr.png
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
    parser.add_argument("--output", "-o", type=str, default="figures/sfr.png")
    parser.add_argument("--bin-width", type=float, default=0.1,
                        help="Bin width in Gyr")
    args = parser.parse_args()

    cat_path = os.path.join(args.output_dir, "catalogue.hdf5")
    reader = HDF5CatalogueReader(cat_path)
    hdr = reader.read_header()

    births = reader.read_births()
    if births.empty:
        print("No birth records found")
        return

    birth_times = births.get("birth_time")
    if birth_times is None:
        print("No birth_time column in births (only birth_id available)")
        return

    t_min, t_max = birth_times.min(), birth_times.max()
    bins = np.arange(t_min, t_max + args.bin_width, args.bin_width)
    counts, edges = np.histogram(birth_times, bins=bins)

    # SFR = counts * particle_mass / bin_width
    particle_mass = 1e4  # Msun per particle (from mock default)
    sfr = counts * particle_mass / args.bin_width  # Msun / Gyr
    t_center = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_center, sfr, drawstyle="steps-mid")
    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel("SFR [M$_\odot$ / Gyr]")
    ax.set_title(f"Star Formation Rate — {len(births)} particles")
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

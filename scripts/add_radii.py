#!/usr/bin/env python3
"""Recompute galaxy radii (r20, rh, r80) from catalogue + merger tree.

Usage:
  python scripts/add_radii.py --output-dir ./output \
      --merger-tree /path/to/merger_tree.csv
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.io.hdf5_reader import HDF5CatalogueReader
from roadrunner.postprocessing.properties import half_mass_radius, find_center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./output")
    args = parser.parse_args()

    cat_path = os.path.join(args.output_dir, "catalogue.hdf5")
    assign_path = os.path.join(args.output_dir, "assignment.hdf5")
    parts_path = os.path.join(args.output_dir, "particles.hdf5")

    reader = HDF5CatalogueReader(cat_path)
    snapshots = reader.read_header()["snapshots"]

    if not os.path.exists(parts_path):
        print("No particles.hdf5 — cannot recompute radii")
        return

    for snap in snapshots:
        print(f"Recomputing radii for snapshot {snap}...")

        with h5py.File(assign_path, "r") as hf:
            hard = pd.DataFrame.from_records(
                hf[f"/snapshots/{snap}/hard_assignment"][()]
            )

        with h5py.File(parts_path, "r") as hf:
            grp = hf[f"/snapshots/{snap}"]
            mean = grp["scaler/mean"][:6]
            scale = grp["scaler/scale"][:6]
            pos = grp["positions"][:] / scale[:3] + mean[:3]
            vel = grp["velocities"][:] / scale[3:6] + mean[3:6]
            masses = grp["masses"][:]

        results = []
        for gid in hard["Sub_tree_id"].unique():
            if gid == -1:
                continue
            mask = hard["Sub_tree_id"] == gid
            idx = hard.loc[mask, "array_index"].values.astype(int)
            gp, gv = find_center(pos[idx], vel[idx], masses[idx])
            cp, cv = pos[idx] - gp, vel[idx] - gv
            rh = half_mass_radius(cp, masses[idx], np.zeros(3), mass_fraction=0.5)
            results.append({"Sub_tree_id": gid, "rh": rh})

        out = pd.DataFrame(results)
        print(f"  Computed rh for {len(out)} galaxies: "
              f"min={out['rh'].min():.2f}, max={out['rh'].max():.2f}")


if __name__ == "__main__":
    main()

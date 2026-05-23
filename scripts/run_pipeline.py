#!/usr/bin/env python3
"""Run the roadrunner accretion pipeline from the command line.

Reads a merger tree CSV and simulation snapshots, processes all
snapshots, and writes output catalogues.

Usage:
  python scripts/run_pipeline.py \
      --merger-tree /path/to/tree.csv \
      --equivalence /path/to/equiv.csv \
      --code RAMSES \
      --output-dir ./results/ \
      --halo-id 12345

  python scripts/run_pipeline.py --config config.yaml
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.pipeline import run_accretion_history, RunConfig


def build_config_from_args(args):
    return RunConfig(
        merger_tree_path=args.merger_tree,
        particle_data_dir=args.particle_dir or args.output_dir,
        equivalence_path=args.equivalence,
        code=args.code,
        ptype=args.ptype,
        fields={
            "index": args.index_field,
            "mass": args.mass_field,
            "position": args.position_field,
            "velocity": args.velocity_field,
            "metallicity": args.metallicity_field,
        } if args.metallicity_field else {
            "index": args.index_field,
            "mass": args.mass_field,
            "position": args.position_field,
            "velocity": args.velocity_field,
        },
        accretion_id=args.halo_id,
        start_snapshot=args.start_snapshot,
        end_snapshot=args.end_snapshot,
        halo_model=args.halo_model,
        cov_type=args.cov_type,
        max_iter=args.max_iter,
        tol=args.tol,
        min_particles=args.min_particles,
        n_los=args.n_los,
        search_factor=args.search_factor,
        birth_window_factor=args.birth_window,
        output_dir=args.output_dir,
        save_particles=args.save_particles,
        save_assignment=args.save_assignment,
    )


def main():
    parser = argparse.ArgumentParser(description="Accretion pipeline runner")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (alternative to CLI args)")

    # Data
    parser.add_argument("--merger-tree", type=str, required=True,
                        help="Path to merger tree CSV")
    parser.add_argument("--equivalence", type=str, required=True,
                        help="Path to equivalence table CSV")
    parser.add_argument("--particle-dir", type=str, default=None,
                        help="Base directory for snapshot files")

    # Simulation code
    parser.add_argument("--code", type=str, default="RAMSES",
                        choices=["ART", "ART-I", "GEAR", "AURIGA",
                                 "AREPO", "RAMSES", "VINTERGATAN"],
                        help="Simulation code")
    parser.add_argument("--ptype", type=str, default="star",
                        help="Particle type filter")

    # Field names
    parser.add_argument("--index-field", type=str, default="particle_index")
    parser.add_argument("--mass-field", type=str, default="particle_mass")
    parser.add_argument("--position-field", type=str, default="coordinates")
    parser.add_argument("--velocity-field", type=str, default="particle_velocity")
    parser.add_argument("--metallicity-field", type=str, default=None)

    # Target
    parser.add_argument("--halo-id", type=int, default=None,
                        help="Accretion host Sub_tree_id "
                        "(auto-detect from last snapshot if omitted)")
    parser.add_argument("--start-snapshot", type=int, default=None)
    parser.add_argument("--end-snapshot", type=int, default=None)

    # GMM
    parser.add_argument("--cov-type", type=str, default="full",
                        choices=["full", "diagonal", "spherical"])
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--min-particles", type=int, default=10)

    # Physics
    parser.add_argument("--halo-model", type=str, default="kepler",
                        choices=["kepler", "nfw"])
    parser.add_argument("--n-los", type=int, default=15)
    parser.add_argument("--search-factor", type=float, default=1.0)

    # Tracking
    parser.add_argument("--birth-window", type=float, default=5.0)

    # Output
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--save-particles", action="store_true",
                        help="Save particle positions/velocities")
    parser.add_argument("--save-assignment", action="store_true", default=True,
                        help="Save soft assignment data")

    args = parser.parse_args()

    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                config = RunConfig(**yaml.safe_load(f))
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config = build_config_from_args(args)

    print(f"Running accretion pipeline for halo {config.accretion_id}")
    print(f"  Merger tree:  {config.merger_tree_path}")
    print(f"  Equivalence:  {config.equivalence_path}")
    print(f"  Output dir:   {config.output_dir}")
    print(f"  Snapshots:    {config.start_snapshot or 'first'} → "
          f"{config.end_snapshot or 'last'}")

    run_accretion_history(config)
    print(f"\nDone. Output in {config.output_dir}")


if __name__ == "__main__":
    main()

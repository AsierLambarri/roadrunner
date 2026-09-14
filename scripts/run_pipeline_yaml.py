#!/usr/bin/env python3
"""Run the roadrunner accretion pipeline from a YAML config file.

Usage:
  python scripts/run_pipeline_yaml.py --config path/to/config.yaml
  python scripts/run_pipeline_yaml.py -c path/to/config.yaml
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from roadrunner.pipeline import run_accretion_history, RunConfig


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run pipeline from YAML config")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml",
              file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.config) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config file: {e}", file=sys.stderr)
        sys.exit(1)

    config = RunConfig(**data)

    print(f"Running accretion pipeline for halo {config.accretion_id}")
    print(f"  Merger tree:  {config.merger_tree_path}")
    print(f"  Equivalence:  {config.equivalence_path}")
    print(f"  Output dir:   {config.output_dir}")
    print(f"  Snapshots:    {config.start_snapshot or 'first'} \u2192 "
          f"{config.end_snapshot or 'last'}")
    if config.selection_snapshot is not None:
        print(f"  Selection:    snapshot {config.selection_snapshot} "
              f"sphere={config.selection_sphere} bbox={config.selection_bbox}")

    run_accretion_history(config)
    print(f"\nDone. Output in {config.output_dir}")


if __name__ == "__main__":
    main()

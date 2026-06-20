# roadrunner

Unravel the accretion history of simulated galaxies.

roadrunner takes dark-matter and star particles from cosmological hydrodynamical
simulations (RAMSES, AREPO, ART, etc.) together with merger-tree catalogues and
produces a detailed account of how each galaxy assembles its stellar mass over
cosmic time: which particles were born *in situ* versus accreted from satellites,
when and from which progenitor they were stripped, and how the dynamical state
of the accreted material evolves across snapshots.

## Key features

- **Multi-method assignment** — particle-to-halo allocation via GMM, Bayesian GMM,
  or stochastic variational inference (SVI-BGMM), all under a single
  `XGMMAssigner` interface.
- **Boundness + segmentation** — configurable Keplerian or NFW potential models
  for unbound-particle filtering and tidal-radius segmentation.
- **Checkpoint/resume** — per-snapshot state saved with zstandard compression;
  restart from the last successful snapshot on failure.
- **Output** — HDF5 catalogues, per-particle assignments (soft labels, timescales),
  and snapshot-particle data.
- **Downstream analysis** — birth tracking (when a particle first appears in a
  galaxy), assembly tracking (accretion-origin groups), dynamical-state
  classification, and mixing metrics.

## Quick start

```sh
pip install -e .[dev]
python scripts/run_pipeline.py --help
```

Or with a YAML config:

```sh
python scripts/run_pipeline_yaml.py scripts/run_pipeline.yaml.example
```

## Requirements

- Python ≥ 3.12
- numpy, scipy, numba, pandas, h5py, yt, zstandard, PyYAML

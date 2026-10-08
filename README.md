# roadrunner

Unravel the accretion history of simulated galaxies.

roadrunner takes star particles from cosmological simulation snapshots and a halo merger tree, and assigns each star to a halo snapshot by snapshot. For every snapshot it finds the particles gravitationally bound to each halo. It groups halos whose virial spheres overlap, and inside each group it fits a Gaussian mixture in 6-D phase space (GMM or variational Bayesian GMM). Each mixture component is one halo, and the fit is restricted to the particles bound to it. The previous snapshot's responsibilities and fitted parameters are carried forward. From the assignments it builds a galaxy catalogue (structural and kinematic properties, satellite relations, dynamical state). It also tracks where every star was born and how each galaxy's stellar content was assembled, so in-situ stars can be told apart from accreted ones.

## Installation

You need Python 3.12 or newer.

```sh
git clone <repository-url> roadrunner
cd roadrunner
pip install -e ".[dev]"
```

Runtime dependencies (`pyproject.toml`): `numpy<=1.26.4`, `numba`, `scipy`, `scikit-learn`, `threadpoolctl`, `pandas`, `h5py`, `yt`, `tqdm`, `zstandard`, `PyYAML`. The `dev` extra adds `pytest`, `pytest-cov`, `ruff` and `mypy`. The version comes from git tags via `setuptools_scm`.

Some scripts in `scripts/` need packages that are not declared as dependencies. The mock generators need `limepy`, and the plotting scripts need `matplotlib`:

```sh
pip install limepy matplotlib
```

The runner scripts import the package from `src/` directly unless `PYTHONPATH` already provides `roadrunner`, so they also work without installing.

## Quick start

### Inputs

| Input | Format |
|---|---|
| Merger tree (`merger_tree_path`) | CSV, one row per halo per snapshot. Columns used: `Snapshot`, `Sub_tree_id` (integer), `Redshift`, `mass` (Msun), `position_x/y/z`, `velocity_x/y/z` (km/s), `virial_radius`, `scale_radius` (NaN entries are filled from the Duffy concentration–mass relation). Lengths are comoving kpc when `comoving: true`. |
| Equivalence table (`equivalence_path`) | CSV with columns `snapshot`, `snapname`, `time`, `redshift`. `snapname` is resolved relative to `particle_data_dir`. |
| Particle data (`particle_data_dir`) | Snapshot files read by one of the three readers (see [Readers](#1-read)). |

### Running

Running from a YAML file is the recommended way:

```sh
cp scripts/run_pipeline.yaml.example my_run.yaml   # edit the paths
python scripts/run_pipeline_yaml.py --config my_run.yaml
```

`scripts/run_pipeline_yaml_npz.py` is the same runner. Without `--config` it defaults to `scripts/run_pipeline_pdata.yaml`, and it also prints the reader type:

```sh
python scripts/run_pipeline_yaml_npz.py --config my_run.yaml
```

`scripts/run_pipeline.py` exposes a subset of the options as command-line flags:

```sh
python scripts/run_pipeline.py \
    --merger-tree /path/to/tree.csv \
    --equivalence /path/to/equiv.csv \
    --particle-dir /path/to/snapshots \
    --reader-type yt --code RAMSES --ptype star \
    --assignment-method bgmm \
    --output-dir ./results --halo-id 12345
```

Notes on `run_pipeline.py`:

- Flags: `--reader-type {yt,npz,pdata}`, `--code`, `--ptype`, the `--index-field`, `--mass-field`, `--position-field` and `--velocity-field` mappings, `--extra-field NAME:YT_FIELD` and `--assign-field ATTR` (both repeatable), `--halo-id`, `--start-snapshot`, `--end-snapshot`, `--assignment-method {gmm,bgmm}`, `--use-bgmm-priors true|false`, `--data-precision`, `--math-precision`, `--cov-type`, `--max-iter`, `--tol`, `--min-particles`, `--halo-model {kepler,nfw}`, `--n-los`, `--search-factor`, `--birth-window`, `--output-dir`, `--resume`, `--save-particles`, `--save-assignment`.
- `--save-particles` is off unless you pass it. In YAML, `save_particles` defaults to true.
- `--config FILE` replaces every other flag with the YAML contents. argparse still requires `--merger-tree` and `--equivalence` to be present, so use `run_pipeline_yaml.py` for YAML runs.
- Options without a flag (`mass_weighting`, `reg_covar`, `seed`, `threads`, the selection keys, centring keys, ...) are available only through YAML.

### End-to-end run on mock data

```sh
# 1. a single z=0 mock snapshot of limepy galaxies (needs limepy)
python scripts/generate_mock.py --n-galaxies 40 --n-groups 3 --max-particles 40000 \
    --output-dir test_data/mock_snap
# 2. evolve it into a multi-snapshot mock simulation
python scripts/generate_mock_simulation.py --input-dir test_data/mock_snap \
    --output-dir test_data/mock_simulation --n-snap 11 --n-newborn 1000
```

```yaml
# mock.yaml
reader_type: npz
npz_mock_sim: true
merger_tree_path: test_data/mock_simulation/merger_tree.csv
equivalence_path: test_data/mock_simulation/equivalence.csv
particle_data_dir: test_data/mock_simulation/
assignment_method: bgmm
output_dir: ./output_mock
```

```sh
python scripts/run_pipeline_yaml.py -c mock.yaml
```

`scripts/run_pipeline_npz.yaml.example` is a fuller version of this config. Its paths are absolute and must be edited.

The library entry point is `roadrunner.run_accretion_history(config)`. It takes a `RunConfig` or a plain dict with the same keys.

## Pipeline overview

`run_accretion_history` builds the components. `AccretionPipeline` then loops over the snapshots (oldest first), and `SnapshotOrchestrator` runs each snapshot through the steps below. The docstrings in the named modules give the details.

### 1. Read

| `reader_type` | Class (`src/roadrunner/readers/`) | Source |
|---|---|---|
| `yt` | `SnapshotReader` | Any snapshot yt can load. Code-specific setup for `ART`, `ART-I`, `GEAR`, `AURIGA`, `AREPO`, `RAMSES` and `VINTERGATAN`. Fields are mapped through `fields`, and data are converted to Msun, comoving kpc, km/s and Gyr. |
| `npz` | `NPZSnapshotReader` | With `npz_mock_sim: true`, the roadrunner mock format: `indices`, `masses`, and `coords` (N x 6). Otherwise a single 2-D array with columns `[id, mass, x, y, z, vx, vy, vz, metallicity]`. |
| `pdata` | `ParticleDataSnapshotReader` | roadrunner's own `particle_data/snapshotNNNN.hdf5` files (see [Outputs](#outputs)), so a previous run's particle output can be read back. |

All readers support a particle selection. With `selection_snapshot` plus one of `selection_sphere` or `selection_bbox`, the IDs inside the region at that snapshot are the only particles loaded at every snapshot. If `accretion_id` is omitted, the most massive halo in the last snapshot becomes the accretion host.

### 2. Boundness (`physics/boundness.py`, `physics/potentials.py`)

Each merger-tree halo becomes a `HaloModel` with a Kepler (a softened point mass: a Plummer sphere with a = 1e-3 kpc) or NFW potential (`halo_model`). A KD-tree finds the particles within `search_factor x virial_radius`. A particle is bound when its specific energy E = Φ + v²/2 is negative. Its boundness is -E over the halo's binding energy scale: -E/v_vir² for Kepler, E/Φ₀ (in (0, 1)) for NFW. Per-particle dynamical timescales are computed here too.

### 3. Overlap groups and ownership (`clustering/segmentation.py`)

Halos with bound candidates are linked into overlap groups with union-find over overlapping virial spheres. Halos with `min_particles` candidates or fewer are split out of their group into singleton groups. Before any fit, contested particles are resolved: a small halo always keeps its candidates over the large halos of the group it was split from, and between two small halos the higher boundness wins.

### 4. Assignment (`clustering/assignment/gmm.py`, `XGMMAssigner`)

Each group is fitted on standardised 6-D phase-space coordinates. One of three paths applies:

- **Single halo:** every bound particle gets responsibility 1, and the Gaussian parameters are the moments of those particles.
- **Unresolved** (fewer than `10 x n_halos` particles): normalised boundness is used as the responsibilities, with no EM.
- **Resolved:** a full mixture fit (`mixture/weighted_gmm.py` for `gmm`, `mixture/bayesian_gmm.py` for `bgmm`). It runs in the chosen `math_precision` and retries in double precision on a numerical failure.

The E-step is masked by a latent prior a_nk, the row-normalised plausibility of the particle's boundness (`plausibility`, `clustering/assignment/plausibility.py`), so a particle can only belong to halos it is bound to:

- `rank`: log1p of the boundness rank within each halo.
- `energy` (NFW only): a likelihood ratio on the particle's rank u in the halo's dark-matter energy distribution. It is a histogram of the previous snapshot's members in t = -ln u (refit between snapshots, outside EM), mixed with Errani et al. (2022)'s tagging ratio, with r½ = 0.015 R_vir, and floored.
- `phase` (any potential: Kepler in closed form, NFW tabulated): a likelihood ratio on the fraction w of the halo's bound phase space more bound than the particle, with a log-normal member model in t = -ln w. The model is the previous snapshot's members' moments, mixed with circular orbits around r½ = 0.015 R_vir, and floored.
- `kinematic` (any potential): co-movement. The particle's speed relative to each halo in units of that halo's escape speed at its position, u² = \|v − v_k\|² / v_esc²(r), so the potential at the particle's position carries no evidence (ν = u³ is uniform for a smooth background). Members follow an isotropic equilibrium tracer, hot or cold (u² ~ Beta(3/2, q+1)). Each halo's ⟨u²⟩ is learned from its previous members and shrunk to the pooled value, mixed with the Wolf et al. (2010) value v_c²/v_esc² at r½, and floored.

```
log r_nk = log a_nk + log π_k + log N(x_n | μ_k, Σ_k) - log Z_n
```

How the resolved path works:

- **Mass weighting** (`mass_weighting: true`): particles are weighted by their mass instead of counted. The weights are normalised to the group's effective sample size (Σm)²/Σm², so equal masses reproduce the unweighted fit exactly.
- **Temporal smoothing** (`_initial_responsibilities`): the fit starts from the previous snapshot's responsibilities. A particle bound to a halo it had no previous responsibility for (newborn, newly bound, or arriving from another group) has its row filled by a predictive E-step, `log a_nk(t) + log π_k(t-1) + log N(x_n | m_k(t), Σ_k(t-1))`. Here m_k(t) is the halo's current tree position and velocity, and the previous covariance is grown by the tree-mass factor below. Halos without a usable history use the covariance of their exclusive (or D+1 most bound) particles.
- **BGMM priors** (`clustering/assignment/priors.py`, `bgmm` only, `use_bgmm_priors: true`). Every component gets explicit priors built from a reference count n_ref and reference variances. These come from the previous snapshot's fit when it is usable, and otherwise from the halo's own pre-fit estimate:
  - mean prior: the halo's current tree position and velocity;
  - weight concentration: `min(n_ref, n_bound) / 5`; mean precision: `min(n_ref, n_bound) / 20`;
  - covariance prior: `ν0 · g · f(n_ref) · diag(σ²_ref)` with ν0 = D + 4. The growth factor is `g = clip((M_t / M_{t-1})^(2/3), 0.05, 2)`, from the tree masses. `f(n)` goes linearly from 1.0 at n ≤ 10 to 0.7 at n ≥ 300.
- **Rank-deficient fits:** a previous fit from fewer than D + 1 particles is flagged `rank_deficient`. It is not used as history (neither for priors nor for the predictive E-step), and it is left out of the condition-number diagnostics.

The constants live in `src/roadrunner/_defaults.py`.

The hard label of each particle is the argmax of its responsibilities. Particles bound to no halo get `Sub_tree_id = -1`.

> `assignment_method: svi-bgmm` (`mixture/svi_bayesian_gmm.py`, keys `svi_iters` and `svi_batch_size`) is still accepted by the code. It is experimental and unvalidated, and it does not support `mass_weighting`. Do not use it for science runs.

### 5. Trackers (`postprocessing/tracking/`)

- **`BirthTracker`:** a particle is born in a galaxy it is bound to when it first appears. Evidence, weighted by responsibility and by a window in units of the particle's dynamical timescale, is collected for `birth_window_factor x timescale`. The host with the most evidence becomes the birth host. With `enforce_initial_hosts`, only the hosts the particle was bound to at first appearance can compete.
- **`AssemblyTracker`:** keeps an infall list per galaxy, meaning every particle it has ever acquired. That includes stars born in it, stars it acquired directly, and the infall lists of satellites that merged into it, using the merger-tree satellite relations.

Setting `birth_window_factor <= 0` turns off both trackers.

### 6. Reduction (`pipeline/reduction.py`, `postprocessing/`)

Per-galaxy properties are computed from the particles that are both bound and in the galaxy's infall list. These are the centre (from the GMM mean when `use_gmm_centers`, otherwise from shrinking-sphere centring), stellar mass, r20/rh/r80, 3-D σ, projected Rhp and σ_los (median over `n_los` random lines of sight), and the tidal radius. On the snapshots chosen by `dynstate_snapshots`, the Riley dynamical-state criterion labels each satellite as relaxed (0), disturbed (1) or mixed (2).

Runs are deterministic given `seed`. Per-snapshot, per-stage seeds are derived from it (`randomness.py`). `threads` caps numba, BLAS/OpenMP and KD-tree workers together (`threads.py`).

## Configuration

All keys are fields of the frozen dataclass `RunConfig` (`src/roadrunner/pipeline/config.py`). A YAML file is passed as `RunConfig(**yaml)`, so an unknown key is an error.

**Data**

| Key | Default | Meaning |
|---|---|---|
| `merger_tree_path` | `""` | Merger-tree CSV. |
| `equivalence_path` | `""` | Equivalence-table CSV. |
| `particle_data_dir` | `""` | Base directory for `snapname`. |
| `reader_type` | `yt` | `yt`, `npz` or `pdata`. |
| `code` | `RAMSES` | yt reader: `ART`, `ART-I`, `GEAR`, `AURIGA`, `AREPO`, `RAMSES`, `VINTERGATAN`. |
| `ptype` | `star` | yt reader particle type. |
| `fields` | `index: particle_index`, `mass: particle_mass`, `position: coordinates`, `velocity: particle_velocity` | Canonical name to field name. The four keys are required. Extra keys are loaded and saved, but not fed to the assigner. |
| `unit_base` | `null` | yt `unit_base` (GEAR, AURIGA, AREPO). |
| `assign_fields` | `null` (= `[position, velocity]`) | `SnapshotData` attributes that form the assigner input. |
| `npz_mock_sim` | `false` | npz reader: roadrunner mock format. |

**Target and range**

| Key | Default | Meaning |
|---|---|---|
| `accretion_id` | `null` | Host `Sub_tree_id`. When null, the most massive halo at the last snapshot. |
| `start_snapshot`, `end_snapshot` | `null` | Snapshot range. Negative values index from the end (`-1` = last). |
| `selection_snapshot` | `null` | Reference snapshot for the particle selection (negative allowed). |
| `selection_sphere` | `null` | `[[cx, cy, cz], r]`, comoving kpc. |
| `selection_bbox` | `null` | `[[xlo, ylo, zlo], [xhi, yhi, zhi]]`, comoving kpc. |

**Assigner**

| Key | Default | Meaning |
|---|---|---|
| `assignment_method` | `gmm` | `gmm` (EM) or `bgmm` (variational Bayesian). `svi-bgmm` is experimental. |
| `use_bgmm_priors` | `true` | Build BGMM priors from the previous snapshot's fit. |
| `mass_weighting` | `false` | Weight particles by mass in the fits. |
| `cov_type` | `full` | `full`, `diagonal` or `spherical`. |
| `max_iter` | `10` | Maximum EM / variational iterations. |
| `tol` | `1e-2` | Convergence tolerance. |
| `reg_covar` | `1e-6` | Covariance regularisation. |
| `min_particles` | `10` | Halos with at most this many candidates leave their overlap group. |
| `data_precision`, `math_precision` | `single` | `single` or `double`, for loaded data and for compute kernels. |
| `svi_iters`, `svi_batch_size` | `1000`, `10000` | SVI only (experimental). |

**Physics, properties, trackers**

| Key | Default | Meaning |
|---|---|---|
| `halo_model` | `kepler` | Potential: `kepler` or `nfw`. |
| `plausibility` | `rank` | Latent prior: `rank`, `energy` (NFW only), `phase` or `kinematic`. |
| `search_factor` | `1.0` | Boundness search radius in units of the virial radius. |
| `comoving` | `true` | Merger-tree and particle lengths are comoving kpc. |
| `n_los` | `15` | Lines of sight for projected properties. |
| `use_gmm_centers` | `true` | Use the fitted means as galaxy centres. |
| `min_particles_structural` | `30` | Minimum particles for structural properties. |
| `ssc_nmin`, `ssc_alpha` | `30`, `0.9` | Shrinking-sphere minimum particles (raised to at least `min_particles_structural`) and shrink factor. |
| `dynstate_snapshots` | `[-2, -1]` | Snapshots that get the Riley criterion: `null` (all), an int, a list (negative = from end), or a path to a text file of IDs. |
| `birth_window_factor` | `5.0` | Birth window in dynamical timescales. `<= 0` disables both trackers. |
| `enforce_initial_hosts` | `true` | Restrict birth hosts to the hosts bound at first appearance. |

**Run and output**

| Key | Default | Meaning |
|---|---|---|
| `output_dir` | `./output` | Output directory. **A fresh run (`resume: false`) deletes it first.** |
| `resume` | `false` | Resume from `checkpoint.zst`. |
| `seed` | `null` | Root seed. When null, one is generated and stored in the catalogue header and checkpoint. |
| `threads` | `null` | Thread budget (null = all cores). |
| `save_particles` | `true` | Write `particle_data/`. |
| `save_assignment` | `true` | Write `assignment/`. |
| `float_atol` | `1e-4` | Tolerance used to pick the smallest float dtype that keeps saved values within it. |

## Outputs

Everything goes to `output_dir`:

```
catalogue.hdf5
assignment/snapshotNNNN.hdf5     (save_assignment)
assignment/timescales.txt        (save_assignment)
particle_data/snapshotNNNN.hdf5  (save_particles)
run.log
progress.json
warnings.log
error.log                        (after a failure)
checkpoint.zst                   (after a failure)
```

### `catalogue.hdf5`

| Path | Content |
|---|---|
| `header/accretion_id`, `header/snapshots`, `header/last_snapshot` | Host ID, snapshot list, last written snapshot. |
| `header/config` | JSON string: `halo_model`, `n_los`, `search_factor`, `mass_weighting`, `plausibility`, `seed`, `threads`. |
| `header/merger_tree`, `header/equivalence` | The merger tree, including the computed columns (`scale_radius`, `host_id`, distances), and the equivalence table as JSON. |
| `snapshots/<id>/galaxy_properties` | `Sub_tree_id`, `mb_host_id`, `position_x/y/z`, `velocity_x/y/z`, `Mtot`, `r20`, `rh`, `r80`, `Rhp`, `sigma`, `sigma_los`, `r_t`. Attribute `time`. |
| `snapshots/<id>/riley_criterion` | `Sub_tree_id`, `mstar`, `f_bound`, `sigma50`, `dynstate`. Only on `dynstate_snapshots`. |
| `snapshots/<id>/satellite_relations/<host_id>` | Satellite IDs of each host. |
| `final/births` | `particle_index` (simulation ID), `birth_id` (birth galaxy). |
| `final/assembly` | `particle_index`, `galaxy_id`: one row per particle in each galaxy's infall list. |

A star in galaxy G's assembly set is in situ if its `birth_id == G`, and accreted otherwise. `roadrunner.io.HDF5CatalogueReader` reads these tables (`read_header`, `read_galaxy_properties`, `read_riley_criterion`, `read_births`, `read_assembly`).

### `assignment/snapshotNNNN.hdf5`

- `galaxies/<Sub_tree_id>/` has one group per halo with non-zero responsibilities:
  - `indices`: row positions in that snapshot's particle array (the same order as `particle_data/.../data/indices`), not simulation IDs.
  - `log_resp`: natural-log responsibilities. Exact zeros are stored as `-inf`.
  - `boundness` and `boundness_valid`: boundness values joined to `indices` by particle. NaN, with `valid = False`, where the particle is not a boundness candidate of this halo. The group attribute `energy_scale` recovers the energy, E = -boundness x energy_scale (for NFW, Φ₀ = -energy_scale).
  - `mean` (6-D), `covariance`, and `weight` (mixture weight within its group): in natural units.
  - `count`: effective particle count (summed responsibility, in the fit's weight units). A value below 1 marks a prior-only component.
- `hard_assignment` holds `Sub_tree_id` and `particle_index` (simulation ID) for every loaded particle. `Sub_tree_id = -1` means unbound.
- Attribute `time`.

`assignment/timescales.txt` is a tab-separated file with columns `particle_index` and `timescale`. Each particle's dynamical timescale is recorded once, at its first appearance.

### `particle_data/snapshotNNNN.hdf5`

`header` has the attributes `snapshot`, `time` and `redshift`. `data/positions` and `data/velocities` are stored standardised; `data/scaler/mean` and `data/scaler/scale` undo the scaling. `data/masses` and `data/indices` hold masses and simulation IDs, and any extra fields are stored under their own names. These files are the input format of `reader_type: pdata`.

### `run.log`

A header (`output_dir`, `halo_model`, `cov_type`) is followed by one row per snapshot and `--- Run complete ---` at the end. Missing or NaN values are shown as `--`.

| Column | Meaning |
|---|---|
| `RUNTIME` | Wall time since the start, `HH:MM:SS.fff`. |
| `SNAP`, `REDSHIFT` | Snapshot ID and redshift. |
| `LOAD`, `PROCESS` | Seconds spent reading and processing, where processing includes the reduction. |
| `REDUCTION` | Not filled at present (always `--`). |
| `BOUND`, `GROUPS` | Bound particles; overlap groups. |
| `FRAGS` | Galaxies with fewer than 10 assigned particles. |
| `UNASSIGNED` | Particles with `Sub_tree_id = -1`. |
| `AVG_CONF` | Mean maximum responsibility over contested particles. |
| `AVG_ENTROPY` | Mean responsibility entropy, normalised by log K, over contested particles. |
| `LOG_COND` | Median log10 condition number of the fitted (scaled) covariances, rank-deficient fits excluded. |
| `BAD_COND` | Components with condition number > 1e6 or non-finite, rank-deficient fits excluded. |
| `EMPTY` | Components with fitted `count` < 1 (parameters come from the prior only). |
| `MED_RET` | Median over galaxies of the fraction of bound candidates assigned to them. |

### Logs, failures and resuming

- `warnings.log` collects every warning raised during a snapshot, with timestamp and snapshot ID. Runs with many unresolved groups can make it large.
- `error.log` gets the traceback when a snapshot fails.
- `progress.json` records `snapshots`, `last_completed_snapshot` and `is_finished`.
- A checkpoint is written only when a snapshot fails, including on Ctrl-C. `checkpoint.zst` is a zstd-compressed pickle of the state after the last completed snapshot: responsibilities, fitted parameters, tracker state, seed and precision.

To resume, rerun with the same configuration and `resume: true` (or `--resume`). The run continues at the snapshot after the last completed one. Resume is refused (`RestartError`) when:

- the run already finished;
- the snapshot range differs;
- `data_precision` or `math_precision` differs;
- an explicit `seed` conflicts with the checkpointed one;
- the checkpoint and `progress.json` disagree.

If you leave `seed` out, the checkpointed seed is used. Remember that `resume: false` deletes `output_dir`.

## Mock data and analysis scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `generate_mock.py` | A single z = 0 mock snapshot of limepy galaxies in overlapping groups (`particles.npz`, `merger_tree.csv`, `equivalence.csv`, with the truth in `galaxy_id`). Units kpc, km/s, Msun. |
| `generate_mock_simulation.py` | Evolves a base mock snapshot into `--n-snap` snapshots (`particles_XXX.npz`). Adds newborn particles, a bulk drift `--v-sys`, flat-ΛCDM redshifts and comoving storage. Writes the tree, the equivalence table and `assignment.csv`. |
| `generate_mock_mass_loss.py` | Like `generate_mock_simulation.py`, plus three controlled mass-loss patterns applied to nine galaxies. |
| `generate_mock_priors.py` | A five-galaxy, eight-snapshot mock with controlled mass loss and gain, for testing the BGMM priors. |
| `inject_mock_galaxies.py` | Injects limepy galaxies, dark subhalos or interacting pairs into a mock snapshot or simulation, following a scenario YAML (`inject_mock_galaxies.yaml.example`). Truth is written for the injected stars. |
| `swap_mock_particles.py` | Swaps particles between halos (`--scope any/same-group/cross-group` or `--halos A B`), or ejects them out of all halos, over a snapshot range. Updates the truth and writes `moves.csv`. |
| `plot_mock.py`, `plot_mock_vel.py` | XY positions or vx–vy of a mock snapshot. |
| `plot_mock_comparison.py` | Ground truth against pipeline assignment for one mock snapshot. |
| `plot_mock_comparison_simulation.py` | Truth against pipeline output for each mock-simulation snapshot, with centres, rh and σ overlaid. |
| `analyze_snapshot.py` | Plots the particle assignment of one snapshot from a run's output. |
| `compute_sfr.py` | Star-formation rate from the birth catalogue. |
| `movie_properties.py` | Per-snapshot property plots for a movie of one galaxy. |
| `add_radii.py` | Recomputes r20, rh and r80 from a catalogue and a merger tree. |
| `mixtures.py` | Ad-hoc comparison of the roadrunner mixtures against scikit-learn on synthetic blobs. |

Every script with arguments prints its options with `--help`. The usage lines in some docstrings say `test_scripts/`; all of these scripts live in `scripts/`.

## Tests

```
tests/
  unit/         per-module tests (mixtures, priors, assigner, boundness, segmentation, sparse,
                readers, writers, trackers, checkpointing, config, ...)
  integration/  full AccretionPipeline and SnapshotOrchestrator runs (resume, error checkpoints, logs)
  benchmarks/   standalone timing / memory scripts (bench_*.py, not collected by pytest)
```

```sh
pytest tests/unit            # unit tests
pytest                       # unit + integration (testpaths = tests)
python tests/benchmarks/bench_sparse_align.py
```

Some tests read local fixtures from `test_data/`, which is not tracked in git. `test_merger_tree*.py` uses `test_data/test_tree.csv`, while `test_processing.py` and the integration tests use `test_data/mock_snap_tight/`. Those tests fail when the fixtures are missing.

The numba kernels are compiled on first use and cached (`cache=True`), so the first run of the tests or the pipeline is noticeably slower.

## Repository layout

```
src/roadrunner/
  pipeline/        RunConfig, run_accretion_history, AccretionPipeline (loop, checkpoint/resume),
                   SnapshotOrchestrator, processing (boundness, groups, assignment), reduction,
                   translation (array index <-> simulation ID)
  readers/         merger-tree CSV, equivalence table, yt / npz / particle-data snapshot readers
  physics/         halo models, Kepler/NFW/Plummer/Hernquist/shell potentials, boundness, merger-tree handler (scale radii,
                   hosts, satellites), dynamical timescales, scaler, constants
  clustering/      sparse CSC/CSR matrices, union-find, overlap segmentation and ownership
    assignment/    XGMMAssigner (gmm.py), BGMM priors (priors.py), run statistics (statistics.py)
  mixture/         weighted GMM, weighted variational BGMM, SVI-BGMM (experimental), k-means/k-means++,
                   Gaussian coresets, shared numba math
  postprocessing/  galaxy properties, centring, Riley dynamical state (mixing.py)
    tracking/      BirthTracker, AssemblyTracker
  io/              HDF5 catalogue / assignment / particle writers, catalogue reader, run log,
                   zstd checkpoints
  _defaults.py     algorithmic constants, precision scope, integer ID dtypes
  randomness.py    deterministic per-snapshot seeds
  threads.py       shared thread budget
scripts/           runners, example configs, mock generators, plotting
tests/             unit, integration, benchmarks
```

## License and author

GNU General Public License v3.0 (see `LICENSE`).

Author: Asier Lambarri Martinez.

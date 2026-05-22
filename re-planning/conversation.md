# Refactoring review: SnapshotProcessor & AccretionPipeline

**Session ID:** ses_1b380c71effe6fT2BRPn7ex4v0
**Created:** 5/21/2026, 11:44:26 PM
**Updated:** 5/21/2026, 11:51:40 PM

---

## User

this folder contains a repository of a big project that i have refactored recently using opencode, to make it more portable and upgradable. The original code is inside src/roadrunner/_accme, while the rest of the folders are part of the new package. The refactor was made with pythonic best practices and scientific computing in mind, and with the aim: stateless -> function, with state -> class. The folder planning/ has markdown files and other utilities that where created along the refactor, but you may only trust the PHASES and Steps in it, the small-items inside each Step usually do not reflect what was ultimatelly made. Only the first eleven phases where completed as the rest is marked as optional. All written tests and scripts work. There is still old pieces of refactor code lying around,dont be bothered by them. I am happy with how the first 8 phases turned out: the code is quite modular and each class follows the simple premise of only doing one thing (boundness, segmentation, assignment, reading, writing catalogues, handling haloes...) but things got a bit out of hand with the SnapshotProcessor and AccretionPipeline. I do like the idea of the RunConfig  dataclass and the entry-point. However, as i said, the SnapshotProcessor and AccretionPipeline classes seem a bit badly done, if we stick to the philosophy described above. The main issues i do not like is the separation between array index (i.e. having particle indices go from 0...N-1) and sim ID (i.e. the persistant simulation IDs of the particles) space is made haphazardly, in part due to the trackers. I also dont like the general structure of the code in AccretionPipeline and SnapshotProcessor since i feel that they break, somehow, the philosophy stated above. Read the repository and have a peek at planning/planB.md (only this file, and only phases and steps, no small-items), and identify the main weak points/violations of code quality and philosophy, particularly from step 9 onwards (for step 8 or bellow i welcome ideas, but the code is i think very stable as it is). Do only this, so that afterwards we can disect everything into a plan for refactoring a bit more the code.

---

## Assistant (Plan · GLM-5.1 · 6.0s)

_Thinking:_

The user wants me to read the repository, particularly the planning/planB.md file (only phases and steps, no small-items), and identify main weak points/violations of code quality and philosophy, particularly from step 9 onwards. Let me start by exploring the repository structure and reading the relevant files.

Let me do this systematically:
1. First, read the repository structure
2. Read planB.md (only phases and steps)
3. Read the key files: SnapshotProcessor, AccretionPipeline, and related code
4. Identify weak points

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner"
}
```

**Output:**
```
<path>/home/asier/roadrunner</path>
<type>directory</type>
<entries>
.git/
.gitignore
.pytest_cache/
.ruff_cache/
figures/
planning/
pyproject.toml
scripts/
src/
test_data/
tests/

(11 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/planning"
}
```

**Output:**
```
<path>/home/asier/roadrunner/planning</path>
<type>directory</type>
<entries>
dependency_graph.dot
dependency.png
expanded_SP.md
expanded2_1.md
expanded2_6.md
expanded5_-1.md
expanded5_-2.md
expanded5_0.md
expanded5_3.md
expanded9_2.md
FLOWCHARTS.md
function_migration.dot
function_migration.png
MODULE_MAP.md
module_migration.dot
module_migration.png
planA.md
planB.md
QUICK_REFERENCE.md
roadrunner_v2.md
scheme_after_compaction.md

(21 entries)
</entries>
```

---

## Assistant (Plan · GLM-5.1 · 3.7s)

_Thinking:_

Let me read the planB.md and the source code structure.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/planning/planB.md"
}
```

**Output:**
```
<path>/home/asier/roadrunner/planning/planB.md</path>
<type>file</type>
<content>
1: # ACCME → roadrunner Refactoring Plan — Phase-by-Phase
2: 
3: ## Key Issues Identified
4: 
5: | # | Problem | Severity |
6: |---|---|---|
7: | 1 | **God class**: `BaseAccretionHistory` has 8 distinct responsibilities (config, loading, output, logging, query, processing, orchestration, particle selection) across 24+ methods | Critical |
8: | 2 | **880-line monolith**: `gaussians.py` mixes boundness, segmentation orchestration, GMM init, assignment, timescales into one file | Critical |
9: | 3 | **Temporal coupling**: 7 instance attributes set in `select_accreted_particles()` rather than `__init__` — object is incomplete until call order is followed | High |
10: | 4 | **Private convention violation**: 40+ underscore-prefixed "private" functions are imported and re-exported across modules — they function as public API but are named as private | High |
11: | 5 | **Magic numbers duplicated 17+ times**: `4.3E-6` in 3 files, `0.97779` undocumented, `1E-4` vs `1E-5` inconsistent softening lengths | High |
12: | 6 | **Hardcoded paths in all 11 scripts**: 30+ absolute paths tied to `/home/asier/` and `/data1/agertz/` — zero use of CLI args or config | High |
13: | 7 | **No clear interface between layers**: The `_priv_reduce_snapshot` method depends on 3 own-modules simultaneously, creating a tightly coupled mesh | Medium |
14: | 8 | **Redundant `*`+explicit `__init__.py` pattern**: All three `__init__.py` files use unnecessary star imports | Medium |
15: | 9 | **Hidden subpackage**: `clustering/_fast/` has no `__init__.py` and is not imported by `clustering/__init__.py` | Medium |
16: | 10 | **`row_squared_norms` duplicated**: Same Numba function in two files (`_kmeans_plusplus.py` and `weighted_gaussian_mixture.py`) | Low |
17: | 11 | **`HaloSegmentator` misspelling**: Should be `HaloSegmenter`; `_find_populated_haloes` British spelling inconsistent | Low |
18: | 12 | **No unit tests**: All `scripts/tests/` are ad-hoc visual/validation scripts, not structured test suites | Critical |
19: | 13 | **ThreadPoolExecutor over-engineering**: Used as a simple double-buffer for one pre-load; adds concurrency complexity without benefit | Low |
20: 
21: ---
22: 
23: ## Guiding Principles
24: 
25: 1. **Stateless → function, internal state → class**
26: 2. **Single Responsibility**: each module, class, and function does exactly one thing
27: 3. **Dependency Inversion**: high-level modules (pipeline) depend on abstractions, not concrete implementations
28: 4. **Explicit interfaces**: public API is clearly separated from private internals
29: 5. **Configuration injection**: no hardcoded paths, no magic numbers in business logic
30: 6. **Testable**: every unit can be tested in isolation with mock data
31: 7. **Test-first per step**: each step writes tests before committing code
32: 8. **Git-tracked with per-phase tags**: clean history for rollback
33: 
34: ---
35: 
36: ## Target Repository Structure
37: 
38: ```
39: roadrunner/                          ← git repo root
40: ├── .gitignore
41: ├── .github/
42: │   └── workflows/
43: │       └── ci.yml
44: ├── pyproject.toml
45: ├── README.md
46: ├── LICENSE
47: │
48: ├── src/
49: │   └── roadrunner/                  ← import as `import roadrunner`
50: │       ├── __init__.py              ← Clean public API
51: │       ├── _exceptions.py           ← Custom exception types
52: │       ├── _type_defs.py            ← Type aliases, protocol classes, dataclasses
53: │       │
54: │       ├── physics/
55: │       │   ├── __init__.py
56: │       │   ├── constants.py         ← G, softening defaults, conversion factors
57: │       │   ├── potentials.py        ← KeplerPotential, NFWPotential classes
58: │       │   ├── boundness.py         ← compute_halo_bound_particles()
59: │       │   └── timescales.py        ← dynamical timescales, tidal radius
60: │       │
61: │       ├── readers/
62: │       │   ├── __init__.py
63: │       │   ├── merger_tree.py       ← MergerTreeReader
64: │       │   ├── snapshot.py          ← SnapshotReader (YT backend)
65: │       │   └── equivalence.py       ← EquivalenceTable
66: │       │
67: │       ├── clustering/
68: │       │   ├── __init__.py
69: │       │   ├── union_find.py        ← UnionFind, overlapping_groups
70: │       │   ├── segmentation.py      ← HaloSegmenter
71: │       │   ├── sparse.py            ← SparseCSC, build_dense, row_norm
72: │       │   └── assignment/
73: │       │       ├── __init__.py
74: │       │       ├── gmm.py           ← GMMAssigner
75: │       │       ├── fast.py          ← FastAssigner
76: │       │       └── orchestrator.py  ← assign_particles()
77: │       │
78: │       ├── mixture/
79: │       │   ├── __init__.py
80: │       │   ├── base.py              ← BaseMixture (abstract EM)
81: │       │   ├── _math.py             ← Shared Numba kernels
82: │       │   ├── kmeans_plusplus.py   ← kmeans_plusplus_prior
83: │       │   ├── weighted_gmm.py      ← WeightedGaussianMixture
84: │       │   ├── bayesian_gmm.py      ← WeightedBayesianGaussianMixture
85: │       │   └── coresets.py          ← GaussianCoreset
86: │       │
87: │       ├── postprocessing/
88: │       │   ├── __init__.py
89: │       │   ├── properties.py        ← Galaxy property functions (stateless)
90: │       │   ├── mixing.py            ← Riley criterion functions (stateless)
91: │       │   └── tracking/
92: │       │       ├── __init__.py
93: │       │       ├── birth.py         ← BirthTracker (stateful)
94: │       │       └── assembly.py      ← AssemblyTracker (stateful)
95: │       │
96: │       ├── io/
97: │       │   ├── __init__.py
98: │       │   ├── hdf5_writer.py       ← HDF5ArchiveWriter
99: │       │   ├── hdf5_reader.py       ← HDF5ArchiveReader
100: │       │   ├── serialization.py     ← Checkpoint save/load (zstd)
101: │       │   └── logging.py           ← RunLogger
102: │       │
103: │       └── pipeline/
104: │           ├── __init__.py
105: │           ├── config.py            ← RunConfig dataclass
106: │           ├── snapshot_processor.py ← SnapshotProcessor
107: │           └── accretion_pipeline.py ← AccretionPipeline
108: │
109: ├── tests/
110: │   ├── conftest.py                  ← Shared fixtures
111: │   ├── data/
112: │   │   ├── sample_merger_tree.csv
113: │   │   └── sample_equivalence.csv
114: │   ├── unit/
115: │   │   ├── test_constants.py
116: │   │   ├── test_exceptions.py
117: │   │   ├── test_type_defs.py
118: │   │   ├── test_helpers.py
119: │   │   ├── test_merger_tree.py
120: │   │   ├── test_snapshot_reader.py
121: │   │   ├── test_equivalence.py
122: │   │   ├── test_union_find.py
123: │   │   ├── test_segmentation.py
124: │   │   ├── test_sparse.py
125: │   │   ├── test_potentials.py
126: │   │   ├── test_boundness.py
127: │   │   ├── test_assignment_base.py
128: │   │   ├── test_assignment_gmm.py
129: │   │   ├── test_assignment_fast.py
130: │   │   ├── test_orchestrator.py
131: │   │   ├── test_kmeans_plusplus.py
132: │   │   ├── test_weighted_gmm.py
133: │   │   ├── test_bayesian_gmm.py
134: │   │   ├── test_coresets.py
135: │   │   ├── test_timescales.py
136: │   │   ├── test_properties.py
137: │   │   ├── test_mixing.py
138: │   │   ├── test_birth_tracker.py
139: │   │   ├── test_assembly_tracker.py
140: │   │   ├── test_hdf5_writer.py
141: │   │   ├── test_hdf5_reader.py
142: │   │   └── test_serialization.py
143: │   └── integration/
144: │       ├── test_snapshot_processor.py
145: │       ├── test_accretion_pipeline.py
146: │       └── test_end_to_end.py
147: │
148: ├── scripts/
149: │   ├── run_pipeline.py              ← was full_simulation.py
150: │   ├── analyze_snapshot.py          ← was single_snapshot.py
151: │   ├── compute_sfr.py               ← was sfr.py
152: │   ├── add_radii.py
153: │   └── movie_properties.py
154: │
155: └── figures/                         ← generated plots output
156: ```
157: 
158: ---
159: 
160: ## PHASE 0: Project Scaffolding & Environment
161: 
162: **Goal**: Initialize the repository structure, git tracking, conda environment, and build system.
163: 
164: ### Step 0.1 — Create conda environment
165: 
166: - [X] Check if `roadrunner` conda env exists: `conda env list | grep roadrunner`
167: - [X] If not, create it: `conda create -n roadrunner python=3.10 -y`
168: - [X] Install core dependencies: `conda run -n roadrunner pip install numpy numba scipy pandas h5py yt tqdm zstandard`
169: - [X] Install dev dependencies: `conda run -n roadrunner pip install pytest pytest-cov ruff mypy`
170: - [X] Document environment name in `README.md`
171: - [X] **Validate**: `conda run -n roadrunner python -c "import numpy, numba, scipy, pandas, h5py, yt, tqdm, zstandard; print('All imports OK')"`
172: 
173: ### Step 0.2 — Initialize git repository
174: 
175: - [X] `cd /home/asier/roadrunner && git init`
176: - [X] Create `.gitignore`:
177:   ```
178:   **/__pycache__/
179:   *.pyc
180:   *.pyo
181:   *.pyd
182:   build/
183:   dist/
184:   *.egg-info/
185:   .venv/
186:   venv/
187:   src/roadrunner/_version.py
188:   scripts_results/
189:   .pytest_cache/
190:   .coverage
191:   htmlcov/
192:   .ruff_cache/
193:   figures/
194:   ```
195: - [X] `git add .gitignore && git commit -m "Phase 0.2: .gitignore"`
196: - [X] **Validate**: `git status` shows clean working tree
197: 
198: ### Step 0.3 — Create `pyproject.toml` with src layout
199: 
200: - [X] Write `pyproject.toml` at repo root:
201:   ```toml
202:   [build-system]
203:   requires = ["setuptools>=61", "setuptools.scm"]
204:   build-backend = "setuptools.build_meta"
205: 
206:   [project]
207:   name = "roadrunner"
208:   dynamic = ["version"]
209:   description = "Unravel the accretion history of simulated galaxies"
210:   authors = [{name = "Asier Lambarri Martinez"}]
211:   readme = "README.md"
212:   requires-python = ">=3.9"
213:   dependencies = [
214:       "numpy",
215:       "numba",
216:       "scipy",
217:       "pandas",
218:       "h5py",
219:       "yt",
220:       "tqdm",
221:       "zstandard",
222:   ]
223: 
224:   [project.optional-dependencies]
225:   dev = ["pytest", "pytest-cov", "ruff", "mypy"]
226: 
227:   [tool.setuptools.packages.find]
228:   where = ["src"]
229: 
230:   [tool.setuptools_scm]
231:   version_scheme = "post-release"
232:   local_scheme = "node-and-date"
233:   write_to = "src/roadrunner/_version.py"
234: 
235:   [tool.pytest.ini_options]
236:   testpaths = ["tests"]
237:   ```
238: - [X] Create directory: `mkdir -p src/roadrunner/ tests/unit/ tests/integration/ tests/data/ scripts/ figures/`
239: - [X] Create `src/roadrunner/__init__.py` (empty, placeholder)
240: - [X] **Validate**: `conda run -n roadrunner python -c "import roadrunner; print('Package imported OK')"`
241: 
242: ### Step 0.4 — Copy existing accme as untracked reference subpackage
243: 
244: Rather than refactoring in-place, accme is stored as an untracked subpackage `roadrunner._accme`
245: to serve as read-only reference code. It will be deleted after all functionality is migrated.
246: 
247: #### Step 0.4.1 — Copy accme into `_accme/` subpackage
248: 
249: - [X] Copy all files from `/home/asier/accme-main/src/accme/` into `src/roadrunner/_accme/`
250:   - Preserve subdirectory structure: `clustering/`, `mixture/`, `postprocessing/`
251: - [X] Copy all scripts from `/home/asier/accme-main/scripts/` into `scripts/`
252:   - Exclude the `tests/` subdirectory under scripts — they become `tests/unit/`
253: - [X] **Note**: No search-and-replace needed — accme uses only relative imports (e.g., `from .xxx import`),
254:   so it works as `roadrunner._accme` without any import modification.
255: 
256: #### Step 0.4.2 — Keep `_accme/` untracked by git
257: 
258: - [X] Add `src/roadrunner/_accme/` to `.gitignore`
259: - [X] **Validate**: `git status` shows `src/roadrunner/_accme/` as ignored (`!!` prefix)
260: 
261: #### Step 0.4.3 — Make importable as `roadrunner._accme`
262: 
263: - [X] No extra files needed — `_accme/` has its own `__init__.py` and editable install resolves it.
264: - [X] **Validate**: `conda run -n roadrunner python -c "import roadrunner; print(roadrunner._accme); print('Import OK')"`
265: - [X] **Validate**: `conda run -n roadrunner python -c "from roadrunner._accme import AccretionHistory; print('Old API import OK')"`
266: - [X] **Validate**: `conda run -n roadrunner python -c "from roadrunner._accme import BaseAccretionHistory; print('Base class import OK')"`
267: 
268: #### Step 0.4.4 — Commit scaffolding
269: 
270: - [X] `git add .gitignore scripts/`
271: - [X] `git commit -m "Phase 0.4: Copy accme as untracked _accme reference subpackage"`
272: 
273: **Phase 0 completion**: `git tag phase-0-scaffolding`
274: 
275: ---
276: 
277: ## PHASE 1: Foundation — Constants, Exceptions, Types, Utilities
278: 
279: **Goal**: Establish the shared vocabulary that every other module depends on.
280: 
281: ### Validation Protocol Per Step
282: 
283: Every numbered step in every phase follows this exact sequence:
284: ```
285: [ ] 1. Write production code in src/roadrunner/...
286: [ ] 2. Write tests in tests/unit/...
287: [ ] 3. conda run -n roadrunner pytest tests/path/to/test_file.py -v
288:      └── if FAIL: fix code, return to step 3
289: [ ] 4. conda run -n roadrunner ruff check src/roadrunner/   (lint)
290: [ ] 5. git add .
291: [ ] 6. git commit -m "Step X.Y: description"
292: ```
293: 
294: ### Step 1.1 — Physical Constants Module
295: 
296: - [X] Create `src/roadrunner/physics/__init__.py`
297: - [X] Create `src/roadrunner/physics/constants.py`
298: - [X] Define `G = 4.3e-6` with unit annotation `kpc^3 / (Msun * Gyr^2)`
299: - [X] Define `G_ALT = 4.3e-6` (same value, documented as alternative convention `kpc * (km/s)^2 / Msun`)
300: - [X] Define `DYN_TIME_FACTOR = 0.97779` (conversion factor to Gyr)
301: - [X] Define `SOFTENING_KEPLER = 1e-5` (named constant for Kepler potential)
302: - [X] Define `SOFTENING_NFW = 1e-4` (named constant for NFW potential, intentionally different)
303: - [X] Add docstrings with physical units for each constant
304: - [X] Add `__all__` export list
305: - [X] Write `tests/unit/test_constants.py`:
306:   - Test constants are positive floats
307:   - Test G has expected value `4.3e-6`
308:   - Test softening constants differ (documented reason)
309:   - Test `DYN_TIME_FACTOR` is within reasonable range (0.9–1.0)
310: - [X] **Test and validate**: `conda run -n roadrunner pytest tests/unit/test_constants.py -v`
311: - [X] Commit
312: 
313: ### Step 1.2 — Custom Exceptions
314: 
315: - [X] Create `src/roadrunner/_exceptions.py`:
316:   - `RoadrunnerError(Exception)` — base exception
317:   - `ConfigurationError(RoadrunnerError)` — invalid setup
318:   - `SnapshotLoadError(RoadrunnerError)` — YT load failure
319:   - `NoBoundParticlesError(RoadrunnerError)` — halo without bound particles
320:   - `ConvergenceError(RoadrunnerError)` — GMM/EM not converging
321:   - `RestartError(RoadrunnerError)` — corrupted restart state
322: - [X] Write `tests/unit/test_exceptions.py`:
323:   - Test each exception can be raised and caught
324:   - Test inheritance hierarchy (`isinstance(ConvergenceError(), RoadrunnerError)`)
325:   - Test exception message strings
326: - [X] **Test and validate**: `pytest tests/unit/test_exceptions.py -v`
327: - [X] Commit
328: 
329: ### Step 1.3 — Type Definitions and Protocols
330: 
331: - [X] Create `src/roadrunner/_type_defs.py`:
332:   - Type aliases: `ParticleArray`, `HaloID`, `SnapshotID`, `CSCResp`
333:   - `SnapshotData` dataclass with fields: `indices, masses, positions, velocities, metallicity (optional), redshift, time`
334:   - `BoundnessResult` dataclass with fields: `candidate_indices, boundness_values, dynamical_times`
335:   - `AssignmentResult` dataclass with fields: `particle_df, responsibilities, fitted_parameters, statistics`
336:   - Protocol class `Assigner` with `.assign(...)` signature
337:   - Protocol class `PotentialModel` with `.potential(r)`, `.dynamical_time(...)`, `.tidal_denominator(...)`
338: - [X] Write `tests/unit/test_type_defs.py`:
339:   - Test dataclass construction and field access
340:   - Test protocol class can be used for `isinstance` check with a mock
341:   - Test type alias usage
342: - [X] **Test and validate**: `pytest tests/unit/test_type_defs.py -v`
343: - [X] Commit
344: 
345: ### Step 1.4 — Helpers (cleaned up)
346: 
347: - [X] Create `src/roadrunner/helpers.py`:
348:   - `select_uint_dtype(max_val)` — replaces `_select_uint_dtype`
349:   - `select_int_dtype(max_val)` — replaces `_select_int_dtype`
350:   - `select_float_dtype(max_value, abs_tol=1e-4)` — replaces `_select_float_dtype`
351:   - `check_particle_uniqueness(data)` — replaces `_check_particle_uniqueness`
352:   - `remove_duplicates(data)` — replaces `_remove_duplicates`
353:   - Remove `_format_runtime` (moved to `io/logging.py` in Phase 8)
354:   - No leading underscores on public functions
355: - [X] Write `tests/unit/test_helpers.py`:
356:   - Test dtype selectors return correct types for various max values
357:   - Test `check_particle_uniqueness` with unique and duplicate data
358:   - Test `remove_duplicates` preserves first occurrence
359:   - Test edge cases: empty dict, single element, all duplicates, negative values for int
360: - [X] **Test and validate**: `pytest tests/unit/test_helpers.py -v`
361: - [X] Commit
362: 
363: **Phase 1 completion**: `git tag phase-1-foundation`
364: 
365: ---
366: 
367: ## PHASE 2: Data Layer — Readers
368: 
369: **Goal**: Cleanly separate simulation data loading from processing logic.
370: 
371: ### Step 2.1 — Merger Tree Reader
372: 
373: - [X] Create `src/roadrunner/readers/__init__.py`
374: - [X] Create `src/roadrunner/readers/merger_tree.py` with class `MergerTreeReader`:
375:   - Constructor: file path OR DataFrame
376:   - Properties: `.dataframe`, `.snapshots`, `.subtree_ids`
377:   - Methods (I/O + query only):
378:     - `.select_snapshots(snap_ids)` — replaces `BaseAccretionHistory.select_snapshots()`
379:     - `.select_subtrees(tree_ids)` — replaces `BaseAccretionHistory.select_subtrees()`
380:     - `.select_accretion_host(snapshot_id, criterion="max_mass")` — find main host
381:     - `.to_numeric()` — replaces `_convert_pandas_to_numeric()`
382: - [X] No test data fixture — use existing `test_data/test_tree.csv`
383: - [X] Write `tests/unit/test_merger_tree.py` — `TestMergerTreeReader` class:
384:   - Test construction from CSV path and from DataFrame
385:   - Test invalid type raises TypeError
386:   - Test `.snapshots` returns sorted ints
387:   - Test `.subtree_ids` returns sorted ints
388:   - Test `.select_snapshots()` filters correctly
389:   - Test `.select_subtrees()` filters correctly
390:   - Test `.select_accretion_host("max_mass")` returns highest-mass row
391:   - Test `.select_accretion_host` raises on unknown criterion
392:   - Test `.to_numeric()` coerces object columns
393: - [X] **Test and validate**: `pytest tests/unit/test_merger_tree.py -v`
394: - [X] Commit
395: 
396: ### Step 2.1b — Physics helper: `nfw_cmz_relation_duffy`
397: 
398: - [X] Create `src/roadrunner/physics/merger_tree.py`:
399:   - `nfw_cmz_relation_duffy(M, z)` — Duffy et al. 2008 concentration relation
400:     - `A, B, C = 7.85, -0.081, -0.71`
401:     - Returns `A * (M / 2e12)**B * (1 + z)**C`
402: - [X] Tests in `tests/unit/test_merger_tree.py` — `TestNfwCmz` class:
403:   - Known input `(1e12, 0)` produces expected output
404:   - Monotonicity: `M↑ ⇒ c↓`, `z↑ ⇒ c↓`
405:   - Edge: large M
406: - [X] **Test and validate**: `pytest tests/unit/test_merger_tree.py -v -k TestNfwCmz`
407: - [X] **Lint**: `ruff check src/roadrunner/physics/merger_tree.py`
408: - [X] Commit
409: 
410: ### Step 2.2 — Snapshot Reader (interface + mock)
411: 
412: - [X] Create `src/roadrunner/readers/snapshot.py`:
413:   - Class `SnapshotReader`:
414:     - Constructor: `code, ptype, fields, unit_base`
415:     - Methods:
416:       - `.load(file_path)` → `SnapshotData`
417:       - `.load_with_filter(file_path, particle_indices)` → `SnapshotData`
418:       - `.apply_particle_filter(ds, particle_indices)` — YT filter creation
419:     - Internal code-specific methods (private):
420:       - `_load_art()`, `_load_gear()`, `_load_auriga()`, `_load_ramses()`
421:     - Static method: `.mock_data(n_particles, seed=42)` → `SnapshotData` (for testing)
422: - [X] Write `tests/unit/test_snapshot_reader.py`:
423:   - Test construction with different code strings
424:   - Test `mock_data()` returns correct data types and shapes
425:   - Test that `SnapshotData` has all required fields
426:   - Test field name configuration is stored correctly
427:   - Test error on invalid code string
428: - [X] **Test and validate**: `pytest tests/unit/test_snapshot_reader.py -v`
429: - [X] Commit
430: 
431: ### Step 2.3 — Equivalence Table Reader
432: 
433: - [X] Create `src/roadrunner/readers/equivalence.py`:
434:   - Class `EquivalenceTable`:
435:     - Constructor: file path, DataFrame, or `(snapshots, paths, times, redshifts)` arrays
436:     - Methods:
437:       - `.snapshot_path(snap_id)` → absolute file path (resolved against base dir)
438:       - `.snapshot_time(snap_id)` → cosmic time
439:       - `.snapshot_redshift(snap_id)` → redshift
440:     - Property: `.snapshots` (sorted list)
441:     - Property: `.max_snapshot`, `.min_snapshot`
442: - [X] Create test fixture: `tests/data/sample_equivalence.csv`
443: - [X] Write `tests/unit/test_equivalence.py`:
444:   - Test path resolution with relative paths
445:   - Test time/redshift lookups return correct values
446:   - Test `.snapshots` sorted property
447:   - Test `max_snapshot` / `min_snapshot` properties
448:   - Test error on missing snapshot
449: - [X] **Test and validate**: `pytest tests/unit/test_equivalence.py -v`
450: - [X] Commit
451: 
452: ### Step 2.4 — Particle Birth Assignment
453: 
454: - [X] Create `src/roadrunner/physics/birth_assignment.py` (new module, extracted from `BaseAccretionHistory._assign_t_snap` and `select_accreted_particles`):
455:   - `assign_birth_snapshots(creation_times, snapshot_times, snapshot_ids)` → array of snapshot IDs
456:   - `build_birth_dict(particle_indices, birth_snapshot_ids)` → `dict[snapshot, list[particle_id]]`
457:   - `sanitize_birth_dict(birth_dict)` → removes duplicates, returns cleaned dict
458: - [X] Write `tests/unit/test_birth_assignment.py`:
459:   - Test `assign_birth_snapshots()` with known times: particles born between snapshots go to next snapshot
460:   - Test particle born before first snapshot → assigned to first snapshot
461:   - Test particle born after last snapshot → assigned to last snapshot
462:   - Test `build_birth_dict()` produces correct mapping
463:   - Test `sanitize_birth_dict()` removes duplicates preserving first occurrence
464:   - Test edge case: empty input
465: - [X] **Test and validate**: `pytest tests/unit/test_birth_assignment.py -v`
466: - [X] Commit
467: 
468: ### Step 2.6 — Merger Tree Physics Handler
469: 
470: **Goal**: Physical operations on merger tree data (scale radii, satellite detection, distances).
471: 
472: Detailed expanded plan in `planning/expanded2_6.md`.
473: 
474: - [X] Create `src/roadrunner/physics/merger_tree.py` (if not already created in 2.1b):
475:   - Class `MergerTreeHandler(MergerTreeReader)`:
476:     - Inherits all `MergerTreeReader` methods
477:     - Methods (physics only):
478:       - `.compute_scale_radii(snapshot_df)` — replaces `_compute_rs`
479:         - Fills NaN `scale_radius` via `virial_radius / c(M,z)` using `nfw_cmz_relation_duffy`
480:       - `.compute_satellites(snapshot_df, rvir_factor=1.0)` — replaces `_compute_satellites`
481:         - KDTree + binding energy (`E_bind = |v_rel|² − 2·G·M_i / r`)
482:         - Returns `dict[int, set[int]]`: host_id → set of satellite_ids
483:       - `.compute_most_bound_satellite(snapshot_df, rvir_factor=1.0)` — replaces `_compute_most_bound_satellite`
484:         - Same KDTree + E_bind, assigns each satellite to host with minimum E_bind
485:         - Returns `dict[int, int]`: satellite_id → host_id
486:       - `.compute_distance_to_host(snapshot_df, column="host_id")` — replaces `_compute_distance_to_host`
487:         - Self-merge on host column, 3D Euclidean distance
488:         - Returns DataFrame with new `distance_to_{column}` column
489:     - Imports:
490:       - `from roadrunner.physics.constants import G_KM`
491:       - `from scipy.spatial import KDTree`
492:       - `numpy`, `pandas`
493: - [X] Write `tests/unit/test_merger_tree_handler.py` — `TestMergerTreeHandler` class:
494:   - `test_compute_scale_radii`: all NaN → filled; mixed → only NaN rows; none NaN → unchanged
495:   - `test_compute_satellites`: 2 halos with known binding → satellite; isolated → none; rvir_factor sensitivity
496:   - `test_compute_most_bound_satellite`: satellite bound to 2 hosts → assigned to most negative E
497:   - `test_compute_distance_to_host`: A host of B → B gets distance > 0
498: - [X] **Test and validate**: `pytest tests/unit/test_merger_tree_handler.py -v`
499: - [X] **Lint**: `ruff check src/roadrunner/physics/merger_tree.py`
500: - [X] Commit
501: 
502: **Phase 2 completion**: `git tag phase-2-readers`
503: 
504: ---
505: 
506: ## PHASE 3: Clustering Infrastructure
507: 
508: **Goal**: Clean, composable building blocks for particle assignment.
509: 
510: ### Step 3.1 — Generic Union-Find
511: 
512: - [X] Create `src/roadrunner/clustering/union_find.py`:
513:   - Class `UnionFind`:
514:     - `__init__(n)` — initialize n elements
515:     - `.find(x)` — path compression
516:     - `.union(x, y)` — union by rank
517:     - `.groups()` → `list[list[int]]` — connected components
518:   - Function `overlapping_groups(positions, radii, linking_length_func)`:
519:     - Criterion: `linking_length_func(p1, r1, p2, r2) → bool`
520:     - Uses `UnionFind` internally
521:     - Returns `list[list[int]]` of groups
522: - [X] Write `tests/unit/test_union_find.py`:
523:   - Test `find()` returns correct root
524:   - Test `union()` merges correctly
525:   - Test path compression works (indirectly via subsequent `find()`)
526:   - Test `groups()` with no connections (all singletons)
527:   - Test `groups()` with all connected (one group)
528:   - Test `overlapping_groups` with simple 2-point case: `d <= r1+r2`
529:   - Test `overlapping_groups` with custom criterion function (e.g., overlapping bounding boxes)
530:   - Test `UnionFind` with 0 elements (edge case)
531: - [X] **Test and validate**: `pytest tests/unit/test_union_find.py -v`
532: - [X] Commit
533: 
534: ### Step 3.2 — HaloSegmenter
535: 
536: - [X] Create `src/roadrunner/clustering/segmentation.py`:
537:   - `find_populated_haloes(candidates_list)` → `(empty_indices, populated_indices)` — replaces `_find_populated_haloes`
538:   - Class `HaloSegmenter` (fixed spelling from `HaloSegmentator`):
539:     - `__init__(positions, virial_radii)`
540:     - `.overlap_groups()` → uses `overlapping_groups` with `d <= r1 + r2`
541:     - `.prune(candidate_list, min_particles, discard=False)` → isolates or discards small haloes
542: - [X] Keep `_check_groups` logic integrated into `UnionFind.groups()`
543: - [X] Write `tests/unit/test_segmentation.py`:
544:   - Test `find_populated_haloes()` with mixed empty/populated
545:   - Test `find_populated_haloes()` with all empty
546:   - Test `find_populated_haloes()` with all populated
547:   - Test `HaloSegmenter.overlap_groups()` with 3 haloes: 1-2 and 2-3 overlap but not 1-3
548:   - Test `HaloSegmenter.overlap_groups()` with no overlapping haloes
549:   - Test `prune()` removes groups below threshold
550:   - Test `prune()` with `discard=True` drops small halos entirely
551:   - Test `prune()` with `discard=False` isolates small halos as singletons
552:   - Test edge case: 0 haloes
553: - [X] **Test and validate**: `pytest tests/unit/test_segmentation.py -v`
554: - [X] Commit
555: 
556: ### Step 3.3 — Sparse Matrix Utilities
557: 
558: - [X] Create `src/roadrunner/clustering/sparse.py`:
559:   - `build_dense_from_csc(matrix_shape, true_indices, column_indices, column_values)` → dense matrix
560:   - `_csc_to_dense_kernel(...)` — private Numba parallel kernel (unchanged logic)
561:   - `stitch_zero_rows(matrix1, matrix2)` — merge zero rows from matrix2 into matrix1
562:   - `row_softmax_with_offset(resp)` — row-wise stabilized softmax with `exp(x) - 1` offset
563:   - `row_l1_normalize(resp)` — row-wise L1 normalization
564:   - Class `SparseCSC`:
565:     - `__init__(column_indices, column_values)`
566:     - `.to_dense(columns=None)` → dense matrix
567: - [X] Write `tests/unit/test_sparse.py`:
568:   - Test `build_dense_from_csc` with known 3x2 matrix
569:   - Test `SparseCSC.to_dense()` with subset of columns
570:   - Test `SparseCSC.to_dense()` with all columns
571:   - Test `row_l1_normalize()` rows sum to 1
572:   - Test `row_softmax_with_offset()` against scipy reference on small matrix
573:   - Test `stitch_zero_rows()` with simple 2x2 case
574:   - Test empty column handling
575:   - Test Numba kernels produce same results as pure Python reference
576: - [X] **Test and validate**: `pytest tests/unit/test_sparse.py -v`
577: - [X] Commit
578: 
579: ### Step 3.4 — Potential Models
580: 
581: - [X] Move potential code to `src/roadrunner/physics/potentials.py`:
582:   - Import constants from `physics.constants` instead of inline `1e-4`, `1e-5`
583:   - Define protocol `PotentialModel`:
584:     - `.potential(r)` → gravitational potential
585:     - `.dynamical_time(E_or_r)` → dynamical time
586:     - `.tidal_denominator()` → `(3 - dlnM/dlnr)*M`
587:   - Class `KeplerPotential(PotentialModel)`:
588:     - Constructor: `M, G`
589:     - `.potential(r)` → `-G*M / sqrt(r^2 + SOFTENING_KEPLER^2)`
590:     - `.dynamical_time(E)` → sqrt(a^3/(G*M)) where a = -0.5*G*M/E
591:     - `.tidal_denominator()` → 3*M
592:   - Class `NFWPotential(PotentialModel)`:
593:     - Constructor: `M, Rs, c, G`
594:     - `.potential(r)` → NFW analytic potential
595:     - `.enclosed_mass(r)` → M(<r)
596:     - `.dynamical_time(r)` → sqrt(r^3 / (G*M(<r)))
597:     - `.tidal_denominator(r)` → (3 - dlnM/dlnr)*M(r)
598:   - Keep `potential()`, `dynamical_time()`, `tidal_denominator()` as factory functions for backward compatibility
599: - [X] Write `tests/unit/test_potentials.py`:
600:   - Test `KeplerPotential.potential(r)` against analytic value `-G*M/r` (for large r)
601:   - Test `NFWPotential.potential(r)` is negative and monotonic
602:   - Test `dynamical_time()` returns positive values
603:   - Test `tidal_denominator()` for Kepler returns 3M
604:   - Test that softening prevents division by zero at r=0
605:   - Test Protocol pattern: mock object passes `isinstance` check
606:   - Test factory functions produce correct type
607: - [X] **Test and validate**: `pytest tests/unit/test_potentials.py -v`
608: - [X] Commit
609: 
610: **Phase 3 completion**: `git tag phase-3-clustering-infra`
611: 
612: ---
613: 
614: ## PHASE 4: Gaussian Mixture Models (Cleaned)
615: 
616: **Goal**: Clean separation of EM algorithms. No functional changes — only file reorganization and deduplication.
617: 
618: ### Step 4.1 — Shared Math Module
619: 
620: - [X] Create `src/roadrunner/mixture/_math.py`:
621:   - `logsumexp(log_probs)` — parallel numba log-sum-exp (from `_base_weighted_gaussian_mixture.py`)
622:   - `row_squared_norms(X)` — single canonical copy (was duplicated in `_kmeans_plusplus.py` and `weighted_gaussian_mixture.py`)
623:   - `compute_precision_cholesky(covariances, cov_type)` — from `weighted_bayesian_mixture.py`
624:   - `compute_log_det_cholesky(choleskys, cov_type, n_features)` — from `weighted_bayesian_mixture.py`
625: - [X] Write `tests/unit/test_mixture_math.py`:
626:   - Test `logsumexp` matches `scipy.special.logsumexp` with `axis=1, keepdims=True`
627:   - Test `row_squared_norms` matches `np.sum(X**2, axis=1)`
628:   - Test `compute_precision_cholesky` for spherical, diagonal, full
629:   - Test `compute_log_det_cholesky` matches `np.log(np.linalg.det())`
630: - [X] **Test and validate**: `pytest tests/unit/test_mixture_math.py -v`
631: - [X] Commit
632: 
633: ### Step 4.2 — Base Mixture
634: 
635: - [X] Move `_base_weighted_gaussian_mixture.py` → `src/roadrunner/mixture/base.py`
636: - [X] Keep class name `BaseMixture`
637: - [X] Import `logsumexp`, `row_squared_norms` from `._math`
638: - [X] No functional changes
639: - [X] Write `tests/unit/test_mixture_base.py`:
640:   - Test abstract methods raise `NotImplementedError` when called
641:   - Test `_initialize_weights_and_prior` with None inputs
642:   - Test `_initialize_weights_and_prior` with custom prior
643:   - Test `fit()` raises appropriate error on abstract methods
644: - [X] **Test and validate**: `pytest tests/unit/test_mixture_base.py -v`
645: - [X] Commit
646: 
647: ### Step 4.3 — Weighted Gaussian Mixture
648: 
649: - [X] Move `weighted_gaussian_mixture.py` → `src/roadrunner/mixture/weighted_gmm.py`
650: - [X] Import `row_squared_norms` from `._math`
651: - [X] Import parameter validation from `._validate` (new single-file validation helper)
652: - [X] Write `tests/unit/test_weighted_gmm.py`:
653:   - Test on 2D well-separated blobs (3 components)
654:   - Test `fit()` converges and lower bound is monotonic
655:   - Test `predict()` returns integer labels
656:   - Test `predict_proba()` returns probabilities that sum to 1
657:   - Test that point_weights change the result (compare weighted vs unweighted)
658:   - Test that latent_prior changes the result
659:   - Test with `float32` dtype
660:   - Test convergence tolerance
661:   - Test edge case: single component
662:   - Test edge case: more features than samples
663: - [X] **Test and validate**: `pytest tests/unit/test_weighted_gmm.py -v`
664: - [X] Commit
665: 
666: ### Step 4.4 — Bayesian Gaussian Mixture
667: 
668: - [X] Move `weighted_bayesian_mixture.py` → `src/roadrunner/mixture/bayesian_gmm.py`
669: - [X] Rename `_estimate_log_gaussian_prob_pchol` → `_estimate_log_prob_precision_cholesky`
670: - [X] Import Cholesky utilities from `._math`
671: - [X] Write `tests/unit/test_bayesian_gmm.py`:
672:   - Test on toy 2D data with 3 components
673:   - Test that ELBO is monotonic during fit
674:   - Test prior initialization shapes and values
675:   - Test all covariance types (full, diagonal, spherical)
676:   - Test posterior means are finite
677:   - Test `predict()` returns integer labels
678:   - Compare against sklearn `BayesianGaussianMixture` for basic sanity
679: - [X] **Test and validate**: `pytest tests/unit/test_bayesian_gmm.py -v`
680: - [X] Commit
681: 
682: ### Step 4.5 — Coresets
683: 
684: - [X] Move `gaussian_coresets.py` → `src/roadrunner/mixture/coresets.py`
685: - [X] Fix method name: `_initialize_centroids` → `_initialize_centers`
686: - [X] Import shared math from `._math`
687: - [X] Write `tests/unit/test_coresets.py`:
688:   - Test `generate_coreset()` returns correct shape: (n_core, n_features) and (n_core,)
689:   - Test `estimate_error()` returns finite epsilon
690:   - Test `_exact_fudge()` root-finding: verify `expected_unique(fudge*M) ≈ M`
691:   - Test `fit()` produces positive importance estimates
692:   - Test with `float32` dtype
693: - [X] **Test and validate**: `pytest tests/unit/test_coresets.py -v`
694: - [X] Commit
695: 
696: ### Step 4.6 — KMeans++
697: 
698: - [X] Move `_kmeans_plusplus.py` → `src/roadrunner/mixture/kmeans_plusplus.py`
699: - [X] Remove duplicate `row_squared_norms` — import from `_math.py`
700: - [X] Add module docstring
701: - [X] Write `tests/unit/test_kmeans_plusplus.py`:
702:   - Test centers returned are actual data points
703:   - Test correct number of centers
704:   - Test with `cluster_weights=None` (uniform weights)
705:   - Test with custom cluster-dependent weights
706:   - Test edge case: 1 cluster
707:   - Test edge case: all points identical
708:   - Test reproducibility with fixed random_state
709:   - Compare with sklearn's `kmeans_plusplus` on same data
710: - [X] **Test and validate**: `pytest tests/unit/test_kmeans_plusplus.py -v`
711: - [X] Commit
712: 
713: **Phase 4 completion**: `git tag phase-4-gmm-cleanup`
714: 
715: ---
716: 
717: ## PHASE 5: Particle Assignment
718: 
719: **Goal**: Clean separation of the assignment logic into composable strategies. This decomposes the 880-line `gaussians.py`.
720: 
721: ### Step 5.-2 — HaloModel
722: 
723: - [X] Create `src/roadrunner/physics/halo_model.py`:
724:   - Class `HaloModel`:
725:     - Constructor: `(inner: PotentialModel, xcen, velocity, virial_radius, sub_tree_id, redshift, comoving=True)`
726:     - Stores: `._inner` (PotentialModel instance), `.xcen`, `.velocity`, `.virial_radius`, `.sub_tree_id`, `.redshift`, `._1plusz`, `._boundness`
727:     - Methods: `.potential(xyz_or_r)`, `.dynamical_time(xyz_or_r)`, `.tidal_denominator(xyz_or_r)`
728:       - Accepts 2D positions (N,3) or 1D precomputed radii — detects automatically
729:       - Converts comoving → physical before delegating to inner model
730:     - Factory: `.from_snapshot_row(row, model="kepler", comoving=True)`
731:     - `_1plusz = 1/(1+z)` precomputed for efficiency
732:     - `_boundness` added in Step 5.-1
733: - [X] Write `tests/unit/test_halo_model.py`:
734:   - Test construction from merger tree row
735:   - Test `.potential(r)` returns same as `KeplerPotential/NFWPotential` with same params
736:   - Test protocol conformance (`isinstance(HaloModel(...), PotentialModel)`)
737:   - Test factory method handles required columns
738: - [X] **Test and validate**: `pytest tests/unit/test_halo_model.py -v`
739: - [X] **Lint**: `ruff check src/roadrunner/physics/halo_model.py`
740: - [X] Commit
741: 
742: ### Step 5.-1 — Boundness on HaloModel (set_/get_)
743: 
744: - [X] Add to `src/roadrunner/physics/halo_model.py`:
745:   - `_boundness: tuple | None = None` in `__init__`
746:   - `.set_boundness(indices, energies, tdyns)` — stores bound particle data
747:   - `.get_boundness()` → `tuple[NDArray, NDArray, NDArray] | None`
748:   - `.has_boundness` property — `True` after set_boundness is called
749:   - No `ProtoGalaxy` class — boundness is set_/get_ state on HaloModel
750: - [X] Write tests in `tests/unit/test_halo_model.py`:
751:   - Test `has_boundness` is False before, True after
752:   - Test `get_boundness()` returns stored arrays
753:   - Test set overwrites previous data
754:   - Test `get_boundness()` returns `None` when unset
755: - [X] **Test and validate**: `pytest tests/unit/test_halo_model.py -v`
756: - [X] Commit
757: 
758: **Ensemble representation**: `list[HaloModel]` throughout Phase 5. No ensemble class.
759: 
760: ### Step 5.1 — Boundness Computation
761: 
762: - [X] Create `src/roadrunner/physics/boundness.py`:
763:   - Function `compute_halo_bound_particles(halos, particle_coordinates, search_factor=1.0)`:
764:     - Takes `list[HaloModel]` instead of `snap_df` + kwargs
765:     - No class — standalone function
766:     - Uses KDTree for spatial search
767:     - Accesses halo data via: `halo.xcen`, `halo.velocity`, `halo.virial_radius`, `halo._inner.M`, `halo.potential()`, `halo.dynamical_time()`
768:     - Passes 1D distances to `halo.potential(dist)` and `halo.dynamical_time(dist)` (avoids double-centering)
769:     - Calls `halo.set_boundness(indices, energies, tdyns)` on each halo
770:     - Returns same `halos` list (modified in place)
771:   - All `4.3E-6` references use `constants.G_KM`
772:   - Dynamical time returned raw (no `/ 0.97779` — deferred to Step 5.5)
773:   - No `gc.collect()` or `del tree`
774: - [X] Write `tests/unit/test_boundness.py`:
775:   - Test with 2 mock haloes + 100 random particles
776:   - Test empty halo case (no particles within search radius)
777:   - Test `search_factor=2.0` captures more particles
778:   - Test Kepler vs NFW give different boundness values
779:   - Test returns same list (`result is halos`)
780: - [X] **Test and validate**: `pytest tests/unit/test_boundness.py -v`
781: - [X] Commit
782: 
783: ### Step 5.2 — Assignment Strategy Protocol
784: 
785: - [X] `ParticleAssigner` protocol in `_mcf_types.py`:
786:   - Minimal signature:
787:     ```python
788:     def assign(
789:         self,
790:         halos: list,
791:         particle_coords: np.ndarray,
792:         newborn_indices: np.ndarray,
793:         groups: list[list[int]],
794:         **kwargs,
795:     ) -> AssignmentResult: ...
796:     ```
797:   - 4 required params + `**kwargs` — no `previous_resp`, no `BoundnessResult`
798:   - `AssignmentResult` unchanged in `_mcf_types.py`
799: - [X] `clustering/assignment/__init__.py` created (empty)
800: - [X] Old `Assigner` protocol removed from `_mcf_types.py`
801: - [X] Write `tests/unit/test_assignment_base.py`:
802:   - Test protocol works with `isinstance` check (mock passes, wrong class fails)
803:   - Test `AssignmentResult` construction with mock data
804:   - Test extra `**kwargs` ignored by mock assigner
805: - [X] **Test and validate**: `pytest tests/unit/test_assignment_base.py -v`
806: - [X] Commit
807: 
808: ### Step 5.3 — GMM Assignment Strategy
809: 
810: - [X] Create `src/roadrunner/clustering/assignment/gmm.py`:
811:   - Class `GMMAssigner(ParticleAssigner)`:
812:     - Constructor: `cov_type, max_iter, tol, min_particles, reg_covar, prior_type, verbose`
813:     - `.assign(...)` — replaces `phase_space_particle_assignment()`
814:     - Private methods:
815:       - `_build_prior(newborn_idx, true_indices, boundness, prev_resp, group_subtrees)` — merges prev resp with new boundness using rank-based weighting
816:       - `_fit_single_component(particle_coords, group_bounds)` — trivial uniform assignment (single halo)
817:       - `_fit_unresolved_group(particle_coords, group_subtrees, boundness, true_indices)` — N < 10*K case, builds flat responsibility
818:       - `_fit_resolved_group(particle_coords, group_subtrees, prior, group_mean, scaling)` — full GMM fitting with float64 fallback
819:       - `_store_parameters(means, weights, covariances, group_mean, scaling)` — stores inverted scaling
820: - [X] Write `tests/unit/test_assignment_gmm.py`:
821:   - Test on small controlled data: 3 overlapping halos, 30 particles each
822:   - Verify output `AssignmentResult` has expected columns and types
823:   - Test single-component group produces correct assignment (all particles → one subtree)
824:   - Test unresolved group handling (N < 10*K)
825:   - Test that previous responsibilities influence result
826:   - Test newborn particle handling
827:   - Test edge case: 0 particles in a group
828:   - Test edge case: only newborn particles in a group
829: - [X] **Test and validate**: `pytest tests/unit/test_assignment_gmm.py -v`
830: - [X] Commit
831: 
832: ### Step 5.5 — Timescale Computation
833: 
834: - [X] Create `src/roadrunner/physics/timescales.py`:
835:   - `compute_particle_dynamical_timescales(particle_df, groups, candidate_list, dynamical_times_list)` → enriched `particle_df` with timescale column
836:     - Replaces `_tdyn_timescale()` from `gaussians.py`
837:     - Uses halos' `.get_boundness()` for per-halo candidate indices and dynamical times
838:     - Dynamical time from Step 5.1 needs to be multiplied by `DYN_TIME_FACTOR`:
839:       `timescale = tdyns_raw * DYN_TIME_FACTOR` (replaces old `/ 0.97779`)
840:   - `compute_tidal_radius(host_mass, satellite_mass, distance, redshift, potential_model)` → float
841:     - Replaces tidal radius computation from `_galaxy_properties()` in `properties.py`
842: - [X] Write `tests/unit/test_timescales.py`:
843:   - Test `compute_particle_dynamical_timescales` produces positive timescales
844:   - Test `compute_tidal_radius` returns positive finite value
845:   - Test `compute_tidal_radius` with Kepler model (analytic: r_t = D * (Msat / (3*Mhost))^(1/3))
846:   - Test edge case: satellite at large distance → large tidal radius
847:   - Test edge case: satellite at small distance → small tidal radius
848: - [X] **Test and validate**: `pytest tests/unit/test_timescales.py -v`
849: - [X] Commit
850: 
851: 
852: **Phase 5 completion**: `git tag phase-5-assignment`
853: 
854: ---
855: 
856: ## PHASE 6: Galaxy Analysis
857: 
858: **Goal**: Cleanly separate property computation and mixing analysis into pure stateless functions.
859: 
860: ### Step 6.1 — Geometry Utilities
861: 
862: - [X] In `src/roadrunner/postprocessing/properties.py`:
863:   - `random_lines_of_sight(n_lines, half_sphere=True, seed=None)` — replaces `_random_vector_spherical`
864:   - `rotation_matrix_from_los(los_vector)` — replaces `_rotation_matrix`
865:   - Group under `# --- Geometry Utilities ---` section
866: - [X] Write tests in `tests/unit/test_properties.py` (class `TestGeometry`):
867:   - Test `random_lines_of_sight` returns unit vectors (`np.linalg.norm == 1`)
868:   - Test `rotation_matrix_from_los` aligns LOS with +z axis
869:   - Test rotation matrix is orthogonal: `R @ R.T == I`
870:   - Test reproducibility with seed
871: - [X] **Test and validate**: `pytest tests/unit/test_properties.py::TestGeometry -v`
872: - [X] Commit
873: 
874: ### Step 6.2 — Galaxy Centering
875: 
876: - [X] In `src/roadrunner/postprocessing/properties.py`:
877:   - `find_center(positions, velocities, masses)` → `(center_pos, center_vel)`
878:     - No algorithm changes, just cleaner signature
879:     - Replaces standalone `find_center()` function
880: - [X] Write tests in `tests/unit/test_properties.py` (class `TestCentering`):
881:   - Test with known center: particles distributed around (10, 20, 30)
882:   - Test mass-weighted center shifts towards massive particles
883:   - Test with uniform masses (center = mean)
884:   - Test with single particle
885: - [X] **Test and validate**: `pytest tests/unit/test_properties.py::TestCentering -v`
886: - [X] Commit
887: 
888: ### Step 6.3 — Mass Radii
889: 
890: - [X] In `src/roadrunner/postprocessing/properties.py`:
891:   - `enclosed_mass_radius(positions, masses, mass_fraction)` — replaces `_Xmass_radius_solver_searchsorted`
892:   - `projected_enclosed_mass_radius(positions, masses, mass_fraction, los_matrices)` — replaces `Xmass_radius` with rotation_matrices parameter
893:   - `half_mass_radius(positions, masses)` — convenience wrapper for mfrac=0.5
894:   - `projected_half_mass_radius(positions, masses, los_matrices)` — convenience wrapper
895: - [X] Write tests in `tests/unit/test_properties.py` (class `TestRadii`):
896:   - Test `enclosed_mass_radius` with uniform distribution: r_50 ≈ median radius
897:   - Test `enclosed_mass_radius` with known radial profile
898:   - Test `half_mass_radius` wraps correctly
899:   - Test `projected_half_mass_radius` returns different values for different LOS
900:   - Test edge case: 1 particle → returns 0
901:   - Test edge case: all masses zero → returns 0
902: - [X] **Test and validate**: `pytest tests/unit/test_properties.py::TestRadii -v`
903: - [X] Commit
904: 
905: ### Step 6.4 — Kinematics
906: 
907: - [X] In `src/roadrunner/postprocessing/properties.py`:
908:   - `velocity_dispersion(velocities)` — 3D velocity dispersion
909:   - `line_of_sight_velocity_dispersion(positions, velocities, los_matrices, apertures)` — replaces `los_velocity_dispersion`
910: - [X] Write tests in `tests/unit/test_properties.py` (class `TestKinematics`):
911:   - Test `velocity_dispersion` with known velocities
912:   - Test `line_of_sight_velocity_dispersion` with single particle
913:   - Test `line_of_sight_velocity_dispersion` with all particles within aperture
914:   - Test `line_of_sight_velocity_dispersion` with no particles in aperture → nan
915: - [X] **Test and validate**: `pytest tests/unit/test_properties.py::TestKinematics -v`
916: - [X] Commit
917: 
918: ### Step 6.5 — Galaxy Properties Orchestrator
919: 
920: - [X] In `src/roadrunner/postprocessing/properties.py`:
921:   - `compute_galaxy_properties(accretion_id, snap_df, particle_masses, particle_coords, assembly_map, lines_of_sight=None)`:
922:     - Pure function: takes data in, returns DataFrame out
923:     - Calls geometry → centering → radii → kinematics per galaxy
924:     - Uses `PotentialModel` protocol for tidal denominator (not string dispatch)
925:     - Generates lines of sight if not provided
926: - [X] Write tests in `tests/unit/test_properties.py` (class `TestOrchestrator`):
927:   - Test with mock assembly data (2 galaxies, 30 particles each)
928:   - Verify output DataFrame has expected columns
929:   - Verify column order matches documentation
930:   - Test with empty assembly → empty DataFrame
931:   - Test with custom lines of sight
932: - [X] **Test and validate**: `pytest tests/unit/test_properties.py::TestOrchestrator -v`
933: - [X] Commit
934: 
935: ### Step 6.6 — Riley Criterion (cleaned)
936: 
937: - [X] In `src/roadrunner/postprocessing/mixing.py`:
938:   - `compute_bound_mask(snap_df_slice, positions, velocities)` — replaces `_bound_mask`
939:   - `local_phase_space_dispersion(positions, velocities, n_neighbors=10, workers=1)` — replaces `_local_velocity_dispersion`
940:   - `riley_dynamical_state_classification(subtree_id, snap_df_slice, array_indices, bound_array_indices, masses, positions, velocities, suffix)` — replaces `_riley_dynstate_criterion_single`
941:   - `compute_riley_criterion(main_id, snap_df, particle_masses, particle_coords, assembly_map, birth_map, bound_assembly_map)` — same interface, cleaner internals
942: - [X] Write `tests/unit/test_mixing.py`:
943:   - Test `compute_bound_mask` with known Kepler potential
944:   - Test `local_phase_space_dispersion` shape matches input
945:   - Test `local_phase_space_dispersion` with insufficient particles → NaN
946:   - Test `riley_dynamical_state_classification` for intact case (f_bound >= 0.97)
947:   - Test `compute_riley_criterion` with mock assembly data
948:   - Verify output DataFrame columns
949: - [X] **Test and validate**: `pytest tests/unit/test_mixing.py -v`
950: - [X] Commit
951: 
952: **Phase 6 completion**: `git tag phase-6-galaxy-analysis`
953: 
954: ---
955: 
956: ## PHASE 7: Tracking
957: 
958: **Goal**: Clean streaming trackers with clear state management.
959: 
960: ### Step 7.1 — Birth Tracker
961: 
962: - [X] Create `src/roadrunner/postprocessing/tracking/__init__.py`
963: - [X] Create `src/roadrunner/postprocessing/tracking/birth.py`:
964:   - Module-level window functions: `gaussian_window(dt_tau)`, `cauchy_window(dt_tau)`, `exp_window(dt_tau)`
965:   - Class `BirthTracker`:
966:     - Constructor: `factor=5, enforce_initial_hosts=False, window="gaussian"`
967:     - State: `_active: dict`, `_finalized: dict`, `_heap: list`, `_birth_map: dict`, `_last_snapshot: int`
968:     - `.update(t_snap, snapshot_id, particle_ids, host_ids, timescales, weights=None)`:
969:       - `.add_new_particles(...)` — register new particles with initial host scores
970:       - `.update_active_particles(...)` — increment scores with Gaussian-weighted delta
971:       - `.finalize_expired_particles(t_snap)` — pop from heap, finalize expired windows
972:     - `.finalize()` → `pd.DataFrame` with columns: particle_index, birth_id
973:     - `.current_birth_map()` → `dict[galaxy_id, set[particle_id]]`
974:     - `._get_state()` / `._set_state(state)` — serialization for restart
975: - [X] Write `tests/unit/test_birth_tracker.py`:
976:   - Test single particle across 3 snapshots → finalized after factor*tau
977:   - Test competition between two hosts (particle switches allegiance)
978:   - Test `enforce_initial_hosts=True` restricts competition to initial hosts
979:   - Test `current_birth_map()` gives consistent state
980:   - Test `finalize()` returns all particles
981:   - Test empty update: no particles
982:   - Test all particles finalized immediately
983:   - Test multiple particles with different timescales
984:   - Test state serialization round-trip
985: - [X] **Test and validate**: `pytest tests/unit/test_birth_tracker.py -v`
986: - [X] Commit
987: 
988: ### Step 7.2 — Assembly Tracker
989: 
990: - [X] Create `src/roadrunner/postprocessing/tracking/assembly.py`:
991:   - Class `AssemblyTracker`:
992:     - Constructor: `n_sat_history=2, unbound_default=-1`
993:     - State: `_infall_lists: dict`, `_previous_birth_map: dict`, `_frozen: set`, `_sat_history_buffer: deque`, `_last_snapshot: int`
994:     - `.update(snapshot_id, assignment_map, birth_map, satellites_map, freeze_galaxies=None)`:
995:       - `_aggregate_satellite_history(satellites_map)` — merges current + buffer
996:       - `_topological_sort(galaxies, satellites)` → list in bottom-up order (satellites first)
997:         - Raises `CycleError` if cycle detected
998:       - Accumulate infall lists: birth particles + inherited from satellites
999:     - `.current()` → `dict[galaxy_id, set[particle_id]]`
1000:     - `._get_state()` / `._set_state(state)` — serialization for restart
1001: - [X] Write `tests/unit/test_assembly_tracker.py`:
1002:   - Test single galaxy without satellites: only birth particles
1003:   - Test galaxy with one satellite: birth + inherited particles
1004:   - Test satellite chain A→B→C: DFS ordering processes A first, then B, then C
1005:   - Test cycle detection raises `CycleError`
1006:   - Test `freeze_galaxies` prevents further updates
1007:   - Test `current()` returns correct state after multiple updates
1008:   - Test `n_sat_history=1` vs `n_sat_history=3`
1009:   - Test state serialization round-trip
1010: - [X] **Test and validate**: `pytest tests/unit/test_assembly_tracker.py -v`
1011: - [X] Commit
1012: 
1013: ### Step 7.3 — Birth assignment integration (bridge to readers)
1014: 
1015: - [X] Update `src/roadrunner/postprocessing/tracking/__init__.py` to export both trackers
1016: - [X] Write integration test in `tests/integration/`:
1017:   - Wire `BirthTracker` + `AssemblyTracker` together
1018:   - Feed 3 snapshots of mock data
1019:   - Verify consistency: all finalized birth particles appear in assembly of their birth galaxy
1020: - [X] **Test and validate**: `pytest tests/integration/ -v -k tracker -v`
1021: - [X] Commit
1022: 
1023: **Phase 7 completion**: `git tag phase-7-tracking`
1024: 
1025: ---
1026: 
1027: ## PHASE 8: I/O Layer
1028: 
1029: **Goal**: Clean output abstraction, not hardcoded into business logic.
1030: 
1031: ### Step 8.1 — HDF5 Writer
1032: 
1033: - [X] Create `src/roadrunner/io/__init__.py`
1034: - [X] Create `src/roadrunner/io/hdf5_writer.py`:
1035:   - Class `HDF5ArchiveWriter`:
1036:     - Constructor: `output_path: str, mode: str = "w"`
1037:     - `.write_header(main_id, snapshots, particle_ids, birth_times)` — creates HDF5 group structure
1038:     - `.write_snapshot(snapshot_id, time, particle_df, properties_df, dynstate_df, satellites_map)` — per-snapshot datasets

(Output capped at 50 KB. Showing lines 1-1038. Use offset=1039 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src</path>
<type>directory</type>
<entries>
roadrunner.egg-info/
roadrunner/

(2 entries)
</entries>
```

---

## Assistant (Plan · GLM-5.1 · 3.5s)

_Thinking:_

Let me continue reading planB.md from line 1039, and also explore the source code structure.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/planning/planB.md",
  "offset": 1039
}
```

**Output:**
```
<path>/home/asier/roadrunner/planning/planB.md</path>
<type>file</type>
<content>
1039:     - `.write_snapshot_data(snapshot_id, time, redshift, particle_data)` — particle positions/velocities/masses
1040:     - `.write_merger_tree(merger_df, equiv_df, snapnames)` — merger tree + equivalence table
1041:     - `.finalize(birth_tracker, assembly_tracker, reduce=False)` — birth/assembly records + consolidation
1042: - [X] Write `tests/unit/test_hdf5_writer.py`:
1043:   - Use `tmp_path` fixture (pytest built-in)
1044:   - Test writing header → verify HDF5 file exists and has expected groups
1045:   - Test writing snapshot → verify dataset is readable
1046:   - Test writing merger tree → verify values round-trip
1047:   - Test finalize → verify birth/assembly records are stored
1048:   - Test overwrite protection
1049:   - Test writing multiple snapshots and reading them back
1050: - [X] **Test and validate**: `pytest tests/unit/test_hdf5_writer.py -v`
1051: - [X] Commit
1052: 
1053: ### Step 8.2 — HDF5 Reader
1054: 
1055: - [X] Create `src/roadrunner/io/hdf5_reader.py`:
1056:   - Class `HDF5ArchiveReader`:
1057:     - Constructor: `path: str`
1058:     - `.read_header()` → `HeaderData` namedtuple
1059:     - `.read_last_snapshot()` → `int | None`
1060:     - `.read_snapshot(snapshot_id)` → `SnapshotOutput` namedtuple
1061:     - `.read_particle_data(snapshot_id)` → `ParticleData` namedtuple
1062:     - `.read_birth_catalogue()` → `pd.DataFrame`
1063:     - `.read_assembly_catalogue()` → `pd.DataFrame`
1064: - [X] Write `tests/unit/test_hdf5_reader.py`:
1065:   - Write file with `HDF5ArchiveWriter`, read with `HDF5ArchiveReader`
1066:   - Test round-trip: header values match
1067:   - Test round-trip: snapshot values match
1068:   - Test `read_last_snapshot()` returns correct value
1069:   - Test reading non-existent snapshot raises `SnapshotLoadError`
1070:   - Test reading empty file raises appropriate error
1071: - [X] **Test and validate**: `pytest tests/unit/test_hdf5_reader.py -v`
1072: - [X] Commit
1073: 
1074: ### Step 8.3 — Restart Serialization
1075: 
1076: - [X] Create `src/roadrunner/io/serialization.py`:
1077:   - `save_checkpoint(path: str, data: dict, level: int = 3)`:
1078:     - Pickle + zstd compression
1079:     - Versioned format: `{"version": 2, "data": ...}`
1080:   - `load_checkpoint(path: str)` → `dict`:
1081:     - Handles version migration (v1 → v2)
1082:     - Raises `RestartError` on corrupted data
1083: - [X] Write `tests/unit/test_serialization.py`:
1084:   - Save dict → load dict → verify data matches
1085:   - Test with nested structures
1086:   - Test with numpy arrays
1087:   - Test compression levels
1088:   - Test corrupted file raises `RestartError`
1089:   - Test version migration
1090: - [X] **Test and validate**: `pytest tests/unit/test_serialization.py -v`
1091: - [X] Commit
1092: 
1093: ### Step 8.4 — Logging
1094: 
1095: - [X] Create `src/roadrunner/io/logging.py`:
1096:   - `format_runtime(seconds)` — moved from `helpers.py`
1097:   - Class `RunLogger`:
1098:     - Constructor: `log_path: str, logo_path: str | None = None`
1099:     - `.write_header(config: "RunConfig")` — prints logo + config summary
1100:     - `.write_snapshot(stats: dict)` — formatted per-snapshot stats line
1101:     - `.write_summary()` — final summary block
1102: - [X] Write `tests/unit/test_logging.py`:
1103:   - Use `StringIO` to capture output (no real files)
1104:   - Test `format_runtime` format
1105:   - Test `write_header` output contains key config fields
1106:   - Test `write_snapshot` output contains expected columns
1107: - [X] **Test and validate**: `pytest tests/unit/test_logging.py -v`
1108: - [X] Commit
1109: 
1110: **Phase 8 completion**: `git tag phase-8-io`
1111: 
1112: ---
1113: 
1114: ## PHASE 9: Pipeline
1115: 
1116: **Goal**: Replace the god `BaseAccretionHistory` with composable pipeline.
1117: 
1118: ### Step 9.1 — Snapshot Processor
1119: 
1120: - [X] Create `src/roadrunner/pipeline/__init__.py`
1121: - [X] Create `src/roadrunner/pipeline/snapshot_processor.py`:
1122:   - Class `SnapshotProcessor`:
1123:     - Constructor: `assigner, potential_model, birth_tracker, assembly_tracker, min_particles=10, riley_criterion_snapshots="all"`
1124:     - `.process(snapshot_data: SnapshotInput, previous_resp: CSCResp | None)` → `SnapshotOutput`
1125:     - `SnapshotInput` dataclass: `(snap_df, particle_coords, newborn_idx, particle_masses, particle_indices, time, redshift)`
1126:     - `SnapshotOutput` dataclass: `(particle_df, properties_df, dynstate_df, satellites_map, responsibilities, fitted_parameters, statistics)`
1127:     - Orchestrates: `assign_particles()` → `birth_tracker.update()` → `assembly_tracker.update()` → `compute_galaxy_properties()` → `compute_riley_criterion()`
1128: - [X] Write `tests/integration/test_snapshot_processor.py`:
1129:   - Wire together mock readers + `GMMAssigner` + `BirthTracker` + `AssemblyTracker`
1130:   - Process one snapshot of mock data → verify `SnapshotOutput` structure
1131:   - Process two snapshots sequentially → verify state passes correctly (birth tracker continues)
1132:   - Verify `properties_df` has non-empty values
1133:   - Verify `dynstate_df` has expected columns
1134: - [X] **Test and validate**: `pytest tests/integration/test_snapshot_processor.py -v`
1135: - [X] Commit
1136: 
1137: ### Step 9.2 — Accretion Pipeline
1138: 
1139: - [X] Create `src/roadrunner/pipeline/accretion_pipeline.py`:
1140:   - Class `AccretionPipeline`:
1141:     - Constructor: `merger_reader, snapshot_reader, equiv_table, snapshot_processor, output_writer, logger`
1142:     - `.run(output_dir, start_snapshot, end_snapshot)`:
1143:       1. Initialize output (header, groups)
1144:       2. Load first snapshot (simple preload, no ThreadPoolExecutor)
1145:       3. For each snapshot:
1146:          - Process via `snapshot_processor`
1147:          - Write results via `output_writer`
1148:          - Log via `logger`
1149:          - Preload next snapshot
1150:       4. Finalize (merger tree, birth/assembly records)
1151:     - Error handling:
1152:       - `except SnapshotLoadError` → log warning, skip to next
1153:       - `except ConvergenceError` → fallback to `FastAssigner` for remaining
1154:       - `except Exception` → `save_checkpoint()` and re-raise
1155:     - `.resume_from(output_dir, checkpoint)` → resume interrupted run
1156: - [X] Write `tests/integration/test_accretion_pipeline.py`:
1157:   - Full mock pipeline with 3 snapshots (5 halos each)
1158:   - Verify loop executes all snapshots
1159:   - Verify HDF5 output is readable after run
1160:   - Verify restart: save checkpoint mid-run, create new pipeline, verify resumes correctly
1161:   - Verify `FastAssigner` fallback on convergence error
1162: - [X] **Test and validate**: `pytest tests/integration/test_accretion_pipeline.py -v`
1163: - [X] Commit
1164: 
1165: ### Step 9.3 — Pipeline entry point
1166: 
1167: - [X] In `src/roadrunner/pipeline/__init__.py`:
1168:   - `run_accretion_history(config)`:
1169:     - Constructs all dependencies from a `RunConfig`
1170:     - Calls `AccretionPipeline.run()`
1171: - [X] Write test:
1172:   - Test with minimal `RunConfig` using mock data
1173:   - Verify end-to-end: config → `run_accretion_history` → valid output
1174: - [X] **Test and validate**: `pytest tests/integration/ -v -k run_accretion`
1175: - [X] Commit
1176: 
1177: **Phase 9 completion**: `git tag phase-9-pipeline`
1178: 
1179: ---
1180: 
1181: ## PHASE 10: Pipeline Integration & Cleanup
1182: 
1183: **Goal**: Remove old code, finalize all integration paths, run full test suite.
1184: 
1185: ### Step 10.1 — Create `RunConfig`
1186: 
1187: - [X] Create `src/roadrunner/pipeline/config.py`:
1188:   ```python
1189:   @dataclass
1190:   class RunConfig:
1191:       # Simulation data
1192:       merger_tree_path: str | pd.DataFrame
1193:       particle_data_dir: str
1194:       equivalence_path: str | pd.DataFrame
1195:       code: str
1196:       particle_type: str
1197:       fields: dict[str, str]
1198:       unit_base: dict | None
1199: 
1200:       # Target selection
1201:       accretion_id: int | None = None  # auto-detect if None
1202:       final_snapshot: int | None = None  # use max if None
1203:       search_radius_factor: float = 1.5
1204: 
1205:       # Processing
1206:       assigner: str = "gmm"  # "gmm" or "fast"
1207:       cov_type: str = "full"
1208:       max_iter: int = 10
1209:       tol: float = 1e-3
1210:       min_particles: int = 10
1211:       halo_model: str = "kepler"
1212:       softening: float = 0.0
1213: 
1214:       # Tracking
1215:       birth_window_factor: float = 5.0
1216:       enforce_initial_hosts: bool = False
1217: 
1218:       # Output
1219:       output_dir: str
1220:       riley_criterion_snapshots: list[int] | str = "all"
1221:       verbose: bool = True
1222:   ```
1223:   - `config_from_yaml(path: str)` → `RunConfig`
1224:   - `config_from_cli(args: list)` → `RunConfig`
1225: - [X] Write tests:
1226:   - Test defaults
1227:   - Test YAML round-trip
1228:   - Test CLI parsing
1229:   - Test validation errors
1230: - [X] **Test and validate**: `pytest tests/unit/test_config.py -v`
1231: - [X] Commit
1232: 
1233: ### Step 10.2 — Remove old classes
1234: 
1235: - [ ] Verify coverage: confirm old `AccretionHistory` + `BaseAccretionHistory` functionality is fully replaced
1236: - [ ] Remove old `print_logs.py` (replaced by `io/logging.py`)
1237: - [ ] Remove old `write_catalogues.py` (replaced by `io/hdf5_writer.py`)
1238: - [ ] Remove old `merger_utils.py` (replaced by `readers/merger_tree.py`)
1239: - [ ] Remove old `load.py` (replaced by `readers/snapshot.py`)
1240: - [ ] Remove old `helpers.py` (replaced by `helpers.py` in new layout — keep the new one)
1241: - [ ] Update all imports across the codebase
1242: - [ ] Run full test suite: `conda run -n roadrunner pytest tests/ -v --tb=short`
1243: - [ ] **Test and validate**: all tests pass
1244: - [ ] Commit: `git commit -m "Phase 10.2: Remove old classes, finalize module structure"`
1245: 
1246: ### Step 10.3 — End-to-end integration test
1247: 
1248: - [X] Create `tests/integration/test_end_to_end.py`:
1249:   - Generate full mock dataset: merger tree (5 snapshots, 5 halos each), particle data (500 particles), equivalence table
1250:   - Build `RunConfig` pointing at mock data
1251:   - Call `run_accretion_history(config)`
1252:   - Verify all HDF5 output files exist
1253:   - Read back with `HDF5ArchiveReader`
1254:   - Verify birth records match expectations
1255:   - Verify assembly records are consistent
1256:   - Verify galaxy properties have expected columns and non-NaN values
1257:   - Verify merger tree is stored correctly
1258: - [X] **Test and validate**: `pytest tests/integration/test_end_to_end.py -v`
1259: - [X] Commit
1260: 
1261: ### Step 10.4 — Clean up `__init__.py` files
1262: 
1263: - [X] Replace all `from .x import *` with explicit name lists across all `__init__.py`
1264: - [X] `src/roadrunner/__init__.py`:
1265:   ```python
1266:   from .pipeline import run_accretion_history
1267:   from .pipeline.config import RunConfig
1268:   from .readers.merger_tree import MergerTreeReaderCSV
1269:   from .readers.snapshot import SnapshotReader
1270:   from .readers.equivalence import EquivalenceTable
1271:   from .mixture.weighted_gmm import WeightedGaussianMixture
1272:   from .mixture.bayesian_gmm import WeightedBayesianGaussianMixture
1273:   from .mixture.coresets import GaussianCoreset
1274:   from .postprocessing.tracking.birth import BirthTracker
1275:   from .postprocessing.tracking.assembly import AssemblyTracker
1276:   from .postprocessing.mixing import compute_riley_criterion
1277:   from .postprocessing.properties import compute_galaxy_properties
1278:   from .clustering.assignment.gmm import GMMAssigner
1279:   from .clustering.assignment.orchestrator import assign_particles
1280:   from .physics.halo_model import HaloModel
1281:   from .physics.potentials import get_potential, KeplerPotential, NFWPotential
1282:   # FastAssigner — only if Step 5.4 (extra) has been implemented
1283:   ```
1284: - [X] Run full test suite: `conda run -n roadrunner pytest tests/ -v`
1285: - [X] **Test and validate**: all tests pass
1286: - [X] Commit
1287: 
1288: **Phase 10 completion**: `git tag phase-10-integration`
1289: 
1290: ---
1291: 
1292: ## PHASE 11: Scripts & Testing Infrastructure
1293: 
1294: **Goal**: Portable scripts with CLI arguments, proper CI setup, complete test coverage.
1295: 
1296: ### Step 11.1 — Refactored Pipeline Script
1297: 
1298: - [X] `scripts/run_pipeline.py` (replaces `full_simulation.py`):
1299:   - CLI via `argparse`:
1300:     ```python
1301:     python scripts/run_pipeline.py \
1302:         --merger-tree /path/to/tree.csv \
1303:         --particle-dir /data/sims/halo685/ \
1304:         --equivalence /path/to/equiv.csv \
1305:         --code RAMSES \
1306:         --output-dir ./results/halo685/ \
1307:         --halo-id 12345 \
1308:         --max-snapshot 200 \
1309:         --halo-model kepler \
1310:         --cov-type full
1311:     ```
1312:   - Also supports `--config config.yaml` for full config
1313:   - No hardcoded paths
1314: - [X] Test: `conda run -n roadrunner python scripts/run_pipeline.py --help` → prints usage
1315: - [X] Commit
1316: 
1317: ### Step 11.2 — Analysis Scripts
1318: 
1319: - [X] `scripts/analyze_snapshot.py` (replaces `single_snapshot.py`):
1320:   - CLI: `--catalogue /path/to/output.hdf5 --snapshot 175 --output figure.png --max-particles 500`
1321:   - Reads from `HDF5ArchiveReader`
1322:   - Generates scatter plot of particles colored by galaxy assignment
1323: - [X] `scripts/compute_sfr.py` (replaces `sfr.py`):
1324:   - CLI: `--catalogue /path/to/output.hdf5 --output figure.pdf --bin-width 0.1`
1325:   - Reads birth times from HDF5
1326:   - Computes and plots SFR(t)
1327: - [X] `scripts/add_radii.py`:
1328:   - CLI: `--catalogue /path/to/output.hdf5 --merger-tree /path/to/tree.csv`
1329:   - Same logic as before, no hardcoded paths
1330: - [X] `scripts/movie_properties.py`:
1331:   - CLI: `--catalogue /path/to/output.hdf5 --merger-tree /path/to/tree.csv --subtree 23599 --output-dir ./frames/`
1332: - [X] Test each: `python scripts/analyze_snapshot.py --help` prints usage
1333: - [X] Commit
1334: 
1335: ### Step 11.3 — CI Configuration
1336: 
1337: - [ ] Create `.github/workflows/ci.yml`:
1338:   ```yaml
1339:   name: CI
1340:   on: [push, pull_request]
1341:   jobs:
1342:     test:
1343:       runs-on: ubuntu-latest
1344:       steps:
1345:         - uses: actions/checkout@v4
1346:         - uses: conda-incubator/setup-miniconda@v3
1347:           with:
1348:             activate-environment: roadrunner
1349:             python-version: "3.10"
1350:         - name: Install dependencies
1351:           run: |
1352:             conda install -n roadrunner numpy numba scipy pandas h5py yt tqdm zstandard -y
1353:             pip install -e ".[dev]"
1354:         - name: Lint
1355:           run: conda run -n roadrunner ruff check src/roadrunner/
1356:         - name: Test
1357:           run: conda run -n roadrunner pytest tests/ -v --cov=roadrunner --cov-report=xml
1358:         - name: Upload coverage
1359:           uses: codecov/codecov-action@v4
1360:   ```
1361: - [ ] Commit
1362: 
1363: ### Step 11.4 — Final `__init__.py` and release
1364: 
1365: - [X] Final review of all `__init__.py` files — no `*` imports anywhere
1366: - [X] Run full test suite + lint: `conda run -n roadrunner pytest tests/ -v && conda run -n roadrunner ruff check src/roadrunner/`
1367: - [X] **Test and validate**: zero failures
1368: - [X] `git tag v1.0.0`
1369: - [X] Commit: `git commit --allow-empty -m "Phase 11: v1.0.0 release"`
1370: 
1371: **Phase 11 completion**: `git tag phase-11-final`
1372: 
1373: ---
1374: 
1375: ## PHASE 12: Mock Data Generation
1376: 
1377: **Goal**: Create a `roadrunner.mock` module that generates realistic mock datasets for testing the full pipeline.
1378: 
1379: ### Dependency
1380: 
1381: - Requires `astro-limepy` (`pip install astro-limepy`) for generating spherical galaxy models with limepy distribution functions.
1382: 
1383: ### Target Repository Structure
1384: 
1385: ```
1386: src/roadrunner/
1387: └── mock/                           ← new module (not in original accme)
1388:     ├── __init__.py                  ← generate_mock_run() entry point
1389:     ├── population.py                ← Mass function + galaxy population utilities
1390:     ├── layout.py                    ← Spatial placement (box, groups, overlaps)
1391:     ├── galaxy.py                    ← MockGalaxy (limepy wrapper)
1392:     ├── tree.py                      ← Merger tree + equivalence table builders
1393:     └── snapshot.py                  ← Snapshot builder + ground-truth assignment
1394: ```
1395: 
1396: ### Step 12.1 — Galaxy Population Utilities
1397: 
1398: **`src/roadrunner/mock/population.py`:**
1399: 
1400: | Function | Input | Output | Description |
1401: |---|---|---|---|
1402: | `sample_schechter_masses(m_total, n_galaxies, alpha=-1.3, M_star=10**10.75, seed=None)` | total stellar mass, target count, Schechter params | `ndarray (n_galaxies,)` | Draw masses from Schechter function, normalised to `m_total`. Uses inverse CDF sampling + renormalisation. |
1403: | `assign_halo_masses(m_stellar, redshift=0)` | stellar mass array, redshift | `(m_halo, r_vir, c, r_s)` — each `ndarray` | Stellar-to-halo mass relation (e.g. Behroozi+13 or Moster+13). `r_vir` from Bryan & Norman 1998 overdensity. `c` from `nfw_cmz_relation_duffy`. |
1404: | `assign_birth_times(n_particles, formation_redshift=3, snapshot_redshift=0, seed=None)` | count, formation z, current z | `ndarray (n_particles,)` | Random birth times between formation and current snapshot (for `BirthTracker` tests). |
1405: 
1406: - [ ] Write production code in `src/roadrunner/mock/population.py`
1407: - [ ] Write tests in `tests/unit/test_mock_population.py`:
1408:   - Test `sample_schechter_masses` sums to `m_total`
1409:   - Test all masses are positive
1410:   - Test reproducibility with seed
1411:   - Test `assign_halo_masses` returns physically reasonable values (M_halo > M_stellar, r_vir > 0, c > 1)
1412:   - Test edge case: single galaxy → one mass equal to m_total
1413: - [ ] **Test and validate**: `pytest tests/unit/test_mock_population.py -v`
1414: - [ ] Commit
1415: 
1416: ### Step 12.2 — Spatial Layout
1417: 
1418: **`src/roadrunner/mock/layout.py`:**
1419: 
1420: | Function | Input | Output | Description |
1421: |---|---|---|---|
1422: | `place_galaxies(n_galaxies, n_groups, box_size, group_spread, random_seed)` | counts, box size, group dispersion | `(positions (N,3), velocities (N,3), group_ids (N,))` | Places galaxies in a simulation box. A fraction form `n_groups` overlapping groups (virial radii overlap) for `HaloSegmenter` tests; the rest are isolated. |
1423: | `assign_virial_motion(positions, masses, redshift)` | positions, masses, z | `(N,3) velocities` | Assigns Hubble flow + random peculiar velocities scaled to mass. |
1424: 
1425: - [ ] Write production code in `src/roadrunner/mock/layout.py`
1426: - [ ] Write tests in `tests/unit/test_mock_layout.py`:
1427:   - Test correct number of groups
1428:   - Test galaxies within a group have overlapping virial radii
1429:   - Test isolated galaxies have no overlaps
1430:   - Test reproducibility with seed
1431: - [ ] **Test and validate**: `pytest tests/unit/test_mock_layout.py -v`
1432: - [ ] Commit
1433: 
1434: ### Step 12.3 — Mock Galaxy (limepy wrapper)
1435: 
1436: **`src/roadrunner/mock/galaxy.py`:**
1437: 
1438: #### `MockGalaxy` class
1439: 
1440: **Purpose**: Represents a single galaxy generated from a limepy model. Holds the limepy model, particle samples, and metadata.
1441: 
1442: **Internal state**:
1443: | Field | Type | Description |
1444: |---|---|---|
1445: | `_model` | `limepy.limepy` | The limepy distribution function model |
1446: | `position` | `ndarray (3,)` | Galaxy centre in simulation box (kpc) |
1447: | `velocity` | `ndarray (3,)` | Galaxy bulk velocity (km/s) |
1448: | `mass` | `float` | Total stellar mass (Msun) |
1449: | `r_half` | `float` | 3D half-mass radius (kpc) |
1450: | `sub_tree_id` | `int` | Unique galaxy identifier |
1451: 
1452: | Method | Signature | Returns | Description |
1453: |---|---|---|---|
1454: | `__init__(mass, r_half, position, velocity, phi0=6.0, g=1.5, seed=42)` | constructor params | — | Builds limepy model with `M=mass`, `r0=r_half`, `phi0=phi0`, `g=g`. Stores metadata. |
1455: | `sample_particles(N)` | `int` | `(pos (N,3), vel (N,3), masses (N,))` | Calls `limepy.sample(N)`, offsets positions by `self.position` and velocities by `self.velocity`. Returns arrays. |
1456: | `virial_radius` (property) | — | `float` | Estimated as `r_half × 3.0` (rough conversion for limepy King models). |
1457: | `scale_radius` (property) | — | `float` | `r_half / c` where c from concentration. Or estimated from limepy output. |
1458: 
1459: - [ ] Write production code in `src/roadrunner/mock/galaxy.py`
1460: - [ ] Write tests in `tests/unit/test_mock_galaxy.py`:
1461:   - Test `sample_particles` returns correct shapes
1462:   - Test particles are centred around `self.position`
1463:   - Test particles have bulk `self.velocity` offset
1464:   - Test reproducibility with seed
1465:   - Test `virial_radius` and `scale_radius` are positive
1466:   - Test edge case: single particle sample
1467: - [ ] **Test and validate**: `pytest tests/unit/test_mock_galaxy.py -v`
1468: - [ ] Commit
1469: 
1470: ### Step 12.4 — Merger Tree & Equivalence Builders
1471: 
1472: **`src/roadrunner/mock/tree.py`:**
1473: 
1474: | Function | Input | Output | Description |
1475: |---|---|---|---|
1476: | `build_merger_tree(galaxies, snapshot_id, host_id_map=None)` | `list[MockGalaxy]`, snapshot ID, optional host→satellite mapping | `pd.DataFrame` | Builds a DataFrame matching `MergerTreeReaderCSV` format: `Snapshot, Sub_tree_id, mass, virial_radius, scale_radius, position_{x,y,z}, velocity_{x,y,z}, Redshift`. If `host_id_map` provided, adds `host_id` and `distance_to_acc_id` columns. |
1477: | `build_equivalence_table(snapshot_ids, redshifts, times, base_dir="")` | snapshot IDs, redshifts, times | `pd.DataFrame` | Returns DataFrame with columns `snapshot, snapname, time, redshift`. Compatible with `EquivalenceTable` constructor. |
1478: 
1479: - [ ] Write production code in `src/roadrunner/mock/tree.py`
1480: - [ ] Write tests in `tests/unit/test_mock_tree.py`:
1481:   - Test `build_merger_tree` output columns match `MergerTreeReaderCSV` expectations
1482:   - Test `build_equivalence_table` output is accepted by `EquivalenceTable`
1483:   - Test with host_id_map: distance column present
1484:   - Test empty galaxy list
1485: - [ ] **Test and validate**: `pytest tests/unit/test_mock_tree.py -v`
1486: - [ ] Commit
1487: 
1488: ### Step 12.5 — Snapshot & Ground-Truth Assignment
1489: 
1490: **`src/roadrunner/mock/snapshot.py`:**
1491: 
1492: | Function | Input | Output | Description |
1493: |---|---|---|---|
1494: | `build_mock_snapshot(galaxies, snapshot_id, redshift=0.0, time=13.8, seed=42)` | `list[MockGalaxy]`, snapshot metadata | `SnapshotData` | Samples particles from each galaxy, merges into single arrays with unique `indices`. Returns `SnapshotData`. |
1495: | `build_ground_truth_assignment(galaxies)` | `list[MockGalaxy]` | `AssignmentResult` | Creates `particle_df` mapping `array_index → Sub_tree_id` (ground truth: each particle belongs to its parent galaxy). `resp_map` has 1.0 responsibility to parent. `fitted_parameters` is empty dict (no GMM fitting on mock data). |
1496: 
1497: - [ ] Write production code in `src/roadrunner/mock/snapshot.py`
1498: - [ ] Write tests in `tests/unit/test_mock_snapshot.py`:
1499:   - Test `build_mock_snapshot` returns valid `SnapshotData`
1500:   - Test total particle count equals sum of per-galaxy samples
1501:   - Test `build_ground_truth_assignment` has correct Sub_tree_id for every particle
1502:   - Test no particles are assigned to -1
1503: - [ ] **Test and validate**: `pytest tests/unit/test_mock_snapshot.py -v`
1504: - [ ] Commit
1505: 
1506: ### Step 12.6 — High-Level Orchestrator
1507: 
1508: **`src/roadrunner/mock/__init__.py`:**
1509: 
1510: #### `generate_mock_run(galaxy_config, layout_config, snapshot_config, seed=42) -> tuple`
1511: 
1512: **Purpose**: Top-level entry point for creating a complete mock dataset. Wires together all sub-steps.
1513: 
1514: ```python
1515: def generate_mock_run(
1516:     galaxy_config: dict,
1517:     layout_config: dict,
1518:     snapshot_config: dict,
1519:     seed: int = 42,
1520: ) -> tuple[pd.DataFrame, pd.DataFrame, SnapshotData, AssignmentResult]:
1521: ```
1522: 
1523: **`galaxy_config`**: `{"total_stellar_mass": 1e11, "n_galaxies": 20, "alpha": -1.3, "M_star": 10**10.75}`
1524: 
1525: **`layout_config`**: `{"n_groups": 3, "box_size": 1000.0, "group_spread": 50.0}`
1526: 
1527: **`snapshot_config`**: `{"snapshot_id": 100, "redshift": 0.1, "time": 13.0, "particles_per_galaxy": 1000}`
1528: 
1529: **Returns**: `(merger_tree_df, equivalence_df, snapshot_data, ground_truth_assignment)`
1530: 
1531: - [ ] Write `src/roadrunner/mock/__init__.py` with `generate_mock_run`
1532: - [ ] Write integration test:
1533:   - Call `generate_mock_run` with realistic parameters
1534:   - Verify merger tree, equivalence table, SnapshotData are self-consistent
1535:   - Feed ground-truth `AssignmentResult` into `compute_galaxy_properties` and verify sensible output
1536:   - Feed into `compute_riley_criterion` and verify dynstate values
1537: - [ ] **Test and validate**: `pytest tests/integration/test_mock_generator.py -v`
1538: - [ ] Commit
1539: 
1540: **Phase 12 completion**: `git tag phase-12-mock-data`
1541: 
1542: ---
1543: 
1544: ### Target repository structure (updated)
1545: 
1546: ```
1547: src/roadrunner/
1548: ├── mock/                          ← NEW
1549: │   ├── __init__.py                ← generate_mock_run()
1550: │   ├── population.py              ← sample_schechter_masses, assign_halo_masses
1551: │   ├── layout.py                  ← place_galaxies, assign_virial_motion
1552: │   ├── galaxy.py                  ← MockGalaxy (limepy wrapper)
1553: │   ├── tree.py                    ← build_merger_tree, build_equivalence_table
1554: │   └── snapshot.py                ← build_mock_snapshot, build_ground_truth_assignment
1555: │   ...
1556: ```
1557: 
1558: ### Git workflow summary (updated)
1559: 
1560: ```
1561: Phase 11 ── scripts, CI, final __init__ + commit
1562:                │  tag: v1.0.0, phase-11-final
1563: Phase 12 ── mock data generation module
1564:                │  tag: phase-12-mock-data
1565: ```
1566: 
1567: ---
1568: 
1569: ## EXTRA STEPS: Nice-to-Have Improvements
1570: 
1571: **Goal**: Non-critical enhancements collected per step. Added on demand.
1572: 
1573: ### Step 1.3 — Type Defs: `__post_init__` validation
1574: 
1575: - [ ] Add `__post_init__` to `SnapshotData`, `BoundnessResult`, `AssignmentResult` for runtime type/shape validation
1576:   - Validate array dtypes (e.g., `indices` is integer, `masses` is float)
1577:   - Validate array shapes match expectations (e.g., `positions` is (N, 3))
1578:   - Raise `TypeError` with clear message on mismatch
1579: 
1580: ### Step 5.0 — ProtoGalaxyEnsemble (container for list[HaloModel])
1581: 
1582: - [ ] Create `src/roadrunner/physics/proto_galaxy_ensemble.py`:
1583:   - Class `ProtoGalaxyEnsemble`:
1584:     - Constructor: `list[HaloModel]`
1585:     - `__getitem__`, `__len__`, `__iter__`
1586:     - Properties: `.positions`, `.velocities`, `.virial_radii`, `.sub_tree_ids` — bulk numpy arrays
1587:     - `.bound_particles_for_group(group_indices)` — consolidated extraction for assignment/timescales
1588:     - `.to_particle_dataframe(particle_sim_ids)`, `.to_csc_responsibility_matrix()`
1589: - [ ] Write `tests/unit/test_proto_galaxy_ensemble.py`
1590: - [ ] When implemented: update all Phase 5 code to accept `ProtoGalaxyEnsemble` instead of `list[HaloModel]`
1591: 
1592: ### Step 5.4 — Fast Assignment Strategy
1593: 
1594: - [ ] Create `src/roadrunner/clustering/assignment/fast.py`:
1595:   - Class `FastAssigner(ParticleAssigner)`:
1596:     - Constructor: binding energy threshold parameters
1597:     - `.assign(...)` — implements fast non-GMM logic from `_fast/fast.py`
1598:     - Internal: processes each halo independently, assigns by highest intensive binding energy
1599:     - Special newborn handling: within 2*r_half AND E_int < median(E_int)
1600: - [ ] Write `tests/unit/test_assignment_fast.py`:
1601:   - Test basic assignment: particles assigned to correct nearest halo
1602:   - Test newborn special handling: particles within 2*r_half with low energy
1603:   - Test that each particle gets at most one assignment
1604:   - Test edge case: single halo, all particles
1605:   - Test edge case: halos with no bound particles
1606:   - Compare results between `FastAssigner` and `GMMAssigner` on simple data (expect similar but not identical)
1607: - [ ] **Test and validate**: `pytest tests/unit/test_assignment_fast.py -v`
1608: - [ ] Commit
1609: 
1610: ### Global — Docstrings for all public functions, classes, and methods
1611: 
1612: - [ ] Add Google-style docstrings to every public function, class, and method across the entire `src/roadrunner/` tree
1613:   - **Functions**: `Summary`, `Args`, `Returns`, `Raises` (if applicable)
1614:   - **Classes**: `Summary` at class level; `Args` on `__init__` or constructor
1615:   - **Methods**: `Summary` + `Args`/`Returns` same as functions
1616:   - **Private methods** (`_*`): short inline comment or one-line docstring only
1617:   - **Static methods**: same as functions
1618:   - **Properties**: `Summary` line describing the property
1619: - [ ] Verify no function/method is left without at least a one-line docstring using `ruff check --select D`
1620: 
1621: ---
1622: 
1623: ## Git Workflow Summary
1624: 
1625: ```
1626: Phase  0 ── init + .gitignore + pyproject.toml + copy existing code
1627:               │
1628: Phase  1 ── foundation modules + tests + commit per step
1629:               │  tag: phase-1-foundation
1630: Phase  2 ── readers + tests per reader + commit per step
1631:               │  tag: phase-2-readers
1632: Phase  3 ── clustering infra + tests + commit per step
1633:               │  tag: phase-3-clustering-infra
1634: Phase  4 ── mixture cleanup + tests + commit per step
1635:               │  tag: phase-4-gmm-cleanup
1636: Phase  5 ── assignment engine + tests + commit per step
1637:               │  tag: phase-5-assignment
1638: Phase  6 ── galaxy analysis + tests + commit per step
1639:               │  tag: phase-6-galaxy-analysis
1640: Phase  7 ── tracking + tests + commit per step
1641:               │  tag: phase-7-tracking
1642: Phase  8 ── I/O layer + tests + commit per step
1643:               │  tag: phase-8-io
1644: Phase  9 ── pipeline + integration tests + commit
1645:               │  tag: phase-9-pipeline
1646: Phase 10 ── integration cleanup + full test suite + commit
1647:               │  tag: phase-10-integration
1648: Phase 11 ── scripts, CI, final __init__ + commit
1649:               │  tag: v1.0.0, phase-11-final
1650: Phase 12 ── mock data generation (astro-limepy)
1651:                │  tag: phase-12-mock-data
1652: ```
1653: 
1654: Each `git commit` is only done after `conda run -n roadrunner pytest path/to/tests -v` passes for that step.
1655: 
1656: ---
1657: 
1658: ## Validation Checklist Per Big Step
1659: 
1660: ```
1661: [ ] 1. Write production code in src/roadrunner/...
1662: [ ] 2. Write tests in tests/unit/test_<module>.py
1663: [ ] 3. conda run -n roadrunner pytest tests/path/to/test_file.py -v
1664:      └── if FAIL: fix code, return to step 3
1665: [ ] 4. conda run -n roadrunner ruff check src/roadrunner/
1666: [ ] 5. git add .
1667: [ ] 6. git commit -m "Step X.Y: description"
1668: ```
1669: 
1670: ---
1671: 
1672: ## Risk Mitigation
1673: 
1674: | Risk | Mitigation |
1675: |---|---|
1676: | **conda environment not available** | Step 0.1 creates it; `conda activate roadrunner` is the single command for all testing |
1677: | **Tests fail due to environment differences** | All tests run inside `conda run -n roadrunner pytest`, guaranteeing isolation |
1678: | **Large test suite slows development** | Step-level tests are focused (one file per module); run only relevant tests during development, full suite at phase end |
1679: | **Git history becomes messy** | Each step is a focused commit; phase tags allow easy rollback to any phase boundary |
1680: | **YAML dependency for config** | Use stdlib `configparser` as fallback; YAML is optional via `PyYAML` |
1681: | **Old `accme` name in import strings** | Single find-and-replace pass at Phase 0.4; all references updated before any new code is written |
1682: | **Numba compilation fails in new env** | Free to install any conda package; add `numba` to conda install in CI |
1683: | **Breaking old scripts during rename** | Phase 11 is last; old scripts work throughout by using old import paths until then |
1684: | **HaloSegmentator → HaloSegmenter breaks callers** | Add `HaloSegmentator` as deprecated alias that delegates to `HaloSegmenter` |

(End of file - total 1684 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner</path>
<type>directory</type>
<entries>
__init__.py
__pycache__/
_accme/
_exceptions.py
_mcf_types.py
_version.py
clustering/
helpers.py
io/
mixture/
physics/
pipeline/
postprocessing/
readers/

(14 entries)
</entries>
```

---

## Assistant (Plan · GLM-5.1 · 5.6s)

_Thinking:_

Now let me read the key files that the user is concerned about: SnapshotProcessor, AccretionPipeline, and related pipeline code. I also want to understand the trackers and the index/sim-ID confusion.

Now let me read the critical pipeline files and the tracking code to understand the issues:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/pipeline"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/pipeline</path>
<type>directory</type>
<entries>
__init__.py
__pycache__/
accretion_pipeline.py
config.py
entry.py
snapshot_processor.py

(6 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/postprocessing/tracking"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/postprocessing/tracking</path>
<type>directory</type>
<entries>
__init__.py
__pycache__/
assembly.py
birth.py

(4 entries)
</entries>
```

---

## Assistant (Plan · GLM-5.1 · 5.1s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/pipeline/snapshot_processor.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/pipeline/snapshot_processor.py</path>
<type>file</type>
<content>
1: import numpy as np
2: import pandas as pd
3: 
4: from roadrunner._mcf_types import ParticleAssigner, SnapshotData
5: from roadrunner.physics.boundness import compute_halo_bound_particles
6: from roadrunner.physics.halo_model import HaloModel
7: from roadrunner.physics.halo_ensemble import HaloEnsemble
8: from roadrunner.physics.timescales import compute_particle_dynamical_timescales
9: from roadrunner.physics.constants import DYN_TIME_FACTOR
10: from roadrunner.clustering.segmentation import HaloSegmenter
11: from roadrunner.postprocessing.properties import compute_galaxy_properties
12: from roadrunner.postprocessing.mixing import compute_riley_criterion
13: 
14: 
15: class SnapshotProcessor:
16:     def __init__(
17:         self,
18:         assigner: ParticleAssigner,
19:         halo_model: str,
20:         accretion_id: int,
21:         n_los: int = 11,
22:         search_factor: float = 2.0,
23:         min_particles: int = 10,
24:     ):
25:         self.assigner = assigner
26:         self.halo_model = halo_model
27:         self.accretion_id = accretion_id
28:         self.n_los = n_los
29:         self.search_factor = search_factor
30:         self.min_particles = min_particles
31: 
32:     def process(
33:         self,
34:         snap_df: pd.DataFrame,
35:         snapshot_data: SnapshotData,
36:         newborn_indices: np.ndarray,
37:         previous_resp=None,
38:     ):
39:         particle_coords = np.column_stack([
40:             snapshot_data.positions, snapshot_data.velocities,
41:         ])
42: 
43:         ensemble = HaloEnsemble([
44:             HaloModel.from_snapshot_row(row, model=self.halo_model, comoving=False)
45:             for _, row in snap_df.iterrows()
46:         ])
47:         compute_halo_bound_particles(
48:             ensemble, particle_coords, search_factor=self.search_factor,
49:         )
50: 
51:         csc_b, _ = ensemble.get_particles()
52:         pop_idx = ensemble.populated_indices()
53:         candidates = [csc_b.column_indices[i] for i in pop_idx]
54: 
55:         seg = HaloSegmenter(
56:             ensemble.positions[pop_idx],
57:             ensemble.virial_radii[pop_idx],
58:         )
59:         seg.overlap_groups().prune(candidates, self.min_particles, discard=False)
60: 
61:         groups = sorted(
62:             [pop_idx[g] for g in seg.pruned_groups],
63:             key=len, reverse=True,
64:         ) if seg.pruned_groups else []
65: 
66:         result = self.assigner.assign(
67:             ensemble, particle_coords, newborn_indices, groups,
68:             previous_resp=previous_resp,
69:         )
70: 
71:         result.particle_df = compute_particle_dynamical_timescales(
72:             result.particle_df, ensemble, groups,
73:             td_factor=DYN_TIME_FACTOR,
74:         )
75: 
76:         return ensemble, result

(End of file - total 76 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/pipeline/accretion_pipeline.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/pipeline/accretion_pipeline.py</path>
<type>file</type>
<content>
1: import os
2: import time
3: import traceback
4: 
5: import numpy as np
6: import pandas as pd
7: 
8: from roadrunner._exceptions import RestartError
9: from roadrunner.clustering.sparse import SparseCSC
10: from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
11: from roadrunner.io.hdf5_particles import HDF5ParticleWriter
12: from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
13: from roadrunner.io.serialization import save_checkpoint, load_checkpoint
14: from roadrunner.io.logging import RunLogger, format_runtime
15: from roadrunner.readers.snapshot import SnapshotReader
16: from roadrunner.readers.equivalence import EquivalenceTable
17: from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
18: from roadrunner.postprocessing.properties import compute_galaxy_properties
19: from roadrunner.postprocessing.mixing import compute_riley_criterion
20: from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
21: 
22: 
23: class AccretionPipeline:
24:     def __init__(
25:         self,
26:         merger_handler: MergerTreeHandlerCSV,
27:         snapshot_reader: SnapshotReader,
28:         equiv_table: EquivalenceTable,
29:         snapshot_processor: SnapshotProcessor,
30:         cat_writer: HDF5CatalogueWriter,
31:         part_writer: HDF5ParticleWriter | None,
32:         assign_writer: HDF5AssignmentWriter | None,
33:         logger: RunLogger,
34:         birth_tracker=None,
35:         assembly_tracker=None,
36:     ):
37:         self.merger_handler = merger_handler
38:         self.snapshot_reader = snapshot_reader
39:         self.equiv_table = equiv_table
40:         self.processor = snapshot_processor
41:         self.cat_writer = cat_writer
42:         self.part_writer = part_writer
43:         self.assign_writer = assign_writer
44:         self.logger = logger
45:         self.birth_tracker = birth_tracker
46:         self.assembly_tracker = assembly_tracker
47: 
48:     # ── ID translation helpers ────────────────────────────────────
49: 
50:     @staticmethod
51:     def _to_sim_space(csc, snap_data):
52:         if csc is None or len(csc) == 0:
53:             return None
54:         src, dst = snap_data.index_to_id_map()
55:         return csc.remap_rows(src, dst)
56: 
57:     @staticmethod
58:     def _from_sim_space(csc, snap_data):
59:         if csc is None or len(csc) == 0:
60:             return None
61:         src, dst = snap_data.id_to_index_map()
62:         return csc.remap_rows(src, dst)
63: 
64:     # ── Logging helper ────────────────────────────────────────────
65: 
66:     def _log_snapshot(self, snap_id, elapsed, result, ensemble, **extra):
67:         stats = {
68:             "snap": snap_id,
69:             "runtime": format_runtime(elapsed),
70:             "z": 0.0,
71:             "load": 0.0,
72:             "process": elapsed,
73:             "reduction": 0.0,
74:             "bound": ensemble.nstars,
75:             "groups": result.statistics.get("groups", 0),
76:         }
77:         stats.update(result.statistics)
78:         stats.update(extra)
79:         self.logger.write_snapshot(stats)
80: 
81:     # ── Tracker helpers ──────────────────────────────────────────
82: 
83:     def _update_trackers(self, snap_id, snap_data, result, satellites,
84:                          birth_tracker, assembly_tracker):
85:         if birth_tracker is not None:
86:             df = result.particle_df
87:             arr_idx = df["array_index"].values
88:             sim_ids = snap_data.indices[arr_idx]
89:             birth_tracker.update(
90:                 t_snap=snap_data.time, snapshot_id=snap_id,
91:                 particle_ids=sim_ids,
92:                 host_ids=df["Sub_tree_id"].values,
93:                 timescales=df.get("timescale", np.full(len(df), 0.1)).values,
94:             )
95:         if assembly_tracker is not None:
96:             assignment_map = {}
97:             for sid, group in result.particle_df.groupby("Sub_tree_id"):
98:                 arr_idx = group["array_index"].values
99:                 assignment_map[int(sid)] = set(
100:                     snap_data.indices[arr_idx].tolist(),
101:                 )
102:             birth_map = (
103:                 birth_tracker.current_birth_map()
104:                 if birth_tracker else {}
105:             )
106:             assembly_tracker.update(
107:                 snap_id, assignment_map, birth_map, satellites,
108:             )
109: 
110:     def _track_and_reduce(self, snap_df, snap_data, ensemble, result,
111:                           assembly_tracker):
112:         bound_csc, _ = ensemble.get_particles()
113:         coords = np.column_stack([snap_data.positions, snap_data.velocities])
114: 
115:         assembly_map = assembly_tracker.current()
116:         galaxy_particles = {}
117:         galaxy_bound = {}
118:         sid_to_col = {sid: i for i, sid in enumerate(bound_csc.column_id)}
119:         for gid, sim_set in assembly_map.items():
120:             col = sid_to_col.get(gid)
121:             if col is None:
122:                 continue
123:             bound_idx = bound_csc.column_indices[col]
124:             sim_arr = np.array(list(sim_set), dtype=np.uint64)
125:             allowed_idx = snap_data.array_index(sim_arr)
126:             allowed_idx = allowed_idx[allowed_idx >= 0]
127:             intersection = np.intersect1d(
128:                 allowed_idx.astype(np.int64), bound_idx.astype(np.int64),
129:             )
130:             if len(intersection) > 0:
131:                 galaxy_particles[int(gid)] = intersection
132:                 galaxy_bound[int(gid)] = bound_idx.astype(np.int64)
133: 
134:         galaxy_table = snap_df[["Sub_tree_id", "host_id", "mass",
135:                                 "distance_to_acc_id"]].copy()
136:         galaxy_table.set_index("Sub_tree_id", inplace=True)
137:         host_row = snap_df[
138:             snap_df["Sub_tree_id"] == self.processor.accretion_id
139:         ]
140:         host_props = host_row.iloc[0] if not host_row.empty else snap_df.iloc[0]
141: 
142:         galaxy_centers = {}
143:         for sid, params in result.fitted_parameters.items():
144:             mean = params.get("mean")
145:             if mean is not None:
146:                 galaxy_centers[int(sid)] = np.asarray(mean)
147: 
148:         properties = compute_galaxy_properties(
149:             accretion_id=self.processor.accretion_id,
150:             particle_masses=snap_data.masses,
151:             particle_coords=coords,
152:             galaxy_particles=galaxy_particles,
153:             galaxy_table=galaxy_table,
154:             host_props=host_props,
155:             halo_model=self.processor.halo_model,
156:             n_los=self.processor.n_los,
157:             galaxy_centers=galaxy_centers if galaxy_centers else None,
158:         )
159: 
160:         dynstate = compute_riley_criterion(
161:             main_id=self.processor.accretion_id,
162:             particle_masses=snap_data.masses,
163:             particle_coords=coords,
164:             galaxy_allowed=galaxy_particles,
165:             galaxy_bound=galaxy_bound,
166:             redshift=snap_data.redshift,
167:         )
168: 
169:         return properties, dynstate
170: 
171:     # ── Run ───────────────────────────────────────────────────────
172: 
173:     def run(
174:         self,
175:         output_dir: str,
176:         *,
177:         start_snapshot: int | None = None,
178:         end_snapshot: int | None = None,
179:         resume: bool = False,
180:     ) -> None:
181:         snapshot_ids = self.merger_handler.snapshots
182:         if start_snapshot is not None:
183:             snapshot_ids = [s for s in snapshot_ids if s >= start_snapshot]
184:         if end_snapshot is not None:
185:             snapshot_ids = [s for s in snapshot_ids if s <= end_snapshot]
186: 
187:         if not snapshot_ids:
188:             print("No snapshots to process.")
189:             return
190: 
191:         checkpoint_path = os.path.join(output_dir, "checkpoint.zst")
192:         log_path = self.logger._log_path
193: 
194:         previous_resp_sim = None
195:         start_idx = 0
196: 
197:         # ── Resume ────────────────────────────────────────────────
198:         if resume:
199:             try:
200:                 ckpt = load_checkpoint(checkpoint_path)
201:                 failed_snap = ckpt["last_snapshot"]
202:                 start_idx = snapshot_ids.index(failed_snap)
203:                 previous_resp_sim = ckpt.get("previous_resp")
204:                 print(f"Resumed. Re-processing snapshot {failed_snap}")
205:             except (RestartError, Exception) as e:
206:                 print(f"Resume failed, starting fresh: {e}")
207:                 resume = False
208: 
209:         # ── Pre-loop (once, skipped on resume) ────────────────────
210:         if not resume:
211:             self.merger_handler.compute_scale_radii()
212:             self.merger_handler.compute_most_bound_satellite()
213:             self.merger_handler.compute_distance_to_host()
214: 
215:             self.cat_writer.write_header(
216:                 accretion_id=self.processor.accretion_id,
217:                 snapshots=snapshot_ids,
218:                 config_dict={
219:                     "halo_model": self.processor.halo_model,
220:                     "n_los": self.processor.n_los,
221:                     "search_factor": self.processor.search_factor,
222:                 },
223:                 merger_tree_df=self.merger_handler.dataframe,
224:                 equivalence_df=self.equiv_table._df,
225:             )
226: 
227:         t_start = time.time()
228: 
229:         # ── Snapshot loop ─────────────────────────────────────────
230:         for idx in range(start_idx, len(snapshot_ids)):
231:             snap_id = snapshot_ids[idx]
232:             print(f"\nSnapshot {snap_id}  ({idx + 1}/{len(snapshot_ids)})")
233: 
234:             # Checkpoint BEFORE processing (in sim ID space)
235:             ckpt_data = {
236:                 "last_snapshot": snap_id,
237:                 "previous_resp": previous_resp_sim,
238:             }
239:             save_checkpoint(checkpoint_path, ckpt_data)
240: 
241:             try:
242:                 snap_df = self.merger_handler.select_snapshots(snap_id)
243:                 if snap_df.empty:
244:                     raise RuntimeError(
245:                         f"Empty merger tree for snapshot {snap_id}"
246:                     )
247: 
248:                 file_path = self.equiv_table.snapshot_path(snap_id)
249:                 snap_data = self.snapshot_reader.load(file_path)
250: 
251:                 coords = np.column_stack([
252:                     snap_data.positions, snap_data.velocities,
253:                 ])
254:                 masses = snap_data.masses
255:                 N = coords.shape[0]
256: 
257:                 # Translate previous_resp: sim ID → array_index
258:                 previous_resp = self._from_sim_space(
259:                     previous_resp_sim, snap_data,
260:                 )
261: 
262:                 # Newborn detection
263:                 if previous_resp is None:
264:                     newborn = np.arange(N, dtype=np.uint64)
265:                 else:
266:                     existing = set(previous_resp.row_id)
267:                     newborn = np.array(
268:                         [i for i in range(N) if i not in existing],
269:                         dtype=np.uint64,
270:                     )
271: 
272:                 # Satellites for this snapshot
273:                 satellites = self.merger_handler.compute_satellites(
274:                     snap_df,
275:                 )
276: 
277:                 # Process (boundness → segment → assign → timescales)
278:                 ensemble, result = self.processor.process(
279:                     snap_df, snap_data, newborn,
280:                     previous_resp=previous_resp,
281:                 )
282: 
283:                 # Trackers (persistent sim ID space)
284:                 self._update_trackers(
285:                     snap_id, snap_data, result, satellites,
286:                     self.birth_tracker, self.assembly_tracker,
287:                 )
288: 
289:                 # Track + reduce (sim ID → array index for bound ∩ allowed)
290:                 properties, dynstate = self._track_and_reduce(
291:                     snap_df, snap_data, ensemble, result,
292:                     self.assembly_tracker,
293:                 ) if self.assembly_tracker is not None else (
294:                     pd.DataFrame(), pd.DataFrame(),
295:                 )
296: 
297:                 # Convert for next snapshot: array_index → sim ID
298:                 previous_resp_sim = self._to_sim_space(
299:                     result.responsibilities, snap_data,
300:                 )
301: 
302:                 # I/O
303:                 if self.cat_writer:
304:                     self.cat_writer.write_snapshot(
305:                         snap_id, snap_data.time,
306:                         properties, dynstate, satellites,
307:                     )
308:                 if self.part_writer:
309:                     self.part_writer.write_snapshot(
310:                         snap_id, snap_data.time, snap_data.redshift, snap_data,
311:                     )
312:                 if self.assign_writer:
313:                     bound_csc, _ = ensemble.get_particles()
314:                     self.assign_writer.write_snapshot(
315:                         snap_id, snap_data.time, result, bound_csc,
316:                     )
317: 
318:                 # Log
319:                 elapsed = time.time() - t_start
320:                 self._log_snapshot(snap_id, elapsed, result, ensemble)
321: 
322:             except Exception as exc:
323:                 with open(log_path, "a") as f:
324:                     f.write(f"\n--- ERROR on snapshot {snap_id} ---\n")
325:                     f.write(f"{type(exc).__name__}: {exc}\n")
326:                     traceback.print_exc(file=f)
327:                     f.write("---\n\n")
328:                 print(f"FATAL: {type(exc).__name__} on snapshot "
329:                       f"{snap_id}: {exc}")
330:                 print(f"Checkpoint saved. See {log_path} for traceback.")
331:                 raise
332: 
333:         # ── Post-loop: finalize ───────────────────────────────────
334:         print("\nFinalizing...")
335:         bt = self.birth_tracker
336:         at = self.assembly_tracker
337: 
338:         if bt is not None:
339:             birth_df = bt.finalize()
340:             timescales = np.array([
341:                 bt._finalized.get(i, {}).get("timescale", -1)
342:                 for i in range(len(bt._finalized))
343:             ]) if bt._finalized else np.array([], dtype=np.float32)
344:         else:
345:             birth_df = pd.DataFrame()
346:             timescales = np.array([], dtype=np.float32)
347: 
348:         assembly_dict = at.current() if at is not None else {}
349:         assembly_records = [
350:             (pid, g) for g, pset in assembly_dict.items() for pid in pset
351:         ]
352:         assembly_df = pd.DataFrame(
353:             assembly_records, columns=["particle_index", "galaxy_id"],
354:         ) if assembly_records else pd.DataFrame()
355: 
356:         self.cat_writer.write_finalize(birth_df, assembly_df)
357: 
358:         if self.assign_writer is not None and len(timescales) > 0:
359:             self.assign_writer.write_timescales(timescales)
360: 
361:         self.logger.write_summary()
362:         print(f"Pipeline complete in {format_runtime(time.time() - t_start)}")

(End of file - total 362 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/pipeline/config.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/pipeline/config.py</path>
<type>file</type>
<content>
1: from dataclasses import dataclass, field
2: 
3: 
4: @dataclass
5: class RunConfig:
6:     merger_tree_path: str = ""
7:     particle_data_dir: str = ""
8:     equivalence_path: str = ""
9:     code: str = "RAMSES"
10:     ptype: str = "star"
11:     fields: dict = field(default_factory=lambda: {
12:         "index": "particle_index",
13:         "mass": "particle_mass",
14:         "position": "coordinates",
15:         "velocity": "particle_velocity",
16:     })
17:     unit_base: dict | None = None
18: 
19:     accretion_id: int | None = None
20:     start_snapshot: int | None = None
21:     end_snapshot: int | None = None
22: 
23:     halo_model: str = "kepler"
24:     cov_type: str = "full"
25:     max_iter: int = 10
26:     tol: float = 1e-2
27:     reg_covar: float = 1e-6
28:     n_los: int = 15
29:     search_factor: float = 1.0
30:     min_particles: int = 10
31: 
32:     birth_window_factor: float = 5.0
33:     enforce_initial_hosts: bool = True
34: 
35:     output_dir: str = "./output"
36:     save_particles: bool = True
37:     save_assignment: bool = True
38:     float_atol: float = 1e-4

(End of file - total 38 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/pipeline/entry.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/pipeline/entry.py</path>
<type>file</type>
<content>
1: import os
2: 
3: from roadrunner.readers.merger_tree import MergerTreeReaderCSV
4: from roadrunner.readers.snapshot import SnapshotReader
5: from roadrunner.readers.equivalence import EquivalenceTable
6: from roadrunner.physics.merger_tree import MergerTreeHandlerCSV
7: from roadrunner.clustering.assignment.gmm import GMMAssigner
8: from roadrunner.pipeline.snapshot_processor import SnapshotProcessor
9: from roadrunner.pipeline.accretion_pipeline import AccretionPipeline
10: from roadrunner.pipeline.config import RunConfig
11: from roadrunner.io.hdf5_catalogue import HDF5CatalogueWriter
12: from roadrunner.io.hdf5_particles import HDF5ParticleWriter
13: from roadrunner.io.hdf5_assignment import HDF5AssignmentWriter
14: from roadrunner.io.logging import RunLogger
15: from roadrunner.postprocessing.tracking.birth import BirthTracker
16: from roadrunner.postprocessing.tracking.assembly import AssemblyTracker
17: 
18: 
19: def run_accretion_history(config: RunConfig | dict) -> None:
20:     if isinstance(config, dict):
21:         config = RunConfig(**config)
22: 
23:     # 1. Readers
24:     reader = MergerTreeReaderCSV(config.merger_tree_path)
25:     merger_handler = MergerTreeHandlerCSV(reader.dataframe)
26: 
27:     snapshot_reader = SnapshotReader(
28:         config.code, config.ptype, config.fields, config.unit_base,
29:     )
30:     equiv_table = EquivalenceTable(
31:         config.equivalence_path, base_dir=config.particle_data_dir,
32:     )
33: 
34:     # 2. Auto-detect accretion_id if not provided
35:     accretion_id = config.accretion_id
36:     if accretion_id is None:
37:         last_snap = merger_handler.snapshots[-1]
38:         host_df = merger_handler.select_accretion_host(
39:             last_snap, criterion="max_mass",
40:         )
41:         accretion_id = int(host_df["Sub_tree_id"].iloc[0])
42: 
43:     # 3. Assigner
44:     assigner = GMMAssigner(
45:         cov_type=config.cov_type,
46:         max_iter=config.max_iter,
47:         tol=config.tol,
48:         min_particles=config.min_particles,
49:         reg_covar=config.reg_covar,
50:         prior_type="",
51:         verbose=1,
52:     )
53: 
54:     # 4. Processor
55:     processor = SnapshotProcessor(
56:         assigner=assigner,
57:         halo_model=config.halo_model,
58:         accretion_id=accretion_id,
59:         n_los=config.n_los,
60:         search_factor=config.search_factor,
61:         min_particles=config.min_particles,
62:     )
63: 
64:     # 5. Trackers (optional)
65:     birth_tracker = BirthTracker(
66:         factor=config.birth_window_factor,
67:         enforce_initial_hosts=config.enforce_initial_hosts,
68:     ) if config.birth_window_factor > 0 else None
69:     assembly_tracker = (
70:         AssemblyTracker() if birth_tracker is not None else None
71:     )
72: 
73:     # 6. Writers
74:     cat_w = HDF5CatalogueWriter(config.output_dir)
75:     part_w = (
76:         HDF5ParticleWriter(config.output_dir, float_atol=config.float_atol)
77:         if config.save_particles else None
78:     )
79:     assign_w = (
80:         HDF5AssignmentWriter(config.output_dir, float_atol=config.float_atol)
81:         if config.save_assignment else None
82:     )
83: 
84:     # 7. Logger
85:     logger = RunLogger(os.path.join(config.output_dir, "run.log"))
86:     logger.write_header({
87:         "output_dir": config.output_dir,
88:         "halo_model": config.halo_model,
89:         "cov_type": config.cov_type,
90:     })
91: 
92:     # 8. Pipeline
93:     pipeline = AccretionPipeline(
94:         merger_handler=merger_handler,
95:         snapshot_reader=snapshot_reader,
96:         equiv_table=equiv_table,
97:         snapshot_processor=processor,
98:         cat_writer=cat_w,
99:         part_writer=part_w,
100:         assign_writer=assign_w,
101:         logger=logger,
102:         birth_tracker=birth_tracker,
103:         assembly_tracker=assembly_tracker,
104:     )
105: 
106:     # 9. Run
107:     pipeline.run(
108:         config.output_dir,
109:         start_snapshot=config.start_snapshot,
110:         end_snapshot=config.end_snapshot,
111:     )

(End of file - total 111 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/pipeline/__init__.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/pipeline/__init__.py</path>
<type>file</type>
<content>
1: from .snapshot_processor import SnapshotProcessor
2: from .accretion_pipeline import AccretionPipeline
3: from .config import RunConfig
4: from .entry import run_accretion_history
5: 
6: __all__ = ["SnapshotProcessor", "AccretionPipeline",
7:            "RunConfig", "run_accretion_history"]

(End of file - total 7 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 6.4s)

_Thinking:_

Now let me read the tracker files and the HaloEnsemble, and the _mcf_types to understand the index/sim-ID space issues better.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/postprocessing/tracking/birth.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/postprocessing/tracking/birth.py</path>
<type>file</type>
<content>
1: import heapq
2: from collections import Counter, defaultdict
3: 
4: import numpy as np
5: import pandas as pd
6: 
7: 
8: def _exp_window(x):
9:     return np.exp(-x)
10: 
11: 
12: def _cauchy_window(x):
13:     return 1.0 / (1 + x ** 2)
14: 
15: 
16: def _gaussian_window(x):
17:     return np.exp(-0.3 * x ** 2)
18: 
19: 
20: _WINDOWS = {
21:     "gaussian": _gaussian_window,
22:     "cauchy": _cauchy_window,
23:     "exp": _exp_window,
24: }
25: 
26: 
27: class BirthTracker:
28:     def __init__(self, factor=5, enforce_initial_hosts=False, window="gaussian"):
29:         if window not in _WINDOWS:
30:             raise ValueError(f"Unknown window: {window}. Choose from {list(_WINDOWS)}")
31:         self._window_fn = _WINDOWS[window]
32:         self._heap = []
33:         self._active = {}
34:         self._finalized = {}
35:         self._birth_map = defaultdict(set)
36:         self.factor = factor - 0.001
37:         self._last_snapshot = -1
38:         self.enforce_initial_hosts = enforce_initial_hosts
39: 
40:     def _add_update_particles(self, t_snap, snapshot_id, particle_ids, host_ids, timescales, weights):
41:         finalized_keys = self._finalized.keys()
42:         mask_not_finalized = ~np.isin(particle_ids, list(finalized_keys))
43:         if not np.any(mask_not_finalized):
44:             return
45: 
46:         pids = particle_ids[mask_not_finalized]
47:         hosts = host_ids[mask_not_finalized]
48:         taus = timescales[mask_not_finalized]
49:         ws = weights[mask_not_finalized]
50: 
51:         is_active = np.isin(pids, list(self._active.keys()))
52:         if np.any(is_active):
53:             p_active = pids[is_active]
54:             host_active = hosts[is_active]
55:             w_active = ws[is_active]
56: 
57:             t0 = np.array([self._active[p]["t0"] for p in p_active])
58:             tau0 = np.array([self._active[p]["tau"] for p in p_active])
59: 
60:             deltas = (w_active * self._window_fn((t_snap - t0) / tau0)).astype(np.float64)
61:             for p, host, delta in zip(p_active, host_active, deltas):
62:                 info = self._active[p]
63:                 if self.enforce_initial_hosts and host not in info["initial_hosts"]:
64:                     continue
65:                 info["counts"][host] += delta
66:                 if info["counts"][host] > info["leader_score"]:
67:                     info["leader_host"] = np.uint64(host)
68:                     info["leader_score"] = info["counts"][host]
69: 
70:         not_active = ~is_active
71:         if np.any(not_active):
72:             p_new = pids[not_active]
73:             host_new = hosts[not_active]
74:             tau_new = taus[not_active]
75:             w_new = ws[not_active]
76: 
77:             order = np.argsort(p_new)
78:             p_sorted = p_new[order]
79:             host_sorted = host_new[order]
80:             tau_sorted = tau_new[order]
81:             w_sorted = w_new[order]
82: 
83:             unique_particles, start_idx = np.unique(p_sorted, return_index=True)
84:             end_idx = np.append(start_idx[1:], len(p_sorted))
85:             for p, start, end in zip(unique_particles, start_idx, end_idx):
86:                 particle_hosts = host_sorted[start:end]
87:                 particle_weights = w_sorted[start:end]
88:                 particle_tau = tau_sorted[start:end].max()
89: 
90:                 hosts_u, inverse_u = np.unique(particle_hosts, return_inverse=True)
91:                 sums = np.zeros(len(hosts_u), dtype=np.float64)
92:                 np.add.at(sums, inverse_u, particle_weights)
93: 
94:                 counts = Counter(dict(zip(hosts_u, sums)))
95:                 initial_hosts = set(hosts_u)
96:                 leader_host, leader_score = counts.most_common(1)[0]
97: 
98:                 self._active[p] = {
99:                     "t0": np.float32(t_snap),
100:                     "snap0": np.uint32(snapshot_id),
101:                     "tau": np.float32(particle_tau),
102:                     "counts": counts,
103:                     "initial_hosts": initial_hosts,
104:                     "leader_host": np.uint64(leader_host),
105:                     "leader_score": np.float32(leader_score),
106:                 }
107:                 heapq.heappush(self._heap, (np.float32(t_snap + self.factor * particle_tau), np.uint64(p)))
108: 
109:     def _finalize_particles(self, t_snap):
110:         while self._heap and self._heap[0][0] <= t_snap:
111:             _, p_to_finalize = heapq.heappop(self._heap)
112:             if p_to_finalize not in self._active:
113:                 continue
114: 
115:             info = self._active.pop(p_to_finalize)
116:             birth_id = info["leader_host"]
117: 
118:             self._finalized[p_to_finalize] = {
119:                 "particle_index": p_to_finalize,
120:                 "birth_time": info["t0"],
121:                 "birth_snap": info["snap0"],
122:                 "birth_id": birth_id,
123:                 "timescale": info["tau"],
124:             }
125: 
126:             self._birth_map[birth_id].add(p_to_finalize)
127: 
128:     def update(self, t_snap, snapshot_id, particle_ids, host_ids, timescales, weights=None):
129:         if snapshot_id > self._last_snapshot:
130:             if weights is None:
131:                 weights = np.full(particle_ids.shape, 1.0)
132:             self._add_update_particles(t_snap, snapshot_id, particle_ids, host_ids, timescales, weights)
133:             self._finalize_particles(t_snap)
134:             self._last_snapshot = snapshot_id
135: 
136:     def finalize(self):
137:         self._finalize_particles(np.inf)
138:         records = [
139:             {"particle_index": p, "birth_id": self._finalized[p]["birth_id"]}
140:             for particles in self._birth_map.values()
141:             for p in particles
142:         ]
143:         birth_df = pd.DataFrame.from_records(
144:             records, columns=["particle_index", "birth_id"],
145:         )
146:         return birth_df
147: 
148:     def current_birth_map(self):
149:         birth_map = {k: set(v) for k, v in self._birth_map.items()}
150:         for p, info in self._active.items():
151:             birth_map.setdefault(info["leader_host"], set()).add(p)
152:         return birth_map
153: 
154:     def _get_state(self):
155:         return {
156:             "active": dict(self._active),
157:             "finalized": dict(self._finalized),
158:             "heap": list(self._heap),
159:             "birth_map": {k: list(v) for k, v in self._birth_map.items()},
160:             "factor": self.factor,
161:             "last_snapshot": self._last_snapshot,
162:             "enforce_initial_hosts": self.enforce_initial_hosts,
163:         }
164: 
165:     def _set_state(self, state):
166:         self._active = state["active"]
167:         self._finalized = state["finalized"]
168:         self._heap = state["heap"]
169:         heapq.heapify(self._heap)
170:         self._birth_map = defaultdict(set, {k: set(v) for k, v in state["birth_map"].items()})
171:         self.factor = state["factor"]
172:         self._last_snapshot = state["last_snapshot"]
173:         self.enforce_initial_hosts = state["enforce_initial_hosts"]

(End of file - total 173 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/postprocessing/tracking/assembly.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/postprocessing/tracking/assembly.py</path>
<type>file</type>
<content>
1: from collections import defaultdict, deque
2: 
3: from tqdm import tqdm
4: 
5: from roadrunner._exceptions import CycleError
6: 
7: 
8: class AssemblyTracker:
9:     def __init__(self, n_sat_history=2, unbound_default=-1):
10:         self._infall_lists = defaultdict(set)
11:         self._previous_birth_map = defaultdict(set)
12:         self._unbound_default = unbound_default
13:         self._frozen = set([unbound_default])
14:         self._last_snapshot = -1
15:         self._sat_history_buffer = deque(maxlen=n_sat_history - 1)
16: 
17:     def _get_current_sat_history(self, satellites_map):
18:         sat_history = {g: set(sats) for g, sats in satellites_map.items()}
19:         for snap_map in self._sat_history_buffer:
20:             for g in sat_history:
21:                 sat_history[g].update(snap_map.get(g, []))
22:         return sat_history
23: 
24:     def _order_galaxies_bottom_up(self, warm_galaxies, satellites_history):
25:         ordered = []
26:         visited = set()
27:         visiting = set()
28: 
29:         def dfs(g):
30:             if g in visited:
31:                 return
32:             if g in visiting:
33:                 raise CycleError(
34:                     f"Cycle detected in satellite graph at galaxy {g}"
35:                 )
36:             visiting.add(g)
37:             for h in satellites_history.get(g, []):
38:                 if h in warm_galaxies:
39:                     dfs(h)
40:             visiting.remove(g)
41:             visited.add(g)
42:             ordered.append(g)
43: 
44:         for g in warm_galaxies:
45:             if g not in visited:
46:                 dfs(g)
47:         return ordered
48: 
49:     def update(self, snapshot_id, assignment_map, birth_map, satellites_map, freeze_galaxies=None):
50:         if snapshot_id > self._last_snapshot:
51:             self._last_snapshot = snapshot_id
52:             all_galaxies = (
53:                 set(assignment_map.keys())
54:                 | set(satellites_map.keys())
55:                 | set(h for sats in satellites_map.values() for h in sats)
56:             )
57:             ordered_galaxies = self._order_galaxies_bottom_up(all_galaxies, satellites_map)
58: 
59:             self._update(
60:                 snapshot_id,
61:                 ordered_galaxies,
62:                 assignment_map,
63:                 birth_map,
64:                 satellites_map,
65:                 [] if freeze_galaxies is None else freeze_galaxies,
66:             )
67:             self._sat_history_buffer.append(satellites_map)
68: 
69:     def _update(self, snapshot_id, ordered_galaxies, assignment_map, birth_map, satellites_map, freeze_galaxies):
70:         for g in freeze_galaxies:
71:             self._frozen.add(g)
72: 
73:         satellites_history = self._get_current_sat_history(satellites_map)
74:         for g in tqdm(ordered_galaxies, desc="Updating each galaxy..."):
75:             prev_birth = self._previous_birth_map.get(g, set())
76:             curr_birth = birth_map.get(g, set())
77: 
78:             erase = prev_birth - curr_birth
79:             add = curr_birth - prev_birth
80: 
81:             self._infall_lists[g].update(add)
82:             self._infall_lists[g].difference_update(erase)
83: 
84:             if g in self._frozen:
85:                 continue
86: 
87:             A_g = assignment_map.get(g, set())
88:             valid = curr_birth | self._infall_lists.get(self._unbound_default, set())
89:             for h in satellites_history.get(g, []):
90:                 valid |= self._infall_lists.get(h, set())
91: 
92:             self._infall_lists[g].update(A_g & valid)
93: 
94:         self._previous_birth_map = dict(birth_map)
95: 
96:     def current(self):
97:         return self._infall_lists
98: 
99:     def _get_state(self):
100:         return {
101:             "infall_lists": {k: list(v) for k, v in self._infall_lists.items()},
102:             "previous_birth_map": {k: list(v) for k, v in self._previous_birth_map.items()},
103:             "frozen": set(self._frozen),
104:             "sat_history_buffer": list(self._sat_history_buffer),
105:             "last_snapshot": self._last_snapshot,
106:             "unbound_default": self._unbound_default,
107:         }
108: 
109:     def _set_state(self, state):
110:         self._infall_lists = defaultdict(set, {k: set(v) for k, v in state["infall_lists"].items()})
111:         self._previous_birth_map = defaultdict(set, {k: set(v) for k, v in state["previous_birth_map"].items()})
112:         self._frozen = state["frozen"]
113:         self._sat_history_buffer = deque(state["sat_history_buffer"], maxlen=self._sat_history_buffer.maxlen)
114:         self._last_snapshot = state["last_snapshot"]
115:         self._unbound_default = state["unbound_default"]

(End of file - total 115 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/_mcf_types.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/_mcf_types.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
3: 
4: import numpy as np
5: import pandas as pd
6: from numpy.typing import NDArray
7: 
8: if TYPE_CHECKING:
9:     from roadrunner.clustering.sparse import SparseCSC
10: 
11: RespMap = dict[int, tuple[np.ndarray, np.ndarray]]
12: 
13: 
14: class SnapshotData:
15:     indices: NDArray[np.integer]
16:     masses: NDArray[np.floating]
17:     positions: NDArray[np.floating]
18:     velocities: NDArray[np.floating]
19:     metallicity: NDArray[np.floating] | None
20:     redshift: float
21:     time: float
22: 
23:     def __init__(
24:         self,
25:         indices: NDArray[np.integer],
26:         masses: NDArray[np.floating],
27:         positions: NDArray[np.floating],
28:         velocities: NDArray[np.floating],
29:         redshift: float,
30:         time: float,
31:         metallicity: NDArray[np.floating] | None = None,
32:     ) -> None:
33:         self.indices = indices
34:         self.masses = masses
35:         self.positions = positions
36:         self.velocities = velocities
37:         self.metallicity = metallicity
38:         self.redshift = redshift
39:         self.time = time
40:         self._index_sorter: np.ndarray | None = None
41: 
42:     def array_index(self, sim_ids: np.ndarray) -> np.ndarray:
43:         if self._index_sorter is None:
44:             self._index_sorter = np.argsort(self.indices)
45:         sorted_ids = self.indices[self._index_sorter]
46:         pos = np.clip(np.searchsorted(sorted_ids, sim_ids), 0, len(sorted_ids) - 1)
47:         found = sorted_ids[pos] == sim_ids
48:         result = np.full(len(sim_ids), -1, dtype=np.int64)
49:         result[found] = self._index_sorter[pos[found]]
50:         return result
51: 
52:     def index_to_id_map(self) -> tuple[np.ndarray, np.ndarray]:
53:         idx = np.arange(len(self.indices), dtype=np.int64)
54:         return idx, self.indices.astype(np.int64, copy=False)
55: 
56:     def id_to_index_map(self) -> tuple[np.ndarray, np.ndarray]:
57:         return self.indices.astype(np.int64, copy=False), np.arange(len(self.indices), dtype=np.int64)
58: 
59: 
60: class BoundnessResult:
61:     candidate_indices: NDArray[np.integer]
62:     boundness_values: NDArray[np.floating]
63:     dynamical_times: list[NDArray[np.floating]]
64: 
65:     def __init__(
66:         self,
67:         candidate_indices: NDArray[np.integer],
68:         boundness_values: NDArray[np.floating],
69:         dynamical_times: list[NDArray[np.floating]],
70:     ) -> None:
71:         self.candidate_indices = candidate_indices
72:         self.boundness_values = boundness_values
73:         self.dynamical_times = dynamical_times
74: 
75: 
76: class AssignmentResult:
77:     particle_df: pd.DataFrame
78:     responsibilities: object  # SparseCSC at runtime
79:     fitted_parameters: dict
80:     statistics: dict
81: 
82:     def __init__(
83:         self,
84:         particle_df: pd.DataFrame,
85:         responsibilities: object,
86:         fitted_parameters: dict,
87:         statistics: dict,
88:     ) -> None:
89:         self.particle_df = particle_df
90:         self.responsibilities = responsibilities
91:         self.fitted_parameters = fitted_parameters
92:         self.statistics = statistics
93: 
94: 
95: @runtime_checkable
96: class ParticleAssigner(Protocol):
97:     def assign(
98:         self,
99:         halos: list,
100:         particle_coords: np.ndarray,
101:         newborn_indices: np.ndarray,
102:         groups: list[list[int]],
103:         **kwargs,
104:     ) -> AssignmentResult: ...
105: 
106: 
107: @runtime_checkable
108: class PotentialModel(Protocol):
109:     def potential(self, r: np.ndarray) -> np.ndarray: ...
110:     def dynamical_time(self, x: np.ndarray) -> np.ndarray: ...
111:     def tidal_denominator(self, r: np.ndarray) -> np.ndarray: ...
112: 
113: 
114: @runtime_checkable
115: class AssignmentStatistics(Protocol):
116:     @property
117:     def values(self) -> dict[str, Any]: ...

(End of file - total 117 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 5.0s)

_Thinking:_

Now let me also look at the HaloEnsemble, the sparse module, and the snapshot reader to understand the CSC and index/sim-ID space issue more deeply.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/physics"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/physics</path>
<type>directory</type>
<entries>
__init__.py
__pycache__/
birth_assignment.py
boundness.py
constants.py
halo_ensemble.py
halo_model.py
merger_tree.py
potentials.py
scaler.py
timescales.py

(11 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/clustering/sparse.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/clustering/sparse.py</path>
<type>file</type>
<content>
1: import numpy as np
2: from numba import njit, prange
3: 
4: 
5: def build_dense_from_csc(matrix_shape, true_indices, column_indices, column_values,
6:                          fill_value=0.0):
7:     n_samples, n_components = matrix_shape
8: 
9:     flat_candidates = np.concatenate(column_indices)
10:     flat_values = np.concatenate(column_values)
11: 
12:     offsets = np.zeros(n_components + 1, dtype=np.int64)
13:     offsets[1:] = np.cumsum([len(c) for c in column_indices])
14: 
15:     return _csc_to_dense_kernel(
16:         matrix_shape, true_indices,
17:         flat_candidates, flat_values, offsets,
18:         fill_value,
19:     )
20: 
21: 
22: @njit(parallel=True)
23: def _csc_to_dense_kernel(
24:     matrix_shape, true_indices, flat_indices, flat_values, offsets,
25:     fill_value,
26: ):
27:     n_samples, n_components = matrix_shape
28: 
29:     imax = true_indices.max()
30:     idx_map = np.full(int(imax + 1), imax + 2, dtype=np.int64)
31:     idx_map[true_indices] = np.arange(true_indices.size)
32: 
33:     dense = np.full(matrix_shape, fill_value, dtype=flat_values.dtype)
34:     for k in prange(n_components):
35:         start, end = offsets[k], offsets[k + 1]
36: 
37:         indices = flat_indices[start:end]
38:         values = flat_values[start:end]
39: 
40:         dense[idx_map[indices], k] = values
41: 
42:     return np.asarray(dense, dtype=flat_values.dtype)
43: 
44: 
45: @njit(parallel=True)
46: def stitch_zero_rows(matrix1, matrix2):
47:     assert matrix1.shape == matrix2.shape, (
48:         f"Matrices cannot be stitched together: shape {matrix1.shape} "
49:         f"is incompatible with {matrix2.shape}."
50:     )
51:     nrows, ncols = matrix1.shape
52: 
53:     for i in prange(nrows):
54:         if matrix2[i, :].sum() > 0:
55:             continue
56:         matrix2[i, :] = matrix1[i, :]
57: 
58:     return matrix2
59: 
60: 
61: def _build_row_id(indices_list):
62:     all_rows = np.unique(np.concatenate(indices_list))
63:     return all_rows if all_rows.size > 0 else np.array([], dtype=np.uint64)
64: 
65: 
66: class SparseCSC:
67:     def __init__(self, column_indices, column_values, column_id=None):
68:         self.column_indices = column_indices
69:         self.column_values = column_values
70:         n = len(column_indices)
71:         self.column_id = (
72:             np.arange(n) if column_id is None
73:             else np.asarray(column_id, dtype=np.int64)
74:         )
75:         self.row_id = _build_row_id(column_indices)
76: 
77:     def __len__(self):
78:         return len(self.column_indices)
79: 
80:     def to_dense(self, columns=None, col_func=None, fill_value=0.0):
81:         if columns is None:
82:             col_idx_list = self.column_indices
83:             col_val_list = self.column_values
84:         else:
85:             mask = np.isin(self.column_id, columns)
86:             if not np.any(mask):
87:                 return np.full((self.row_id.size, 0), fill_value, dtype=np.float32)
88:             idx = np.where(mask)[0]
89:             col_idx_list = [self.column_indices[i] for i in idx]
90:             col_val_list = [self.column_values[i] for i in idx]
91: 
92:         if self.row_id.size == 0:
93:             return np.full((0, len(col_idx_list)), fill_value)
94: 
95:         if col_func is not None:
96:             col_val_list = [col_func(v) for v in col_val_list]
97: 
98:         return build_dense_from_csc(
99:             matrix_shape=(self.row_id.size, len(col_idx_list)),
100:             true_indices=self.row_id,
101:             column_indices=col_idx_list,
102:             column_values=col_val_list,
103:             fill_value=fill_value,
104:         )
105: 
106:     def align(self, other, how="left"):
107:         if how == "left":
108:             new_rows = self.row_id.copy()
109:             new_cols = self.column_id.copy()
110:         elif how == "both":
111:             new_rows = np.union1d(self.row_id, other.row_id).astype(np.int64)
112:             new_cols = np.union1d(self.column_id, other.column_id).astype(np.int64)
113:         else:
114:             raise ValueError(f"how must be 'left' or 'both', got '{how}'")
115: 
116:         if new_rows.size == 0 or new_cols.size == 0:
117:             empty = SparseCSC([], [], column_id=np.array([], dtype=np.int64))
118:             return empty, empty
119: 
120:         max_id = max(
121:             int(self.row_id.max()) if self.row_id.size > 0 else 0,
122:             int(other.row_id.max()) if other.row_id.size > 0 else 0,
123:         )
124:         row_map = np.full(max_id + 1, -1, dtype=np.int64)
125:         row_map[new_rows] = np.arange(new_rows.size)
126:         new_cols_arr = new_cols.copy()
127: 
128:         def _align_one(csc):
129:             col_map = {cid: i for i, cid in enumerate(csc.column_id)}
130:             new_idx, new_val = [], []
131:             for cid in new_cols_arr:
132:                 k = col_map.get(cid)
133:                 if k is None:
134:                     new_idx.append(np.array([], dtype=np.int64))
135:                     new_val.append(np.array([], dtype=np.float32))
136:                     continue
137:                 old_idx = csc.column_indices[k]
138:                 old_val = csc.column_values[k]
139:                 mapped = row_map[old_idx]
140:                 valid = mapped >= 0
141:                 new_idx.append(old_idx[valid])
142:                 new_val.append(old_val[valid])
143: 
144:             result = SparseCSC(new_idx, new_val, column_id=new_cols_arr)
145:             result.row_id = new_rows.copy()
146:             return result
147: 
148:         return _align_one(self), _align_one(other)
149: 
150:     def remap_rows(self, src_ids: np.ndarray, dst_ids: np.ndarray) -> "SparseCSC":
151:         idx_order = np.argsort(src_ids)
152:         sorted_src = src_ids[idx_order]
153:         sorted_dst = dst_ids[idx_order]
154: 
155:         new_indices = []
156:         new_values = []
157:         for idx_arr, val_arr in zip(self.column_indices, self.column_values):
158:             if idx_arr.size == 0:
159:                 new_indices.append(np.array([], dtype=np.int64))
160:                 new_values.append(np.array([], dtype=np.float32))
161:                 continue
162:             pos = np.searchsorted(sorted_src, idx_arr)
163:             pos = np.clip(pos, 0, len(sorted_src) - 1)
164:             found = sorted_src[pos] == idx_arr
165:             new_indices.append(sorted_dst[pos[found]].astype(np.int64, copy=False))
166:             new_values.append(val_arr[found].copy())
167: 
168:         return SparseCSC(new_indices, new_values, column_id=self.column_id.copy())

(End of file - total 168 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 4.9s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/physics/halo_ensemble.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/physics/halo_ensemble.py</path>
<type>file</type>
<content>
1: import numpy as np
2: 
3: from roadrunner.clustering.sparse import SparseCSC
4: from roadrunner.physics.halo_model import HaloModel
5: 
6: 
7: class HaloEnsemble:
8:     def __init__(self, halos: list[HaloModel]):
9:         self._halos = list(halos)
10: 
11:     def __len__(self) -> int:
12:         return len(self._halos)
13: 
14:     def __getitem__(self, i) -> HaloModel:
15:         return self._halos[i]
16: 
17:     def __iter__(self):
18:         return iter(self._halos)
19: 
20:     @property
21:     def nhalo(self) -> int:
22:         return len(self._halos)
23: 
24:     @property
25:     def nstars(self) -> int:
26:         total = 0
27:         for h in self._halos:
28:             if h.has_boundness:
29:                 total += len(h.get_boundness()[0])
30:         return total
31: 
32:     @property
33:     def empty(self) -> bool:
34:         return len(self._halos) == 0
35: 
36:     @property
37:     def positions(self) -> np.ndarray:
38:         return np.array([h.xcen for h in self._halos], dtype=np.float64)
39: 
40:     @property
41:     def virial_radii(self) -> np.ndarray:
42:         return np.array([h.virial_radius for h in self._halos], dtype=np.float64)
43: 
44:     @property
45:     def sub_tree_ids(self) -> np.ndarray:
46:         return np.array([h.sub_tree_id for h in self._halos], dtype=int)
47: 
48:     def select(self, indices: list[int]) -> "HaloEnsemble":
49:         return HaloEnsemble([self._halos[i] for i in indices])
50: 
51:     def get_particles(self) -> tuple[SparseCSC, SparseCSC]:
52:         candidates = []
53:         boundness = []
54:         tdyns = []
55: 
56:         for h in self._halos:
57:             if h.has_boundness:
58:                 inds, ener, tdyn_arr = h.get_boundness()
59:                 candidates.append(inds)
60:                 boundness.append(ener)
61:                 tdyns.append(tdyn_arr)
62:             else:
63:                 empty_idx = np.array([], dtype=np.uint64)
64:                 empty_val = np.array([], dtype=np.float32)
65:                 candidates.append(empty_idx)
66:                 boundness.append(empty_val)
67:                 tdyns.append(empty_val)
68: 
69:         col_id = np.array([h.sub_tree_id for h in self._halos], dtype=np.int64)
70:         return SparseCSC(candidates, boundness, column_id=col_id), SparseCSC(candidates, tdyns, column_id=col_id)
71: 
72:     def populated_indices(self) -> np.ndarray:
73:         csc, _ = self.get_particles()
74:         return np.array(
75:             [i for i, col in enumerate(csc.column_indices) if col.size > 0],
76:             dtype=np.int64,
77:         )
78: 
79:     def empty_indices(self) -> np.ndarray:
80:         csc, _ = self.get_particles()
81:         return np.array(
82:             [i for i, col in enumerate(csc.column_indices) if col.size == 0],
83:             dtype=np.int64,
84:         )

(End of file - total 84 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/physics/boundness.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/physics/boundness.py</path>
<type>file</type>
<content>
1: import numpy as np
2: from scipy.spatial import KDTree
3: 
4: from roadrunner.physics.constants import G_KM
5: 
6: 
7: def compute_halo_bound_particles(
8:     halos: list,
9:     particle_coordinates: np.ndarray,
10:     search_factor: float = 1.0,
11: ) -> list:
12:     positions = particle_coordinates[:, :3]
13:     velocities = particle_coordinates[:, 3:6]
14:     tree = KDTree(positions)
15: 
16:     empty_indices = np.array([], dtype=np.uint64)
17:     empty_values = np.array([], dtype=np.float32)
18: 
19:     for halo in halos:
20:         local = np.asarray(
21:             tree.query_ball_point(
22:                 halo.xcen, r=search_factor * halo.virial_radius, workers=-1
23:             )
24:         )
25:         if local.size == 0:
26:             halo.set_boundness(empty_indices, empty_values, empty_values)
27:             continue
28: 
29:         rel_pos = positions[local] - halo.xcen
30:         rel_vel = velocities[local] - halo.velocity
31:         dist = np.linalg.norm(rel_pos, axis=1)
32:         vel_mags = np.linalg.norm(rel_vel, axis=1)
33: 
34:         v_vir_sq = G_KM * halo._inner.M / halo.virial_radius * (1 + halo.redshift)
35:         phi = halo.potential(dist)
36:         v_esc = np.sqrt(2 * np.abs(phi))
37:         boundness = 0.5 * (v_esc**2 - vel_mags**2) / v_vir_sq
38:         tdyns = halo.dynamical_time(dist)
39: 
40:         bound = vel_mags <= v_esc
41:         valid = local[bound]
42:         if valid.size == 0:
43:             halo.set_boundness(empty_indices, empty_values, empty_values)
44:         else:
45:             halo.set_boundness(
46:                 valid.astype(np.uint64),
47:                 boundness[bound].astype(np.float32),
48:                 tdyns[bound].astype(np.float32),
49:             )
50: 
51:     return halos

(End of file - total 51 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/clustering/assignment/gmm.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/clustering/assignment/gmm.py</path>
<type>file</type>
<content>
1: import warnings
2: 
3: import numpy as np
4: import pandas as pd
5: from numba import njit, prange
6: from scipy.stats import rankdata
7: 
8: from roadrunner._mcf_types import AssignmentResult
9: from roadrunner.clustering.assignment.statistics import GMMAssignerStatistics
10: from roadrunner.clustering.sparse import SparseCSC, stitch_zero_rows
11: from roadrunner.mixture._math import row_l1_normalize
12: from roadrunner.mixture.weighted_gmm import (
13:     WeightedGaussianMixture,
14:     _estimate_gaussian_parameters,
15: )
16: from roadrunner.physics.halo_ensemble import HaloEnsemble
17: 
18: 
19: def _no_transform(values):
20:     return values
21: 
22: 
23: def _rank_transform(values, func_rank=np.log1p):
24:     if len(values) == 0:
25:         return values
26:     ranks = rankdata(values, method="ordinal")
27:     return func_rank(ranks).astype(np.float32, copy=False)
28: 
29: 
30: @njit(parallel=True)
31: def _merge_resp_kernel(raw, bound, n_components, newborn_1d):
32:     """Merge previous responsibilities with boundness.
33: 
34:     Parameters
35:     ----------
36:     raw : float32 (n, k) — prev_dense on input, overwritten with merged result.
37:     bound : float32 (n, k) — dense boundness matrix.
38:     n_components : int
39:     newborn_1d : bool (n,) — True for newborn particles.
40:     """
41:     n = raw.shape[0]
42:     inv_ncomp = 1.0 / n_components
43:     for n_idx in prange(n):
44:         is_newb = newborn_1d[n_idx]
45:         for k_idx in range(n_components):
46:             b = bound[n_idx, k_idx]
47:             p = raw[n_idx, k_idx]
48:             if b > 0:
49:                 if p > 0:
50:                     raw[n_idx, k_idx] = p
51:                 elif not is_newb:
52:                     raw[n_idx, k_idx] = inv_ncomp
53:                 else:
54:                     raw[n_idx, k_idx] = 0.0
55:             else:
56:                 raw[n_idx, k_idx] = 0.0
57: 
58: 
59: class GMMAssigner:
60:     def __init__(
61:         self,
62:         cov_type="full",
63:         max_iter=10,
64:         tol=5e-2,
65:         min_particles=10,
66:         reg_covar=1e-6,
67:         prior_type="",
68:         verbose=1,
69:     ):
70:         self.cov_type = cov_type
71:         self.max_iter = max_iter
72:         self.tol = tol
73:         self.min_particles = min_particles
74:         self.reg_covar = reg_covar
75:         self.prior_type = prior_type
76:         self.verbose = verbose
77:         self.statistics = GMMAssignerStatistics()
78: 
79:     def assign(
80:         self, halos, particle_coords, newborn_indices, groups, **kwargs
81:     ) -> AssignmentResult:
82:         self.ensemble = HaloEnsemble(halos) if not isinstance(halos, HaloEnsemble) else halos
83:         self.particle_coords = particle_coords
84:         self.newborn_indices = newborn_indices
85:         self.previous_resp = kwargs.get("previous_resp", {})
86: 
87:         N = particle_coords.shape[0]
88:         self.particles_df = pd.DataFrame(
89:             {
90:                 "array_index": np.arange(N, dtype=np.uint64),
91:                 "Sub_tree_id": -1,
92:             }
93:         ).set_index("array_index")
94:         self.resp_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
95:         self.parameters: dict[int, dict] = {}
96: 
97:         ngal = np.concatenate(groups).size if groups else 0
98:         print(f"{ngal} in a total of {len(groups)} groups")
99: 
100:         for group in groups:
101:             self._process_group(group)
102: 
103:         self.particles_df.reset_index(inplace=True)
104:         csc_b, _ = self.ensemble.get_particles()
105:         self.statistics.compute(
106:             self.particles_df, self.resp_map, self.parameters,
107:             boundness_csc=csc_b,
108:         )
109: 
110:         gids = np.array(sorted(self.resp_map.keys()), dtype=np.int64)
111:         resp_csc = SparseCSC(
112:             [self.resp_map[g][0] for g in gids],
113:             [self.resp_map[g][1] for g in gids],
114:             column_id=gids,
115:         )
116:         return AssignmentResult(
117:             self.particles_df, resp_csc, self.parameters, {
118:                 "groups": len(groups),
119:                 "halos_in_groups": sum(len(g) for g in groups),
120:                 "bound_particles": self.ensemble.nstars,
121:                 **self.statistics.values,
122:         })
123: 
124:     def _process_group(self, group):
125:         sub_ensemble = self.ensemble.select(group)
126:         csc_b, _ = sub_ensemble.get_particles()
127:         group_subtrees = sub_ensemble.sub_tree_ids
128:         gp_idx = np.unique(np.concatenate(csc_b.column_indices))
129: 
130:         if gp_idx.size == 0:
131:             warnings.warn(
132:                 "WARNING: A non-empty group turns out to be empty of particles!"
133:             )
134:             return
135: 
136:         n_comp = len(group_subtrees)
137: 
138:         if n_comp == 1:
139:             post_prob, nonz = self._fit_single(gp_idx)
140:         elif gp_idx.size < 10 * n_comp:
141:             warnings.warn(
142:                 f"WARNING: {n_comp} galaxies are unresolved inside a group!"
143:             )
144:             post_prob, nonz = self._fit_unresolved(gp_idx, csc_b)
145:         else:
146:             post_prob, nonz = self._fit_resolved(
147:                 gp_idx, group_subtrees, csc_b
148:             )
149: 
150:         for i in range(len(group)):
151:             mask = nonz[:, i] > 0
152:             if np.any(mask):
153:                 self.resp_map[group_subtrees[i]] = (
154:                     gp_idx[mask],
155:                     post_prob[mask, i],
156:                 )
157: 
158:         assigned = group_subtrees[post_prob.argmax(axis=1)]
159:         self.particles_df.loc[gp_idx, "Sub_tree_id"] = assigned
160: 
161:     def _fit_single(self, gp_idx):
162:         post_prob = np.ones((gp_idx.size, 1))
163:         nonz = (post_prob > 0).astype(np.uint8)
164:         return post_prob, nonz
165: 
166:     def _fit_unresolved(self, gp_idx, csc_b):
167:         post_prob = row_l1_normalize(
168:             csc_b.to_dense(col_func=_no_transform)
169:         ).astype(np.float32, copy=False)
170:         nonz = (post_prob > 0).astype(np.uint8)
171:         return post_prob, nonz
172: 
173:     def _fit_resolved(self, gp_idx, group_subtrees, csc_b):
174:         from roadrunner.physics.scaler import StandardScaler
175:         scaler = StandardScaler()
176:         coords = scaler.fit_transform(self.particle_coords[gp_idx].astype(np.float64, copy=False))
177: 
178:         prior, w_init, m_init, c_init, cov_t = self._estimate_initial_params(
179:             coords, csc_b
180:         )
181:         nonz = (prior > 0).astype(np.uint8)
182:         n_comp = len(group_subtrees)
183: 
184:         try:
185:             gmm = WeightedGaussianMixture(
186:                 n_components=n_comp,
187:                 means_init=m_init,
188:                 covariance_init=c_init,
189:                 weights_init=w_init,
190:                 cov_type=cov_t,
191:                 init_params="kmeans++",
192:                 cast_dtype=np.float32,
193:                 tol=self.tol,
194:                 verbose=self.verbose,
195:             ).fit(coords, latent_prior=prior)
196:         except np.linalg.LinAlgError:
197:             gmm = WeightedGaussianMixture(
198:                 n_components=n_comp,
199:                 means_init=m_init,
200:                 covariance_init=c_init,
201:                 weights_init=w_init,
202:                 cov_type=cov_t,
203:                 init_params="kmeans++",
204:                 cast_dtype=np.float64,
205:                 tol=self.tol,
206:                 verbose=self.verbose,
207:             ).fit(coords, latent_prior=prior)
208:             warnings.warn("Precision increased to float64.")
209: 
210:         log_prob = gmm.predict_log_proba(coords, latent_prior=prior)
211:         post_prob = np.exp(log_prob)
212: 
213:         scaled = {
214:             "means": {int(sid): gmm.means_[i] for i, sid in enumerate(group_subtrees)},
215:             "weights": {int(sid): gmm.weights_[i] for i, sid in enumerate(group_subtrees)},
216:             "covariances": {int(sid): gmm.covariances_[i] for i, sid in enumerate(group_subtrees)},
217:             "cov_type": gmm.cov_type,
218:         }
219:         natural = self.get_parameters_natural(scaled, scaler)
220: 
221:         for i, sid in enumerate(group_subtrees):
222:             sid = int(sid)
223:             if sid in natural["means"]:
224:                 cov_scaled = gmm.covariances_[i]
225:                 vals = np.linalg.eigvalsh(cov_scaled) if cov_scaled.ndim == 2 else cov_scaled
226:                 cond = float(vals.max() / max(vals.min(), 1e-30))
227:                 self.parameters[sid] = {
228:                     "mean": natural["means"][sid],
229:                     "weight": natural["weights"][sid],
230:                     "covariance": natural["covariances"][sid],
231:                     "covariance_condition": cond,
232:                 }
233: 
234:         return post_prob, nonz
235: 
236:     def _estimate_initial_params(self, coords, csc_b):
237:         n_samples = coords.shape[0]
238:         n_components = len(csc_b.column_indices)
239: 
240:         prior = row_l1_normalize(
241:             csc_b.to_dense(col_func=_rank_transform)
242:         ).astype(np.float32, copy=False)
243: 
244:         if self.previous_resp:
245:             _, aligned_p = csc_b.align(self.previous_resp, how="left")
246:             prev_dense = aligned_p.to_dense()
247: 
248:             newborn_1d = np.isin(csc_b.row_id, self.newborn_indices)
249:             _merge_resp_kernel(prev_dense, prior, n_components, newborn_1d)
250: 
251:             resp = stitch_zero_rows(prior, prev_dense)
252:             resp = row_l1_normalize(resp)
253:             if self.prior_type.lower() == "temporal-log-lik":
254:                 prior = resp
255:         else:
256:             resp = prior
257: 
258:         nk, means_init, covs_init = _estimate_gaussian_parameters(
259:             coords, resp, np.ones(n_samples, dtype=np.float32),
260:             self.cov_type, reg_covar=self.reg_covar,
261:         )
262:         weights_init = nk / nk.sum()
263: 
264:         return prior, weights_init, means_init, covs_init, self.cov_type
265: 
266:     @staticmethod
267:     def get_parameters_natural(stored, scaler):
268:         inv_s = 1.0 / scaler.scale_
269:         means, weights, covariances = {}, {}, {}
270:         cov_t = stored.get("cov_type", "full")
271: 
272:         for sid in stored.get("means", {}):
273:             mean_s = stored["means"][sid]
274:             w = stored["weights"][sid]
275:             cov_s = stored["covariances"][sid]
276: 
277:             if cov_t == "spherical":
278:                 scaling_matrix = np.mean(inv_s**2)
279:             elif cov_t == "diagonal":
280:                 scaling_matrix = inv_s**2
281:             else:
282:                 scaling_matrix = np.outer(inv_s, inv_s)
283: 
284:             means[sid] = mean_s * inv_s + scaler.mean_
285:             weights[sid] = float(w)
286:             covariances[sid] = cov_s * scaling_matrix
287: 
288:         return {"means": means, "weights": weights, "covariances": covariances}

(End of file - total 288 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 5.2s)

_Thinking:_

Now let me also read the orchestrator and the readers to understand the full picture.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/clustering/assignment/orchestrator.py"
}
```

**Error:**
```
File not found: /home/asier/roadrunner/src/roadrunner/clustering/assignment/orchestrator.py
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/readers/snapshot.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/readers/snapshot.py</path>
<type>file</type>
<content>
1: import numpy as np
2: import yt
3: 
4: from roadrunner._mcf_types import SnapshotData
5: 
6: 
7: class SnapshotReader:
8:     def __init__(self, code: str, ptype: str, fields: dict, unit_base=None):
9:         self.code = code.upper()
10:         self.ptype = ptype
11:         self.fields = fields
12:         self.unit_base = unit_base
13: 
14:     def load(self, file_path: str) -> SnapshotData:
15:         ds = self._open(file_path)
16:         return self._extract(ds)
17: 
18:     def load_with_filter(
19:         self, file_path: str, particle_indices: np.ndarray
20:     ) -> SnapshotData:
21:         ds = self._open(file_path)
22:         ds = self.apply_particle_filter(ds, particle_indices)
23:         return self._extract(ds)
24: 
25:     def apply_particle_filter(self, ds, particle_indices):
26:         filter_name = f"_snap_filter_{id(particle_indices)}"
27:         yt.add_particle_filter(
28:             filter_name,
29:             function=lambda pfilter, data: np.isin(
30:                 data[pfilter.filtered_type, self.fields["index"]],
31:                 particle_indices,
32:             ),
33:             filtered_type=self.ptype,
34:             requires=[self.fields["index"]],
35:         )
36:         ds.add_particle_filter(filter_name)
37:         return ds
38: 
39:     @staticmethod
40:     def mock_data(n_particles: int, seed: int = 42) -> SnapshotData:
41:         rng = np.random.default_rng(seed)
42:         indices = np.arange(n_particles, dtype=np.uint64)
43:         masses = rng.uniform(0.1, 10.0, n_particles).astype(np.float64)
44:         positions = rng.uniform(-100, 100, (n_particles, 3)).astype(np.float64)
45:         velocities = rng.uniform(-200, 200, (n_particles, 3)).astype(np.float64)
46:         metallicity = rng.uniform(-2.0, 0.5, n_particles).astype(np.float64)
47:         return SnapshotData(
48:             indices, masses, positions, velocities,
49:             redshift=0.0, time=13.8, metallicity=metallicity,
50:         )
51: 
52:     def _open(self, file_path: str):
53:         if self.code in ("ART", "ART-I"):
54:             return yt.load(file_path)
55: 
56:         if self.code in ("GEAR",):
57:             return yt.load(file_path, unit_base=self.unit_base)
58: 
59:         if self.code in ("AURIGA", "AREPO"):
60:             ds = yt.load(file_path, unit_base=self.unit_base)
61:             yt.add_particle_filter(
62:                 self.ptype,
63:                 function=lambda pfilter, data: (
64:                     data[pfilter.filtered_type, "GFM_StellarFormationTime"] >= 0
65:                 ),
66:                 requires=[
67:                     "GFM_StellarFormationTime",
68:                     "particle_index",
69:                 ],
70:                 filtered_type="PartType4",
71:             )
72:             ds.add_particle_filter(self.ptype)
73:             return ds
74: 
75:         if self.code in ("RAMSES", "VINTERGATAN"):
76:             yt.add_particle_filter(
77:                 self.ptype,
78:                 function=lambda pfilter, data: (
79:                     data[pfilter.filtered_type, "particle_birth_time"] > 0
80:                 ) & (data[pfilter.filtered_type, "particle_metallicity0"] > 0),
81:                 requires=["particle_birth_time"],
82:                 filtered_type="all",
83:             )
84:             ds = yt.load(
85:                 file_path,
86:                 extra_particle_fields=[
87:                     ("particle_potential", "d"),
88:                     ("conformal_birth_time", "d"),
89:                     ("particle_metallicity0", "d"),
90:                     ("particle_metallicity1", "d"),
91:                     ("particle_tag", "d"),
92:                     ("particle_birth_time", "d"),
93:                 ],
94:             )
95:             ds.add_particle_filter(self.ptype)
96:             ds.add_field(
97:                 (self.ptype, "particle_metallicity"),
98:                 function=lambda field, data: (
99:                     2.09 * data[self.ptype, "particle_metallicity0"]
100:                     + 1.06 * data[self.ptype, "particle_metallicity1"]
101:                 ) / 0.02,
102:                 sampling_type="particle",
103:                 units="",
104:             )
105:             return ds
106: 
107:         raise ValueError(f"Unsupported code: {self.code}")
108: 
109:     def _extract(self, ds) -> SnapshotData:
110:         ad = ds.all_data()
111:         ptype = self.ptype
112:         f = self.fields
113: 
114:         indices = ad[ptype, f["index"]].value.astype(np.uint64)
115:         masses = ad[ptype, f["mass"]].to("Msun").value.astype(np.float64)
116:         positions = (
117:             ad[ptype, f["position"]].to("kpccm").value.astype(np.float64)
118:         )
119:         velocities = (
120:             ad[ptype, f["velocity"]].to("km/s").value.astype(np.float64)
121:         )
122:         redshift = ds.current_redshift
123:         time = ds.current_time.to("Gyr").value
124: 
125:         metallicity = None
126:         if "metallicity" in f:
127:             metallicity = ad[ptype, f["metallicity"]].value.astype(np.float64)
128: 
129:         return SnapshotData(
130:             indices, masses, positions, velocities,
131:             redshift, time, metallicity,
132:         )

(End of file - total 132 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/physics/timescales.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/physics/timescales.py</path>
<type>file</type>
<content>
1: import numpy as np
2: 
3: 
4: def compute_particle_dynamical_timescales(particle_df, ensemble, groups, td_factor=1.0):
5:     particle_df["timescale"] = 0.0
6: 
7:     for group in groups:
8:         sub = ensemble.select(group)
9:         _, csc_t = sub.get_particles()
10: 
11:         tdyn_dense = csc_t.to_dense()
12: 
13:         max_tdyn = np.minimum(np.nanmax(tdyn_dense, axis=1) * td_factor, 0.4)
14:         gp_idx = csc_t.row_id
15:         particle_df.loc[gp_idx, "timescale"] = max_tdyn
16: 
17:     return particle_df
18: 
19: 
20: def compute_tidal_radius(halo, satellite_mass, distance):
21:     denom = np.asarray(halo.tidal_denominator(np.array([distance]))).flat[0]
22:     return distance * (satellite_mass / denom) ** (1.0 / 3.0)

(End of file - total 22 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 6.9s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/postprocessing/properties.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/postprocessing/properties.py</path>
<type>file</type>
<content>
1: import warnings
2: 
3: import numpy as np
4: import pandas as pd
5: 
6: from roadrunner.physics.potentials import get_potential
7: from roadrunner.physics.timescales import compute_tidal_radius
8: 
9: 
10: def random_lines_of_sight(N, half_sphere=True, seed=None):
11:     rng = np.random.default_rng(seed)
12: 
13:     phi = rng.uniform(0, 2 * np.pi, size=N)
14:     if half_sphere:
15:         cos_theta = rng.uniform(0, 1, size=N)
16:     else:
17:         cos_theta = rng.uniform(-1, 1, size=N)
18:     theta = np.arccos(cos_theta)
19: 
20:     x = np.sin(theta) * np.cos(phi)
21:     y = np.sin(theta) * np.sin(phi)
22:     z = cos_theta
23: 
24:     return np.column_stack((x, y, z))
25: 
26: 
27: def rotation_matrix_from_los(los):
28:     los = np.asarray(los)
29:     if los.shape != (3,):
30:         raise ValueError("los must be a 3-element vector")
31:     ez = los / np.linalg.norm(los)
32: 
33:     for i in range(10):
34:         randvecs = random_lines_of_sight(
35:             100, half_sphere=False, seed=42 + i
36:         )[0]
37: 
38:         dots = np.abs(randvecs @ ez)
39:         mask = dots < 0.5
40:         if np.any(mask):
41:             idx = np.argmin(dots[mask])
42:             randvec = randvecs[mask][idx]
43:             break
44:     else:
45:         warnings.warn(
46:             "Failed to draw a suitable random vector"
47:         )
48:         if abs(np.dot([0, 0, 2], ez)) < 0.9:
49:             randvec = np.array([0, 0, 1]) - ez
50:         else:
51:             randvec = np.array([0, 1, 0]) - ez
52: 
53:     a = ez + randvec
54:     ex = a - np.dot(a, ez) * ez
55:     ex /= np.linalg.norm(ex)
56:     ey = np.cross(ez, ex)
57: 
58:     return np.vstack((ex, ey, ez))
59: 
60: 
61: def find_center(particle_positions, particle_velocities, particle_masses):
62:     radii = np.linalg.norm(
63:         particle_positions - np.median(particle_positions, axis=0),
64:         axis=1,
65:     )
66:     quantile = np.quantile(radii, 0.95)
67:     rc_scale = 0.5
68:     inner = radii <= rc_scale * quantile
69:     center_pos = np.average(
70:         particle_positions[inner], axis=0, weights=particle_masses[inner],
71:     )
72:     center_vel = np.average(
73:         particle_velocities[inner], axis=0, weights=particle_masses[inner],
74:     )
75:     return center_pos, center_vel
76: 
77: 
78: def enclosed_mass_radius(radii, masses, mass_fraction):
79:     if mass_fraction > 1 or mass_fraction < 0:
80:         raise ValueError(
81:             f"mass_fraction must be in [0, 1], got {mass_fraction}"
82:         )
83:     tot_mass = masses.sum()
84:     if tot_mass <= 0.0:
85:         return 0.0
86: 
87:     idx_sort = np.argsort(radii)
88:     radii_sorted = radii[idx_sort]
89:     mass_sorted = masses[idx_sort]
90:     cum_mass = np.cumsum(mass_sorted) / tot_mass
91: 
92:     i = np.searchsorted(cum_mass, mass_fraction)
93: 
94:     if i == 0:
95:         return radii_sorted[0]
96:     elif i >= len(radii_sorted):
97:         return radii_sorted[-1]
98:     else:
99:         r_low, r_high = radii_sorted[i - 1], radii_sorted[i]
100:         f_low, f_high = cum_mass[i - 1], cum_mass[i]
101:         return r_low + (mass_fraction - f_low) / (f_high - f_low) * (r_high - r_low)
102: 
103: 
104: def half_mass_radius(positions, masses, center, mass_fraction=0.5):
105:     radii = np.sqrt(np.sum((positions - center) ** 2, axis=1))
106:     return enclosed_mass_radius(radii, masses, mass_fraction)
107: 
108: 
109: def projected_half_mass_radius(positions, masses, center, los_matrices, mass_fraction=0.5):
110:     Nlos = los_matrices.shape[0]
111:     centered = positions - center
112:     result = np.empty(Nlos, dtype=np.float64)
113:     for i in range(Nlos):
114:         pos_rot = centered @ los_matrices[i].T
115:         radii = np.sqrt(pos_rot[:, 0] ** 2 + pos_rot[:, 1] ** 2)
116:         result[i] = enclosed_mass_radius(radii, masses, mass_fraction)
117:     return result if Nlos > 1 else result[0]
118: 
119: 
120: def velocity_dispersion(velocities):
121:     sigma_per_axis = np.std(velocities, axis=0, ddof=1)
122:     return np.sqrt(np.sum(sigma_per_axis ** 2))
123: 
124: 
125: def line_of_sight_velocity_dispersion(positions, velocities, los_matrices, apertures):
126:     los_matrices = np.asarray(los_matrices)
127:     apertures = np.asarray(apertures)
128: 
129:     if los_matrices.shape[0] != apertures.shape[0]:
130:         raise ValueError(
131:             "Number of los_matrices must match number of apertures"
132:         )
133: 
134:     Nlos = los_matrices.shape[0]
135:     sigma_los = np.full(Nlos, np.nan)
136: 
137:     for i in range(Nlos):
138:         R = los_matrices[i]
139:         pos_rot = positions @ R.T
140:         vel_rot = velocities @ R.T
141: 
142:         r_proj = np.sqrt(pos_rot[:, 0] ** 2 + pos_rot[:, 1] ** 2)
143:         mask = r_proj <= apertures[i]
144: 
145:         if np.any(mask):
146:             v_los = vel_rot[mask, 2]
147:             sigma_los[i] = np.std(v_los, ddof=1)
148: 
149:     return sigma_los
150: 
151: 
152: def compute_galaxy_properties(
153:     accretion_id,
154:     particle_masses,
155:     particle_coords,
156:     galaxy_particles,
157:     galaxy_table,
158:     host_props,
159:     halo_model,
160:     n_los=11,
161:     galaxy_centers=None,
162: ):
163:     columns = [
164:         "Sub_tree_id", "mb_host_id",
165:         "position_x", "position_y", "position_z",
166:         "velocity_x", "velocity_y", "velocity_z",
167:         "Mtot",
168:         "r20", "rh", "r80",
169:         "Rhp",
170:         "sigma",
171:         "sigma_los",
172:         "r_t",
173:     ]
174: 
175:     los_vectors = random_lines_of_sight(n_los)
176:     los_matrices = np.array([rotation_matrix_from_los(los) for los in los_vectors])
177: 
178:     host_mass = host_props["mass"]
179:     if halo_model.lower() == "kepler":
180:         host_potential = get_potential(halo_model, M=host_mass)
181:     else:
182:         host_Rs = host_props["scale_radius"] / (1 + host_props["Redshift"])
183:         host_c = host_props["virial_radius"] / host_props["scale_radius"]
184:         host_potential = get_potential(halo_model, M=host_mass, Rs=host_Rs, c=host_c)
185: 
186:     if -1 in galaxy_particles:
187:         del galaxy_particles[-1]
188: 
189:     records = []
190:     for sid, indices in galaxy_particles.items():
191:         if indices.size == 0:
192:             continue
193: 
194:         row = galaxy_table.loc[sid]
195:         host_id = row["host_id"]
196:         sat_mass = row["mass"]
197:         distance = row["distance_to_acc_id"]
198: 
199:         r_t = compute_tidal_radius(host_potential, sat_mass, distance)
200: 
201:         indices_int = np.asarray(indices, dtype=np.intp)
202:         gal_pos = particle_coords[indices_int, :3]
203:         gal_vel = particle_coords[indices_int, 3:6]
204:         gal_masses = particle_masses[indices_int]
205:         Mtot = gal_masses.sum()
206:         npart = indices.size
207: 
208:         if npart <= 2:
209:             records.append(dict(
210:                 Sub_tree_id=sid, mb_host_id=host_id,
211:                 position_x=np.nan, position_y=np.nan, position_z=np.nan,
212:                 velocity_x=np.nan, velocity_y=np.nan, velocity_z=np.nan,
213:                 Mtot=Mtot, r20=np.nan, rh=np.nan, r80=np.nan,
214:                 Rhp=np.nan, sigma=np.nan, sigma_los=np.nan, r_t=r_t,
215:             ))
216:             continue
217: 
218:         fitted = None if galaxy_centers is None else galaxy_centers.get(sid)
219:         if fitted is not None:
220:             center_pos = fitted[:3]
221:             center_vel = fitted[3:6]
222:         else:
223:             center_pos, center_vel = find_center(gal_pos, gal_vel, gal_masses)
224: 
225:         if npart < 30:
226:             records.append(dict(
227:                 Sub_tree_id=sid, mb_host_id=host_id,
228:                 position_x=center_pos[0], position_y=center_pos[1],
229:                 position_z=center_pos[2],
230:                 velocity_x=center_vel[0], velocity_y=center_vel[1],
231:                 velocity_z=center_vel[2],
232:                 Mtot=Mtot, r20=np.nan, rh=np.nan, r80=np.nan,
233:                 Rhp=np.nan, sigma=np.nan, sigma_los=np.nan, r_t=r_t,
234:             ))
235:             continue
236: 
237:         centered_pos = gal_pos - center_pos
238:         centered_vel = gal_vel - center_vel
239: 
240:         r20 = half_mass_radius(centered_pos, gal_masses, np.zeros(3), mass_fraction=0.2)
241:         rh = half_mass_radius(centered_pos, gal_masses, np.zeros(3), mass_fraction=0.5)
242:         r80 = half_mass_radius(centered_pos, gal_masses, np.zeros(3), mass_fraction=0.8)
243:         sigma = velocity_dispersion(centered_vel)
244: 
245:         Rhp = projected_half_mass_radius(
246:             centered_pos, gal_masses, np.zeros(3), los_matrices, mass_fraction=0.5,
247:         )
248:         sigma_los_arr = line_of_sight_velocity_dispersion(
249:             centered_pos, centered_vel, los_matrices, Rhp,
250:         )
251: 
252:         records.append(dict(
253:             Sub_tree_id=sid, mb_host_id=host_id,
254:             position_x=center_pos[0], position_y=center_pos[1],
255:             position_z=center_pos[2],
256:             velocity_x=center_vel[0], velocity_y=center_vel[1],
257:             velocity_z=center_vel[2],
258:             Mtot=Mtot,
259:             r20=r20, rh=rh, r80=r80,
260:             Rhp=np.nanmedian(Rhp) if isinstance(Rhp, np.ndarray) else Rhp,
261:             sigma=sigma,
262:             sigma_los=np.nanmedian(sigma_los_arr),
263:             r_t=r_t,
264:         ))
265: 
266:     if not records:
267:         return pd.DataFrame(columns=columns, dtype=np.float32)
268:     return pd.DataFrame.from_records(records)[columns]

(End of file - total 268 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/postprocessing/mixing.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/postprocessing/mixing.py</path>
<type>file</type>
<content>
1: import numpy as np
2: import pandas as pd
3: from scipy.spatial import KDTree
4: 
5: from roadrunner.physics.constants import (
6:     RILEY_BOUND_THRESHOLD,
7:     RILEY_SVM_SLOPE,
8:     RILEY_SVM_INTERCEPT,
9: )
10: 
11: 
12: def _local_velocity_dispersion(pos, vel, nmin=10):
13:     N = pos.shape[0]
14:     n = int(max(0.01 * N, nmin))
15:     if N <= nmin + 1:
16:         return np.full(N, np.nan)
17: 
18:     wi = np.std(vel, axis=0) / np.std(pos, axis=0)
19:     data = np.empty((N, 6), dtype=np.float32)
20:     data[:, :3] = pos
21:     data[:, 3:] = vel / wi
22: 
23:     tree = KDTree(data)
24:     _, idx = tree.query(data, k=n, workers=-1)
25: 
26:     local_disp = np.empty(N, dtype=np.float64)
27:     for k in range(N):
28:         local_disp[k] = np.linalg.norm(np.std(vel[idx[k]], axis=0))
29: 
30:     return local_disp
31: 
32: 
33: def _riley_criterion_single(
34:     subtree_id,
35:     mstar,
36:     f_bound,
37:     subset_positions,
38:     subset_velocities,
39: ):
40:     if f_bound >= RILEY_BOUND_THRESHOLD:
41:         sigma50 = np.linalg.norm(np.std(subset_velocities, axis=0))
42:         dynstate = 0
43:     else:
44:         local_disp = _local_velocity_dispersion(
45:             subset_positions, subset_velocities,
46:         )
47:         sigma50 = np.nanmedian(local_disp)
48:         svm = RILEY_SVM_SLOPE * np.log10(mstar) + RILEY_SVM_INTERCEPT
49:         dynstate = 2 if (sigma50 >= svm or np.isnan(sigma50)) else 1
50: 
51:     return {
52:         "Sub_tree_id": int(subtree_id),
53:         "mstar": mstar,
54:         "f_bound": f_bound,
55:         "sigma50": sigma50,
56:         "dynstate": dynstate,
57:     }
58: 
59: 
60: def compute_riley_criterion(
61:     main_id,
62:     particle_masses,
63:     particle_coords,
64:     galaxy_allowed,
65:     galaxy_bound,
66:     redshift,
67: ):
68:     positions = particle_coords[:, :3] / (1 + redshift)
69:     velocities = particle_coords[:, 3:6]
70: 
71:     skip_ids = {main_id, -1}
72:     empty = np.array([], dtype=np.int64)
73: 
74:     records = []
75:     for sid in galaxy_allowed:
76:         if sid in skip_ids:
77:             continue
78: 
79:         allowed_idx = galaxy_allowed[sid]
80:         if allowed_idx.size == 0:
81:             continue
82: 
83:         bound_idx = galaxy_bound.get(sid, empty)
84:         mstar = particle_masses[allowed_idx].sum()
85:         if mstar > 0:
86:             bound_overlap = np.intersect1d(
87:                 allowed_idx.astype(np.int64), bound_idx.astype(np.int64),
88:             )
89:             f_bound = particle_masses[bound_overlap].sum() / mstar
90:         else:
91:             f_bound = 0.0
92: 
93:         result = _riley_criterion_single(
94:             sid, mstar, f_bound,
95:             positions[allowed_idx], velocities[allowed_idx],
96:         )
97:         records.append(result)
98: 
99:     columns = [
100:         "Sub_tree_id", "mstar", "f_bound", "sigma50", "dynstate",
101:     ]
102:     if not records:
103:         return pd.DataFrame(columns=columns, dtype=np.float32)
104:     return pd.DataFrame.from_records(records)[columns]

(End of file - total 104 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 5.7s)

_Thinking:_

Now I have a good understanding of the codebase. Let me also look at the IO modules to see how they interact with the pipeline.

Let me also check the IO layer and the remaining relevant files:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/io"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/io</path>
<type>directory</type>
<entries>
__init__.py
__pycache__/
hdf5_assignment.py
hdf5_catalogue.py
hdf5_particles.py
hdf5_reader.py
logging.py
serialization.py

(8 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/physics/halo_model.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/physics/halo_model.py</path>
<type>file</type>
<content>
1: import numpy as np
2: 
3: from roadrunner.physics.constants import G_KM
4: from roadrunner.physics.potentials import get_potential
5: from roadrunner._mcf_types import PotentialModel
6: 
7: 
8: class HaloModel:
9:     def __init__(
10:         self,
11:         inner: PotentialModel,
12:         xcen: np.ndarray,
13:         velocity: np.ndarray,
14:         virial_radius: float,
15:         sub_tree_id: int,
16:         redshift: float,
17:         comoving: bool = True,
18:     ):
19:         self._inner = inner
20:         self.xcen = np.asarray(xcen, dtype=np.float64)
21:         self.velocity = np.asarray(velocity, dtype=np.float64)
22:         self.virial_radius = float(virial_radius)
23:         self.sub_tree_id = sub_tree_id
24:         self.redshift = float(redshift)
25:         self.comoving = comoving
26:         self._1plusz  = 1 / (1 + self.redshift) if comoving else 1
27:         self._boundness: tuple | None = None
28: 
29:     def set_boundness(
30:         self, indices: np.ndarray, energies: np.ndarray, tdyns: np.ndarray
31:     ) -> None:
32:         self._boundness = (indices, energies, tdyns)
33: 
34:     def get_boundness(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
35:         return self._boundness
36: 
37:     @property
38:     def has_boundness(self) -> bool:
39:         return self._boundness is not None
40: 
41:     def potential(self, xyz_or_r: np.ndarray) -> np.ndarray:
42:         if xyz_or_r.ndim == 2:
43:             r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
44:         else:
45:             r = xyz_or_r * self._1plusz
46:         return self._inner.potential(r)
47: 
48:     def dynamical_time(self, xyz_or_r: np.ndarray) -> np.ndarray:
49:         if xyz_or_r.ndim == 2:
50:             r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
51:         else:
52:             r = xyz_or_r * self._1plusz
53:         return self._inner.dynamical_time(r)
54: 
55:     def tidal_denominator(self, xyz_or_r: np.ndarray) -> np.ndarray:
56:         if xyz_or_r.ndim == 2:
57:             r = np.linalg.norm(xyz_or_r - self.xcen, axis=1) * self._1plusz
58:         else:
59:             r = xyz_or_r * self._1plusz
60:         return self._inner.tidal_denominator(r)
61: 
62:     @classmethod
63:     def from_snapshot_row(cls, row, model="kepler", comoving=True):
64:         xcen = np.array([
65:             row["position_x"], row["position_y"], row["position_z"],
66:         ], dtype=np.float64)
67:         velocity = np.array([
68:             row["velocity_x"], row["velocity_y"], row["velocity_z"],
69:         ], dtype=np.float64)
70:         conc = row["virial_radius"] / row["scale_radius"]
71:         kwargs = {"M": row["mass"], "G": G_KM}
72:         if model.lower() == "nfw":
73:             kwargs["Rs"] = row["scale_radius"] / (1 + row["Redshift"])
74:             kwargs["c"] = conc
75:         inner = get_potential(model, **kwargs)
76:         return cls(
77:             inner=inner,
78:             xcen=xcen,
79:             velocity=velocity,
80:             virial_radius=float(row["virial_radius"]),
81:             sub_tree_id=int(row["Sub_tree_id"]),
82:             redshift=float(row["Redshift"]),
83:             comoving=comoving,
84:         )

(End of file - total 84 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 5.3s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/io/hdf5_catalogue.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/io/hdf5_catalogue.py</path>
<type>file</type>
<content>
1: import json
2: import os
3: 
4: import h5py
5: import numpy as np
6: import pandas as pd
7: 
8: from roadrunner.helpers import select_uint_dtype
9: 
10: 
11: class HDF5CatalogueWriter:
12:     def __init__(self, output_dir):
13:         os.makedirs(output_dir, exist_ok=True)
14:         self._path = os.path.join(output_dir, "catalogue.hdf5")
15: 
16:     def _del_existing(self, group, name):
17:         if name in group:
18:             del group[name]
19: 
20:     def write_header(self, accretion_id, snapshots, config_dict,
21:                      merger_tree_df, equivalence_df):
22:         with h5py.File(self._path, "a") as hf:
23:             hdr = hf.require_group("header")
24:             self._del_existing(hdr, "accretion_id")
25:             hdr.create_dataset("accretion_id", data=accretion_id)
26:             self._del_existing(hdr, "snapshots")
27:             snap_arr = np.asarray(snapshots, dtype=select_uint_dtype(max(snapshots)))
28:             hdr.create_dataset("snapshots", data=snap_arr, compression="gzip")
29:             self._del_existing(hdr, "config")
30:             config_json = json.dumps(config_dict, default=str)
31:             hdr.create_dataset("config", data=config_json,
32:                                dtype=h5py.string_dtype())
33:             self._del_existing(hdr, "merger_tree")
34:             rec_merger = merger_tree_df.to_records(index=False)
35:             hdr.create_dataset("merger_tree", data=rec_merger,
36:                                compression="gzip")
37:             self._del_existing(hdr, "equivalence")
38:             equiv_json = equivalence_df.to_json(orient="records")
39:             hdr.create_dataset("equivalence", data=equiv_json,
40:                                dtype=h5py.string_dtype())
41:             self._del_existing(hdr, "last_snapshot")
42:             hdr.create_dataset("last_snapshot", data=-1, dtype=np.int32)
43: 
44:     def write_snapshot(self, snapshot_id, time,
45:                        properties_df, dynstate_df,
46:                        satellites_map):
47:         with h5py.File(self._path, "a") as hf:
48:             snap_grp = hf.require_group(f"/snapshots/{snapshot_id}")
49: 
50:             snap_grp.attrs["time"] = float(time)
51: 
52:             if not properties_df.empty:
53:                 ds = snap_grp.create_dataset(
54:                     "galaxy_properties",
55:                     data=properties_df.to_records(index=False),
56:                     compression="gzip",
57:                 )
58:                 ds.attrs["Snapshot"] = int(snapshot_id)
59: 
60:             if not dynstate_df.empty:
61:                 ds = snap_grp.create_dataset(
62:                     "riley_criterion",
63:                     data=dynstate_df.to_records(index=False),
64:                     compression="gzip",
65:                 )
66:                 ds.attrs["Snapshot"] = int(snapshot_id)
67: 
68:             sat_grp = snap_grp.require_group("satellite_relations")
69:             for gal_id, sats in satellites_map.items():
70:                 arr = np.asarray(list(sats), dtype=np.int64)
71:                 sat_grp.create_dataset(str(gal_id), data=arr, compression="gzip")
72: 
73:             # Update last_snapshot
74:             hf["header"]["last_snapshot"][()] = int(snapshot_id)
75: 
76:     def write_finalize(self, birth_df, assembly_df):
77:         with h5py.File(self._path, "a") as hf:
78:             final = hf.require_group("final")
79:             if "births" in final:
80:                 del final["births"]
81:             if not birth_df.empty:
82:                 final.create_dataset(
83:                     "births",
84:                     data=birth_df.to_records(index=False),
85:                     compression="gzip",
86:                 )
87:             if "assembly" in final:
88:                 del final["assembly"]
89:             if not assembly_df.empty:
90:                 final.create_dataset(
91:                     "assembly",
92:                     data=assembly_df.to_records(index=False),
93:                     compression="gzip",
94:                 )

(End of file - total 94 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/io/hdf5_assignment.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/io/hdf5_assignment.py</path>
<type>file</type>
<content>
1: import os
2: 
3: import h5py
4: import numpy as np
5: 
6: from roadrunner.helpers import select_float_dtype, select_uint_dtype
7: 
8: 
9: class HDF5AssignmentWriter:
10:     def __init__(self, output_dir, float_atol=1e-4):
11:         os.makedirs(output_dir, exist_ok=True)
12:         self._path = os.path.join(output_dir, "assignment.hdf5")
13:         self._float_atol = float_atol
14: 
15:     def _gal_uint_dtype(self, gids):
16:         return select_uint_dtype(int(max(gids)) if len(gids) > 0 else 1)
17: 
18:     def _pick_float_dtype(self, arr):
19:         max_abs = float(np.abs(arr).max()) if arr.size > 0 else 1.0
20:         return select_float_dtype(max_abs, self._float_atol)
21: 
22:     def write_snapshot(self, snapshot_id, time,
23:                        assignment_result, boundness_csc):
24:         with h5py.File(self._path, "a") as hf:
25:             snap_grp = hf.require_group(f"/snapshots/{snapshot_id}")
26:             snap_grp.attrs["time"] = float(time)
27: 
28:             galaxies_grp = snap_grp.require_group("galaxies")
29: 
30:             resp_csc = assignment_result.responsibilities
31:             all_gids = list(resp_csc.column_id)
32:             uint_dtype = self._gal_uint_dtype(all_gids) if all_gids else np.uint32
33:             if all_gids:
34:                 all_vals = np.concatenate(resp_csc.column_values)
35:                 float_dtype = self._pick_float_dtype(all_vals)
36:             else:
37:                 float_dtype = np.float32
38: 
39:             for j, gid in enumerate(all_gids):
40:                 resp_idx = resp_csc.column_indices[j]
41:                 resp_vals = resp_csc.column_values[j]
42: 
43:                 col_mask = boundness_csc.column_id == gid
44:                 col_pos = np.where(col_mask)[0]
45:                 bound_vals = (
46:                     boundness_csc.column_values[col_pos[0]]
47:                     if len(col_pos) > 0 and col_pos[0] < len(boundness_csc.column_values)
48:                     else np.array([], dtype=np.float32)
49:                 )
50: 
51:                 grp = galaxies_grp.require_group(str(gid))
52: 
53:                 pid_dtype = select_uint_dtype(
54:                     int(resp_idx.max()) if resp_idx.size > 0 else 1
55:                 )
56:                 grp.create_dataset(
57:                     "indices",
58:                     data=resp_idx.astype(pid_dtype, copy=False),
59:                     compression="gzip",
60:                 )
61:                 grp.create_dataset(
62:                     "log_resp",
63:                     data=resp_vals.astype(float_dtype, copy=False),
64:                     compression="gzip",
65:                 )
66:                 grp.create_dataset(
67:                     "boundness",
68:                     data=bound_vals.astype(float_dtype, copy=False),
69:                     compression="gzip",
70:                 )
71: 
72:                 params = assignment_result.fitted_parameters.get(int(gid), {})
73:                 mean = params.get("mean")
74:                 if mean is not None:
75:                     mean_dtype = self._pick_float_dtype(mean)
76:                     grp.create_dataset(
77:                         "mean",
78:                         data=np.asarray(mean).astype(mean_dtype, copy=False),
79:                         compression="gzip",
80:                     )
81:                 weight = params.get("weight")
82:                 if weight is not None:
83:                     w = float(weight)
84:                     w_dtype = select_float_dtype(w, self._float_atol)
85:                     grp.create_dataset(
86:                         "weight",
87:                         data=np.array([w], dtype=w_dtype),
88:                         compression="gzip",
89:                     )
90:                 cov = params.get("covariance")
91:                 if cov is not None:
92:                     cov_arr = np.asarray(cov)
93:                     cov_dtype = self._pick_float_dtype(cov_arr)
94:                     kw = {} if cov_arr.ndim == 0 else {"compression": "gzip"}
95:                     grp.create_dataset(
96:                         "covariance",
97:                         data=cov_arr.astype(cov_dtype, copy=False),
98:                         **kw,
99:                     )
100: 
101:             # Hard assignment
102:             hard_path = "/".join(["", "snapshots", str(snapshot_id), "hard_assignment"])
103:             if hard_path in hf:
104:                 del hf[hard_path]
105:             snap_grp.create_dataset(
106:                 "hard_assignment",
107:                 data=assignment_result.particle_df.to_records(index=False),
108:                 compression="gzip",
109:             )
110: 
111:     def write_timescales(self, particle_timescales):
112:         with h5py.File(self._path, "a") as hf:
113:             hdr = hf.require_group("header")
114:             ts_dtype = select_float_dtype(
115:                 float(particle_timescales.max()) if particle_timescales.size > 0 else 1.0,
116:                 self._float_atol,
117:             )
118:             if "timescales" in hdr:
119:                 del hdr["timescales"]
120:             hdr.create_dataset(
121:                 "timescales",
122:                 data=particle_timescales.astype(ts_dtype, copy=False),
123:                 compression="gzip",
124:             )

(End of file - total 124 lines)
</content>
```

---

## Assistant (Plan · GLM-5.1 · 54.0s)

_Thinking:_

Now I have a thorough understanding of the codebase. Let me analyze the main weak points, particularly from step 9 onwards (phases 9-11), focusing on:

1. **Array index vs sim ID confusion** - The user specifically called this out
2. **SnapshotProcessor and AccretionPipeline structure** - The user said these break the philosophy
3. **Tracker-related index space issues**
4. **General violations of the "stateless → function, with state → class" philosophy**

Let me also quickly check the entry.py more carefully and the hdf5_assignment writer for the index confusion.

Now let me compile my analysis:

## Main Weak Points

### 1. Array Index vs Sim ID Space Confusion (CRITICAL)

The codebase has a fundamental confusion between two coordinate systems:
- **Array index space**: 0 to N-1, the position in the loaded arrays for a given snapshot
- **Sim ID space**: The persistent particle identifiers from the simulation (stored in `SnapshotData.indices`)

The problem is that:
- `SnapshotProcessor.process()` works entirely in array index space (0 to N-1)
- `BirthTracker` and `AssemblyTracker` work in sim ID space
- The conversion between these two happens in `AccretionPipeline._update_trackers()` and `AccretionPipeline._track_and_reduce()`
- The responsibilities matrix (`SparseCSC`) gets translated between spaces via `_to_sim_space()` / `_from_sim_space()` in AccretionPipeline

This is haphazard because:
1. The `BirthTracker` stores `particle_ids` which are sim IDs, but calls them `particle_index` in its output (see `finalize()` line 139: `"particle_index"` but the values are actually sim IDs from `snap_data.indices[arr_idx]`)
2. The `AssemblyTracker` stores sim IDs in `assignment_map`, but `_track_and_reduce` has to convert them back to array indices using `snap_data.array_index(sim_arr)` with a `-1` check for not-found particles
3. The `HDF5AssignmentWriter` writes out array indices (since it gets `AssignmentResult` which is in array index space), but `BirthTracker.finalize()` returns sim IDs labeled as "particle_index"
4. The `_to_sim_space`/`_from_sim_space` methods are static methods on `AccretionPipeline` but they're really just `SparseCSC.remap_rows()` calls - the pipeline shouldn't own this logic

### 2. SnapshotProcessor: Does too much and too little at the same time

**Violations of single responsibility:**
- `SnapshotProcessor.process()` creates `HaloEnsemble`, runs boundness, does segmentation, calls assigner, then computes timescales - this is orchestration, not "one thing"
- But it also DOESN'T handle the tracker updates or the postprocessing (properties/riley), which are done by `AccretionPipeline`
- This creates a weird split where:
  - `SnapshotProcessor` does: boundness → segment → assign → timescales
  - `AccretionPipeline` does: load data → coordinate translation → call SnapshotProcessor → update trackers → compute properties → compute riley → write output
- The `SnapshotProcessor` also creates `HaloEnsemble` internally (lines 43-46), which is a factory responsibility mixed into orchestration

**Specific issues:**
- `SnapshotProcessor` returns `ensemble` AND `result`, but doesn't do anything with the ensemble itself - it's just returned for the pipeline to use. This suggests the ensemble is part of a different concern.
- The `process()` method takes `previous_resp` but doesn't deal with the sim-ID ↔ array-index translation. The pipeline has to handle that separately.
- The SnapshotProcessor stores `accretion_id`, `n_los`, `halo_model` which are really config values, not processor state. This is configuration injection mixed into the processor.

### 3. AccretionPipeline: God class in disguise

Despite being named "pipeline", `AccretionPipeline` does way too much:
1. **ID translation** (`_to_sim_space`, `_from_sim_space`) - should be in SnapshotData or a dedicated coordinator
2. **Tracker updates** (`_update_trackers`) - mixes sim ID / array index concerns
3. **Postprocessing orchestration** (`_track_and_reduce`) - 60 lines of code that builds `galaxy_particles`, `galaxy_bound`, `galaxy_table`, calls `compute_galaxy_properties` and `compute_riley_criterion`
4. **I/O coordination** (writing catalogue, particles, assignment)
5. **Logging**
6. **Error handling & checkpointing**
7. **Newborn detection** (lines 263-268) - this is a pure function that happens inside the pipeline loop
8. **Resume logic** (lines 198-207)

The `_track_and_reduce` method is particularly problematic:
- It takes 5 arguments and does complex sim-ID ↔ array-index translation
- It rebuilds `galaxy_particles` and `galaxy_bound` from scratch using `assembly_tracker.current()` + `bound_csc`
- It builds `galaxy_table`, `host_props` from `snap_df`
- It calls `compute_galaxy_properties` and `compute_riley_criterion`
- This is essentially a second "process" step that lives in the pipeline instead of the processor

### 4. Trackers: Sim ID space leakage

The `BirthTracker` operates in sim ID space (it receives `particle_ids=sim_ids`), which is correct for persistence across snapshots. But:
- Its `finalize()` method labels the output column as `"particle_index"` when the values are actually sim IDs. This is misleading.
- The `current_birth_map()` returns `{galaxy_id: set(particle_sim_ids)}` which the `AssemblyTracker` consumes directly
- The `AssemblyTracker` then passes sim IDs through, and `_track_and_reduce` has to convert them back to array indices

### 5. Data flow is tangled

The data flow for a single snapshot is:
1. `AccretionPipeline.run()` loads snapshot → gets `SnapshotData` (indices = sim IDs)
2. Translates `previous_resp` from sim ID → array index
3. Computes newborns in array index space
4. Calls `SnapshotProcessor.process()` which works in array index space
5. Gets back `ensemble` (array index) + `result` (array index)
6. `_update_trackers`: converts result.particle_df["array_index"] → sim IDs for trackers
7. `_track_and_reduce`: converts assembly_tracker.current() sim IDs → array indices, intersects with bound_csc array indices
8. `_to_sim_space`: converts result.responsibilities → sim IDs for next snapshot

This ping-pong between spaces is fragile and error-prone. The conversion happens in at least 4 different places.

### 6. _track_and_reduce is doing work that should be in SnapshotProcessor or a dedicated class

The `_track_and_reduce` method:
- Rebuilds `galaxy_particles` by intersecting assembly tracker's sim-ID sets with bound CSC's array-index data
- This intersection requires converting sim IDs → array indices AND comparing with array indices from bound CSC
- It also builds `galaxy_bound` from the same data
- This is essentially a "reduce" or "aggregate" operation that should be encapsulated

### 7. The `particle_df` DataFrame is a leaky abstraction

`AssignmentResult.particle_df` contains `array_index` and `Sub_tree_id` columns. The `array_index` is always just `np.arange(N)` (see gmm.py line 90), making it redundant. The actual useful data is the `Sub_tree_id` mapping, but it's stored in a DataFrame with an index that's just 0..N-1.

### 8. AccretionPipeline directly accesses internal state of its dependencies

- `self.logger._log_path` (line 192) - accessing private attribute
- `self.processor.accretion_id` (line 139) - reaching into processor's config
- `self.processor.halo_model` (line 155) - same
- `self.processor.n_los` (line 156) - same
- `birth_tracker._finalized` (line 342) - accessing private state directly for timescale extraction

### 9. Step 8 (I/O layer) notes

The I/O layer is mostly clean, but:
- The pipeline directly calls `cat_writer`, `part_writer`, `assign_writer` with ad-hoc parameter lists - there's no unified output interface
- The `HDF5CatalogueWriter`, `HDF5ParticleWriter`, `HDF5AssignmentWriter` are separate classes but they're always used together and their creation/coordination is in `entry.py`

### 10. The newborn detection logic is buried in the pipeline loop

Lines 263-268 of AccretionPipeline:
```python
if previous_resp is None:
    newborn = np.arange(N, dtype=np.uint64)
else:
    existing = set(previous_resp.row_id)
    newborn = np.array([i for i in range(N) if i not in existing], dtype=np.uint64)
```
This is a pure computation that should be a standalone function (or method on SnapshotData/SparseCSC).

### Summary of key philosophical violations:

1. **SnapshotProcessor does orchestration, not a single thing** - it's a mini-pipeline itself
2. **AccretionPipeline is a god class** - ID translation, tracker updates, postprocessing, I/O, logging, checkpointing, resume, newborn detection
3. **Array index / sim ID space is handled haphazardly** - conversions scattered across pipeline, trackers, and SparseCSC
4. **_track_and_reduce mixes concerns** - it's doing data reduction, ID translation, and postprocessing orchestration
5. **Private state access** - pipeline reaches into private internals of its dependencies
6. **particle_df is redundant** - array_index column is always 0..N-1

Now I have a thorough picture. Let me also quickly verify the test structure and the HDF5ParticleWriter to confirm the last pieces:

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/src/roadrunner/io/hdf5_particles.py"
}
```

**Output:**
```
<path>/home/asier/roadrunner/src/roadrunner/io/hdf5_particles.py</path>
<type>file</type>
<content>
1: import os
2: 
3: import h5py
4: import numpy as np
5: 
6: from roadrunner.helpers import select_float_dtype, select_uint_dtype
7: from roadrunner.physics.scaler import StandardScaler
8: 
9: 
10: class HDF5ParticleWriter:
11:     def __init__(self, output_dir, float_atol=1e-4):
12:         os.makedirs(output_dir, exist_ok=True)
13:         self._path = os.path.join(output_dir, "particles.hdf5")
14:         self._float_atol = float_atol
15: 
16:     def write_snapshot(self, snapshot_id, time, redshift, snapshot_data):
17:         with h5py.File(self._path, "a") as hf:
18:             snap_grp = hf.require_group(f"/snapshots/{snapshot_id}")
19:             snap_grp.attrs["time"] = float(time)
20:             snap_grp.attrs["redshift"] = float(redshift)
21: 
22:             coords = np.column_stack([
23:                 snapshot_data.positions,
24:                 snapshot_data.velocities,
25:             ])
26: 
27:             scaler = StandardScaler()
28:             scaled = scaler.fit_transform(coords.astype(np.float64, copy=False))
29: 
30:             scaler_grp = snap_grp.require_group("scaler")
31:             scaler_grp.create_dataset("mean", data=scaler.mean_)
32:             scaler_grp.create_dataset("scale", data=scaler.scale_)
33: 
34:             float_dtype = select_float_dtype(
35:                 max(abs(scaled).max(), 1e-10), self._float_atol
36:             )
37: 
38:             snap_grp.create_dataset(
39:                 "positions",
40:                 data=scaled[:, :3].astype(float_dtype, copy=False),
41:                 compression="gzip",
42:             )
43:             snap_grp.create_dataset(
44:                 "velocities",
45:                 data=scaled[:, 3:6].astype(float_dtype, copy=False),
46:                 compression="gzip",
47:             )
48: 
49:             mass_dtype = select_float_dtype(
50:                 float(snapshot_data.masses.max()), self._float_atol
51:             )
52:             snap_grp.create_dataset(
53:                 "masses",
54:                 data=snapshot_data.masses.astype(mass_dtype, copy=False),
55:                 compression="gzip",
56:             )
57: 
58:             idx_dtype = select_uint_dtype(int(snapshot_data.indices.max()))
59:             snap_grp.create_dataset(
60:                 "indices",
61:                 data=snapshot_data.indices.astype(idx_dtype, copy=False),
62:                 compression="gzip",
63:             )

(End of file - total 63 lines)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/asier/roadrunner/tests"
}
```

**Output:**
```
<path>/home/asier/roadrunner/tests</path>
<type>directory</type>
<entries>
benchmarks/
data/
integration/
unit/

(4 entries)
</entries>
```

---

## Assistant (Plan · GLM-5.1 · 41.2s)

_Thinking:_

I now have a comprehensive understanding. Let me formulate my analysis of the main weak points.

Here is my analysis of the main weak points and philosophical violations, particularly from Phase 9 onwards:

---

## Main Weak Points / Philosophical Violations

### 1. Array Index vs Sim ID: Scattered, Not Encapsulated (CRITICAL)

The two coordinate systems (array index `0..N-1` and persistent simulation ID) are converted in **at least 5 different places**, none of which "owns" the translation:

| Location | Direction | How |
|---|---|---|
| `AccretionPipeline._to_sim_space()` | array→sim | `SparseCSC.remap_rows()` via `SnapshotData.index_to_id_map()` |
| `AccretionPipeline._from_sim_space()` | sim→array | `SparseCSC.remap_rows()` via `SnapshotData.id_to_index_map()` |
| `AccretionPipeline._update_trackers()` line 88 | array→sim | `snap_data.indices[arr_idx]` |
| `AccretionPipeline._update_trackers()` line 99 | array→sim | `snap_data.indices[arr_idx]` again (duplicate pattern) |
| `AccretionPipeline._track_and_reduce()` line 125 | sim→array | `snap_data.array_index(sim_arr)` with manual `-1` guard |

This is the core source of fragility. The translation is conceptually "SnapshotData's job", but the pipeline drives it imperatively each time. Worse:
- `BirthTracker.finalize()` labels its output column `"particle_index"` when the values are **sim IDs** (`birth.py:139`). This is actively misleading.
- The checkpoint stores `previous_resp` in sim ID space, but the processor consumes array index space — the pipeline must translate at the boundary of each snapshot.

**Verdict**: There should be a single abstraction boundary where the conversion happens. Right now it leaks everywhere.

---

### 2. AccretionPipeline Is a God Class in Disguise

`AccretionPipeline.run()` is ~130 lines that does **7 distinct things**:

1. **ID space translation** (`_to_sim_space`, `_from_sim_space`) — should be a `SnapshotData` responsibility or a thin adapter
2. **Newborn detection** (lines 263–268) — a pure function, buried in the loop
3. **Tracker updates** (`_update_trackers`) — mixes index translation with tracker logic
4. **Postprocessing orchestration** (`_track_and_reduce`) — 60 lines that rebuild `galaxy_particles`, `galaxy_bound`, call properties + riley. This is a second "process" step.
5. **I/O coordination** — three separate writers called with ad-hoc argument lists
6. **Checkpointing / resume** — error handling + serialization concerns
7. **Logging**

This violates **Principle 1** (stateless→function, with state→class) and **Principle 2** (single responsibility). The original god class `BaseAccretionHistory` was decomposed into many clean modules (phases 1–8), but the pipeline re-concentrates too many concerns back into one place.

---

### 3. SnapshotProcessor: Orchestration Posing as a Single Responsibility

`SnapshotProcessor.process()` does: create `HaloEnsemble` → run boundness → segment → assign → compute timescales. This is **5 steps of orchestration**, not one thing. Meanwhile it does **not** handle the tracker updates or postprocessing, which are awkwardly split into `AccretionPipeline._update_trackers()` and `_track_and_reduce()`.

The result is a confusing split where:
- `SnapshotProcessor` does "physics orchestration" (boundness → assign → timescales)
- `AccretionPipeline` does "post-assignment orchestration" (trackers → properties → riley) plus everything else

Neither class has a clean, single purpose. The `SnapshotProcessor` also:
- Stores `accretion_id`, `n_los`, `halo_model` — these are **config**, not processor state
- Returns `ensemble` only so the pipeline can extract `bound_csc` from it (the ensemble is leaked as implementation detail)
- Creates `HaloEnsemble` internally from `snap_df.iterrows()` — factory logic mixed into orchestration

---

### 4. `_track_and_reduce` Is the Worst Offender

`AccretionPipeline._track_and_reduce()` (lines 110–169) is 60 lines that:
1. Gets `bound_csc` from ensemble (array index space)
2. Gets `assembly_map` from `assembly_tracker.current()` (sim ID space)
3. Builds `sid_to_col` mapping to translate between them
4. Converts sim IDs → array indices via `snap_data.array_index()` with `-1` guard
5. Computes `galaxy_particles` (intersection of assembly sim IDs ∩ bound array indices)
6. Computes `galaxy_bound` (raw bound indices per galaxy)
7. Builds `galaxy_table` from `snap_df`
8. Extracts `host_props`
9. Extracts `galaxy_centers` from `fitted_parameters`
10. Calls `compute_galaxy_properties()` and `compute_riley_criterion()`

This is **data reduction + ID translation + postprocessing orchestration** crammed into one method. It's the kind of thing that should be a class (or at least a function with a clear data-in/data-out signature).

---

### 5. Private State Access Across Classes

`AccretionPipeline` directly reaches into private internals:
- `self.logger._log_path` (`accretion_pipeline.py:192`) — private attribute
- `birth_tracker._finalized` (`accretion_pipeline.py:342`) — private dict, iterated to extract timescales
- `self.processor.accretion_id`, `.halo_model`, `.n_los` — the pipeline reaches into the processor's config instead of having its own copy or receiving it as shared config

This breaks encapsulation and creates hidden coupling.

---

### 6. Redundant `particle_df["array_index"]`

In `GMMAssigner.assign()` (gmm.py:90), the `particle_df` is created with `array_index = np.arange(N)`. This column is always just `0..N-1` and is redundant — it's used only as a carrier for `Sub_tree_id`. The actual mapping is done via `SparseCSC`. The `array_index` column adds noise and makes it unclear whether the DataFrame operates in array or sim space.

---

### 7. Newborn Detection Should Be a Pure Function

Lines 263–268 in `AccretionPipeline`:
```python
if previous_resp is None:
    newborn = np.arange(N, dtype=np.uint64)
else:
    existing = set(previous_resp.row_id)
    newborn = np.array([i for i in range(N) if i not in existing], dtype=np.uint64)
```
This is a pure computation (set difference) with no pipeline state, buried in the loop body. It should be a standalone function like `detect_newborn_particles(N, previous_resp)`.

---

### 8. Minor Issues in Phases 8 and Below

- **I/O layer (Phase 8)**: Three separate writer classes (`HDF5CatalogueWriter`, `HDF5ParticleWriter`, `HDF5AssignmentWriter`) are always constructed together in `entry.py` and called sequentially in the pipeline. They could share a facade or at least a common `OutputManager` that coordinates the three files.
- **`HaloEnsemble.get_particles()`** is called multiple times per snapshot (by `SnapshotProcessor`, by `AccretionPipeline._track_and_reduce`, by `AccretionPipeline` I/O writing). Each call rebuilds `SparseCSC` from scratch. This should be cached or returned once.
- **`compute_particle_dynamical_timescales`** mutates `particle_df` in place (timescales.py:5) — violates the "pure function" ideal for a stateless module.

---

### Summary of Root Causes

| Root Cause | Manifests As |
|---|---|
| **No clear "space boundary" abstraction** | Array↔sim ID translation scattered in 5+ places |
| **SnapshotProcessor + AccretionPipeline split is wrong** | Physics orchestration in one, post-assignment orchestration in the other, plus pipeline does 5 other things |
| **Trackers operate in sim ID space but consumers need array indices** | `_track_and_reduce` is a complex adapter between spaces |
| **Config distributed across classes** | Pipeline reaches into processor's `accretion_id`, `halo_model`, `n_los` |
| **No "reduce/aggregate" step** | `_track_and_reduce` does data reduction + postprocessing in one tangled method |

The core refactoring need is: **define a clean boundary between array-index space and sim-ID space, and put the conversion in exactly one place** (likely `SnapshotData` or a thin wrapper). Then restructure `SnapshotProcessor` and `AccretionPipeline` so that each has a single, well-defined responsibility — likely extracting `_track_and_reduce` into its own unit, moving newborn detection to a pure function, and having the pipeline be a thin orchestrator that wires pre/post-processing steps together.

---

## User

Create or update `AGENTS.md` for this repository.

The goal is a compact instruction file that helps future OpenCode sessions avoid mistakes and ramp up quickly. Every line should answer: "Would an agent likely miss this without help?" If not, leave it out.

User-provided focus or constraints (honor these):


## How to investigate

Read the highest-value sources first:
- `README*`, root manifests, workspace config, lockfiles
- build, test, lint, formatter, typecheck, and codegen config
- CI workflows and pre-commit / task runner config
- existing instruction files (`AGENTS.md`, `CLAUDE.md`, `.cursor/rules/`, `.cursorrules`, `.github/copilot-instructions.md`)
- repo-local OpenCode config such as `opencode.json`

If architecture is still unclear after reading config and docs, inspect a small number of representative code files to find the real entrypoints, package boundaries, and execution flow. Prefer reading the files that explain how the system is wired together over random leaf files.

Prefer executable sources of truth over prose. If docs conflict with config or scripts, trust the executable source and only keep what you can verify.

## What to extract

Look for the highest-signal facts for an agent working in this repo:
- exact developer commands, especially non-obvious ones
- how to run a single test, a single package, or a focused verification step
- required command order when it matters, such as `lint -> typecheck -> test`
- monorepo or multi-package boundaries, ownership of major directories, and the real app/library entrypoints
- framework or toolchain quirks: generated code, migrations, codegen, build artifacts, special env loading, dev servers, infra deploy flow
- repo-specific style or workflow conventions that differ from defaults
- testing quirks: fixtures, integration test prerequisites, snapshot workflows, required services, flaky or expensive suites
- important constraints from existing instruction files worth preserving

Good `AGENTS.md` content is usually hard-earned context that took reading multiple files to infer.

## Questions

Only ask the user questions if the repo cannot answer something important. Use the `question` tool for one short batch at most.

Good questions:
- undocumented team conventions
- branch / PR / release expectations
- missing setup or test prerequisites that are known but not written down

Do not ask about anything the repo already makes clear.

## Writing rules

Include only high-signal, repo-specific guidance such as:
- exact commands and shortcuts the agent would otherwise guess wrong
- architecture notes that are not obvious from filenames
- conventions that differ from language or framework defaults
- setup requirements, environment quirks, and operational gotchas
- references to existing instruction sources that matter

Exclude:
- generic software advice
- long tutorials or exhaustive file trees
- obvious language conventions
- speculative claims or anything you could not verify
- content better stored in another file referenced via `opencode.json` `instructions`

When in doubt, omit.

Prefer short sections and bullets. If the repo is simple, keep the file simple. If the repo is large, summarize the few structural facts that actually change how an agent should work.

If `AGENTS.md` already exists at `/home/asier/roadrunner`, improve it in place rather than rewriting blindly. Preserve verified useful guidance, delete fluff or stale claims, and reconcile it with the current codebase.

---

## Assistant (Plan · GLM-5.1 · 8.9s)

_Thinking:_

The user wants me to create or update an `AGENTS.md` file for this repository. I'm in plan mode, so I need to first investigate the repository thoroughly, then plan what should go into the AGENTS.md file. Let me gather all the necessary information first.

I already have a good understanding of the repository from the previous conversation, but let me check for existing instruction files and other config files I haven't looked at yet.

**Tool: glob**

**Input:**
```json
{
  "pattern": "AGENTS.md"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "CLAUDE.md"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": ".cursorrules"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": ".cursor/rules/**"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": ".github/copilot-instructions.md"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "opencode.json"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "opencode.jsonc"
}
```

**Output:**
```
No files found
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "README*"
}
```

**Output:**
```
No files found
```

---

## Assistant (Plan · GLM-5.1 · 5.3s)

---


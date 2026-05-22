# Level 1 Plan: SnapshotOrchestrator — Matrioska Implementation

**Goal**: Build Level 1 (per-snapshot orchestrator) inside-out: core → boundaries → API shell.

---

## BIG STEP 1: Processing Core

*Pure array-space function — no sim IDs, no trackers, no pipeline knowledge.*

### 1.1
Create `src/roadrunner/pipeline/__init__.py` (empty placeholder)

### 1.2
Create `src/roadrunner/pipeline/processing.py` with:
- `ProcessingConfig` dataclass: `search_factor: float = 1.0`, `min_particles: int = 10`
- `process_snapshot(snap_data, snap_df, newborn, previous_resp, assigner, config) → (HaloEnsemble, AssignmentResult)`:
  - Body extracted from `SnapshotProcessor.process()` (lines 35–76 of `snapshot_processor.py`)
  - HaloEnsemble built from `snap_df` via `HaloModel.from_snapshot_row()`
  - Calls: `compute_halo_bound_particles`, `HaloSegmenter`, `assigner.assign`, `compute_particle_dynamical_timescales`
  - All indices are array-space; `accretion_id` not needed here (it's only in reduction)

### 1.3
Unit test `tests/unit/test_processing.py`: mock `SnapshotData` (3 halos, ~30 particles), mock `ParticleAssigner`, verify output shapes/types

---

## BIG STEP 2: Reduction Core

*Pure function computing galaxy properties + dynstate from prepared inputs. No tracker knowledge, no sim↔array translation.*

### 2.1
Create `src/roadrunner/pipeline/reduction.py` with:
- `ReductionConfig` dataclass: `accretion_id: int`, `halo_model: str`, `n_los: int = 11`
- `reduce_snapshot(snap_data, snap_df, ensemble, result, galaxy_particles, galaxy_bound, satellites, config) → (DataFrame, DataFrame)`:
  - Extracts `host_props` from `snap_df` at `config.accretion_id`
  - Extracts `galaxy_centers` from `result.fitted_parameters`
  - Builds `galaxy_table` from `snap_df[["Sub_tree_id", "host_id", "mass", "distance_to_acc_id"]]`
  - Delegates to existing `compute_galaxy_properties()` + `compute_riley_criterion()`
  - All `galaxy_particles`/`galaxy_bound` keys are sim IDs, values are array indices (already the interface of `compute_galaxy_properties` and `compute_riley_criterion`)

### 2.2
Unit test `tests/unit/test_reduction.py`: mock input dicts, verify output columns (`Sub_tree_id`, `Mtot`, `dynstate`, etc.)

---

## BIG STEP 3: Space Translation & Newborn Detection

*Thin pure functions extracted from `AccretionPipeline` static methods and loop body. No class, no state.*

### 3.1
Create `src/roadrunner/pipeline/translation.py` with:
- `responsibilities_to_sim(resp_csc, snap_data) → SparseCSC | None` — extracted from `AccretionPipeline._to_sim_space()`: `csc.remap_rows(*snap_data.index_to_id_map())`, returns None for empty
- `responsibilities_from_sim(resp_csc, snap_data) → SparseCSC | None` — extracted from `AccretionPipeline._from_sim_space()`: `csc.remap_rows(*snap_data.id_to_index_map())`, returns None for empty
- `detect_newborns(previous_resp_array, n_particles) → np.ndarray` — extracted from pipeline loop: if None → `np.arange(N, dtype=np.uint64)`, else set-difference complement of `previous_resp.row_id`. Returns array-space uint64 indices

### 3.2
Unit test `tests/unit/test_translation.py`: round-trip sim↔array on small `SparseCSC`; `detect_newborns` with None and with known `previous_resp`

---

## BIG STEP 4: Tracker Boundary Adapters

*Bridge between array-space core output and sim-space trackers, and between tracker output and reduction input.*

### 4.1
Add to `src/roadrunner/pipeline/translation.py` (same file, same conceptual layer):

- `update_birth_tracker(birth_tracker, snap_id, snap_data, result)`:
  - Extracts `result.particle_df["array_index"]` and `result.particle_df["Sub_tree_id"]`
  - Translates `array_index → sim ID` via `snap_data.indices[arr_idx]`
  - Extracts timescales from `result.particle_df.get("timescale", ...)`
  - Calls `birth_tracker.update(t_snap, snap_id, sim_ids, host_ids, timescales)`

- `update_assembly_tracker(assembly_tracker, snap_id, snap_data, result, satellites, birth_tracker)`:
  - Groups `result.particle_df` by `Sub_tree_id`; translates `array_index → sim ID` per group → `assignment_map: dict[int, set[int]]` (sim-space)
  - Gets `birth_map = birth_tracker.current_birth_map()` if birth_tracker exists, else `{}`
  - Calls `assembly_tracker.update(snap_id, assignment_map, birth_map, satellites)`

- `build_reduction_input(snap_data, ensemble, result, assembly_tracker) → (galaxy_particles, galaxy_bound)`:
  - Gets `bound_csc` from `ensemble.get_particles()`; builds `sid_to_col = {sid: i for i, sid in enumerate(bound_csc.column_id)}`
  - Gets `assembly_map = assembly_tracker.current()` (sim-space)
  - For each `(gid, sim_set)` in assembly_map: translates `sim_set → array_index` via `snap_data.array_index()`, intersects with `bound_csc.column_indices[sid_to_col[gid]]`
  - Returns `(galaxy_particles: dict[int, ndarray], galaxy_bound: dict[int, ndarray])` where keys are sim IDs (galaxy Sub_tree_ids) and values are array indices

### 4.2
Unit test additions to `tests/unit/test_translation.py`:
- Mock `BirthTracker` and `AssignmentResult`; verify `update_birth_tracker` passes sim IDs (not array indices)
- Mock `AssemblyTracker` + `HaloEnsemble`; verify `build_reduction_input` produces correct intersection

---

## BIG STEP 5: SnapshotOrchestrator Class

*The shell that owns config + trackers and wires Steps 1–4 in the correct sequence. Sim-space in, sim-space out.*

### 5.1
Create `src/roadrunner/pipeline/snapshot_orchestrator.py` with:

- `SnapshotResult` dataclass: `ensemble`, `result`, `previous_resp_sim` (SparseCSC | None), `properties` (DataFrame | None), `dynstate` (DataFrame | None)

- `SnapshotOrchestrator` class:
  - `__init__(self, processing_config, reduction_config, assigner, birth_tracker=None, assembly_tracker=None)`
  - Stores: `ProcessingConfig`, `ReductionConfig`, `ParticleAssigner` (immutable); `BirthTracker | None`, `AssemblyTracker | None` (mutable)
  - `process_snapshot(self, snap_id, snap_df, snap_data, satellites, previous_resp_sim=None) → SnapshotResult`:
    1. `previous_resp = responsibilities_from_sim(previous_resp_sim, snap_data)` — Step 3 boundary
    2. `newborn = detect_newborns(previous_resp, snap_data.positions.shape[0])` — Step 3
    3. `ensemble, result = process_snapshot(snap_data, snap_df, newborn, previous_resp, self.assigner, self.processing_config)` — Step 1 core
    4. If `self.birth_tracker`: `update_birth_tracker(...)` — Step 4 adapter
    5. If `self.assembly_tracker`: `update_assembly_tracker(...)` — Step 4 adapter
    6. If `self.assembly_tracker`:
       - `galaxy_particles, galaxy_bound = build_reduction_input(snap_data, ensemble, result, self.assembly_tracker)` — Step 4
       - `properties, dynstate = reduce_snapshot(...)` — Step 2 core
    7. Else: `properties, dynstate = None, None`
    8. `previous_resp_sim_out = responsibilities_to_sim(result.responsibilities, snap_data)` — Step 3 boundary
    9. Return `SnapshotResult(ensemble, result, previous_resp_sim_out, properties, dynstate)`

### 5.2
Integration test `tests/integration/test_snapshot_orchestrator.py`:
- Mock/small `snap_data` + `snap_df`, real `GMMAssigner`, real `BirthTracker` + `AssemblyTracker`
- Verify: calling `process_snapshot` twice accumulates tracker state
- Verify: `previous_resp_sim` round-trips correctly (sim-space in → sim-space out)
- Verify: `properties` and `dynstate` DataFrames have expected columns

---

## BIG STEP 6: Module Wiring & API Contract

*Connect to package, ensure Level 2 can import cleanly.*

### 6.1
Copy `RunConfig` dataclass to `src/roadrunner/pipeline/config.py` (from `__pipeline/config.py`, unchanged)

### 6.2
Populate `src/roadrunner/pipeline/__init__.py` exports: `SnapshotOrchestrator`, `SnapshotResult`, `ProcessingConfig`, `ReductionConfig`, `RunConfig`

### 6.3
Update `entry.py` to import from `roadrunner.pipeline` and construct `SnapshotOrchestrator` from `RunConfig` (replace old `SnapshotProcessor` usage). Keep old `AccretionPipeline` — it will be rewritten in Level 2 later.

### 6.4
Run `ruff check src/roadrunner/pipeline/` — fix any lint issues

### 6.5
Run `pytest tests/ -v` — ensure no regressions

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `process_snapshot` is a function, not a class | Stateless processing; config is a dataclass, assigner injected |
| `reduce_snapshot` is a function, not a class | Stateless reduction; config is a dataclass |
| Space translation lives in `translation.py` as free functions | No state, no class — pure mapping utilities |
| Tracker adapters are free functions in `translation.py` | Thin wrappers; they don't own state, just translate at the boundary |
| `SnapshotOrchestrator` owns mutable trackers | Only mutable state in Level 1; everything else is immutable config |
| `SnapshotResult` is a dataclass return type | Clean contract with Level 2 — no leaking internals |
| `accretion_id` lives in `ReductionConfig`, not `ProcessingConfig` | Processing doesn't need it; only reduction (for host row lookup) needs it |
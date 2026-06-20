"""Pipeline orchestration: processing, reduction, and coordinate translation."""

from .processing import ProcessingConfig, process_snapshot
from .reduction import ReductionConfig, reduce_snapshot
from .translation import (
    responsibilities_to_sim,
    responsibilities_from_sim,
    detect_newborns,
    update_birth_tracker,
    update_assembly_tracker,
    build_reduction_input,
)
from .snapshot_orchestrator import SnapshotOrchestrator, SnapshotResult
from .config import RunConfig
from .accretion_pipeline import AccretionPipeline
from .entry import run_accretion_history

__all__ = [
    "ProcessingConfig",
    "process_snapshot",
    "ReductionConfig",
    "reduce_snapshot",
    "responsibilities_to_sim",
    "responsibilities_from_sim",
    "detect_newborns",
    "update_birth_tracker",
    "update_assembly_tracker",
    "build_reduction_input",
    "SnapshotOrchestrator",
    "SnapshotResult",
    "RunConfig",
    "AccretionPipeline",
    "run_accretion_history",
]
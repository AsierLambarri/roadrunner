#############################################################################
#
# package:   roadrunner
# file:      __init__.py
# brief:     Public API and symbol exports for the roadrunner package.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   12 May 2026 - Created
#
#############################################################################

"""roadrunner: Unravel the accretion history of simulated galaxies.

This package provides tools for analysing the assembly history of
simulated galaxies by combining halo merger trees with particle data.
The pipeline supports multiple assignment methods (GMM, variational
Bayesian GMM, and stochastic variational inference), track-based
birth and assembly tracking, and a variety of I/O backends.

Key modules
-----------
pipeline : Full accretion-history pipeline orchestration.
mixture : Gaussian mixture models (standard, variational Bayesian, SVI).
clustering : Sparse matrix formats, halo segmentation, and the XGMM assigner.
physics : Halo models, potentials, boundness, and scaling relations.
io : HDF5 writers and checkpoint serialization.
readers : Data ingestion for merger trees, snapshot files, and equivalence tables.
postprocessing : Galaxy property computation, centering, and dynamical state.
"""

from .pipeline import (
    run_accretion_history,
    RunConfig,
    AccretionPipeline,
    SnapshotOrchestrator,
    SnapshotResult,
    ProcessingConfig,
    ReductionConfig,
    process_snapshot,
    reduce_snapshot,
    responsibilities_to_sim,
    responsibilities_from_sim,
    detect_newborns,
    update_birth_tracker,
    update_assembly_tracker,
    build_reduction_input,
)
from .readers.merger_tree import MergerTreeReaderCSV
from .readers.snapshot import SnapshotReader
from .readers.equivalence import EquivalenceTable
from .mixture.weighted_gmm import WeightedGaussianMixture
from .mixture.bayesian_gmm import WeightedBayesianGaussianMixture
from .mixture.coresets import GaussianCoreset
from .postprocessing.tracking.birth import BirthTracker
from .postprocessing.tracking.assembly import AssemblyTracker
from .postprocessing.mixing import compute_riley_criterion
from .postprocessing.properties import compute_galaxy_properties
from .clustering.assignment.gmm import XGMMAssigner
from .physics.halo_model import HaloModel
from .physics.potentials import get_potential, KeplerPotential, NFWPotential

__all__ = [
    "run_accretion_history",
    "RunConfig",
    "AccretionPipeline",
    "SnapshotOrchestrator",
    "SnapshotResult",
    "ProcessingConfig",
    "ReductionConfig",
    "process_snapshot",
    "reduce_snapshot",
    "responsibilities_to_sim",
    "responsibilities_from_sim",
    "detect_newborns",
    "update_birth_tracker",
    "update_assembly_tracker",
    "build_reduction_input",
    "MergerTreeReaderCSV",
    "SnapshotReader",
    "EquivalenceTable",
    "WeightedGaussianMixture",
    "WeightedBayesianGaussianMixture",
    "GaussianCoreset",
    "BirthTracker",
    "AssemblyTracker",
    "compute_riley_criterion",
    "compute_galaxy_properties",
    "XGMMAssigner",
    "HaloModel",
    "get_potential",
    "KeplerPotential",
    "NFWPotential",
]
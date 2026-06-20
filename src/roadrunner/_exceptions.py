#############################################################################
#
# package:   roadrunner
# file:      _exceptions.py
# brief:     Custom exception types for the roadrunner package.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   12 May 2026 - Created
#            18 May 2026 - Last edit
#
#############################################################################

"""Package-specific exception types."""


class RoadrunnerError(Exception):
    """Base exception for all roadrunner-specific errors."""


class ConfigurationError(RoadrunnerError):
    """Raised when the pipeline configuration is invalid or inconsistent."""


class SnapshotLoadError(RoadrunnerError):
    """Raised when a snapshot file cannot be read or parsed."""


class NoBoundParticlesError(RoadrunnerError):
    """Raised when a halo has no bound particles within the search radius."""


class ConvergenceError(RoadrunnerError):
    """Raised when an iterative solver fails to converge within the allowed
    number of iterations."""


class RestartError(RoadrunnerError):
    """Raised when a checkpoint-based restart cannot proceed.

    Indicates a corrupt or missing checkpoint file that prevents
    the pipeline from resuming a previous run.
    """


class CycleError(RoadrunnerError):
    """Raised when a cyclic dependency is detected in a data structure.

    Typically used in tree processing when a node references itself
    indirectly through its ancestry.
    """

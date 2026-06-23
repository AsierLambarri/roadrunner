#############################################################################
#
# package:   roadrunner
# file:      _mcf_types.py
# brief:     Core data contracts, protocols, and result types.
#
# Defines the fundamental data carriers used across the entire pipeline:
# SnapshotData for particle-frame data, BoundnessResult for binding
# information, AssignmentResult for mixture-model fitting output, and
# runtime-checkable protocols (ParticleAssigner, PotentialModel,
# AssignmentStatistics) that define the assigner and physics interfaces.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   12 May 2026 - Created
#            21 May 2026 - Last edit
#
#############################################################################

"""Core data types and runtime protocols for the roadrunner pipeline.

.. highlight:: python

The types in this module are used as internal contracts between
pipeline stages. Every assigner must implement the ParticleAssigner
protocol, and every potential must implement the PotentialModel
protocol.
"""

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from roadrunner.clustering.sparse import SparseCSC

RespMap = dict[int, tuple[np.ndarray, np.ndarray]]


class SnapshotData:
    """Container for a single snapshot's particle data.

    Stores per-particle data as attributes and maintains two registries:
    ``.fields`` (all particle-field names) and ``.extra_fields`` (field
    names that are saved to disk but not passed to the assigner).  The
    ``.data`` property returns the column-concatenated array of whichever
    fields are designated for assignment.

    All parameters are keyword-only for clarity.

    Parameters
    ----------
    indices : NDArray of int
        Global simulation particle IDs.
    masses : NDArray of float
        Particle masses.
    positions : NDArray of float, shape (n, 3)
        Spatial coordinates.
    velocities : NDArray of float, shape (n, 3)
        Velocities.
    redshift : float
        Snapshot redshift.
    time : float
        Cosmic time.
    assign_fields : list of str, optional
        Attribute names that make up the assigner input array
        (``.data``).  Defaults to ``["positions", "velocities"]``.
    **kwargs : NDArray
        Arbitrary additional per-particle fields (e.g. ``metallicity``,
        ``birth_density``).  Each becomes an instance attribute and is
        automatically tracked in ``.extra_fields`` unless listed in
        ``assign_fields``.
    """

    indices: NDArray[np.integer]
    masses: NDArray[np.floating]
    positions: NDArray[np.floating]
    velocities: NDArray[np.floating]
    redshift: float
    time: float
    fields: list[str]
    extra_fields: list[str]
    _assign_fields: list[str]

    def __init__(
        self,
        *,
        indices,
        masses,
        positions,
        velocities,
        redshift,
        time,
        assign_fields=None,
        **kwargs,
    ):
        self.indices = np.asarray(indices)
        self.masses = np.asarray(masses)
        self.positions = np.asarray(positions)
        self.velocities = np.asarray(velocities)
        self.redshift = float(redshift)
        self.time = float(time)

        for name, arr in kwargs.items():
            setattr(self, name, np.asarray(arr))

        # ── Field registries ──────────────────────────────────────────
        all_fields = ["indices", "masses", "positions", "velocities"] + list(kwargs.keys())
        self.fields = all_fields

        if assign_fields is not None:
            self._assign_fields = list(assign_fields)
        else:
            self._assign_fields = ["positions", "velocities"]

        _internal = {"indices", "masses"}
        self.extra_fields = [
            f for f in self.fields
            if f not in self._assign_fields and f not in _internal
        ]

        self._index_sorter: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        """Column-concatenated array of fields used by the assigner.

        Returns
        -------
        coords : ndarray of shape (n_particles, D)
            Where D is the sum of the last dimensions of each field
            in ``assign_fields``.
        """
        arrays = [getattr(self, f) for f in self._assign_fields]
        return np.column_stack(arrays)

    def array_index(self, sim_ids: np.ndarray) -> np.ndarray:
        """Map global simulation IDs to local array indices.

        Parameters
        ----------
        sim_ids : ndarray of int
            Global simulation particle IDs to look up.

        Returns
        -------
        result : ndarray of int64
            Local array indices for each queried ID; ``-1`` for IDs
            that are not present in this snapshot.
        """
        if self._index_sorter is None:
            self._index_sorter = np.argsort(self.indices)
        sorted_ids = self.indices[self._index_sorter]
        pos = np.clip(np.searchsorted(sorted_ids, sim_ids), 0, len(sorted_ids) - 1)
        found = sorted_ids[pos] == sim_ids
        result = np.full(len(sim_ids), -1, dtype=np.int64)
        result[found] = self._index_sorter[pos[found]]
        return result

    def index_to_id_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a mapping from array index to simulation ID.

        Returns
        -------
        idx : ndarray of int64
            Array indices.
        ids : ndarray of int64
            Corresponding simulation particle IDs.
        """
        idx = np.arange(len(self.indices), dtype=np.int64)
        return idx, self.indices.astype(np.int64, copy=False)

    def id_to_index_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a mapping from simulation ID to array index.

        Returns
        -------
        ids : ndarray of int64
            Simulation particle IDs.
        idx : ndarray of int64
            Corresponding array indices.
        """
        return self.indices.astype(np.int64, copy=False), np.arange(len(self.indices), dtype=np.int64)


class BoundnessResult:
    """Result of a boundness computation for a set of halos.

    Attributes
    ----------
    candidate_indices : NDArray of int
        Array indices of particles considered as candidates.
    boundness_values : NDArray of float
        Boundness energy values (negative = bound).
    dynamical_times : list of NDArray
        Per-halo dynamical times for the bound particles.
    """

    candidate_indices: NDArray[np.integer]
    boundness_values: NDArray[np.floating]
    dynamical_times: list[NDArray[np.floating]]

    def __init__(
        self,
        candidate_indices: NDArray[np.integer],
        boundness_values: NDArray[np.floating],
        dynamical_times: list[NDArray[np.floating]],
    ) -> None:
        self.candidate_indices = candidate_indices
        self.boundness_values = boundness_values
        self.dynamical_times = dynamical_times


class AssignmentResult:
    """Result of a single assigner invocation.

    Attributes
    ----------
    particle_df : DataFrame
        Particle-level assignment data with at least a ``Sub_tree_id`` column.
    responsibilities : object
        Soft assignment matrix. At runtime this is a SparseCSC where
        ``responsibilities[i, k]`` is the posterior probability that
        particle ``i`` belongs to component ``k``.
    fitted_parameters : dict
        Per-component fitted mixture parameters (means, covariances, counts).
    statistics : dict
        Summary statistics from the assignment process.
    """

    particle_df: pd.DataFrame
    responsibilities: object  # SparseCSC at runtime
    fitted_parameters: dict
    statistics: dict

    def __init__(
        self,
        particle_df: pd.DataFrame,
        responsibilities: object,
        fitted_parameters: dict,
        statistics: dict,
    ) -> None:
        self.particle_df = particle_df
        self.responsibilities = responsibilities
        self.fitted_parameters = fitted_parameters
        self.statistics = statistics


@runtime_checkable
class ParticleAssigner(Protocol):
    """Protocol for particle-to-halo assigners.

    Any class implementing this protocol can be used as the
    assigner in the roadrunner pipeline. The central method is
    ``assign()`` which takes halo information, particle coordinates,
    and group definitions and returns an AssignmentResult.
    """

    def assign(
        self,
        halos: list,
        particle_coords: np.ndarray,
        newborn_indices: np.ndarray,
        groups: list[list[int]],
        **kwargs,
    ) -> AssignmentResult: ...


@runtime_checkable
class PotentialModel(Protocol):
    """Protocol for gravitational potential models.

    Implementations provide potential energy, dynamical time, and
    tidal denominator computations for a given radius or position array.
    """

    def potential(self, r: np.ndarray) -> np.ndarray: ...
    def dynamical_time(self, x: np.ndarray) -> np.ndarray: ...
    def tidal_denominator(self, r: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class AssignmentStatistics(Protocol):
    """Protocol for assignment statistics trackers.

    Provides a ``values`` property returning a dictionary of
    summary metrics.
    """

    @property
    def values(self) -> dict[str, Any]: ...

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ParticleArray = NDArray[np.floating]
HaloID = int
SnapshotID = int
CSCResp = tuple[list[list[int]], list[list[float]]]
RespMap = dict[int, tuple[np.ndarray, np.ndarray]]


class SnapshotData:
    indices: NDArray[np.integer]
    masses: NDArray[np.floating]
    positions: NDArray[np.floating]
    velocities: NDArray[np.floating]
    metallicity: NDArray[np.floating] | None
    redshift: float
    time: float

    def __init__(
        self,
        indices: NDArray[np.integer],
        masses: NDArray[np.floating],
        positions: NDArray[np.floating],
        velocities: NDArray[np.floating],
        redshift: float,
        time: float,
        metallicity: NDArray[np.floating] | None = None,
    ) -> None:
        self.indices = indices
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.metallicity = metallicity
        self.redshift = redshift
        self.time = time


class BoundnessResult:
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
    particle_df: pd.DataFrame
    responsibilities: CSCResp
    fitted_parameters: dict
    statistics: dict

    def __init__(
        self,
        particle_df: pd.DataFrame,
        responsibilities: CSCResp,
        fitted_parameters: dict,
        statistics: dict,
    ) -> None:
        self.particle_df = particle_df
        self.responsibilities = responsibilities
        self.fitted_parameters = fitted_parameters
        self.statistics = statistics


@runtime_checkable
class ParticleAssigner(Protocol):
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
    def potential(self, r: np.ndarray) -> np.ndarray: ...
    def dynamical_time(self, x: np.ndarray) -> np.ndarray: ...
    def tidal_denominator(self, r: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class AssignmentStatistics(Protocol):
    @property
    def values(self) -> dict[str, Any]: ...

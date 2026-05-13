from typing import Protocol, runtime_checkable

import numpy as np

from roadrunner._mcf_types import AssignmentResult


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

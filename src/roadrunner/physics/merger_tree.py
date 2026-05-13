import numpy as np


def nfw_cmz_relation_duffy(M: float | np.ndarray, z: float) -> float | np.ndarray:
    A, B, C = 7.85, -0.081, -0.71
    return A * (M / 2e12) ** B * (1 + z) ** C

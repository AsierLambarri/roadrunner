import numpy as np

from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_model import HaloModel


class HaloEnsemble:
    def __init__(self, halos: list[HaloModel]):
        self._halos = list(halos)

    def __len__(self) -> int:
        return len(self._halos)

    def __getitem__(self, i) -> HaloModel:
        return self._halos[i]

    def __iter__(self):
        return iter(self._halos)

    @property
    def nhalo(self) -> int:
        return len(self._halos)

    @property
    def nstars(self) -> int:
        total = 0
        for h in self._halos:
            if h.has_boundness:
                total += len(h.get_boundness()[0])
        return total

    @property
    def empty(self) -> bool:
        return len(self._halos) == 0

    @property
    def positions(self) -> np.ndarray:
        return np.array([h.xcen for h in self._halos], dtype=np.float64)

    @property
    def velocities(self) -> np.ndarray:
        return np.array([h.velocity for h in self._halos], dtype=np.float64)

    @property
    def virial_radii(self) -> np.ndarray:
        return np.array([h.virial_radius for h in self._halos], dtype=np.float64)

    @property
    def sub_tree_ids(self) -> np.ndarray:
        return np.array([h.sub_tree_id for h in self._halos], dtype=int)

    def select(self, indices: list[int]) -> "HaloEnsemble":
        return HaloEnsemble([self._halos[i] for i in indices])

    def get_particles(self) -> tuple[SparseCSC, SparseCSC]:
        candidates = []
        boundness = []
        tdyns = []

        for h in self._halos:
            if h.has_boundness:
                inds, ener, tdyn_arr = h.get_boundness()
                candidates.append(inds)
                boundness.append(ener)
                tdyns.append(tdyn_arr)
            else:
                empty_idx = np.array([], dtype=np.uint64)
                empty_val = np.array([], dtype=np.float32)
                candidates.append(empty_idx)
                boundness.append(empty_val)
                tdyns.append(empty_val)

        col_id = np.array([h.sub_tree_id for h in self._halos], dtype=np.int64)
        return SparseCSC(candidates, boundness, column_id=col_id), SparseCSC(candidates, tdyns, column_id=col_id)

    def populated_indices(self) -> np.ndarray:
        csc, _ = self.get_particles()
        return np.array(
            [i for i, col in enumerate(csc.column_indices) if col.size > 0],
            dtype=np.int64,
        )

    def empty_indices(self) -> np.ndarray:
        csc, _ = self.get_particles()
        return np.array(
            [i for i, col in enumerate(csc.column_indices) if col.size == 0],
            dtype=np.int64,
        )

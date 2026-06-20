#############################################################################
#
# package:   roadrunner.physics
# file:      halo_ensemble.py
# brief:     Ensemble of HaloModel objects with sparse particle access.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   14 may 2026 - Created
#            14 may 2026 - Last edit
#
#############################################################################

import numpy as np

from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_model import HaloModel


class HaloEnsemble:
    """Ordered collection of :class:`HaloModel` objects with sparse particle access.

    Provides array-like access to halo properties (positions, velocities,
    virial radii, Sub_tree_ids) and returns combined particle data
    as a pair of :class:`SparseCSC` matrices (boundness and dynamical
    times).

    Parameters
    ----------
    halos : list of HaloModel
        Halos in this ensemble.
    """

    def __init__(self, halos: list[HaloModel]):
        self._halos = list(halos)

    def __len__(self) -> int:
        """Return the number of halos in the ensemble."""
        return len(self._halos)

    def __getitem__(self, i) -> HaloModel:
        """Access a single halo by index.

        Parameters
        ----------
        i : int
            Index.

        Returns
        -------
        halo : HaloModel
        """
        return self._halos[i]

    def __iter__(self):
        """Iterate over all halos."""
        return iter(self._halos)

    @property
    def nhalo(self) -> int:
        """Number of halos in this ensemble."""
        return len(self._halos)

    @property
    def nstars(self) -> int:
        """Total number of bound star particles across all halos."""
        total = 0
        for h in self._halos:
            if h.has_boundness:
                total += len(h.get_boundness()[0])
        return total

    @property
    def empty(self) -> bool:
        """``True`` if the ensemble contains no halos."""
        return len(self._halos) == 0

    @property
    def positions(self) -> np.ndarray:
        """Positions of all halos, shape ``(n_halos, 3)``."""
        return np.array([h.xcen for h in self._halos], dtype=np.float64)

    @property
    def velocities(self) -> np.ndarray:
        """Velocities of all halos, shape ``(n_halos, 3)``."""
        return np.array([h.velocity for h in self._halos], dtype=np.float64)

    @property
    def virial_radii(self) -> np.ndarray:
        """Virial radii of all halos, shape ``(n_halos,)``."""
        return np.array([h.virial_radius for h in self._halos], dtype=np.float64)

    @property
    def sub_tree_ids(self) -> np.ndarray:
        """``Sub_tree_id`` of each halo, shape ``(n_halos,)``."""
        return np.array([h.sub_tree_id for h in self._halos], dtype=int)

    def select(self, indices: list[int]) -> "HaloEnsemble":
        """Return a sub-ensemble containing the halos at the given indices.

        Parameters
        ----------
        indices : list of int
            Indices of halos to include.

        Returns
        -------
        sub : HaloEnsemble
        """
        return HaloEnsemble([self._halos[i] for i in indices])

    def get_particles(self) -> tuple[SparseCSC, SparseCSC]:
        """Return boundness and dynamical-time matrices as :class:`SparseCSC`.

        Columns correspond to halos in the same order as
        ``self.sub_tree_ids``.  Empty halos contribute empty columns.

        Returns
        -------
        boundness : SparseCSC
            Boundness energy matrix.
        tdyns : SparseCSC
            Dynamical time matrix.
        """
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
        """Indices of halos that have at least one bound particle.

        Returns
        -------
        idx : ndarray of int64
        """
        csc, _ = self.get_particles()
        return np.array(
            [i for i, col in enumerate(csc.column_indices) if col.size > 0],
            dtype=np.int64,
        )

    def empty_indices(self) -> np.ndarray:
        """Indices of halos with no bound particles.

        Returns
        -------
        idx : ndarray of int64
        """
        return np.array(
            [i for i, col in enumerate(csc.column_indices) if col.size == 0],
            dtype=np.int64,
        )

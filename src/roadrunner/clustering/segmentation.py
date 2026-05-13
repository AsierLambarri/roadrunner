import numpy as np

from roadrunner.clustering.union_find import overlapping_groups


def find_populated_haloes(candidates_list):
    empty = [i for i, c in enumerate(candidates_list) if len(c) == 0]
    populated = [i for i, c in enumerate(candidates_list) if len(c) > 0]
    return np.array(empty, dtype=np.int64), np.array(populated, dtype=np.int64)


class HaloSegmenter:
    def __init__(self, positions, virial_radii):
        self.positions = np.asarray(positions)
        self.rvirs = np.asarray(virial_radii)
        self.groups = None
        self.pruned_groups = None

    def overlap_groups(self):
        if len(self.positions) == 0:
            self.groups = []
            return self
        self.groups = overlapping_groups(self.positions, self.rvirs)
        return self

    def prune(self, candidates_list, min_particles=0, discard=False):
        if self.groups is None:
            raise RuntimeError(
                "Groups have not been computed. Call `overlap_groups()` first."
            )
        pruned = []
        for group in self.groups:
            large = [h for h in group if len(candidates_list[h]) > min_particles]
            small = [h for h in group if len(candidates_list[h]) <= min_particles]
            if large:
                pruned.append(large)
            if small and not discard:
                for h in small:
                    pruned.append([h])
        self.pruned_groups = pruned
        return self

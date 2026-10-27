"""What flows between the pipeline and the trackers of one snapshot.

The birth tracker reads the soft memberships of a snapshot
(:class:`Memberships`) and answers with what its birth hosts did
(:class:`BirthChanges`); the assembly tracker reads that answer. Neither
tracker needs to know the other, nor the pipeline types.
"""

from typing import NamedTuple

import numpy as np

from roadrunner._defaults import GALAXY_ID, SIM_ID


class Memberships(NamedTuple):
    """The soft memberships of one output: what a birth tracker reads.

    Attributes
    ----------
    t_snap : float
        Cosmic time of the output, in the unit of ``t_dyn``.
    snapshot_id : int
    particle_ids : ndarray of SIM_ID
        One entry per (particle, host) pair; a particle may appear with several hosts.
    host_ids : ndarray of GALAXY_ID
        ``UNBOUND`` for a particle that is bound to no galaxy.
    weights : ndarray of float
        Evidence weight of each pair: the responsibility, 1 for an unbound particle.
    t_dyn : ndarray of float
        Dynamical time of the particle in the host of each pair, ``NaN`` where there is none.
    alive : ndarray of GALAXY_ID
        The galaxies that exist at this output.
    """

    t_snap: float
    snapshot_id: int
    particle_ids: np.ndarray
    host_ids: np.ndarray
    weights: np.ndarray
    t_dyn: np.ndarray
    alive: np.ndarray


def _group(hosts, pids):
    """``{host: set of particle IDs}`` of parallel arrays."""
    if hosts.size == 0:
        return {}
    order = np.argsort(hosts, kind="stable")
    hosts, pids = hosts[order], pids[order]
    starts = np.flatnonzero(np.r_[True, hosts[1:] != hosts[:-1]])
    return {int(g): set(grp.tolist()) for g, grp in zip(hosts[starts], np.split(pids, starts[1:]))}


class BirthChanges(NamedTuple):
    """What an update of the birth tracker changed in the birth hosts.

    A particle's birth host is its current leader while its window is open
    and its final host after: it is *born* in the first host seen, and
    *moves* when a later output makes another host the leader. Nothing
    else changes it, so these are the whole difference between two
    snapshots' birth maps.

    Attributes
    ----------
    born_pid, born_host : ndarray
        Particles first seen in this update, with their first host.
    moved_pid, moved_from, moved_to : ndarray
        Particles whose leader changed, with the host they left and the one they entered.
    stats : dict of {str: int}
        Counters of the update (``new``, ``at_once``, ``opened``, ``open``, ``moved``, ``closed``).
    """

    born_pid: np.ndarray
    born_host: np.ndarray
    moved_pid: np.ndarray
    moved_from: np.ndarray
    moved_to: np.ndarray
    stats: dict

    @classmethod
    def nothing(cls):
        """No change at all."""
        return cls(np.empty(0, SIM_ID), np.empty(0, GALAXY_ID), np.empty(0, SIM_ID), np.empty(0, GALAXY_ID), np.empty(0, GALAXY_ID), {})

    def by_galaxy(self):
        """The changes seen from each galaxy's birth set.

        Returns
        -------
        gained, lost : dict of {int: set of int}
            Per galaxy, the particles that entered its birth set and the ones that left it.
        """
        return (_group(np.concatenate([self.born_host, self.moved_to]), np.concatenate([self.born_pid, self.moved_pid])),
                _group(self.moved_from, self.moved_pid))

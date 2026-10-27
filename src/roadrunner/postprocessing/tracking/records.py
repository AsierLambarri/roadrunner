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
    snapshot_time : float
        Cosmic time of the output, in the unit of ``dynamical_times``.
    snapshot_id : int
        Index of the snapshot.
    particle_ids : ndarray of SIM_ID, shape (n_pairs,)
        One entry per (particle, host) pair; a particle may appear with
        several hosts.
    host_ids : ndarray of GALAXY_ID, shape (n_pairs,)
        The host of each pair; ``UNBOUND`` for a particle that is bound to no
        galaxy.
    weights : ndarray of float, shape (n_pairs,)
        Evidence weight of each pair: the responsibility, 1 for an unbound
        particle.
    dynamical_times : ndarray of float, shape (n_pairs,)
        Dynamical time of the particle in the host of each pair, ``NaN`` where
        there is none.
    alive : ndarray of GALAXY_ID
        The galaxies that exist at this output.
    """

    snapshot_time: float
    snapshot_id: int
    particle_ids: np.ndarray
    host_ids: np.ndarray
    weights: np.ndarray
    dynamical_times: np.ndarray
    alive: np.ndarray


def _group(host_ids, particle_ids):
    """Group particles by host.

    Parameters
    ----------
    host_ids, particle_ids : ndarray of shape (n,)
        The host and the particle of each entry.

    Returns
    -------
    groups : dict of {int: set of int}
        ``{host ID: set of particle IDs}``.
    """
    if host_ids.size == 0:
        return {}
    order = np.argsort(host_ids, kind="stable")
    host_ids, particle_ids = host_ids[order], particle_ids[order]
    starts = np.flatnonzero(np.r_[True, host_ids[1:] != host_ids[:-1]])
    return {
        int(host_id): set(group.tolist())
        for host_id, group in zip(host_ids[starts], np.split(particle_ids, starts[1:]))
    }


class BirthChanges(NamedTuple):
    """What an update of the birth tracker changed in the birth hosts.

    A particle's birth host is its current leader while its window is open
    and its final host after: it is *born* in the first host seen, and
    *moves* when a later output makes another host the leader. Nothing
    else changes it, so these are the whole difference between two
    snapshots' birth maps.

    Attributes
    ----------
    born_ids : ndarray of SIM_ID
        Particles first seen in this update.
    born_hosts : ndarray of GALAXY_ID
        The first host of each of ``born_ids``.
    moved_ids : ndarray of SIM_ID
        Particles whose leader changed.
    moved_from : ndarray of GALAXY_ID
        The host each of ``moved_ids`` left.
    moved_to : ndarray of GALAXY_ID
        The host each of ``moved_ids`` entered.
    stats : dict of {str: int}
        Counters of the update: ``new``, ``at_once``, ``opened``, ``open``,
        ``moved`` and ``closed``.
    """

    born_ids: np.ndarray
    born_hosts: np.ndarray
    moved_ids: np.ndarray
    moved_from: np.ndarray
    moved_to: np.ndarray
    stats: dict

    @classmethod
    def nothing(cls):
        """Make the changes of an update that changed nothing.

        Returns
        -------
        changes : BirthChanges
            No particle born or moved, no counters.
        """
        return cls(
            np.empty(0, SIM_ID),
            np.empty(0, GALAXY_ID),
            np.empty(0, SIM_ID),
            np.empty(0, GALAXY_ID),
            np.empty(0, GALAXY_ID),
            {},
        )

    def by_galaxy(self):
        """Split the changes by the galaxy whose birth set they touch.

        Returns
        -------
        gained : dict of {int: set of int}
            Per galaxy, the particles that entered its birth set.
        lost : dict of {int: set of int}
            Per galaxy, the particles that left it.
        """
        gained = _group(
            np.concatenate([self.born_hosts, self.moved_to]),
            np.concatenate([self.born_ids, self.moved_ids]),
        )
        lost = _group(self.moved_from, self.moved_ids)
        return gained, lost

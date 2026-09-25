#############################################################################
#
# package:   roadrunner
# file:      randomness.py
# brief:     Deterministic, restartable per-snapshot/per-stage seed derivation.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   25 Sep 2026 - Created
#
#############################################################################

"""Deterministic, restartable per-snapshot/per-stage seed derivation.

This module only computes plain integer seeds -- it never seeds an RNG
itself and never mutates any shared/global random state. Every
consumer (mixture classes, ``random_lines_of_sight``, ...) receives an
ordinary ``int`` and seeds its own generator exactly as it already
does when called directly with an explicit seed. This keeps those
modules fully standalone-usable outside the pipeline, with zero
dependency on this module.

Two axes of derivation:

- ``consumer_seed(base_seed, snapshot_id, name)`` -- one seed per named
  pipeline stage ("fit", "los", ...) per snapshot. Plain arithmetic
  over a small, fixed, known set of consumer names (see ``_CONSUMERS``);
  the snapshot term is multiplied (not added) by a generous fixed cap
  on the number of named consumers so that no (snapshot, consumer)
  pair can ever collide with another, regardless of how many snapshots
  a run has.
- ``spawn_seeds(seed, n)`` -- ``n`` independent child seeds from one
  parent, for the one genuinely unbounded axis (the number of
  halo-groups fit within a single snapshot's "fit" stage). Uses
  :class:`numpy.random.SeedSequence`, which guarantees independence
  between children regardless of the parent's numeric value -- plain
  arithmetic here (e.g. ``fit_seed + group_index``) would reintroduce
  the same cross-snapshot collision risk one level down.

A small contextvar-based bridge (``random_seed`` / ``current_seed``)
lets orchestration code that doesn't itself care about seeds (the
snapshot orchestrator, ``process_snapshot``, ``reduce_snapshot``) pass
a snapshot's consumer seeds through unchanged signatures, while the
actual leaf consumers (``XGMMAssigner.assign``,
``compute_galaxy_properties``) still take a plain explicit ``seed``
parameter and never import this module.
"""

import contextvars
from contextlib import contextmanager

import numpy as np

__all__ = [
    "consumer_seed",
    "spawn_seeds",
    "random_seed",
    "current_seed",
]

_CONSUMERS = {"fit": 0, "los": 1}
_MAX_CONSUMERS = 16  # generous headroom for future named stages

_seeds_var = contextvars.ContextVar("roadrunner_consumer_seeds", default=None)


def consumer_seed(base_seed, snapshot_id, name):
    """Deterministic seed for one named consumer within one snapshot.

    Parameters
    ----------
    base_seed : int
    snapshot_id : int
    name : str
        Must be a key of ``_CONSUMERS``.

    Returns
    -------
    seed : int
    """
    return int(base_seed) + int(snapshot_id) * _MAX_CONSUMERS + _CONSUMERS[name]


def spawn_seeds(seed, n):
    """``n`` deterministic, independent child seeds from one parent seed.

    Parameters
    ----------
    seed : int
    n : int

    Returns
    -------
    seeds : list of int
        Length ``n``, in a fixed order.
    """
    ss = np.random.SeedSequence(int(seed))
    return [int(child.generate_state(1)[0]) for child in ss.spawn(n)]


@contextmanager
def random_seed(**seeds):
    """Scope this snapshot's consumer seeds (nestable, thread-safe).

    Parameters
    ----------
    **seeds : int
        Consumer name -> seed, e.g. ``random_seed(fit=1234, los=5678)``.
    """
    token = _seeds_var.set(seeds)
    try:
        yield
    finally:
        _seeds_var.reset(token)


def current_seed(name):
    """The active seed for one named consumer.

    Parameters
    ----------
    name : str

    Returns
    -------
    seed : int or None
        ``None`` outside any ``random_seed(...)`` scope, or if ``name``
        wasn't passed to the active scope -- callers should fall back
        to their own default (e.g. OS entropy) exactly as they do today.
    """
    seeds = _seeds_var.get()
    return None if seeds is None else seeds.get(name)

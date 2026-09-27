#############################################################################
#
# package:   roadrunner
# file:      threads.py
# brief:     One scoped thread budget for numba, BLAS/OpenMP and KDTrees.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   27 Sep 2026 - Created
#
#############################################################################

"""One scoped thread budget for every parallel backend.

Three independent thread pools run inside a pipeline snapshot: numba's
parallel kernels, the BLAS/OpenMP pools behind numpy and scipy, and the
``workers=`` of scipy KDTree queries. ``threads(n)`` sets all three at
once for the duration of a ``with`` block, like ``precision()`` does for
dtypes:

- numba: ``numba.set_num_threads`` (capped at ``NUMBA_NUM_THREADS``,
  fixed when numba starts), restored on exit;
- BLAS/OpenMP: ``threadpoolctl.threadpool_limits``;
- KDTree ``workers=``: a contextvar read through ``tree_workers()`` at
  each query site.

Outside a scope, or with ``threads(None)``, every backend keeps its own
default (all cores).
"""

import contextvars
import os
from contextlib import contextmanager

import numba
from threadpoolctl import threadpool_limits

_budget_var = contextvars.ContextVar("roadrunner_threads", default=None)


def _available_cpus():
    """CPUs this process may run on (its affinity set where available)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@contextmanager
def threads(n=None):
    """Scope the thread budget for numba, BLAS/OpenMP and KDTree queries.

    Parameters
    ----------
    n : int or None, optional
        Number of threads, capped at the CPUs available to the process.
        ``None`` leaves every backend at its default.

    Raises
    ------
    ValueError
        If ``n`` is smaller than 1.
    """
    if n is None:
        yield
        return
    if int(n) < 1:
        raise ValueError(f"threads must be a positive integer, got {n}")
    n = min(int(n), _available_cpus())
    previous = numba.get_num_threads()
    token = _budget_var.set(n)
    numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        numba.set_num_threads(previous)
        _budget_var.reset(token)


def thread_budget():
    """The active thread budget, or ``None`` outside a ``threads(n)`` scope."""
    return _budget_var.get()


def tree_workers():
    """``workers=`` for scipy KDTree queries: the scoped budget, else all cores."""
    n = _budget_var.get()
    return -1 if n is None else n


__all__ = ["threads", "thread_budget", "tree_workers"]

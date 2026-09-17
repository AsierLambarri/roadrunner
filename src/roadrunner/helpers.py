#############################################################################
#
# package:   roadrunner
# file:      helpers.py
# brief:     Auxiliary numerical helpers and data utilities.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   12 May 2026 - Created
#            16 Jun 2026 - Last edit
#
#############################################################################

"""Numerical dtype selection and data uniqueness utilities."""

import warnings

import numpy as np


def select_uint_dtype(max_val, msg=None):
    """Return the smallest unsigned integer dtype that can hold ``max_val``.

    Parameters
    ----------
    max_val : int
        Maximum value to be stored.
    msg : str, optional
        Extra context appended to the warning when no dtype suffices.

    Returns
    -------
    dtype : np.dtype
        One of ``uint16``, ``uint32``, or ``uint64``.

    Raises
    ------
    ValueError
        If ``max_val`` is negative: unsigned storage cannot hold
        sentinel values such as ``-1`` (see ``UNBOUND``); use a signed
        dtype (``LOCAL_IDX``/``GALAXY_ID``) for those arrays.
    """
    if max_val < 0:
        raise ValueError(
            f"select_uint_dtype got negative max_val={max_val}. "
            f"Unsigned dtypes cannot hold sentinel values like -1; "
            f"use a signed dtype instead.{f' {msg}' if msg else ''}"
        )
    bits = int(np.floor(np.log2(max_val))) + 1 if max_val != 0 else 1
    if bits <= 16:
        return np.uint16
    elif bits <= 32:
        return np.uint32
    elif bits <= 64:
        return np.uint64
    detail = f" {msg}" if msg else ""
    warnings.warn(f"uint64 insufficient for requested precision{detail}")
    return np.uint64


def select_float_dtype(max_value, abs_tol=1e-4, f128=False, msg=None):
    """Return the smallest floating-point dtype with sufficient precision.

    Parameters
    ----------
    max_value : float
        Maximum absolute value to be represented.
    abs_tol : float, default=1e-4
        Required absolute tolerance (half the spacing at ``max_value``).
    f128 : bool, default=False
        Include ``float128`` at the top of the precision ladder.
    msg : str, optional
        Extra context appended to the warning when no dtype suffices.

    Returns
    -------
    dtype : np.dtype
        One of ``float16``, ``float32``, ``float64`` (or ``float128``
        when ``f128=True``).
    """
    max_value = abs(max_value)
    ladder = [np.float16, np.float32, np.float64]
    if f128:
        float128 = getattr(np, "float128", None)
        if float128 is not None:
            ladder.append(float128)
    for dtype in ladder:
        with np.errstate(all='ignore'):
            x = dtype(max_value)
            if np.spacing(x) / 2 <= abs_tol:
                return dtype
    limit = "float128" if f128 else "float64"
    detail = f" {msg}" if msg else ""
    warnings.warn(f"{limit} insufficient for requested precision{detail}")
    return np.float64


def check_particle_uniqueness(data):
    """Check that all particle IDs across all groups are unique.

    Parameters
    ----------
    data : dict of list
        Dictionary mapping group keys to lists of particle IDs.

    Returns
    -------
    unique : bool
        True if every ID appears at most once across all groups.
    """
    all_values = [value for sublist in data.values() for value in sublist]
    unique_values = set(all_values)
    return len(all_values) == len(unique_values)


def remove_duplicates(data):
    """Remove duplicate particle IDs while preserving group structure.

    Parameters
    ----------
    data : dict of list
        Dictionary mapping group keys to lists of particle IDs.

    Returns
    -------
    new_data : dict of ndarray
        Same structure with duplicates removed; groups are processed
        in sorted key order so earlier groups retain IDs first.
    """
    seen = set()
    new_data = {}
    for key in sorted(data.keys()):
        filtered_list = [x for x in data[key] if x not in seen and (seen.add(x) or True)]
        new_data[key] = np.array(filtered_list)
    return new_data

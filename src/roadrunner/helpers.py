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


def select_uint_dtype(max_val):
    """Return the smallest unsigned integer dtype that can hold ``max_val``.

    Parameters
    ----------
    max_val : int
        Maximum value to be stored.

    Returns
    -------
    dtype : np.dtype
        One of ``uint16``, ``uint32``, or ``uint64``.
    """
    if max_val < 0:
        max_val = abs(max_val)
    bits = int(np.floor(np.log2(max_val))) + 1 if max_val != 0 else 1
    if bits <= 16:
        return np.uint16
    elif bits <= 32:
        return np.uint32
    elif bits <= 64:
        return np.uint64
    warnings.warn("uint64 insufficient for requested precision")
    return np.uint64


def select_int_dtype(max_val):
    """Return the smallest signed integer dtype that can hold ``max_val``.

    Parameters
    ----------
    max_val : int
        Maximum absolute value to be stored.

    Returns
    -------
    dtype : np.dtype
        One of ``int16``, ``int32``, or ``int64``.
    """
    if max_val < 0:
        max_val = abs(max_val)
    bits = int(np.floor(np.log2(max_val))) + 2 if max_val != 0 else 1
    if bits <= 16:
        return np.int16
    elif bits <= 32:
        return np.int32
    elif bits <= 64:
        return np.int64
    warnings.warn("int64 insufficient for requested precision")
    return np.int64


def select_float_dtype(max_value, abs_tol=1e-4):
    """Return the smallest floating-point dtype with sufficient precision.

    Parameters
    ----------
    max_value : float
        Maximum absolute value to be represented.
    abs_tol : float, default=1e-4
        Required absolute tolerance (half the spacing at ``max_value``).

    Returns
    -------
    dtype : np.dtype
        One of ``float16``, ``float32``, ``float64``, or ``float128``.
    """
    max_value = abs(max_value)
    for dtype in (np.float16, np.float32, np.float64, np.float128):
        with np.errstate(all='ignore'):
            x = dtype(max_value)
            if np.spacing(x) / 2 <= abs_tol:
                return dtype
    warnings.warn("float128 insufficient for requested precision")
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

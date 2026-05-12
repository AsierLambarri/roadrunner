import warnings

import numpy as np


def select_uint_dtype(max_val):
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
    max_value = abs(max_value)
    for dtype in (np.float16, np.float32, np.float64, np.float128):
        x = dtype(max_value)
        if np.spacing(x) / 2 <= abs_tol:
            return dtype
    warnings.warn("float128 insufficient for requested precision")
    return np.float64


def check_particle_uniqueness(data):
    all_values = [value for sublist in data.values() for value in sublist]
    unique_values = set(all_values)
    return len(all_values) == len(unique_values)


def remove_duplicates(data):
    seen = set()
    new_data = {}
    for key in sorted(data.keys()):
        filtered_list = [x for x in data[key] if x not in seen and (seen.add(x) or True)]
        new_data[key] = np.array(filtered_list)
    return new_data

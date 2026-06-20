#############################################################################
#
# package:   roadrunner.io
# file:      serialization.py
# brief:     Checkpoint serialization (zstd-compressed pickle).
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   19 may 2026 - Created
#            19 may 2026 - Last edit
#
#############################################################################

"""Checkpoint save/load with zstd-compressed pickle.

Supports version migration for backward compatibility.
"""

import pickle
import os

import zstandard as zstd
from roadrunner._exceptions import RestartError
from roadrunner._defaults import ZSTD_COMPRESSION_LEVEL


_VERSION = 2


def save_checkpoint(path, data, level=ZSTD_COMPRESSION_LEVEL):
    """Save data to a zstd-compressed checkpoint file.

    Parameters
    ----------
    path : str
        Output file path.
    data : object
        Any picklable object to save.
    level : int, default=ZSTD_COMPRESSION_LEVEL
        Zstd compression level.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"version": _VERSION, "data": data}
    pickled = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(pickled)

    with open(path, "wb") as f:
        f.write(compressed)


def load_checkpoint(path):
    """Load data from a zstd-compressed checkpoint file.

    Parameters
    ----------
    path : str
        Checkpoint file path.

    Returns
    -------
    data : object
        The saved data.

    Raises
    ------
    RestartError
        If the file is missing or corrupt.
    """
    try:
        with open(path, "rb") as f:
            compressed = f.read()
    except FileNotFoundError:
        raise RestartError(f"Checkpoint not found: {path}")
    except Exception as e:
        raise RestartError(f"Failed to read checkpoint: {e}")

    try:
        dctx = zstd.ZstdDecompressor()
        pickled = dctx.decompress(compressed)
        payload = pickle.loads(pickled)
    except Exception as e:
        raise RestartError(f"Corrupted checkpoint data: {e}")

    version = payload.get("version", 0)
    if version < _VERSION:
        payload = _migrate(payload, version)

    return payload["data"]


def _migrate(payload, version):
    """Migrate checkpoint from older versions."""
    if version == 0:
        payload["version"] = 1
        payload["data"] = payload.get("data", payload)
        version = 1
    if version == 1:
        payload["version"] = _VERSION
        version = _VERSION
    return payload

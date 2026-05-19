import pickle
import os

import zstandard as zstd

from roadrunner._exceptions import RestartError

_VERSION = 2


def save_checkpoint(path, data, level=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"version": _VERSION, "data": data}
    pickled = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(pickled)

    with open(path, "wb") as f:
        f.write(compressed)


def load_checkpoint(path):
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

"""Particle-data snapshot reader for custom binary formats.

The HDF5 file must contain:
- ``data/indices``, ``data/masses``, ``data/positions``,
  ``data/velocities``, ``data/scaler/mean``, ``data/scaler/scale``
- ``header/redshift`` and ``header/time`` attributes

An optional ``header/cosmology`` group, with the fields of
:class:`roadrunner.cosmology.Cosmology` as attributes, carries the run's
cosmology.

Extra datasets (e.g. ``data/metallicity``) are optional and are
loaded as-is (stored in raw physical units).

With a particle-ID filter, only ``data/indices`` is read in full; every
other dataset is read in contiguous row blocks and masked block by
block, so peak memory is one block plus the selected rows rather than
the whole snapshot.
"""

from __future__ import annotations

import h5py
import numpy as np

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers.equivalence import EquivalenceTable
from roadrunner._defaults import COSMOLOGY_GROUP, SIM_ID, data_dtype, math_dtype

READ_BLOCK_ROWS = 1 << 20   # rows per block for masked reads


def _block_rows(ds):
    """Block length for masked reads: ``READ_BLOCK_ROWS`` rounded up to
    a whole number of the dataset's HDF5 chunks (when chunked)."""
    chunk = ds.chunks[0] if ds.chunks else 1
    return -(-READ_BLOCK_ROWS // chunk) * chunk


def _read_rows(ds, mask):
    """Read the rows of an HDF5 dataset selected by a boolean mask.

    Walks the dataset in contiguous row blocks and keeps the selected
    rows of each, in file order; blocks without a selected row are not
    read at all. Sequential block reads avoid the cost of scattered
    HDF5 fancy indexing.

    Parameters
    ----------
    ds : h5py.Dataset
    mask : ndarray of bool, shape (ds.shape[0],)

    Returns
    -------
    rows : ndarray of shape (mask.sum(),) + ds.shape[1:]
    """
    out = np.empty((int(np.count_nonzero(mask)),) + ds.shape[1:], dtype=ds.dtype)
    block = _block_rows(ds)
    pos = 0
    for start in range(0, ds.shape[0], block):
        m = mask[start:start + block]
        k = int(np.count_nonzero(m))
        if k:
            out[pos:pos + k] = ds[start:start + block][m]
            pos += k
    return out


class ParticleDataSnapshotReader:
    """Snapshot reader for pre-processed HDF5 particle data.

    Parameters
    ----------
    equiv_table : EquivalenceTable
        Snapshot equivalence table.
    base_dir : str, default=''
        Base directory for data files.
    assign_fields : list of str or None, optional
        Attribute names for the assigner input.
    extra_fields : list of str or None, optional
        Dataset names under ``data/`` to load as extra particle
        fields (e.g. ``["metallicity"]``).
    """

    _particle_filter = None

    def __init__(self, equiv_table: EquivalenceTable, base_dir: str = "",
                 assign_fields=None, extra_fields=None):
        self._equiv = equiv_table
        self._base_dir = base_dir
        self._assign_fields = assign_fields
        self._extra_fields = extra_fields or []

    @property
    def particle_filter(self):
        """Currently configured persistent particle-ID filter (or None)."""
        return self._particle_filter

    def set_particle_filter(self, particle_indices) -> None:
        """Set a persistent ID filter applied to every subsequent load()."""
        self._particle_filter = np.asarray(particle_indices, dtype=SIM_ID)

    def erase_particle_filter(self) -> None:
        """Remove the persistent ID filter (back to loading all particles)."""
        self._particle_filter = None

    @staticmethod
    def _region_mask(positions, sphere=None, bbox=None):
        if (sphere is None) == (bbox is None):
            raise ValueError("Provide exactly one of `sphere` or `bbox`.")
        if sphere is not None:
            center = np.asarray(sphere[0], dtype=math_dtype())
            radius = float(sphere[1])
            return np.sum((positions - center) ** 2, axis=1) <= radius ** 2
        lower = np.asarray(bbox[0], dtype=math_dtype())
        upper = np.asarray(bbox[1], dtype=math_dtype())
        return np.all((positions >= lower) & (positions <= upper), axis=1)

    def select_indices(self, file_path: str, sphere=None, bbox=None) -> np.ndarray:
        """Return simulation IDs inside a comoving region.

        Parameters
        ----------
        file_path : str
            Path to the snapshot file.
        sphere : tuple or None, optional
            Sphere selection ``((cx, cy, cz), radius)`` in comoving kpc.
        bbox : tuple or None, optional
            Box selection ``((xlo, ylo, zlo), (xhi, yhi, zhi))`` in comoving kpc.

        Returns
        -------
        indices : ndarray of uint64
            Simulation IDs of the particles inside the region.
        """
        # Only indices and positions are needed: read positions block by
        # block and keep the region mask, ignoring any particle filter.
        with h5py.File(file_path, "r") as hf:
            indices = hf["data/indices"][:]
            mean = hf["data/scaler/mean"][:]
            scale = hf["data/scaler/scale"][:]
            ds = hf["data/positions"]
            inv_s = 1.0 / scale
            dt = data_dtype()
            inside = np.empty(indices.size, dtype=bool)
            block = _block_rows(ds)
            for start in range(0, ds.shape[0], block):
                pos = (ds[start:start + block] * inv_s[:3] + mean[:3]).astype(dt)
                inside[start:start + block] = self._region_mask(pos, sphere=sphere, bbox=bbox)
        return indices[inside]

    def read_cosmology(self, file_path: str) -> dict:
        """Cosmology carried by the file's ``header/cosmology`` group.

        Parameters
        ----------
        file_path : str
            Path to the ``.hdf5`` file.

        Returns
        -------
        params : dict
            Empty if the file has no ``header/cosmology`` group.
        """
        with h5py.File(file_path, "r") as hf:
            group = hf.get(COSMOLOGY_GROUP)
            return {} if group is None else {k: float(v) for k, v in group.attrs.items()}

    def load(self, file_path: str, particle_indices=None) -> SnapshotData:
        """Load a snapshot from an HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the ``.hdf5`` file.
        particle_indices : ndarray or None, optional
            If given, keep only these simulation IDs for this call only,
            overriding any persistent filter.

        Returns
        -------
        snap_data : SnapshotData
            Loaded particle data.
        """
        ids = (particle_indices if particle_indices is not None
               else self._particle_filter)
        with h5py.File(file_path, "r") as hf:
            indices = hf["data/indices"][:]
            mask = None if ids is None else np.isin(indices, ids)
            if mask is not None:
                indices = indices[mask]

            def read(ds):
                return ds[:] if mask is None else _read_rows(ds, mask)

            masses = read(hf["data/masses"])
            pos_scaled = read(hf["data/positions"])
            vel_scaled = read(hf["data/velocities"])
            mean = hf["data/scaler/mean"][:]
            scale = hf["data/scaler/scale"][:]
            redshift = hf["header"].attrs["redshift"]
            time = hf["header"].attrs["time"]

            extra = {}
            for name in self._extra_fields:
                key = f"data/{name}"
                if key in hf:
                    extra[name] = read(hf[key])

        inv_s = 1.0 / scale
        dt = data_dtype()
        masses = np.asarray(masses, dtype=dt)
        positions = (pos_scaled * inv_s[:3] + mean[:3]).astype(dt)
        velocities = (vel_scaled * inv_s[3:6] + mean[3:6]).astype(dt)
        for name in extra:
            if np.issubdtype(np.asanyarray(extra[name]).dtype, np.floating):
                extra[name] = np.asarray(extra[name], dtype=dt)

        return SnapshotData(
            index=indices, mass=masses, position=positions,
            velocity=velocities, redshift=redshift, time=time,
            assign_fields=self._assign_fields,
            **extra,
        )

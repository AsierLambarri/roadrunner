"""YT-based snapshot reader for cosmological simulations."""

import numpy as np
import yt

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers._defaults import METALLICITY_COEFF0, METALLICITY_COEFF1, SOLAR_METALLICITY
from roadrunner._defaults import SIM_ID, data_dtype


_TRACK_FILTER_NAME = "_rr_track"


class SnapshotReader:
    """YT-based snapshot reader for cosmological simulations.

    Parameters
    ----------
    code : str
        Simulation code name.
    ptype : str
        Particle type.
    fields : dict
        Mapping from canonical names to YT field names.
    unit_base : dict or None, optional
        YT unit base.
    assign_fields : list of str or None, optional
        Attribute names for the assigner input.  Default preserved
        in :class:`SnapshotData` (``["positions", "velocities"]``).
    """

    def __init__(self, code: str, ptype: str, fields: dict, unit_base=None,
                 assign_fields=None):
        self.code = code.upper()
        self.ptype = ptype
        self.fields = fields
        self.unit_base = unit_base
        self._assign_fields = assign_fields
        self._particle_filter = None
        self._filter_name = f"{_TRACK_FILTER_NAME}_{id(self):x}"
        self._active_ptype = self.ptype

    @property
    def particle_filter(self) -> np.ndarray | None:
        """Currently configured persistent particle-ID filter (or None)."""
        return self._particle_filter

    def load(self, file_path: str,
             particle_indices: np.ndarray | None = None) -> SnapshotData:
        """Open and extract particle data from a snapshot file.

        Parameters
        ----------
        file_path : str
            Path to the snapshot file.
        particle_indices : ndarray or None, optional
            If given, keep only these simulation IDs for this call only,
            overriding any persistent filter.

        Returns
        -------
        snap_data : SnapshotData
        """
        ids = (particle_indices if particle_indices is not None
               else self._particle_filter)
        ds = self._open(file_path)
        self._active_ptype = self.ptype
        if ids is not None:
            self._active_ptype = self._attach_index_filter(ds, ids)
        return self._extract(ds)

    def set_particle_filter(self, particle_indices: np.ndarray) -> None:
        """Set a persistent ID filter applied to every subsequent load()."""
        self._particle_filter = np.asarray(particle_indices, dtype=SIM_ID)

    def erase_particle_filter(self) -> None:
        """Remove the persistent ID filter (back to loading all particles)."""
        self._particle_filter = None

    def select_indices(self, file_path: str, sphere=None, bbox=None) -> np.ndarray:
        """Return simulation IDs inside a comoving region (YT-native).

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
        if (sphere is None) == (bbox is None):
            raise ValueError("Provide exactly one of `sphere` or `bbox`.")
        ds = self._open(file_path)
        if sphere is not None:
            (cx, cy, cz), radius = sphere
            container = ds.sphere(ds.arr((cx, cy, cz), "kpccm"),
                                  ds.arr(radius, "kpccm"))
        else:
            (xlo, ylo, zlo), (xhi, yhi, zhi) = bbox
            container = ds.box(ds.arr((xlo, ylo, zlo), "kpccm"),
                               ds.arr((xhi, yhi, zhi), "kpccm"))
        return container[self.ptype, self.fields["index"]].value.astype(SIM_ID)

    def _attach_index_filter(self, ds, particle_indices: np.ndarray) -> str:
        """Register and attach a YT particle filter for the given IDs.

        Layers on top of the code-specific ``self.ptype`` filter installed
        by ``_open``; re-created on every opened dataset.

        Returns
        -------
        ptype : str
            Name of the active filtered particle type.
        """
        ids = np.asarray(particle_indices, dtype=SIM_ID)
        index_field = self.fields["index"]
        yt.add_particle_filter(
            self._filter_name,
            function=lambda pfilter, data: np.isin(
                data[pfilter.filtered_type, index_field], ids),
            requires=[index_field],
            filtered_type=self.ptype,
        )
        ds.add_particle_filter(self._filter_name)
        return self._filter_name

    @staticmethod
    def mock_data(n_particles: int, seed: int = 42) -> SnapshotData:
        """Generate random mock snapshot data for testing.

        Parameters
        ----------
        n_particles : int
            Number of particles.
        seed : int, default=42
            Random seed.

        Returns
        -------
        snap_data : SnapshotData
        """
        rng = np.random.default_rng(seed)
        indices = np.arange(n_particles, dtype=SIM_ID)
        dt = data_dtype()
        masses = rng.uniform(0.1, 10.0, n_particles).astype(dt)
        positions = rng.uniform(-100, 100, (n_particles, 3)).astype(dt)
        velocities = rng.uniform(-200, 200, (n_particles, 3)).astype(dt)
        metallicity = rng.uniform(-2.0, 0.5, n_particles).astype(dt)
        return SnapshotData(
            index=indices, mass=masses, position=positions,
            velocity=velocities, redshift=0.0, time=13.8,
            metallicity=metallicity,
        )

    def _open(self, file_path: str):
        """Open a snapshot file with code-specific settings.

        Parameters
        ----------
        file_path : str
            Path to the snapshot file.

        Returns
        -------
        ds : yt.Dataset
        """
        if self.code in ("ART", "ART-I"):
            return yt.load(file_path)

        if self.code in ("GEAR",):
            return yt.load(file_path, unit_base=self.unit_base)

        if self.code in ("AURIGA", "AREPO"):
            ds = yt.load(file_path, unit_base=self.unit_base)
            yt.add_particle_filter(
                self.ptype,
                function=lambda pfilter, data: (
                    data[pfilter.filtered_type, "GFM_StellarFormationTime"] >= 0
                ),
                requires=[
                    "GFM_StellarFormationTime",
                    "particle_index",
                ],
                filtered_type="PartType4",
            )
            ds.add_particle_filter(self.ptype)
            return ds

        if self.code in ("RAMSES", "VINTERGATAN"):
            yt.add_particle_filter(
                self.ptype,
                function=lambda pfilter, data: (
                    data[pfilter.filtered_type, "particle_birth_time"] > 0
                ) & (data[pfilter.filtered_type, "particle_metallicity0"] > 0),
                requires=["particle_birth_time"],
                filtered_type="all",
            )
            ds = yt.load(
                file_path,
                extra_particle_fields=[
                    ("particle_potential", "d"),
                    ("conformal_birth_time", "d"),
                    ("particle_metallicity0", "d"),
                    ("particle_metallicity1", "d"),
                    ("particle_tag", "d"),
                    ("particle_birth_time", "d"),
                ],
            )
            ds.add_particle_filter(self.ptype)
            ds.add_field(
                (self.ptype, "particle_metallicity"),
                function=lambda field, data: (
                    METALLICITY_COEFF0 * data[self.ptype, "particle_metallicity0"]
                    + METALLICITY_COEFF1 * data[self.ptype, "particle_metallicity1"]
                ) / SOLAR_METALLICITY,
                sampling_type="particle",
                units="",
            )
            return ds

        raise ValueError(f"Unsupported code: {self.code}")

    def _extract(self, ds) -> SnapshotData:
        """Extract particle data from an opened YT dataset.

        Reads from ``self._active_ptype``: the raw type, or the tracking
        filter name when a particle-ID filter is active.

        Parameters
        ----------
        ds : yt.Dataset
            The opened dataset.

        Returns
        -------
        snap_data : SnapshotData
        """
        ad = ds.all_data()
        ptype = self._active_ptype
        f = self.fields

        indices = ad[ptype, f["index"]].value.astype(SIM_ID)
        dt = data_dtype()
        masses = ad[ptype, f["mass"]].to("Msun").value.astype(dt)
        positions = (
            ad[ptype, f["position"]].to("kpccm").value.astype(dt)
        )
        velocities = (
            ad[ptype, f["velocity"]].to("km/s").value.astype(dt)
        )
        redshift = ds.current_redshift
        time = ds.current_time.to("Gyr").value

        _required = {"index", "mass", "position", "velocity"}
        extra = {}
        for key, yt_field in f.items():
            if key not in _required:
                extra[key] = ad[ptype, yt_field].value.astype(dt)

        return SnapshotData(
            index=indices, mass=masses, position=positions,
            velocity=velocities, redshift=redshift, time=time,
            assign_fields=self._assign_fields,
            **extra,
        )

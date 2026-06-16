import numpy as np
import yt

from roadrunner._mcf_types import SnapshotData
from roadrunner.readers._defaults import METALLICITY_COEFF0, METALLICITY_COEFF1, SOLAR_METALLICITY


class SnapshotReader:
    def __init__(self, code: str, ptype: str, fields: dict, unit_base=None):
        self.code = code.upper()
        self.ptype = ptype
        self.fields = fields
        self.unit_base = unit_base

    def load(self, file_path: str) -> SnapshotData:
        ds = self._open(file_path)
        return self._extract(ds)

    def load_with_filter(
        self, file_path: str, particle_indices: np.ndarray
    ) -> SnapshotData:
        ds = self._open(file_path)
        ds = self.apply_particle_filter(ds, particle_indices)
        return self._extract(ds)

    def apply_particle_filter(self, ds, particle_indices):
        filter_name = f"_snap_filter_{id(particle_indices)}"
        yt.add_particle_filter(
            filter_name,
            function=lambda pfilter, data: np.isin(
                data[pfilter.filtered_type, self.fields["index"]],
                particle_indices,
            ),
            filtered_type=self.ptype,
            requires=[self.fields["index"]],
        )
        ds.add_particle_filter(filter_name)
        return ds

    @staticmethod
    def mock_data(n_particles: int, seed: int = 42) -> SnapshotData:
        rng = np.random.default_rng(seed)
        indices = np.arange(n_particles, dtype=np.uint64)
        masses = rng.uniform(0.1, 10.0, n_particles).astype(np.float64)
        positions = rng.uniform(-100, 100, (n_particles, 3)).astype(np.float64)
        velocities = rng.uniform(-200, 200, (n_particles, 3)).astype(np.float64)
        metallicity = rng.uniform(-2.0, 0.5, n_particles).astype(np.float64)
        return SnapshotData(
            indices, masses, positions, velocities,
            redshift=0.0, time=13.8, metallicity=metallicity,
        )

    def _open(self, file_path: str):
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
        ad = ds.all_data()
        ptype = self.ptype
        f = self.fields

        indices = ad[ptype, f["index"]].value.astype(np.uint64)
        masses = ad[ptype, f["mass"]].to("Msun").value.astype(np.float64)
        positions = (
            ad[ptype, f["position"]].to("kpccm").value.astype(np.float64)
        )
        velocities = (
            ad[ptype, f["velocity"]].to("km/s").value.astype(np.float64)
        )
        redshift = ds.current_redshift
        time = ds.current_time.to("Gyr").value

        metallicity = None
        if "metallicity" in f:
            metallicity = ad[ptype, f["metallicity"]].value.astype(np.float64)

        return SnapshotData(
            indices, masses, positions, velocities,
            redshift, time, metallicity,
        )

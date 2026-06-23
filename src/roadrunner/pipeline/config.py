"""RunConfig frozen dataclass for pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunConfig:
    """Frozen dataclass for pipeline configuration.

    All fields have defaults.  See the source for the full list of
    available parameters.
    """
    merger_tree_path: str = ""
    particle_data_dir: str = ""
    equivalence_path: str = ""
    code: str = "RAMSES"
    ptype: str = "star"
    reader_type: str = "yt"
    npz_mock_sim: bool = False
    fields: dict = field(default_factory=lambda: {
        "index": "particle_index",
        "mass": "particle_mass",
        "position": "coordinates",
        "velocity": "particle_velocity",
    })
    unit_base: dict | None = None
    assign_fields: list | None = None

    accretion_id: int | None = None
    start_snapshot: int | None = None
    end_snapshot: int | None = None

    assignment_method: str = "gmm"
    use_bgmm_priors: bool = True
    dtype_math: str = "float64"
    svi_iters: int = 1000
    svi_batch_size: int = 10000
    halo_model: str = "kepler"
    cov_type: str = "full"
    max_iter: int = 10
    tol: float = 1e-2
    reg_covar: float = 1e-6
    n_los: int = 15
    search_factor: float = 1.0
    min_particles: int = 10
    comoving: bool = True

    birth_window_factor: float = 5.0
    enforce_initial_hosts: bool = True
    use_gmm_centers: bool = True
    min_particles_structural: int = 30
    ssc_nmin: int = 30
    ssc_alpha: float = 0.9
    dynstate_snapshots: int | list[int] | str | None = field(default_factory=lambda: [-2, -1])

    output_dir: str = "./output"
    resume: bool = False
    save_particles: bool = True
    save_assignment: bool = True
    float_atol: float = 1e-4

    def __post_init__(self):
        """Validate and coerce configuration values.

        Checks that ``reader_type`` is one of the allowed values,
        and coerces numeric fields to the correct type when provided
        as strings or other types.
        """
        if self.reader_type not in ("yt", "npz", "pdata"):
            raise ValueError(f"reader_type must be 'yt', 'npz', or 'pdata', got '{self.reader_type}'")

        for field_name in (
            "accretion_id", "max_iter", "n_los", "min_particles",
            "start_snapshot", "end_snapshot",
        ):
            val = getattr(self, field_name, None)
            if val is not None:
                object.__setattr__(self, field_name, int(val))

        for field_name in (
            "tol", "reg_covar", "search_factor",
            "birth_window_factor", "float_atol", "ssc_alpha",
        ):
            val = getattr(self, field_name, None)
            if val is not None:
                object.__setattr__(self, field_name, float(val))
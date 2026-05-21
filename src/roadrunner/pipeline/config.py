from dataclasses import dataclass, field


@dataclass
class RunConfig:
    merger_tree_path: str = ""
    particle_data_dir: str = ""
    equivalence_path: str = ""
    code: str = "RAMSES"
    ptype: str = "star"
    fields: dict = field(default_factory=lambda: {
        "index": "particle_index",
        "mass": "particle_mass",
        "position": "coordinates",
        "velocity": "particle_velocity",
    })
    unit_base: dict | None = None

    accretion_id: int | None = None
    start_snapshot: int | None = None
    end_snapshot: int | None = None

    halo_model: str = "kepler"
    cov_type: str = "full"
    max_iter: int = 10
    tol: float = 1e-2
    reg_covar: float = 1e-6
    n_los: int = 15
    search_factor: float = 1.0
    min_particles: int = 10

    birth_window_factor: float = 5.0
    enforce_initial_hosts: bool = True

    output_dir: str = "./output"
    save_particles: bool = True
    save_assignment: bool = True
    float_atol: float = 1e-4

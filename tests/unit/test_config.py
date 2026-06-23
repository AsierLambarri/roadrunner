import pytest

from roadrunner.pipeline.config import RunConfig


class TestRunConfigDefaults:
    def test_defaults(self):
        c = RunConfig()
        assert c.halo_model == "kepler"
        assert c.cov_type == "full"
        assert c.n_los == 15
        assert c.search_factor == 1.0
        assert c.min_particles == 10
        assert c.max_iter == 10
        assert c.tol == 1e-2
        assert c.reg_covar == 1e-6
        assert c.birth_window_factor == 5.0
        assert c.enforce_initial_hosts is True
        assert c.output_dir == "./output"
        assert c.save_particles is True
        assert c.save_assignment is True
        assert c.float_atol == 1e-4
        assert c.accretion_id is None
        assert c.start_snapshot is None
        assert c.end_snapshot is None
        assert c.code == "RAMSES"
        assert c.ptype == "star"
        assert c.unit_base is None

    def test_fields_default_factory(self):
        c1 = RunConfig()
        c2 = RunConfig()
        assert c1.fields == c2.fields
        assert c1.fields is not c2.fields

    def test_fields_contents(self):
        c = RunConfig()
        assert c.fields == {
            "index": "particle_index",
            "mass": "particle_mass",
            "position": "coordinates",
            "velocity": "particle_velocity",
        }


class TestRunConfigFromDict:
    def test_from_dict(self):
        d = {"halo_model": "nfw", "cov_type": "diagonal"}
        c = RunConfig(**d)
        assert c.halo_model == "nfw"
        assert c.cov_type == "diagonal"

    def test_from_dict_extra_keys_rejected(self):
        with pytest.raises(TypeError, match="unknown_key"):
            RunConfig(**{"halo_model": "kepler", "unknown_key": 42})

    def test_from_dict_all_fields(self):
        c = RunConfig(
            merger_tree_path="/path/tree.csv",
            particle_data_dir="/path/data",
            equivalence_path="/path/equiv.csv",
            code="GEAR",
            ptype="star",
            fields={
                "index": "my_index",
                "mass": "my_mass",
                "position": "my_position",
                "velocity": "my_velocity",
            },
            unit_base={"length": 1.0},
            accretion_id=42,
            start_snapshot=10,
            end_snapshot=20,
            halo_model="nfw",
            cov_type="diag",
            max_iter=20,
            tol=0.01,
            reg_covar=1e-5,
            n_los=25,
            search_factor=2.0,
            min_particles=5,
            birth_window_factor=3.0,
            enforce_initial_hosts=False,
            output_dir="/out",
            save_particles=False,
            save_assignment=False,
            float_atol=0.001,
        )
        assert c.merger_tree_path == "/path/tree.csv"
        assert c.accretion_id == 42
        assert c.n_los == 25
        assert c.enforce_initial_hosts is False


class TestRunConfigFrozen:
    def test_frozen_attribute(self):
        c = RunConfig()
        with pytest.raises(AttributeError):
            c.halo_model = "nfw"

    def test_frozen_fields(self):
        c = RunConfig()
        with pytest.raises(AttributeError):
            c.fields = {}

    def test_frozen_attribute_from_dict(self):
        c = RunConfig(halo_model="nfw")
        with pytest.raises(AttributeError):
            c.halo_model = "kepler"
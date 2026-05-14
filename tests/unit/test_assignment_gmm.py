import numpy as np

from roadrunner._mcf_types import ParticleAssigner
from roadrunner.clustering.assignment.gmm import GMMAssigner
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


def _make_halo(xcen, mass=1e12, rvir=RVR, sub_tree_id=1):
    inner = KeplerPotential(M=mass, G=4.3e-6)
    return HaloModel(
        inner, np.asarray(xcen, dtype=np.float64),
        VCENTER, rvir, sub_tree_id=sub_tree_id, redshift=0.0,
    )


def _setup_mock_halos(n_halos=2, n_particles=100):
    rng = np.random.default_rng(42)
    halos = [
        _make_halo([0.0, 0.0, 0.0], sub_tree_id=i + 1)
        for i in range(n_halos)
    ]
    coords = rng.uniform(-50, 50, (n_particles, 6)).astype(np.float64)
    # Assign bound particles: first half to halo 0, second half to halo 1
    for i, h in enumerate(halos):
        start = i * (n_particles // n_halos)
        end = (i + 1) * (n_particles // n_halos)
        h.set_boundness(
            np.arange(start, end, dtype=np.uint64),
            np.ones(end - start, dtype=np.float32) * 0.5,
            np.ones(end - start, dtype=np.float32) * 0.1,
        )
    return halos, coords


class TestGMMAssigner:
    def test_is_particle_assigner(self):
        assert isinstance(GMMAssigner(), ParticleAssigner)

    def test_single_halo(self):
        halos, coords = _setup_mock_halos(n_halos=1, n_particles=50)
        groups = [[0]]
        assigner = GMMAssigner(verbose=0)
        result = assigner.assign(halos, coords, np.array([], dtype=np.uint64), groups)
        assert result.particle_df is not None
        assert isinstance(result.responsibilities, dict)

    def test_two_halos_assigned(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        groups = [[0], [1]]
        assigner = GMMAssigner(verbose=0)
        result = assigner.assign(halos, coords, np.array([], dtype=np.uint64), groups)
        df = result.particle_df
        assert len(df) == 100
        assert df["Sub_tree_id"].nunique() == 2
        assert set(df["Sub_tree_id"].unique()) == {1, 2}

    def test_assign_returns_respmap(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=50)
        groups = [[0], [1]]
        assigner = GMMAssigner(verbose=0)
        result = assigner.assign(halos, coords, np.array([], dtype=np.uint64), groups)
        assert isinstance(result.responsibilities, dict)
        for key, val in result.responsibilities.items():
            assert isinstance(key, (int, np.integer))
            assert len(val) == 2
            assert isinstance(val[0], np.ndarray)
            assert isinstance(val[1], np.ndarray)

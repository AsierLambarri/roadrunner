import numpy as np

from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


def _make_halo(xcen, model="kepler", mass=1e12, rvir=RVR, redshift=0.0, vel=VCENTER):
    inner = KeplerPotential(M=mass, G=4.3e-6) if model == "kepler" else NFWPotential(M=mass, Rs=10.0, c=10.0, G=4.3e-6)
    return HaloModel(inner, np.asarray(xcen, dtype=np.float64), vel, rvir, sub_tree_id=1, redshift=redshift)


class TestFunction:
    def test_importable(self):
        assert callable(compute_halo_bound_particles)

    def test_returns_same_list(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(-50, 50, (100, 6)).astype(np.float64)
        halos = [_make_halo([0.0, 0.0, 0.0])]
        result = compute_halo_bound_particles(halos, coords)
        assert result is halos

    def test_two_halos_non_overlapping(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(-50, 50, (100, 6)).astype(np.float64)
        halos = [
            _make_halo([-200.0, 0.0, 0.0]),
            _make_halo([200.0, 0.0, 0.0]),
        ]
        result = compute_halo_bound_particles(halos, coords, search_factor=1.0)
        for halo in result:
            assert halo.has_boundness

    def test_halo_far_from_particles(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(-50, 50, (100, 6)).astype(np.float64)
        halos = [_make_halo([1e6, 0.0, 0.0])]
        result = compute_halo_bound_particles(halos, coords)
        inds, vals, tdyns = result[0].get_boundness()
        assert len(inds) == 0
        assert len(vals) == 0
        assert len(tdyns) == 0

    def test_search_factor_captures_more(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(-150, 150, (200, 6)).astype(np.float64)
        halos = [_make_halo([0.0, 0.0, 0.0], rvir=50.0)]
        r1 = compute_halo_bound_particles(halos, coords, search_factor=1.0)
        n1 = len(r1[0].get_boundness()[0])
        r2 = compute_halo_bound_particles(halos, coords, search_factor=4.0)
        n2 = len(r2[0].get_boundness()[0])
        assert n2 >= n1


class TestDifferentModels:
    def test_kepler_vs_nfw_different(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-50, 50, (100, 3)).astype(np.float64)
        V = np.zeros((100, 3), dtype=np.float64)
        coords = np.hstack([X, V])
        h_k = _make_halo([0.0, 0.0, 0.0], model="kepler", mass=1e12)
        h_n = _make_halo([0.0, 0.0, 0.0], model="nfw", mass=1e12)
        r_k = compute_halo_bound_particles([h_k], coords)
        r_n = compute_halo_bound_particles([h_n], coords)
        _, ek, _ = r_k[0].get_boundness()
        _, en, _ = r_n[0].get_boundness()
        assert not np.array_equal(ek, en)

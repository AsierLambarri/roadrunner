import numpy as np

from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential
from roadrunner.physics.timescales import compute_tidal_radius, pair_dynamical_timescales

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


def _make_halo(xcen, mass=1e12, rvir=RVR, sub_tree_id=1, inner=None):
    if inner is None:
        inner = KeplerPotential(M=mass, G=4.3e-6)
    return HaloModel(
        inner, np.asarray(xcen, dtype=np.float64),
        VCENTER, rvir, sub_tree_id=sub_tree_id, redshift=0.0,
    )


class TestTidalRadius:
    def test_kepler_analytic(self):
        M, Ms = 1e12, 1e10
        D = 100.0
        halo = _make_halo([0.0, 0.0, 0.0], mass=M)
        rt = compute_tidal_radius(halo.potential_model, Ms, np.array([[D, 0.0, 0.0]]))
        expected = D * (Ms / (3 * M)) ** (1.0 / 3.0)
        assert np.isclose(rt, expected)

    def test_nfw_finite(self):
        inner = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        halo = _make_halo([0.0, 0.0, 0.0], inner=inner)
        rt = compute_tidal_radius(halo.potential_model, 1e10, np.array([[100.0, 0.0, 0.0]]))
        assert np.isfinite(rt)
        assert rt > 0
        assert compute_tidal_radius(halo.potential_model, 1e10, np.zeros((1, 3))) == 0   # the main host itself: the denominator vanishes at d = 0


class TestPairDynamicalTimescales:
    def test_each_entry_gets_the_time_stored_for_it_in_its_galaxy(self):
        h1 = _make_halo([0.0, 0.0, 0.0], sub_tree_id=1)
        h2 = _make_halo([50.0, 0.0, 0.0], sub_tree_id=2)
        h1.set_bound(np.array([0, 1, 2, 5]), np.full(4, -0.5, dtype=np.float32), np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        h2.set_bound(np.array([1, 2, 7]), np.full(3, -0.5, dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32))
        resp = SparseCSC([np.array([2, 0]), np.array([1, 2, 7])], [np.ones(2, np.float32), np.ones(3, np.float32)], column_id=np.array([2, 1]))
        out = pair_dynamical_timescales(HaloEnsemble([h1, h2]), resp, td_factor=2.0)
        # the columns come in the order of the matrix (galaxy 2 first), the entries in the order of the column
        np.testing.assert_allclose(out, [4.0, np.nan, 0.4, 0.6, np.nan], rtol=1e-6)    # galaxy 2: rows 2 and 0 (not bound); galaxy 1: rows 1, 2 and 7 (not bound)

    def test_entry_without_a_stored_time_is_nan(self):
        h1 = _make_halo([0.0, 0.0, 0.0], sub_tree_id=1)
        h1.set_bound(np.array([0, 3]), np.full(2, -0.5, dtype=np.float32), np.array([0.1, 0.3], dtype=np.float32))
        resp = SparseCSC([np.array([0, 1, 3, 9])], [np.ones(4, np.float32)], column_id=np.array([1]))
        out = pair_dynamical_timescales(HaloEnsemble([h1]), resp)
        np.testing.assert_allclose(out, [0.1, np.nan, 0.3, np.nan], rtol=1e-6)

    def test_empty_matrix(self):
        resp = SparseCSC([], [], column_id=np.array([], dtype=np.int64))
        assert pair_dynamical_timescales(HaloEnsemble([_make_halo([0.0, 0.0, 0.0])]), resp).size == 0

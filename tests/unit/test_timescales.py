import numpy as np

from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential
from roadrunner.physics.timescales import (
    compute_tidal_radius,
    pair_dynamical_timescales,
)

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


def _set_bound(halo, rows, dynamical_times):
    """Bind the particles ``rows`` to a halo, with the given dynamical times."""
    halo.set_bound(
        np.array(rows),
        np.full(len(rows), -0.5, dtype=np.float32),
        np.array(dynamical_times, dtype=np.float32),
    )


def _responsibilities(columns):
    """Make a responsibility matrix of unit weights from ``{galaxy: rows}``."""
    return SparseCSC(
        [np.array(rows) for rows in columns.values()],
        [np.ones(len(rows), np.float32) for rows in columns.values()],
        column_id=np.array(list(columns)),
    )


class TestPairDynamicalTimescales:
    def test_each_entry_gets_the_time_stored_for_it_in_its_galaxy(self):
        halo_1 = _make_halo([0.0, 0.0, 0.0], sub_tree_id=1)
        halo_2 = _make_halo([50.0, 0.0, 0.0], sub_tree_id=2)
        _set_bound(halo_1, [0, 1, 2, 5], [0.1, 0.2, 0.3, 0.4])
        _set_bound(halo_2, [1, 2, 7], [1.0, 2.0, 3.0])
        responsibilities = _responsibilities({2: [2, 0], 1: [1, 2, 7]})
        times = pair_dynamical_timescales(
            HaloEnsemble([halo_1, halo_2]), responsibilities, td_factor=2.0
        )
        # The columns come in the order of the matrix (galaxy 2 first), the
        # entries in the order of the column. Galaxy 2: row 2 is bound, row 0
        # is not; galaxy 1: rows 1 and 2 are bound, row 7 is not.
        np.testing.assert_allclose(times, [4.0, np.nan, 0.4, 0.6, np.nan], rtol=1e-6)

    def test_entry_without_a_stored_time_is_nan(self):
        halo = _make_halo([0.0, 0.0, 0.0], sub_tree_id=1)
        _set_bound(halo, [0, 3], [0.1, 0.3])
        times = pair_dynamical_timescales(
            HaloEnsemble([halo]), _responsibilities({1: [0, 1, 3, 9]})
        )
        np.testing.assert_allclose(times, [0.1, np.nan, 0.3, np.nan], rtol=1e-6)

    def test_empty_matrix(self):
        empty = SparseCSC([], [], column_id=np.array([], dtype=np.int64))
        ensemble = HaloEnsemble([_make_halo([0.0, 0.0, 0.0])])
        assert pair_dynamical_timescales(ensemble, empty).size == 0

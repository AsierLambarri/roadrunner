import numpy as np

from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


def _halo(xcen=(0.0, 0.0, 0.0), sid=1, n_bound=0):
    inner = KeplerPotential(M=1e12, G=4.3e-6)
    h = HaloModel(
        inner, np.array(xcen, dtype=np.float64),
        VCENTER, RVR, sub_tree_id=sid, redshift=0.0,
    )
    if n_bound > 0:
        h.set_boundness(
            np.arange(n_bound, dtype=np.uint64),
            np.full(n_bound, 0.5, dtype=np.float32),
            np.full(n_bound, 0.1, dtype=np.float32),
        )
    return h


class TestConstruction:
    def test_empty(self):
        ens = HaloEnsemble([])
        assert ens.empty
        assert len(ens) == 0
        assert ens.nhalo == 0

    def test_nhalo(self):
        ens = HaloEnsemble([_halo(), _halo(), _halo()])
        assert len(ens) == 3
        assert ens.nhalo == 3

    def test_nstars(self):
        ens = HaloEnsemble([
            _halo(n_bound=10),
            _halo(n_bound=20),
            _halo(n_bound=30),
        ])
        assert ens.nstars == 60

    def test_iteration(self):
        halos = [_halo(sid=i) for i in range(3)]
        ens = HaloEnsemble(halos)
        for h in ens:
            assert h.sub_tree_id in (0, 1, 2)


class TestArrays:
    def test_positions(self):
        ens = HaloEnsemble([
            _halo(xcen=(1.0, 2.0, 3.0)),
            _halo(xcen=(4.0, 5.0, 6.0)),
        ])
        assert ens.positions.shape == (2, 3)
        assert np.allclose(ens.positions[0], [1.0, 2.0, 3.0])

    def test_virial_radii(self):
        inner = KeplerPotential(M=1e12, G=4.3e-6)
        h1 = HaloModel(inner, np.zeros(3), np.zeros(3), 50.0, sub_tree_id=1, redshift=0.0)
        h2 = HaloModel(inner, np.zeros(3), np.zeros(3), 100.0, sub_tree_id=2, redshift=0.0)
        ens = HaloEnsemble([h1, h2])
        assert np.allclose(ens.virial_radii, [50.0, 100.0])

    def test_sub_tree_ids(self):
        ens = HaloEnsemble([_halo(sid=10), _halo(sid=20)])
        assert np.allclose(ens.sub_tree_ids, [10, 20])


class TestSelection:
    def test_select(self):
        ens = HaloEnsemble([
            _halo(sid=1), _halo(sid=2), _halo(sid=3),
            _halo(sid=4), _halo(sid=5),
        ])
        sub = ens.select([1, 3])
        assert isinstance(sub, HaloEnsemble)
        assert len(sub) == 2
        assert list(sub.sub_tree_ids) == [2, 4]

    def test_select_chained(self):
        ens = HaloEnsemble([
            _halo(sid=1), _halo(sid=2), _halo(sid=3),
            _halo(sid=4), _halo(sid=5),
        ])
        sub = ens.select([0, 2, 4]).select([2])
        assert len(sub) == 1
        assert sub.sub_tree_ids[0] == 5


class TestGetParticles:
    def test_returns_two_csc(self):
        ens = HaloEnsemble([_halo(n_bound=5), _halo(n_bound=3)])
        csc_b, csc_t = ens.get_particles()
        assert isinstance(csc_b, SparseCSC)
        assert isinstance(csc_t, SparseCSC)
        assert len(csc_b.column_indices) == 2
        assert len(csc_t.column_indices) == 2

    def test_candidate_lengths(self):
        ens = HaloEnsemble([_halo(n_bound=10), _halo(n_bound=5)])
        csc_b, _ = ens.get_particles()
        assert len(csc_b.column_indices[0]) == 10
        assert len(csc_b.column_indices[1]) == 5

    def test_empty_halo(self):
        ens = HaloEnsemble([_halo(n_bound=0)])
        csc_b, _ = ens.get_particles()
        assert len(csc_b.column_indices[0]) == 0

    def test_shared_indices(self):
        ens = HaloEnsemble([_halo(n_bound=4), _halo(n_bound=6)])
        csc_b, csc_t = ens.get_particles()
        for i in range(2):
            assert np.array_equal(csc_b.column_indices[i], csc_t.column_indices[i])


class TestPopulated:
    def test_populated_indices(self):
        ens = HaloEnsemble([_halo(n_bound=0), _halo(n_bound=5), _halo(n_bound=3)])
        assert list(ens.populated_indices()) == [1, 2]

    def test_empty_indices(self):
        ens = HaloEnsemble([_halo(n_bound=0), _halo(n_bound=5), _halo(n_bound=0)])
        assert list(ens.empty_indices()) == [0, 2]

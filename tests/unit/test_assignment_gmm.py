import numpy as np

from roadrunner._mcf_types import ParticleAssigner
from roadrunner.clustering.assignment.gmm import (
    XGMMAssigner,
    _merge_resp_kernel,
    _no_transform,
    _rank_transform,
)
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.potentials import KeplerPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


# ── Helpers ─────────────────────────────────────────────

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
    for i, h in enumerate(halos):
        start = i * (n_particles // n_halos)
        end = (i + 1) * (n_particles // n_halos)
        h.set_boundness(
            np.arange(start, end, dtype=np.uint64),
            np.ones(end - start, dtype=np.float32) * 0.5,
            np.ones(end - start, dtype=np.float32) * 0.1,
        )
    return halos, coords


# ── Module-level function tests ─────────────────────────

class TestTransformFunctions:
    def test_no_transform_identity(self):
        vals = np.array([1.0, 2.0, 3.0])
        result = _no_transform(vals)
        assert np.array_equal(result, vals)

    def test_rank_transform_shape(self):
        vals = np.array([0.1, 0.5, 2.0], dtype=np.float32)
        result = _rank_transform(vals)
        assert result.shape == (3,)
        assert result.dtype == np.float32
        assert np.all(np.diff(result) >= 0)

    def test_rank_transform_empty(self):
        result = _rank_transform(np.array([], dtype=np.float32))
        assert result.size == 0


class TestMergeRespKernel:
    def setup_method(self):
        self.n, self.k = 100, 5
        rng = np.random.default_rng(42)
        self.prev = rng.uniform(0, 1, (self.n, self.k)).astype(np.float32)
        self.bound = (rng.uniform(0, 1, (self.n, self.k)) > 0.3).astype(np.float32)
        self.newborn = rng.random(self.n) < 0.1

    def test_carry_over_preserved(self):
        prev_copy = self.prev.copy()
        _merge_resp_kernel(prev_copy, self.bound, self.k, self.newborn)
        both = (self.bound > 0) & (self.prev > 0)
        assert np.allclose(prev_copy[both], self.prev[both])
        assert np.all(prev_copy[self.bound == 0] == 0.0)

    def test_single_component(self):
        prev = np.array([[0.5], [0.0]], dtype=np.float32)
        bound = np.array([[1.0], [1.0]], dtype=np.float32)
        newborn = np.array([False, True])
        _merge_resp_kernel(prev, bound, 1, newborn)
        assert prev[0, 0] == 0.5   # carry over
        assert prev[1, 0] == 0.0   # newborn + bound → 0


# ── XGMMAssigner integration tests ───────────────────────

class TestXGMMAssigner:
    def test_is_particle_assigner(self):
        assert isinstance(XGMMAssigner(), ParticleAssigner)

    def test_two_halos_assigned(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        groups = [[0], [1]]
        assigner = XGMMAssigner(verbose=0)
        result = assigner.assign(halos, coords, np.array([], dtype=np.uint64), groups)
        df = result.particle_df
        assert len(df) == 100
        assert df["Sub_tree_id"].nunique() == 2
        assert isinstance(result.responsibilities, SparseCSC)
        assert len(result.responsibilities.column_id) > 0
        for j in range(len(result.responsibilities)):
            assert isinstance(result.responsibilities.column_id[j], (int, np.integer))
            assert isinstance(result.responsibilities.column_indices[j], np.ndarray)
            assert isinstance(result.responsibilities.column_values[j], np.ndarray)

    def test_resolved_fit_includes_weight(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        groups = [[0], [1]]
        assigner = XGMMAssigner(verbose=0)
        result = assigner.assign(halos, coords, np.array([], dtype=np.uint64), groups)
        for sid, params in result.fitted_parameters.items():
            assert "weight" in params
            assert 0.0 <= params["weight"] <= 1.0

    def test_seed_makes_fit_reproducible(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        groups = [[0], [1]]
        result1 = XGMMAssigner(verbose=0).assign(
            halos, coords, np.array([], dtype=np.uint64), groups, seed=1234,
        )
        result2 = XGMMAssigner(verbose=0).assign(
            halos, coords, np.array([], dtype=np.uint64), groups, seed=1234,
        )
        for sid in result1.fitted_parameters:
            np.testing.assert_array_equal(
                result1.fitted_parameters[sid]["mean"],
                result2.fitted_parameters[sid]["mean"],
            )

    def test_newborn_handled(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        groups = [[0], [1]]
        newborn = np.array([0, 1], dtype=np.uint64)
        assigner = XGMMAssigner(verbose=0)
        result = assigner.assign(halos, coords, newborn, groups)
        assert result.particle_df is not None

    def test_previous_resp_accepted(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=50)
        groups = [[0], [1]]
        # Build a valid previous_resp SparseCSC
        prev_candidates = [
            np.arange(25, dtype=np.int64),
            np.arange(25, 50, dtype=np.int64),
        ]
        prev_values = [
            np.full(25, 0.5, dtype=np.float32),
            np.full(25, 0.5, dtype=np.float32),
        ]
        prev_resp = SparseCSC(prev_candidates, prev_values, column_id=np.array([1, 2], dtype=np.int64))

        assigner = XGMMAssigner(verbose=0)
        result = assigner.assign(
            halos, coords, np.array([], dtype=np.uint64), groups,
            previous_resp=prev_resp,
        )
        assert result.particle_df is not None

    def test_overlapping_groups(self):
        """Two halos that share particles (overlapping boundness)."""
        rng = np.random.default_rng(42)
        h1 = _make_halo([0.0, 0.0, 0.0], sub_tree_id=1)
        h2 = _make_halo([0.5, 0.5, 0.5], sub_tree_id=2)
        coords = rng.uniform(-50, 50, (200, 6)).astype(np.float64)
        # Each halo sees the same particle set (fully overlapping)
        h1.set_boundness(
            np.arange(200, dtype=np.uint64),
            np.full(200, 0.6, dtype=np.float32),
            np.full(200, 0.1, dtype=np.float32),
        )
        h2.set_boundness(
            np.arange(200, dtype=np.uint64),
            np.full(200, 0.4, dtype=np.float32),
            np.full(200, 0.1, dtype=np.float32),
        )
        groups = [[0, 1]]
        assigner = XGMMAssigner(verbose=0, max_iter=5)
        result = assigner.assign([h1, h2], coords, np.array([], dtype=np.uint64), groups)
        assert result.particle_df is not None

    def test_empty_group(self):
        """Group with a halo that has no bound particles."""
        h1 = _make_halo([0.0, 0.0, 0.0], sub_tree_id=1)
        h2 = _make_halo([10.0, 10.0, 10.0], sub_tree_id=2)
        coords = np.random.default_rng(42).uniform(-50, 50, (100, 6)).astype(np.float64)
        h1.set_boundness(
            np.arange(50, dtype=np.uint64),
            np.full(50, 0.5, dtype=np.float32),
            np.full(50, 0.1, dtype=np.float32),
        )
        # h2 has no boundness — stays empty
        groups = [[0, 1]]
        assigner = XGMMAssigner(verbose=0)
        result = assigner.assign([h1, h2], coords, np.array([], dtype=np.uint64), groups)
        assert result.particle_df is not None

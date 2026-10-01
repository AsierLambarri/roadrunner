import numpy as np
import pytest

from roadrunner._mcf_types import ParticleAssigner
from roadrunner.clustering.assignment.gmm import (
    XGMMAssigner,
    _drop_unbound_kernel,
    _no_transform,
    _rank_transform,
    _rows_to_pad_kernel,
)
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.potentials import KeplerPotential
from roadrunner.physics.scaler import StandardScaler

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


class TestPaddingKernels:
    def setup_method(self):
        self.n, self.k = 100, 5
        rng = np.random.default_rng(42)
        self.prev = rng.uniform(0, 1, (self.n, self.k)).astype(np.float32)
        self.bound = (rng.uniform(0, 1, (self.n, self.k)) > 0.3).astype(np.float32)

    def test_rows_to_pad_and_drop_unbound(self):
        prev = self.prev.copy()
        prev[::7, 0] = 0.0                       # some newly bound entries
        orig = prev.copy()
        pad = _rows_to_pad_kernel(prev, self.bound)
        np.testing.assert_array_equal(pad, ((self.bound > 0) & (orig <= 0)).any(axis=1))
        _drop_unbound_kernel(prev, self.bound)
        both = (self.bound > 0) & (orig > 0)
        assert np.allclose(prev[both], orig[both])
        assert np.all(prev[self.bound == 0] == 0.0)

    def test_single_component(self):
        prev = np.array([[0.5], [0.0], [0.0]], dtype=np.float32)
        bound = np.ones((3, 1), dtype=np.float32)
        pad = _rows_to_pad_kernel(prev, bound)
        np.testing.assert_array_equal(pad, [False, True, True])   # carried; no previous resp -> padded


# ── XGMMAssigner integration tests ───────────────────────

class TestXGMMAssigner:
    def test_is_particle_assigner(self):
        assert isinstance(XGMMAssigner(), ParticleAssigner)

    def test_two_halos_assigned(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        groups = [[0], [1]]
        assigner = XGMMAssigner(verbose=0)
        result = assigner.assign(halos, coords, groups)
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
        result = assigner.assign(halos, coords, groups)
        for sid, params in result.fitted_parameters.items():
            assert "weight" in params
            assert 0.0 <= params["weight"] <= 1.0

    def test_seed_makes_fit_reproducible(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        groups = [[0], [1]]
        result1 = XGMMAssigner(verbose=0).assign(
            halos, coords, groups, seed=1234,
        )
        result2 = XGMMAssigner(verbose=0).assign(
            halos, coords, groups, seed=1234,
        )
        for sid in result1.fitted_parameters:
            np.testing.assert_array_equal(
                result1.fitted_parameters[sid]["mean"],
                result2.fitted_parameters[sid]["mean"],
            )

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
            halos, coords, groups,
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
        result = assigner.assign([h1, h2], coords, groups)
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
        result = assigner.assign([h1, h2], coords, groups)
        assert result.particle_df is not None


def _two_halo_group(n=200):
    """Two halos jointly fit as one resolved group, with separated clusters."""
    rng = np.random.default_rng(11)
    coords = np.vstack([
        rng.normal(-10.0, 3.0, (n // 2, 6)),
        rng.normal(10.0, 3.0, (n // 2, 6)),
    ])
    h1 = _make_halo([-10.0, -10.0, -10.0], sub_tree_id=1)
    h2 = _make_halo([10.0, 10.0, 10.0], sub_tree_id=2)
    for h, rows in ((h1, np.arange(0, 3 * n // 4)), (h2, np.arange(n // 4, n))):
        h.set_boundness(rows.astype(np.uint64),
                        np.linspace(0.1, 0.9, rows.size, dtype=np.float32),
                        np.full(rows.size, 0.1, dtype=np.float32))
    return [h1, h2], coords


# A 2-particle fit: rank 1 plus reg_covar, rounded indefinite in float32
# (smallest eigenvalue < 0), like halo 1228 in the halo685x12 v4 run.
_RANK_DEFICIENT_COV = (np.outer(np.r_[1, 1, 1, 9, 6, 2], np.r_[1, 1, 1, 9, 6, 2]) * 1.2
                       + 1e-6 * np.eye(6) - 4e-6 * np.diag([1, 0, 0, 0, 0, 0]))


class TestPredictivePadding:
    @pytest.mark.parametrize("halo2", [
        None,                                                                   # no previous fit
        {"count": 2.0, "covariance": _RANK_DEFICIENT_COV, "rank_deficient": True},
        {"count": 2.0, "covariance": _RANK_DEFICIENT_COV},                      # checkpoint before the flag
    ])
    def test_initial_responsibilities(self, halo2):
        # Halo 1 (bound to rows 0-149, at -10) has a previous fit; halo 2
        # (rows 50-199, at +10) is new, with 50 exclusive particles, or has
        # only a rank-deficient previous fit, which must count as none.
        # Rows 0-9 have no previous responsibility (newborn).
        assert np.linalg.eigvalsh(_RANK_DEFICIENT_COV).min() < 0
        halos, coords = _two_halo_group()
        a = XGMMAssigner(verbose=0)
        a.particle_coords, a.ensemble = coords, HaloEnsemble(halos)
        a.previous_parameters = {
            1: {"count": 100.0, "covariance": 9.0 * np.eye(6), "tree_mass": 1e12}}
        if halo2 is not None:
            a.previous_parameters[2] = {**halo2, "tree_mass": 1e12}
        a.previous_resp = SparseCSC([np.arange(10, 100)], [np.ones(90, np.float32)],
                                    column_id=np.array([1], dtype=np.int64))
        csc_b, _ = a.ensemble.get_particles()
        scaler = StandardScaler()
        X = scaler.fit_transform(coords.astype(np.float32))
        _, resp = a._initial_responsibilities(X, csc_b, np.ones(len(X), np.float32), scaler)
        np.testing.assert_allclose(resp.sum(axis=1), 1.0, rtol=1e-6)
        assert np.allclose(resp[:10], [1.0, 0.0])      # newborn, bound to halo 1 only
        assert np.allclose(resp[10:50], [1.0, 0.0])    # carried over
        assert np.all(resp[50:100, 0] > 0.99)          # newly bound to new halo 2, near halo 1
        assert np.all(resp[100:150, 1] > 0.99)         # bound to both, near halo 2
        assert np.allclose(resp[150:], [0.0, 1.0])     # exclusive to new halo 2


class TestMassWeighting:
    def test_group_weights(self):
        masses = np.array([1.0, 2.0, 3.0, 6.0])
        gp_idx = np.array([0, 2, 3])
        number = XGMMAssigner(verbose=0)
        np.testing.assert_array_equal(number._group_weights(gp_idx), np.ones(3))
        mass = XGMMAssigner(verbose=0, mass_weighting=True)
        mass.particle_masses = masses
        w = mass._group_weights(gp_idx)
        m = masses[gp_idx]
        np.testing.assert_allclose(w, m * m.sum() / (m ** 2).sum(), rtol=1e-6)
        assert np.isclose(w.sum(), m.sum() ** 2 / (m ** 2).sum())    # the ESS
        mass.particle_masses = np.full(4, 3.7e4)
        np.testing.assert_array_equal(mass._group_weights(gp_idx), np.ones(3))

    def test_single_halo_mean_is_mass_weighted(self):
        halos, coords = _setup_mock_halos(n_halos=1, n_particles=60)
        masses = np.random.default_rng(2).uniform(1.0, 10.0, 60)
        result = XGMMAssigner(verbose=0, mass_weighting=True).assign(
            halos, coords, [[0]], particle_masses=masses,
        )
        params = result.fitted_parameters[1]
        np.testing.assert_allclose(params["mean"], np.average(coords, axis=0, weights=masses),
                                   rtol=1e-5)
        assert np.isclose(params["count"], masses.sum() ** 2 / (masses ** 2).sum(), rtol=1e-5)

    @pytest.mark.parametrize("method", ["gmm", "bgmm"])
    def test_equal_masses_reproduce_counting(self, method):
        halos, coords = _two_halo_group()
        masses = np.full(coords.shape[0], 1e4)
        results = [
            XGMMAssigner(verbose=0, method=method, mass_weighting=mw).assign(
                halos, coords, [[0, 1]],
                seed=3, particle_masses=masses,
            )
            for mw in (False, True)
        ]
        number, mass = (r.responsibilities for r in results)
        # GMM: bitwise. BGMM: the data-derived default priors take the
        # weighted formula instead of np.cov, equal up to rounding.
        rtol = 0 if method == "gmm" else 1e-4
        for a, b in zip(number.column_values, mass.column_values):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=1e-6 if rtol else 0)
        np.testing.assert_array_equal(results[0].particle_df["Sub_tree_id"],
                                      results[1].particle_df["Sub_tree_id"])

    def test_requires_masses(self):
        halos, coords = _setup_mock_halos(n_halos=2, n_particles=100)
        with pytest.raises(ValueError, match="particle_masses"):
            XGMMAssigner(verbose=0, mass_weighting=True).assign(
                halos, coords, [[0], [1]],
            )

    def test_rejected_with_svi(self):
        with pytest.raises(ValueError, match="svi-bgmm"):
            XGMMAssigner(method="svi-bgmm", mass_weighting=True)

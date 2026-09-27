import numpy as np
import pytest

from roadrunner._defaults import PRIOR_DOF_OFFSET
from roadrunner.clustering.assignment import priors as P
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential
from roadrunner.physics.scaler import StandardScaler


class TestCovarianceScale:
    def test_linear_between_the_two_counts_and_flat_outside(self):
        f = P.covariance_scale
        assert f(0) == f(10) == pytest.approx(1.0)
        assert f(300) == f(5000) == pytest.approx(0.7)
        assert f(155) == pytest.approx(0.85)
        vals = [f(n) for n in np.linspace(10, 300, 30)]
        assert np.all(np.diff(vals) < 0)

    def test_covariance_prior_formula(self):
        diag_vars, s = np.array([1.0, 4.0]), np.array([2.0, 0.5])
        dof = 6.0
        got = P.covariance_prior(diag_vars, 100.0, 150.0, s, dof, "diag")
        np.testing.assert_allclose(got, dof * 1.5 * P.covariance_scale(150.0) * diag_vars * s ** 2)


def _halo(sid, center):
    return HaloModel(KeplerPotential(M=1e11, G=4.3e-6), np.asarray(center, float),
                     np.zeros(3), 20.0, sub_tree_id=sid, redshift=0.0)


def _prior_kwargs(history, use_bgmm_priors=True, method="bgmm"):
    X = np.random.default_rng(0).normal(size=(400, 6))
    a = XGMMAssigner(method=method, verbose=0, use_bgmm_priors=use_bgmm_priors)
    a.particle_coords = X
    a.ensemble = HaloEnsemble([_halo(1, [0, 0, 0]), _halo(2, [1, 0, 0])])
    a.previous_parameters = history
    scaler = StandardScaler()
    scaler.fit(X)
    csc_b = SparseCSC([np.arange(0, 300), np.arange(250, 400)],
                      [np.ones(300, np.float32), np.ones(150, np.float32)], column_id=[1, 2])
    nk_init = np.array([280.0, 120.0])
    covs_init = np.stack([0.8 * np.eye(6), 0.2 * np.eye(6)])          # scaled coordinates
    kw = a._build_prior_kwargs(np.array([1, 2]), csc_b, scaler, 2, np.arange(400),
                               np.ones(400, np.float32), nk_init, covs_init)
    return kw, scaler


class TestPriorReferences:
    DOF = 6 + PRIOR_DOF_OFFSET

    def test_history_and_newborn(self):
        kw, scaler = _prior_kwargs({1: {"count": 250.0, "covariance": 2.0 * np.eye(6)}})
        s2 = scaler.scale_ ** 2
        # halo 1 has history: previous covariance, growth from the expected count
        np.testing.assert_allclose(np.diag(kw["covariance_prior"][0]),
                                   self.DOF * (280 / 250) * P.covariance_scale(280) * 2.0 * s2, rtol=1e-5)
        # halo 2 is new: its own pre-fit covariance and count are the reference
        np.testing.assert_allclose(np.diag(kw["covariance_prior"][1]),
                                   self.DOF * P.covariance_scale(120) * 0.2, rtol=1e-5)
        np.testing.assert_allclose(kw["weight_concentration_prior"], [250 / 5, 120 / 5], rtol=1e-6)
        np.testing.assert_allclose(kw["mean_precision_prior"], [250 / 10, 120 / 10], rtol=1e-6)
        assert kw["degrees_of_freedom_prior"] == self.DOF

    @pytest.mark.parametrize("history,use", [(None, True), ({1: {"count": 250.0, "covariance": np.eye(6)}}, False)])
    def test_no_history_uses_own_estimate(self, history, use):
        # First snapshot, or temporal priors switched off: every halo is its own reference.
        kw, _ = _prior_kwargs(history, use_bgmm_priors=use)
        for i, (n, v) in enumerate(((280, 0.8), (120, 0.2))):
            np.testing.assert_allclose(np.diag(kw["covariance_prior"][i]),
                                       self.DOF * P.covariance_scale(n) * v, rtol=1e-5)

    def test_gmm_gets_no_priors(self):
        kw, _ = _prior_kwargs(None, method="gmm")
        assert kw == {}


def _halos_for(X, lab, mus, sigs, which):
    """Halos whose candidates are their own particles plus any within 3 sigma."""
    hs = []
    for k in which:
        d = np.linalg.norm(X - mus[k], axis=1) / sigs[k]
        idx = np.flatnonzero((lab == k) | (d < 3.0))
        h = HaloModel(KeplerPotential(M=1e11, G=4.3e-6), mus[k][:3], mus[k][3:], 20.0,
                      sub_tree_id=k + 1, redshift=0.0)
        h.set_boundness(idx.astype(np.uint64), np.exp(-0.5 * d[idx] ** 2).astype(np.float32),
                        np.full(idx.size, 0.1, np.float32))
        hs.append(h)
    return hs


class TestSmallHaloRegressions:
    """A compact 30-particle halo B, 3 sigma from a 3000-particle halo A (6D)."""
    MUS = [np.zeros(6), np.r_[3.0, np.zeros(5)]]
    SIGS = [1.0, 0.3]
    NB = 30

    def _data(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(self.MUS[0], 1.0, (3000, 6)), rng.normal(self.MUS[1], 0.3, (self.NB, 6))])
        return X, np.r_[np.zeros(3000, int), np.ones(self.NB, int)]

    def test_first_snapshot_small_halo_not_inflated(self):
        # The BGMM's own default covariance prior (the whole group's) inflated B ~1.5x.
        X, lab = self._data()
        res = XGMMAssigner(verbose=0, method="bgmm").assign(
            _halos_for(X, lab, self.MUS, self.SIGS, (0, 1)), X, np.array([], np.uint64), [[0, 1]], seed=1)
        B = res.fitted_parameters[2]
        assert np.diag(B["covariance"]).mean() / 0.09 < 1.2
        assert B["count"] / self.NB > 0.85

    def test_newborn_halo_gets_a_proper_prior(self):
        # H06 control: B is new at snapshot 2 (no previous parameters). A zero
        # covariance prior shrank it by N/(N + nu0); now its own estimate is the reference.
        X, lab = self._data()
        a = XGMMAssigner(verbose=0, method="bgmm")
        first = a.assign(_halos_for(X[:3000], lab[:3000], self.MUS, self.SIGS, (0,)), X[:3000],
                         np.array([], np.uint64), [[0]], seed=1)
        res = a.assign(_halos_for(X, lab, self.MUS, self.SIGS, (0, 1)), X,
                       np.arange(3000, 3000 + self.NB, dtype=np.uint64), [[0, 1]], seed=2,
                       previous_resp=first.responsibilities)
        assert set(res.fitted_parameters) == {1, 2}
        for p in res.fitted_parameters.values():
            assert np.all(np.isfinite(p["mean"])) and np.all(np.isfinite(p["covariance"]))
            assert np.linalg.eigvalsh(p["covariance"]).min() > 0
        B = res.fitted_parameters[2]
        sample = np.var(X[lab == 1], axis=0).mean()
        assert np.diag(B["covariance"]).mean() / sample > 0.8

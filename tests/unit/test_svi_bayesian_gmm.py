import numpy as np

from roadrunner.mixture.svi_bayesian_gmm import SVIBayesianGaussianMixture


class TestLatentPriorSmoke:
    def test_fit_with_latent_prior_gives_finite_bound(self):
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal([-2, 0], 0.5, (400, 2)),
            rng.normal([2, 0], 0.5, (400, 2)),
        ])
        prior = rng.random((X.shape[0], 2)) + 0.1
        svi = SVIBayesianGaussianMixture(
            n_components=2, random_state=0, n_svi_iters=20, batch_size=200, eval_every=5
        )
        svi.fit(X, latent_prior=prior)
        assert np.isfinite(svi.lower_bound_)

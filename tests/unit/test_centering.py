import numpy as np
import pytest

from roadrunner.postprocessing.centering import ssc_center


class TestSSCCenterClumpDebris:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("n_clump,n_debris", [(2000, 2000), (20000, 5000), (300, 300)])
    def test_recovers_clump_center(self, dtype, n_clump, n_debris):
        rng = np.random.default_rng(1)

        clump_pos = rng.normal(0.0, 0.5, size=(n_clump, 3))
        clump_vel = np.zeros((n_clump, 3))

        debris_pos = rng.normal(0.0, 5.0, size=(n_debris, 3))
        debris_pos[:, 0] += 30.0
        debris_vel = np.zeros((n_debris, 3))
        debris_vel[:, 0] = 100.0

        positions = np.concatenate([clump_pos, debris_pos]).astype(dtype)
        velocities = np.concatenate([clump_vel, debris_vel]).astype(dtype)
        masses = np.ones(n_clump + n_debris, dtype=dtype)

        center, vcenter = ssc_center(positions, velocities, masses, alpha=0.9, nmin=30)

        # The final sphere holds ~nmin clump particles, so the centre scatters
        # by ~sigma/sqrt(nmin) ~ 0.09; the old COM-only result sat at x >~ 6.
        assert abs(center[0]) < 0.5
        assert abs(vcenter[0]) < 1.0


class TestSSCCenterBelowNmin:
    def test_returns_weighted_mean(self):
        rng = np.random.default_rng(1)
        n = 10
        positions = rng.normal(size=(n, 3))
        velocities = rng.normal(size=(n, 3))
        masses = rng.uniform(1.0, 5.0, size=n)

        center, vcenter = ssc_center(positions, velocities, masses, alpha=0.9, nmin=30)

        expected_center = np.average(positions, axis=0, weights=masses)
        expected_vcenter = np.average(velocities, axis=0, weights=masses)
        np.testing.assert_allclose(center, expected_center)
        np.testing.assert_allclose(vcenter, expected_vcenter)


class TestSSCCenterFallback:
    def test_uses_last_valid_selection(self):
        # A tight 5-particle core at the origin plus two far outliers.
        # nmin=5, alpha=0.5: the first sphere holds all 7 particles; after
        # the outliers drop out the sphere keeps the whole 5-particle core
        # (still valid) for several shrinks, until the radius falls below
        # ~0.1 and only 3 particles remain (< nmin). The loop then breaks
        # and must fall back to the core-only selection, not the sub-nmin one.
        positions = np.array([
            [-0.1, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ])
        velocities = np.array([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ])
        masses = np.ones(7)

        center, vcenter = ssc_center(positions, velocities, masses, alpha=0.5, nmin=5)

        core_pos = positions[:5]
        core_vel = velocities[:5]
        expected_center = np.average(core_pos, axis=0, weights=masses[:5])
        expected_vcenter = np.average(core_vel, axis=0, weights=masses[:5])

        np.testing.assert_allclose(center, expected_center, atol=1e-8)
        np.testing.assert_allclose(vcenter, expected_vcenter, atol=1e-8)

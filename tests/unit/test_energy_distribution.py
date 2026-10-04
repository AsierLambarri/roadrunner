"""NFW energy-distribution tables against direct quadrature."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from roadrunner.physics import energy_distribution as ed


def _g_quad(cal, c):
    """Density of states by adaptive quadrature."""
    e = cal - 1.0
    r_e = brentq(lambda x: ed.nfw_phi(x) - e, 1e-12, 1e12)
    integrand = lambda x: x * x * np.sqrt(max(2.0 * (e - ed.nfw_phi(x)), 0.0))
    return 16 * np.pi**2 * quad(integrand, 0.0, min(c, r_e), limit=200)[0]


class TestNFWEnergyTables:
    def test_density_normalised_and_cdf_monotone(self):
        t = ed.nfw_energy_tables()
        w = ed.trapezoid_weights(ed.CAL_GRID)
        np.testing.assert_allclose(np.exp(t.log_density) @ w, 1.0, rtol=1e-10)
        assert np.all(np.diff(t.cdf, axis=1) >= 0)
        np.testing.assert_allclose(t.cdf[:, 0], 0.0)
        np.testing.assert_allclose(t.cdf[:, -1], 1.0)

    def test_density_of_states_matches_quadrature(self):
        cal = np.array([1e-3, 0.05, 0.3, 0.7, 0.95])
        for c in (3.0, 10.0, 40.0):
            ref = np.array([_g_quad(x, c) for x in cal])
            np.testing.assert_allclose(ed.nfw_density_of_states(cal, c), ref, rtol=1e-3)

    def test_halo_dark_matter_is_uniform_in_u(self):
        # Sample the c = 10 energy distribution: u = F_c(calE) must be U(0, 1).
        rng = np.random.default_rng(0)
        c = 10.0
        cdf = np.interp(ed.CAL_GRID, ed.CAL_GRID, ed._row(ed.nfw_energy_tables().cdf, c))
        cal = np.interp(rng.uniform(size=20000), cdf, ed.CAL_GRID)
        u = ed.nfw_energy_fraction(1.0 - cal, c)
        assert abs(u.mean() - 0.5) < 0.01
        assert abs(np.quantile(u, 0.1) - 0.1) < 0.01

    def test_deeper_is_smaller_u(self):
        u = ed.nfw_energy_fraction(np.array([0.95, 0.7, 0.3]), 8.0)
        assert np.all(np.diff(u) > 0)


def _q_quad(cal, c):
    """Phase-space volume by adaptive quadrature."""
    e = cal - 1.0
    r_e = brentq(lambda x: ed.nfw_phi(x) - e, 1e-12, 1e12)
    integrand = lambda x: x * x * max(2.0 * (e - ed.nfw_phi(x)), 0.0) ** 1.5
    return 16 * np.pi**2 / 3 * quad(integrand, 0.0, min(c, r_e), limit=200)[0]


class TestNFWPhaseSpaceFraction:
    def test_volume_matches_quadrature(self):
        for cal, c in [(1e-3, 10.0), (0.3, 10.0), (0.9, 3.0)]:
            np.testing.assert_allclose(ed.nfw_phase_volume(np.array([cal]), c)[0], _q_quad(cal, c), rtol=1e-6)

    def test_central_slope_and_continuous_extension(self):
        lw = ed.nfw_energy_tables().log_volume_fraction
        slope = np.diff(lw[:, :2], axis=1)[:, 0] / np.diff(np.log(ed.CAL_GRID[:2]))[0]
        np.testing.assert_allclose(slope, ed._CENTRAL_VOLUME_SLOPE, rtol=1e-3)
        cal0 = ed.CAL_GRID[0]
        a, b = ed.nfw_log_phase_space_fraction(1.0 - np.array([cal0 * (1 - 1e-9), cal0]), 10.0)
        assert abs(a - b) < 1e-6

    def test_uniform_phase_space_background_is_uniform_in_w(self):
        # Uniform in the bound phase space of x <= c: x with density ∝ x^2 (-2 phi)^{3/2},
        # speed uniform in the escape ball.
        rng = np.random.default_rng(1)
        c, n = 10.0, 40000
        x_grid = np.linspace(1e-6, c, 200001)
        pdf = x_grid**2 * (-2.0 * ed.nfw_phi(x_grid)) ** 1.5
        cdf = np.cumsum(pdf); cdf /= cdf[-1]
        x = np.interp(rng.uniform(size=n), cdf, x_grid)
        v = np.sqrt(-2.0 * ed.nfw_phi(x)) * rng.uniform(size=n) ** (1.0 / 3.0)
        eps = -(ed.nfw_phi(x) + 0.5 * v**2)                # E / Phi_0 with Phi_0 = -1
        w = np.exp(ed.nfw_log_phase_space_fraction(eps, c))
        assert abs(w.mean() - 0.5) < 0.01
        np.testing.assert_allclose(np.quantile(w, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.01)

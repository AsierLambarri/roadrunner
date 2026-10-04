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

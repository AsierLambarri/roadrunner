"""NFW energy-distribution tables against direct quadrature, and the generic distribution machinery."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from roadrunner.physics import distribution as ed


def _g_quad(cal, c):
    """Density of states by adaptive quadrature."""
    e = cal - 1.0
    r_e = brentq(lambda x: ed.nfw_phi(x) - e, 1e-12, 1e12)
    integrand = lambda x: x * x * np.sqrt(max(2.0 * (e - ed.nfw_phi(x)), 0.0))
    return 16 * np.pi**2 * quad(integrand, 0.0, min(c, r_e), limit=200)[0]


class TestNFWEnergyTables:
    def test_density_normalised_and_cdf_monotone(self):
        # rows run from E = 0 to the deepest node: the fraction more bound falls from 1 to the
        # power-law tail below the deepest node, which completes the normalisation
        t = ed.nfw_energy_tables()
        eps = ed._nfw_nodes(ed.C_GRID)
        norm = [ed.trapezoid_weights(row) @ np.exp(n) for row, n in zip(eps, t.log_density)]
        np.testing.assert_allclose(norm + t.cdf[:, -1], 1.0, rtol=1e-10)
        assert np.all(np.diff(t.cdf, axis=1) <= 0)
        np.testing.assert_allclose(t.cdf[:, 0], 1.0)
        assert np.all((t.cdf[:, -1] > 0) & (t.cdf[:, -1] < 1e-9))

    def test_density_of_states_matches_quadrature(self):
        cal = np.array([1e-3, 0.05, 0.3, 0.7, 0.95])
        for c in (3.0, 10.0, 40.0):
            ref = np.array([_g_quad(x, c) for x in cal])
            np.testing.assert_allclose(ed.nfw_density_of_states(cal, c), ref, rtol=1e-3)

    def test_halo_dark_matter_is_uniform_in_u(self):
        # Sample the c = 10 energy distribution: u = F_c(calE) must be U(0, 1).
        rng = np.random.default_rng(0)
        c = 10.0
        eps = np.interp(rng.uniform(size=20000), ed._row(ed.nfw_energy_tables().cdf, c)[::-1], ed._nfw_nodes(c)[::-1])
        u = ed.nfw_energy_fraction(eps, c)
        assert abs(u.mean() - 0.5) < 0.01
        assert abs(np.quantile(u, 0.1) - 0.1) < 0.01

    def test_interpolation_at_the_kink_and_towards_e_zero(self):
        # c between two rows; depths around the kink k = 1 + phi(c) and up to the last node, -E = (1 - k) 1e-5 |Phi_0|
        c = np.sqrt(ed.C_GRID[23] * ed.C_GRID[24])
        k = 1.0 + ed.nfw_phi(c)
        cal = np.concatenate([k + np.linspace(-0.02, 0.02, 41), 1.0 - np.geomspace(1e-2, (1.0 - k) * 1e-5, 41)])
        log_n = np.log(ed._eddington(1.0 - cal) * ed.nfw_density_of_states(cal, c))
        log_n_table = ed.nfw_log_energy_density(1.0 - cal, c)
        # the tables are normalised: compare shapes
        np.testing.assert_allclose(log_n_table - log_n, np.median(log_n_table - log_n), atol=2e-3)
        log_w = np.log(ed.nfw_phase_volume(cal, c)) - np.log(ed.nfw_phase_volume(np.ones(1), c))
        np.testing.assert_allclose(ed.nfw_log_phase_space_fraction(1.0 - cal, c), log_w, atol=2e-3)

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
        lw, cal = ed.nfw_energy_tables().log_volume_fraction, 1.0 - ed._nfw_nodes(ed.C_GRID)
        slope = np.diff(lw[:, -2:], axis=1)[:, 0] / np.diff(np.log(cal[:, -2:]), axis=1)[:, 0]
        np.testing.assert_allclose(slope, ed._CENTRAL_VOLUME_SLOPE, rtol=1e-3)
        cal0 = ed._DEEPEST
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


class TestSphericalDistribution:
    def test_reproduces_nfw_tables_and_kepler_closed_form(self):
        from roadrunner.physics.potentials import KeplerPotential, NFWPotential, SphericalPotential
        # the bulk, both sides of the kink eps = -phi(c) and towards E = 0, down to the tables' last node
        nfw, R = NFWPotential(M=1e11, Rs=8.0, c=10.0), 80.0
        k = -ed.nfw_phi(10.0)
        eps = np.concatenate([[0.8, 0.5, 0.2], k * (1.0 + np.linspace(-0.02, 0.02, 21)), np.geomspace(1e-2, 1e-5 * k, 21)])
        E = eps * nfw.central_potential()
        for name in ("energy_fraction", "log_energy_density", "log_phase_space_fraction"):
            np.testing.assert_allclose(getattr(SphericalPotential, name)(nfw, E, R), getattr(nfw, name)(E, R), atol=2e-3)
        k, R = KeplerPotential(M=1e11), 60.0
        E = -np.array([0.5, 2.0, 30.0]) * k.G * k.M / R
        np.testing.assert_allclose(SphericalPotential.log_phase_space_fraction(k, E, R),
                                   k.log_phase_space_fraction(E, R), atol=1e-3)

    def test_eddington_matches_analytic_plummer_and_hernquist(self):
        from roadrunner.physics.potentials import HernquistPotential, PlummerPotential
        # from calE = 1e-6 (Hernquist's cusp, where d²ρ/dψ² peaks at the endpoint) down to -E = 1e-12 |Phi_0|,
        # where Hernquist's closed form needs its cancellation-free bracket
        for p in (PlummerPotential(1e10, 0.5), HernquistPotential(1e10, 0.5)):
            E = np.array([1 - 1e-6, 0.8, 0.5, 0.2, 1e-6, 1e-12]) * p.central_potential()
            np.testing.assert_allclose(ed.eddington_distribution_function(p, E), p.distribution_function(E), rtol=1e-3)

    def test_uniform_background_is_uniform_in_w_and_raising_potentials(self):
        from roadrunner.physics.potentials import KeplerPotential, PlummerPotential, SphericalPotential
        p, R, rng = PlummerPotential(1e10, 0.5, G=1.0), 5.0, np.random.default_rng(4)
        r_grid = np.linspace(1e-6, R, 200001)                     # uniform in the bound phase space of r <= R
        pdf = r_grid**2 * (-2.0 * p.potential(r_grid)) ** 1.5
        cdf = np.cumsum(pdf); cdf /= cdf[-1]
        r = np.interp(rng.uniform(size=40000), cdf, r_grid)
        v = np.sqrt(-2.0 * p.potential(r)) * rng.uniform(size=r.size) ** (1.0 / 3.0)
        w = np.exp(p.log_phase_space_fraction(p.potential(r) + 0.5 * v**2, R))
        np.testing.assert_allclose(np.quantile(w, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.01)
        with pytest.raises(NotImplementedError):
            SphericalPotential.energy_fraction(KeplerPotential(1e10), np.array([-1.0]), 10.0)

"""NFW energy-distribution tables against direct quadrature, and the generic distribution machinery."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import beta

from roadrunner.physics import distribution as ed

_X = np.array([1.0, 0.0, 0.0])


def _xyz(r):
    return np.asarray(r, dtype=np.float64).reshape(-1, 1) * _X


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
        eps = np.interp(rng.uniform(size=20000), ed._row(ed.nfw_energy_tables().cdf, ed.C_GRID, c)[::-1], ed._nfw_nodes(c)[::-1])
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
        pdf = r_grid**2 * (-2.0 * p.potential(_xyz(r_grid))) ** 1.5
        cdf = np.cumsum(pdf); cdf /= cdf[-1]
        r = np.interp(rng.uniform(size=40000), cdf, r_grid)
        v = np.sqrt(-2.0 * p.potential(_xyz(r))) * rng.uniform(size=r.size) ** (1.0 / 3.0)
        w = np.exp(p.log_phase_space_fraction(p.potential(_xyz(r)) + 0.5 * v**2, R))
        np.testing.assert_allclose(np.quantile(w, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.01)
        with pytest.raises(NotImplementedError):
            SphericalPotential.energy_fraction(KeplerPotential(1e10), np.array([-1.0]), 10.0)


class TestTwoModelInversion:
    def test_tracer_in_another_potential_matches_the_power_law_closed_form(self):
        from roadrunner.physics.potentials import PlummerPotential
        bg, n = PlummerPotential(1e10, 0.5), 7.0

        class Tracer:                                            # ρ = ψ^n of the background's ψ = -Φ
            def _density(self, r):
                return (-bg._potential(r)) ** n

        E = np.array([1 - 1e-6, 0.8, 0.5, 0.2, 1e-6]) * bg.central_potential()
        ref = n * (n - 1) * beta(n - 1, 0.5) * (-E) ** (n - 1.5) / (np.sqrt(8.0) * np.pi**2)
        np.testing.assert_allclose(ed.eddington_distribution_function(bg, E, density=Tracer()), ref, rtol=1e-3)
        np.testing.assert_array_equal(ed.eddington_distribution_function(bg, E, density=bg), ed.eddington_distribution_function(bg, E))

    def test_negative_f_is_clipped_with_a_warning(self):
        from roadrunner.physics.potentials import NFWPotential, PlummerPotential
        nfw = NFWPotential(1e11, 8.0, 10.0)
        E = nfw.central_potential() * (1.0 - np.geomspace(1e-6, 1e-1, 30))
        with pytest.warns(RuntimeWarning, match="negative"):
            f = ed.eddington_distribution_function(nfw, E, density=PlummerPotential(1e9, 0.5))
        assert np.all(f >= 0.0) and f[0] == 0.0


class TestSelfSimilarTables:
    def test_shared_tables_match_the_generic_route(self):
        # rows on and between the grid, the bulk, both sides of the kink and the approach to E = 0;
        # a second instance with the same r_max / a reads the same values (ln N shifted by ln |Φ₀|);
        # outside the grid the generic route is used as is
        from roadrunner.physics.potentials import HernquistPotential, PlummerPotential, SphericalPotential
        names = ("energy_fraction", "log_energy_density", "log_phase_space_fraction")
        for cls in (PlummerPotential, HernquistPotential):
            for c in (ed.SELF_SIMILAR_C[40], np.sqrt(ed.SELF_SIMILAR_C[40] * ed.SELF_SIMILAR_C[41]), 37.0):
                p, q = cls(1e10, 0.5), cls(3e8, 0.02)
                k = p.potential(_xyz([c * p.a]))[0] / p.central_potential()
                eps = np.concatenate([[1 - 1e-6, 0.9, 0.5, 0.1], k * (1 + np.linspace(-0.02, 0.02, 9)), np.geomspace(1e-2, 1e-10, 9)])
                for name in names:
                    E = eps * p.central_potential()
                    np.testing.assert_allclose(getattr(p, name)(E, c * p.a), getattr(SphericalPotential, name)(p, E, c * p.a),
                                               atol=2e-3)
                    shift = np.log(q.central_potential() / p.central_potential()) if name == "log_energy_density" else 0.0
                    np.testing.assert_allclose(getattr(q, name)(eps * q.central_potential(), c * q.a),
                                               getattr(p, name)(E, c * p.a) - shift, rtol=1e-10, atol=1e-10)
            p = cls(1e10, 0.5)
            E = np.array([0.5, 0.1]) * p.central_potential()
            for name in names:
                np.testing.assert_array_equal(getattr(p, name)(E, 0.01), getattr(SphericalPotential, name)(p, E, 0.01))


class TestCompiledKernels:
    def test_table_query_matches_np_interp_and_the_deep_power_law(self):
        rng = np.random.default_rng(0)
        nodes = np.sort(rng.uniform(0.0, 1.0 - 1e-3, 50))
        row = rng.normal(size=50)
        eps = np.concatenate([rng.uniform(-0.1, 1.2, 1000), nodes, [np.nan]])
        np.testing.assert_allclose(ed.table_query(eps, nodes, row), np.interp(eps, nodes, row), rtol=1e-13, atol=1e-13)
        deep = eps > nodes[-1]
        ref = np.interp(eps, nodes, row)
        ref[deep] = row[-1] + 2.0 * np.log(np.maximum(1.0 - eps[deep], np.finfo(np.float64).tiny) / (1.0 - nodes[-1]))
        np.testing.assert_allclose(ed.table_query(eps, nodes, row, deep_slope=2.0), ref, rtol=1e-13, atol=1e-13)
        assert ed.table_query(np.float64(0.5), nodes, row).shape == ()

    def test_compiled_quadratures_match_numpy(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential, HernquistPotential, KeplerPotential, NFWPotential, PlummerPotential, ShellPotential
        for p, R in ((PlummerPotential(1e10, 0.5), 5.0), (HernquistPotential(1e10, 0.5), 5.0),
                     (NFWPotential(1e11, 8.0, 10.0), 80.0), (KeplerPotential(1e11), 60.0),
                     (ShellPotential(_xyz(np.random.default_rng(0).uniform(0.01, 10.0, 1000)), 1e7), 5.0)):
            d = ed.SphericalDistribution(p, R)
            q, g = d._q_and_g(d._energies)
            d._profile = None                                    # the numpy fallback (still evaluates exact offsets)
            q0, g0 = d._q_and_g(d._energies)
            np.testing.assert_allclose(q, q0, rtol=1e-12)
            np.testing.assert_allclose(g, g0, rtol=1e-12)
        # U2: offset composites (one row offset, the other concentric -> the Hermite-tabulated q/g
        # path for the offset row) across a few concentrations, companion profiles and offsets;
        # measured max relative error over this set: q 3.0e-9, g 6.6e-8 (well inside rtol=1e-4)
        for c, Rs in ((10.0, 8.0), (15.0, 6.0)):
            for cls, a in ((PlummerPotential, 0.4), (HernquistPotential, 0.3)):
                for d_off in (1.5, 3.0):
                    nfw = NFWPotential(1e11, Rs, c)
                    comp = CompositeSphericalPotential(dark_matter=nfw, stars=cls(3e8, a, centre=np.array([d_off, 0.0, 0.0])))
                    d = ed.SphericalDistribution(comp, 80.0)
                    assert comp._offsets.sum() > 0.0                 # one row (DM or stars) is on the monopole pathway
                    q, g = d._q_and_g(d._energies)
                    d._profile = None                                # the numpy fallback (still evaluates exact offsets)
                    q0, g0 = d._q_and_g(d._energies)
                    np.testing.assert_allclose(q, q0, rtol=1e-4)
                    np.testing.assert_allclose(g, g0, rtol=1e-4)
        # Shell plan: a composite's log_phase_space_fraction, concentric and offset Shell, compiled
        # (its tabulated row, offset through U2's Hermite table) vs the numpy fallback
        rng = np.random.default_rng(0)
        for d_off in (0.0, 4.0):
            centre = d_off * np.array([1.0, 0.0, 0.0])
            shell = ShellPotential(centre + _xyz(rng.uniform(0.01, 10.0, 1000)), 1e5, centre=centre)
            comp = CompositeSphericalPotential(dark_matter=NFWPotential(1e11, 8.0, 10.0), stars=shell)
            R = 80.0
            E = comp.central_potential() * np.array([0.9, 0.5, 0.1, 0.01, 1e-6])
            compiled = ed.SphericalDistribution(comp, R).log_phase_space_fraction(E)
            numpy_d = ed.SphericalDistribution(comp, R)
            numpy_d._profile = None
            numpy_w = numpy_d.log_phase_space_fraction(E)
            np.testing.assert_allclose(compiled, numpy_w, rtol=1e-6 if d_off == 0.0 else 1e-4)
        # the Eddington quadrature against its numpy form
        b = np.array([1 - 1e-6, 0.5, 1e-3, 1e-8])
        psi = -ed.nfw_phi(ed._X)
        rho = 1.0 / (4.0 * np.pi * ed._X * (1.0 + ed._X) ** 2)
        d2 = np.gradient(np.gradient(rho, psi), psi)
        depth = 1.0 - b
        t_max = np.arcsinh(np.sqrt(b / depth))
        t = t_max[:, None] * ed._W
        root = np.sqrt(depth)[:, None]
        d2_at = np.interp(b[:, None] - (root * np.sinh(t)) ** 2, psi[::-1], d2[::-1])
        ref = t_max * np.sum(ed._WW * 2.0 * d2_at * root * np.cosh(t), axis=-1) / (np.sqrt(8.0) * np.pi**2)
        np.testing.assert_allclose(ed._eddington(b), ref, rtol=1e-12)

    def test_eddington_preprocess_matches_numpy(self):
        # U7: the fused gradients/filter/reversal against the numpy form they replace, on a real cusp
        from roadrunner.physics.potentials import HernquistPotential
        p = HernquistPotential(1e10, 0.5)
        r = ed._EDDINGTON_R
        psi = -np.asarray(p._potential(r), dtype=np.float64)
        dpsi = -p.G * np.asarray(p._enclosed_mass(r), dtype=np.float64) / r**2
        rho = np.asarray(p._density(r), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            drho = np.gradient(rho, r) / dpsi
            d2 = np.gradient(drho, r) / dpsi
        keep = np.isfinite(d2) & np.concatenate(([True], np.diff(psi) < 0))
        ref_psi, ref_d2 = psi[keep][::-1], d2[keep][::-1]
        psi_k, d2_k = ed._eddington_preprocess(r, psi, rho, dpsi)
        np.testing.assert_array_equal(psi_k, ref_psi)
        np.testing.assert_array_equal(d2_k, ref_d2)

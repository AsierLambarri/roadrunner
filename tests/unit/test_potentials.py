import numpy as np

from roadrunner._mcf_types import PotentialModel
from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER
from roadrunner.physics.potentials import (
    KeplerPotential,
    NFWPotential,
    PlummerPotential,
    ShellPotential,
)


class TestKeplerPotential:
    def test_potential_large_r(self):
        M = 1e12
        G = 4.3e-6
        k = KeplerPotential(M, G=G)
        r = np.array([100.0, 200.0])
        result = k.potential(r)
        expected = -G * M / r
        assert np.allclose(result, expected, rtol=1e-3)

    def test_potential_at_zero(self):
        k = KeplerPotential(1e12, G=4.3e-6)
        r = np.array([0.0])
        result = k.potential(r)
        assert np.isfinite(result).all()

    def test_dynamical_time_positive(self):
        k = KeplerPotential(1e12, G=4.3e-6)
        r = np.array([10.0, 20.0])
        result = k.dynamical_time(r)
        assert np.all(result > 0)

    def test_tidal_denominator(self):
        M = 1e12
        k = KeplerPotential(M)
        assert k.tidal_denominator(np.array([1.0])) == 3 * M
        # Kepler is a Plummer sphere with a = SOFTENING_KEPLER; 3 M(<r) - dM/dln r for any a
        assert isinstance(k, PlummerPotential) and k.a == SOFTENING_KEPLER
        p, r, h = PlummerPotential(M, 0.5), np.array([0.2, 0.5, 3.0]), 1e-5
        menc = lambda x: M * x**3 / (x**2 + 0.25) ** 1.5
        dm_dlnr = (menc(r * np.exp(h)) - menc(r * np.exp(-h))) / (2 * h)
        np.testing.assert_allclose(p.tidal_denominator(r), 3 * menc(r) - dm_dlnr, rtol=1e-6)
        np.testing.assert_allclose(p.dynamical_time(r), 2 * np.pi * np.sqrt(r**3 / (p.G * menc(r))))


class TestNFWPotential:
    def test_potential_monotonic(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([1.0, 10.0, 100.0])
        result = nfw.potential(r)
        assert result[0] < result[1] < result[2]
        assert np.all(result < 0)

    def test_dynamical_time_positive(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([1.0, 5.0, 10.0])
        result = nfw.dynamical_time(r)
        assert np.all(result > 0)

    def test_tidal_denominator_positive(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0)
        r = np.array([1.0, 10.0, 100.0])
        result = nfw.tidal_denominator(r)
        assert np.all(result > 0)

    def test_enclosed_mass_converges(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0)
        R200 = nfw.c * nfw.Rs
        result = nfw.enclosed_mass(np.array([R200]))
        assert np.isclose(result[0], nfw.M, rtol=0.1)

    def test_potential_at_zero(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([0.0])
        result = nfw.potential(r)
        assert np.isfinite(result).all()


class TestShellPotential:
    def test_plummer_sample(self):
        # Inverse-transform Plummer sphere: M(<r) / M = u = r³ / (r² + a²)^(3/2).
        M, a, n = 1e12, 10.0, 200_000
        u = np.random.default_rng(0).uniform(size=n)
        p = ShellPotential(a / np.sqrt(u ** (-2 / 3) - 1), M / n)
        assert isinstance(p, PotentialModel)
        r = np.array([5.0, 10.0, 50.0])
        s2 = r**2 + a**2
        np.testing.assert_allclose(p.potential(r), -G_KM * M / np.sqrt(s2), rtol=5e-3)
        np.testing.assert_allclose(p.dynamical_time(r), 2 * np.pi * np.sqrt(s2**1.5 / (G_KM * M)), rtol=1e-2)
        np.testing.assert_allclose(p.tidal_denominator(r[1:]), 3 * M * (r[1:] ** 2 / s2[1:]) ** 2.5, rtol=5e-2)
        np.testing.assert_allclose(p.central_potential(), -G_KM * M / a, rtol=1e-2)
        np.testing.assert_allclose(p.binding_energy_scale(100.0), G_KM * M * 1e4 / (1e4 + a**2) ** 1.5, rtol=1e-2)


class TestGetPotential:
    def test_get_kepler_case_insensitive(self):
        from roadrunner.physics.potentials import get_potential
        for name in ("kepler", "Kepler", "KEPLER"):
            pot = get_potential(name, M=1e12)
            assert isinstance(pot, KeplerPotential)

    def test_get_unknown_raises(self):
        from roadrunner.physics.potentials import get_potential
        import pytest
        with pytest.raises(ValueError, match="Unknown potential model"):
            get_potential("foo", M=1e12)

    def test_get_kepler_usable(self):
        from roadrunner.physics.potentials import get_potential
        pot = get_potential("kepler", M=1e12, G=4.3e-6)
        r = np.array([100.0])
        expected = KeplerPotential(M=1e12, G=4.3e-6).potential(r)
        assert np.isclose(pot.potential(r), expected).all()

    def test_get_nfw_usable(self):
        from roadrunner.physics.potentials import get_potential
        pot = get_potential("nfw", M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        assert isinstance(pot, NFWPotential)
        r = np.array([10.0])
        expected = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6).potential(r)
        assert np.isclose(pot.potential(r), expected).all()


class TestGetPotentialClass:
    def test_get_kepler_class_no_kwargs(self):
        from roadrunner.physics.potentials import get_potential
        cls = get_potential("kepler")
        assert cls is KeplerPotential
        assert issubclass(cls, PotentialModel)
        assert get_potential("plummer") is PlummerPotential and issubclass(PlummerPotential, PotentialModel)

    def test_get_nfw_class_no_kwargs(self):
        from roadrunner.physics.potentials import get_potential
        cls = get_potential("nfw")
        assert cls is NFWPotential
        assert issubclass(cls, PotentialModel)


class TestBindingEnergyInterface:
    def test_kepler_scale_and_undefined_methods(self):
        import pytest
        p = KeplerPotential(M=1e10)
        assert isinstance(p, PotentialModel)
        np.testing.assert_allclose(p.binding_energy_scale(20.0), G_KM * 1e10 / 20.0)
        for call in (p.central_potential, lambda: p.energy_fraction(np.array([0.5])),
                     lambda: p.log_energy_density(np.array([0.5]))):
            with pytest.raises(NotImplementedError):
                call()
        E = np.array([-1e4, -10.0, 0.0, 5.0])
        s = -0.5 * p.G * p.M / E[:2]
        np.testing.assert_array_equal(p.orbital_time(E, np.ones(4)),
                                      np.r_[2 * np.pi * np.sqrt(s**3 / (p.G * p.M)), 0.0, 0.0])
        q = PlummerPotential(M=1e10, a=0.5)
        np.testing.assert_allclose(q.central_potential(), -G_KM * 1e10 / 0.5)
        with pytest.raises(NotImplementedError):
            q.log_phase_space_fraction(np.array([2.0]))

    def test_nfw_central_potential_and_scale(self):
        p = NFWPotential(M=1e10, Rs=5.0, c=8.0)
        assert isinstance(p, PotentialModel)
        phi0 = -G_KM * 1e10 / (5.0 * (np.log(9.0) - 8.0 / 9.0))
        np.testing.assert_allclose(p.central_potential(), phi0)
        np.testing.assert_allclose(p.binding_energy_scale(40.0), -phi0)
        # The softened potential stays above Phi_0.
        assert p.potential(np.array([0.0]))[0] > phi0
        u = p.energy_fraction(np.array([0.9, 0.5, 0.1]))
        assert np.all((u > 0) & (u < 1)) and np.all(np.diff(u) > 0)


class TestPhaseSpaceFraction:
    @staticmethod
    def _kepler_quad(b):
        from scipy.integrate import quad
        q = quad(lambda r: r * r * max(2.0 * (1.0 / r - b), 0.0) ** 1.5, 0.0, min(1.0, 1.0 / b))[0]
        return q / quad(lambda r: r * r * (2.0 / r) ** 1.5, 0.0, 1.0)[0]

    def test_kepler_closed_form(self):
        p = KeplerPotential(M=1.0, G=1.0)
        b = np.array([0.3, 1.0, 30.0])
        np.testing.assert_allclose(np.exp(p.log_phase_space_fraction(b)),
                                   [self._kepler_quad(x) for x in b], rtol=1e-6)
        np.testing.assert_allclose(p.log_phase_space_fraction(b[1:]),
                                   np.log(3 * np.pi / 32) - 1.5 * np.log(b[1:]), rtol=1e-12)

    def test_kepler_uniform_background_is_uniform_in_w(self):
        # G M = R_vir = 1: x density ∝ r^2 (2/r)^{3/2} ∝ r^{1/2}, so r = U^{2/3}.
        rng = np.random.default_rng(2)
        r = rng.uniform(size=40000) ** (2.0 / 3.0)
        v = np.sqrt(2.0 / r) * rng.uniform(size=r.size) ** (1.0 / 3.0)
        b = 1.0 / r - 0.5 * v**2                          # -E / v_vir^2
        w = np.exp(KeplerPotential(M=1.0, G=1.0).log_phase_space_fraction(b))
        assert abs(w.mean() - 0.5) < 0.01
        np.testing.assert_allclose(np.quantile(w, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.01)

    def test_nfw_delegates_to_tables(self):
        from roadrunner.physics.energy_distribution import nfw_log_phase_space_fraction
        p = NFWPotential(M=1e10, Rs=5.0, c=8.0)
        eps = np.array([0.95, 0.5, 0.05])
        np.testing.assert_array_equal(p.log_phase_space_fraction(eps), nfw_log_phase_space_fraction(eps, 8.0))
        assert np.all(np.diff(p.log_phase_space_fraction(eps)) > 0)

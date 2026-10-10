import warnings

import numpy as np
import pytest

from roadrunner._mcf_types import PotentialModel
from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER
from roadrunner.physics.potentials import (
    get_potential,
    KeplerPotential,
    NFWPotential,
    PlummerPotential,
    HernquistPotential,
    ShellPotential,
    SphericalPotential,
)

_X = np.array([1.0, 0.0, 0.0])


def _xyz(r):
    return np.asarray(r, dtype=np.float64).reshape(-1, 1) * _X


class TestKeplerPotential:
    def test_potential_large_r(self):
        M = 1e12
        G = 4.3e-6
        k = KeplerPotential(M, G=G)
        r = np.array([100.0, 200.0])
        result = k.potential(_xyz(r))
        expected = -G * M / r
        assert np.allclose(result, expected, rtol=1e-3)

    def test_potential_at_zero(self):
        k = KeplerPotential(1e12, G=4.3e-6)
        r = np.array([0.0])
        result = k.potential(_xyz(r))
        assert np.isfinite(result).all()

    def test_orbital_time_positive(self):
        k = KeplerPotential(1e12, G=4.3e-6)
        r = np.array([10.0, 20.0])
        result = k.orbital_time(np.array([-1.0, -2.0]), _xyz(r))
        assert np.all(result > 0)

    def test_tidal_denominator(self):
        M = 1e12
        k = KeplerPotential(M)
        assert k.tidal_denominator(_xyz([1.0])) == 3 * M
        # Kepler is a Plummer sphere with a = SOFTENING_KEPLER; 3 M(<r) - dM/dln r for any a
        assert isinstance(k, PlummerPotential) and k.a == SOFTENING_KEPLER
        p, r, h = PlummerPotential(M, 0.5), np.array([0.2, 0.5, 3.0]), 1e-5
        menc = lambda x: M * x**3 / (x**2 + 0.25) ** 1.5
        dm_dlnr = (menc(r * np.exp(h)) - menc(r * np.exp(-h))) / (2 * h)
        np.testing.assert_allclose(p.tidal_denominator(_xyz(r)), 3 * menc(r) - dm_dlnr, rtol=1e-6)
        np.testing.assert_allclose(p.orbital_time(np.full(r.size, -1.0), _xyz(r)), 2 * np.pi * np.sqrt(r**3 / (p.G * menc(r))))
        # the exact overrides agree with the generic (numerical) forms of SphericalPotential
        np.testing.assert_allclose(SphericalPotential.enclosed_mass(p, _xyz(r)), menc(r), rtol=1e-3)
        np.testing.assert_allclose(SphericalPotential.orbital_time(p, None, _xyz(r)), p.orbital_time(None, _xyz(r)), rtol=1e-12)
        np.testing.assert_allclose(SphericalPotential.central_potential(p), p.central_potential(), rtol=1e-9)


class TestNFWPotential:
    def test_potential_monotonic(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([1.0, 10.0, 100.0])
        result = nfw.potential(_xyz(r))
        assert result[0] < result[1] < result[2]
        assert np.all(result < 0)

    def test_orbital_time_positive(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([1.0, 5.0, 10.0])
        result = nfw.orbital_time(np.zeros(3), _xyz(r))
        assert np.all(result > 0)

    def test_tidal_denominator_positive(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0)
        r = np.array([1.0, 10.0, 100.0, 300.0])
        result = nfw.tidal_denominator(_xyz(r))
        assert np.all(result > 0)
        # untruncated: the closed form (M / A(c)) (3 A(x) - x²/(1 + x)²), also beyond r_vir
        x, A = r / nfw.Rs, lambda y: np.log1p(y) - y / (1 + y)
        np.testing.assert_allclose(result, nfw.M / A(nfw.c) * (3 * A(x) - x**2 / (1 + x) ** 2), rtol=1e-5)

    def test_enclosed_mass_converges(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0)
        R200 = nfw.c * nfw.Rs
        result = nfw.enclosed_mass(_xyz([R200]))
        assert np.isclose(result[0], nfw.M, rtol=0.1)
        # the density is the exact derivative of the enclosed mass
        r = np.array([1.0, 10.0, 99.0])
        np.testing.assert_allclose(SphericalPotential.enclosed_mass(nfw, _xyz(r)), nfw.enclosed_mass(_xyz(r)), rtol=1e-3)

    def test_potential_at_zero(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([0.0])
        result = nfw.potential(_xyz(r))
        assert np.isfinite(result).all()


class TestShellPotential:
    def test_plummer_sample(self):
        # Inverse-transform Plummer sphere: M(<r) / M = u = r³ / (r² + a²)^(3/2).
        M, a, n = 1e12, 10.0, 200_000
        u = np.random.default_rng(0).uniform(size=n)
        p = ShellPotential(_xyz(a / np.sqrt(u ** (-2 / 3) - 1)), M / n)
        assert isinstance(p, PotentialModel)
        r = np.array([5.0, 10.0, 50.0])
        s2 = r**2 + a**2
        np.testing.assert_allclose(p.potential(_xyz(r)), -G_KM * M / np.sqrt(s2), rtol=5e-3)
        np.testing.assert_allclose(p.enclosed_mass(_xyz(r)), M * r**3 / s2**1.5, rtol=2e-2)
        np.testing.assert_allclose(p.orbital_time(np.zeros(3), _xyz(r)), 2 * np.pi * np.sqrt(s2**1.5 / (G_KM * M)), rtol=1e-2)
        np.testing.assert_allclose(p.tidal_denominator(_xyz(r[1:])), 3 * M * (r[1:] ** 2 / s2[1:]) ** 2.5, rtol=5e-2)
        np.testing.assert_allclose(p.central_potential(), -G_KM * M / a, rtol=1e-2)


class TestHernquistPotential:
    def test_closed_forms_and_point_mass_limit(self):
        # M(<r) = M r² / (r + a)²; tidal denominator 3 M(<r) - dM/dln r by central difference
        M, a = 1e10, 0.5
        p = get_potential("hernquist", M=M, a=a)
        assert isinstance(p, HernquistPotential) and isinstance(p, PotentialModel)
        r, h = np.array([0.05, 0.5, 5.0, 100.0]), 1e-5
        menc = lambda x: M * x**2 / (x + a) ** 2
        dm_dlnr = (menc(r * np.exp(h)) - menc(r * np.exp(-h))) / (2 * h)
        np.testing.assert_allclose(p.potential(_xyz(r)), -G_KM * M / (r + a))
        np.testing.assert_allclose(p.orbital_time(np.zeros(r.size), _xyz(r)), 2 * np.pi * np.sqrt(r**3 / (G_KM * menc(r))))
        np.testing.assert_allclose(SphericalPotential.enclosed_mass(p, _xyz(r)), menc(r), rtol=1e-3)
        np.testing.assert_allclose(SphericalPotential.orbital_time(p, None, _xyz(r)), p.orbital_time(None, _xyz(r)), rtol=1e-12)
        np.testing.assert_allclose(SphericalPotential.central_potential(p), p.central_potential(), rtol=1e-5)
        np.testing.assert_allclose(p.tidal_denominator(_xyz(r)), 3 * menc(r) - dm_dlnr, rtol=1e-6)
        np.testing.assert_allclose(p.central_potential(), -G_KM * M / a)
        k = HernquistPotential(M, 1e-8)
        np.testing.assert_allclose(k.potential(_xyz(r)), -G_KM * M / r, rtol=1e-6)
        np.testing.assert_allclose(k.tidal_denominator(_xyz(r)), 3 * M, rtol=1e-6)


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
        expected = KeplerPotential(M=1e12, G=4.3e-6).potential(_xyz(r))
        assert np.isclose(pot.potential(_xyz(r)), expected).all()

    def test_get_nfw_usable(self):
        from roadrunner.physics.potentials import get_potential
        pot = get_potential("nfw", M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        assert isinstance(pot, NFWPotential)
        r = np.array([10.0])
        expected = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6).potential(_xyz(r))
        assert np.isclose(pot.potential(_xyz(r)), expected).all()


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
    def test_kepler_undefined_methods_and_period(self):
        import pytest
        p = KeplerPotential(M=1e10)
        assert isinstance(p, PotentialModel)
        for call in (p.central_potential, lambda: p.energy_fraction(np.array([-0.5]), 1.0),
                     lambda: p.log_energy_density(np.array([-0.5]), 1.0)):
            with pytest.raises(NotImplementedError):
                call()
        E = np.array([-1e4, -10.0, 0.0, 5.0])
        s = -0.5 * p.G * p.M / E[:2]
        np.testing.assert_array_equal(p.orbital_time(E, _xyz(np.ones(4))),
                                      np.r_[2 * np.pi * np.sqrt(s**3 / (p.G * p.M)), 0.0, 0.0])
        q = PlummerPotential(M=1e10, a=0.5)
        np.testing.assert_allclose(q.central_potential(), -G_KM * 1e10 / 0.5)
        assert np.all(q.log_phase_space_fraction(np.array([-1e10 * G_KM / 2.0]), 10.0) < 0)

    def test_nfw_central_potential(self):
        p = NFWPotential(M=1e10, Rs=5.0, c=8.0)
        assert isinstance(p, PotentialModel)
        phi0 = -G_KM * 1e10 / (5.0 * (np.log(9.0) - 8.0 / 9.0))
        np.testing.assert_allclose(p.central_potential(), phi0)
        assert p.potential(_xyz([0.0]))[0] == p.central_potential()
        u = p.energy_fraction(np.array([0.9, 0.5, 0.1]) * phi0, 40.0)
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
        np.testing.assert_allclose(np.exp(p.log_phase_space_fraction(-b, 1.0)),
                                   [self._kepler_quad(x) for x in b], rtol=1e-6)
        np.testing.assert_allclose(p.log_phase_space_fraction(-b[1:], 1.0),
                                   np.log(3 * np.pi / 32) - 1.5 * np.log(b[1:]), rtol=1e-12)

    def test_kepler_uniform_background_is_uniform_in_w(self):
        # G M = R_vir = 1: x density ∝ r^2 (2/r)^{3/2} ∝ r^{1/2}, so r = U^{2/3}.
        rng = np.random.default_rng(2)
        r = rng.uniform(size=40000) ** (2.0 / 3.0)
        v = np.sqrt(2.0 / r) * rng.uniform(size=r.size) ** (1.0 / 3.0)
        b = 1.0 / r - 0.5 * v**2                          # -E / v_vir^2
        w = np.exp(KeplerPotential(M=1.0, G=1.0).log_phase_space_fraction(-b, 1.0))
        assert abs(w.mean() - 0.5) < 0.01
        np.testing.assert_allclose(np.quantile(w, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.01)

    def test_nfw_delegates_to_tables(self):
        from roadrunner.physics.distribution import nfw_log_phase_space_fraction
        p = NFWPotential(M=1e10, Rs=5.0, c=8.0)
        eps = np.array([0.95, 0.5, 0.05])
        E = eps * p.central_potential()
        np.testing.assert_allclose(p.log_phase_space_fraction(E, 40.0), nfw_log_phase_space_fraction(eps, 8.0),
                                   rtol=1e-6)
        assert np.all(np.diff(p.log_phase_space_fraction(E, 40.0)) > 0)


class TestCentre:
    def test_arrays_holding_r_zero(self):
        # each formula runs on the whole array; the 0/0 and 0·inf at the centre become their limits, silently
        r = np.array([0.0, 1e-12, 1e-6, 1.0, 100.0])
        plummer, hernquist, nfw = PlummerPotential(1e10, 0.5), HernquistPotential(1e10, 0.5), NFWPotential(1e11, 8.0, 10.0)
        shell = ShellPotential(_xyz(np.random.default_rng(0).uniform(0.01, 10.0, 1000)), 1e7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for p in (KeplerPotential(1e10), plummer, hernquist, nfw, shell):
                phi, menc, denom = p.potential(_xyz(r)), p.enclosed_mass(_xyz(r)), p.tidal_denominator(_xyz(r))
                t = p.orbital_time(-np.ones(r.size), _xyz(r))
                assert np.all(np.isfinite(phi)) and np.all(np.diff(phi) >= 0)
                assert menc[0] == 0 and np.all(np.diff(menc) >= 0) and np.all(denom >= 0)
                assert t.shape == denom.shape == r.shape and not np.isnan(t).any()
            # Φ(0) = Φ₀; the dynamical time vanishes in a cusp and is the core's 2π sqrt(a³ / (G M)) in a core
            assert nfw.potential(_xyz(r))[0] == nfw.central_potential()
            assert nfw.orbital_time(None, _xyz(r))[0] == 0 and hernquist.orbital_time(None, _xyz(r))[0] == 0
            np.testing.assert_allclose(SphericalPotential.orbital_time(plummer, None, _xyz(r))[0], plummer.orbital_time(None, _xyz(r))[0])
            assert shell.orbital_time(None, _xyz(r))[0] == np.inf   # no mass inside the innermost particle
            # log1p keeps NFW's M(<r) -> M x² / (2 A(c)) accurate at small x
            x, A = r[2] / nfw.Rs, np.log1p(nfw.c) - nfw.c / (1 + nfw.c)
            np.testing.assert_allclose(nfw.enclosed_mass(_xyz(r[2:3])), nfw.M * x**2 / (2 * A), rtol=1e-6)
            np.testing.assert_allclose(nfw.tidal_denominator(_xyz(r[2:3])), nfw.M * x**2 / (2 * A), rtol=1e-6)


class TestCompositeSphericalPotential:
    def test_single_component_is_the_component(self):
        # every method of a one-component composite returns its component's values exactly
        from roadrunner.physics.potentials import CompositeSphericalPotential
        r, E = np.array([0.0, 0.3, 5.0, 80.0]), np.array([-2e4, -5e3, -1e3, 10.0])
        for p, R in ((NFWPotential(1e11, 8.0, 10.0), 80.0), (KeplerPotential(1e11), 80.0),
                     (PlummerPotential(1e10, 0.5), 5.0), (HernquistPotential(1e10, 0.5), 5.0)):
            c = CompositeSphericalPotential(p)
            assert len(c) == 1 and c[0] is p and list(c) == [p] and c.M == p.M
            for name in ("potential", "density", "enclosed_mass", "tidal_denominator"):
                np.testing.assert_array_equal(getattr(c, name)(_xyz(r)), getattr(p, name)(_xyz(r)))
            np.testing.assert_array_equal(c.orbital_time(E, _xyz(r)), p.orbital_time(E, _xyz(r)))
            assert c.well_depth(_xyz([0.1])) == p.well_depth(_xyz([0.1]))
            np.testing.assert_array_equal(c.profile(), p.profile())
            Eb = E[:3] * (1.0 if isinstance(p, KeplerPotential) else -p.central_potential() / 2e4)
            for name in ("energy_fraction", "log_energy_density", "log_phase_space_fraction"):
                if isinstance(p, KeplerPotential) and name != "log_phase_space_fraction":
                    with pytest.raises(NotImplementedError):
                        getattr(c, name)(Eb, R)
                    continue
                np.testing.assert_array_equal(getattr(c, name)(Eb, R), getattr(p, name)(Eb, R))

    def test_sums_orbital_time_and_distributions_of_the_whole(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        nfw, plummer = NFWPotential(1e11, 8.0, 10.0), PlummerPotential(1e10, 0.5)
        c = CompositeSphericalPotential(nfw, plummer)
        r = np.array([0.0, 0.3, 5.0, 80.0])
        for name in ("potential", "density", "enclosed_mass", "tidal_denominator"):
            np.testing.assert_array_equal(getattr(c, name)(_xyz(r)), getattr(nfw, name)(_xyz(r)) + getattr(plummer, name)(_xyz(r)))
        assert c.central_potential() == nfw.central_potential() + plummer.central_potential()
        assert c.well_depth(_xyz([0.1])) == nfw.well_depth(_xyz([0.1])) + plummer.well_depth(_xyz([0.1]))
        np.testing.assert_array_equal(c.profile(), np.vstack([nfw.profile(), plummer.profile()]))
        # dynamical times combine exactly: t^-2 ∝ M(<r)
        np.testing.assert_allclose(c.orbital_time(None, _xyz(r[1:])), 2 * np.pi * np.sqrt(r[1:]**3 / (G_KM * c.enclosed_mass(_xyz(r[1:])))),
                                   rtol=1e-12)
        assert c.orbital_time(None, _xyz(r[:1]))[0] == 0.0                     # the NFW cusp at r = 0
        # particles uniform in the bound phase space inside R are uniform in w
        R, rng = 20.0, np.random.default_rng(5)
        r_grid = np.geomspace(1e-6, R, 200001)
        cdf = np.cumsum(r_grid**3 * (-2.0 * c.potential(_xyz(r_grid))) ** 1.5); cdf /= cdf[-1]
        x = np.interp(rng.uniform(size=40000), cdf, r_grid)
        v = np.sqrt(-2.0 * c.potential(_xyz(x))) * rng.uniform(size=x.size) ** (1.0 / 3.0)
        w = np.exp(c.log_phase_space_fraction(c.potential(_xyz(x)) + 0.5 * v**2, R))
        np.testing.assert_allclose(np.quantile(w, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.01)
        u = c.energy_fraction(np.array([0.9, 0.5, 0.1]) * c.central_potential(), R)
        assert np.all((u > 0) & (u < 1)) and np.all(np.diff(u) > 0)
        # a component without f: the energy pair raises, the phase-space fraction does not
        k = CompositeSphericalPotential(nfw, KeplerPotential(1e10))
        assert not k.has_distribution
        with pytest.raises(NotImplementedError):
            k.energy_fraction(np.array([-1e4]), R)
        assert np.all(k.log_phase_space_fraction(np.array([-1e4]), R) < 0)
        with pytest.raises(ValueError):
            CompositeSphericalPotential(nfw, PlummerPotential(1e10, 0.5, G=1.0))


class TestPositions:
    def test_centres_and_composite_offsets(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        c1, c2 = np.array([1e5, 2e5, -3e4]), np.array([1e5 + 4.0, 2e5, -3e4])
        nfw, pl = NFWPotential(1e11, 8.0, 10.0, centre=c1), PlummerPotential(1e10, 0.5, centre=c2)
        xyz = c1 + np.array([[3.0, 0.0, 0.0], [0.0, 1e-3, 0.0], [10.0, 5.0, -2.0]])
        r1, r2 = np.linalg.norm(xyz - c1, axis=1), np.linalg.norm(xyz - c2, axis=1)
        np.testing.assert_allclose(nfw.potential(xyz), nfw._potential(r1), rtol=1e-12)
        comp = CompositeSphericalPotential(nfw, pl)
        np.testing.assert_array_equal(comp.centre, c1)
        np.testing.assert_allclose(comp.potential(xyz), nfw._potential(r1) + pl._potential(r2), rtol=1e-12)
        np.testing.assert_allclose(comp.enclosed_mass(xyz), nfw._enclosed_mass(r1) + pl._enclosed_mass(r2), rtol=1e-12)
        np.testing.assert_allclose(comp._potential(r1), nfw._potential(r1) + pl._potential(r1), rtol=1e-12)
        k = KeplerPotential(1e10, centre=c1)
        np.testing.assert_allclose(k.well_depth(c1 + np.array([[0.2, 0.0, 0.0]])), -k._potential(np.array([0.2]))[0], rtol=1e-9)


class TestCompositeAdd:
    def test_add_on_the_fly_matches_building_at_once(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        nfw, plummer = NFWPotential(1e11, 8.0, 10.0), PlummerPotential(1e10, 0.5)
        c = CompositeSphericalPotential(nfw)
        E, R = np.array([0.9, 0.5, 0.1]) * nfw.central_potential(), 20.0
        np.testing.assert_array_equal(c.log_phase_space_fraction(E, R), nfw.log_phase_space_fraction(E, R))
        c.add(plummer)
        both = CompositeSphericalPotential(nfw, plummer)
        assert len(c) == 2 and c[1] is plummer and c.M == nfw.M + plummer.M and c.has_distribution
        assert "orbital_time" not in c.__dict__ and "log_phase_space_fraction" not in c.__dict__
        xyz = _xyz([0.0, 0.3, 5.0])
        np.testing.assert_array_equal(c.potential(xyz), both.potential(xyz))
        np.testing.assert_array_equal(c.orbital_time(None, xyz), both.orbital_time(None, xyz))
        E = np.array([0.9, 0.5, 0.1]) * c.central_potential()
        for name in ("energy_fraction", "log_energy_density", "log_phase_space_fraction"):
            np.testing.assert_allclose(getattr(c, name)(E, R), getattr(both, name)(E, R), rtol=1e-12)
        # a composite whose cache is filled drops it on add: the third component is seen
        hernquist = HernquistPotential(1e9, 0.2)
        before = c.energy_fraction(E, R)
        c.add(hernquist)
        three = CompositeSphericalPotential(nfw, plummer, hernquist)
        assert not np.array_equal(c.energy_fraction(E, R), before)
        np.testing.assert_allclose(c.energy_fraction(E, R), three.energy_fraction(E, R), rtol=1e-12)
        c.add(KeplerPotential(1e9))
        assert not c.has_distribution
        with pytest.raises(ValueError):
            c.add(PlummerPotential(1e10, 0.5, G=1.0))


_C0 = np.array([1e3, 2e3, 3e3])


def _segment(c, p, q, n=200001):
    """Minimum of ``c`` sampled on the straight segment between ``p`` and ``q``'s centres."""
    seg = p.centre + np.linspace(0.0, 1.0, n)[:, None] * (q.centre - p.centre)
    phi = c.potential(seg)
    return seg[phi.argmin()], phi.min()


class TestWellMinimum:
    def test_coincident_centres_need_no_search(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential, _locate_minimum
        nfw = NFWPotential(1e11, 8.0, 10.0, centre=_C0)
        pl = PlummerPotential(1e10, 0.5, centre=_C0)
        c = CompositeSphericalPotential(nfw, pl)
        np.testing.assert_array_equal(nfw.x_min, _C0)
        np.testing.assert_array_equal(c.x_min, _C0)
        np.testing.assert_array_equal(_locate_minimum(None, np.array([_C0, _C0])), _C0)
        assert c.central_potential() == nfw.central_potential() + pl.central_potential()

    def test_two_offset_plummers(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        p1 = PlummerPotential(1e10, 1.0, centre=_C0)
        p2 = PlummerPotential(5e9, 0.8, centre=_C0 + np.array([1.5, 0.5, 0.0]))
        c = CompositeSphericalPotential(p1, p2)
        xb, pb = _segment(c, p1, p2)
        L = np.linalg.norm(p2.centre - p1.centre)
        np.testing.assert_allclose(c.x_min, xb, rtol=0, atol=2e-5 * L)
        phi0 = c.central_potential()
        assert phi0 <= pb + 1e-9 * abs(pb)
        assert phi0 == c.potential(c.x_min[None])[0]
        pts = _C0 + np.random.default_rng(1).uniform(-3.0, 4.0, (10000, 3))
        assert np.all(c.potential(pts) >= phi0)

    def test_cusp_holds_the_minimum(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        nfw = NFWPotential(1e11, 8.0, 10.0, centre=_C0)
        # light Plummer's largest pull 0.385 G M/a^2 = 1.2e8 G is below the NFW gradient on the
        # whole segment, >= G M(<1 kpc) = 4.5e8 G; cone slope |Phi0|/(2 Rs) = 5.25e8 G
        light = PlummerPotential(5e9, 4.0, centre=_C0 + np.array([0.6, 0.8, 0.0]))
        c = CompositeSphericalPotential(nfw, light)
        np.testing.assert_array_equal(c.x_min, _C0)
        assert c.central_potential() == nfw.central_potential() + light.potential(_C0[None])[0]
        # heavy Plummer's pull at the cusp, 1.43e9 G, exceeds the slope: the minimum moves off
        heavy = PlummerPotential(1e11, 4.0, centre=_C0 + np.array([0.6, 0.8, 0.0]))
        c = CompositeSphericalPotential(nfw, heavy)
        xb, pb = _segment(c, nfw, heavy)
        np.testing.assert_allclose(c.x_min, xb, rtol=0, atol=2e-5)
        assert np.linalg.norm(c.x_min - _C0) > 0.1

    def test_deeper_of_two_wells(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        p1 = PlummerPotential(1e10, 0.5, centre=_C0)
        p2 = PlummerPotential(1e10, 0.3, centre=_C0 + np.array([20.0, 0.0, 0.0]))
        c = CompositeSphericalPotential(p1, p2)
        np.testing.assert_allclose(c.x_min, p2.centre, rtol=0, atol=1e-3)
        assert c.central_potential() <= c.potential(p2.centre[None])[0] < c.potential(p1.centre[None])[0]

    def test_three_centres_span_a_plane(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        p1 = PlummerPotential(1e10, 1.5, centre=_C0)
        p2 = PlummerPotential(8e9, 1.2, centre=_C0 + np.array([2.0, 0.0, 0.0]))
        p3 = PlummerPotential(6e9, 1.0, centre=_C0 + np.array([0.8, 1.8, 0.5]))
        c = CompositeSphericalPotential(p1, p2, p3)
        a = np.linspace(0.0, 1.0, 1001)
        A, B = (g.reshape(-1, 1) for g in np.meshgrid(a, a, indexing="ij"))
        plane = _C0 + A * (p2.centre - _C0) + B * (p3.centre - _C0)
        phi = c.potential(plane)
        np.testing.assert_allclose(c.x_min, plane[phi.argmin()], rtol=0, atol=3e-3)
        assert c.central_potential() <= phi.min() + 1e-9 * abs(phi.min())

    def test_kepler_component(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        nfw = NFWPotential(1e11, 8.0, 10.0, centre=_C0)
        for k in (KeplerPotential(1e9, centre=_C0), KeplerPotential(1e9, centre=_C0 + np.array([1.0, 0.0, 0.0]))):
            c = CompositeSphericalPotential(nfw, k)
            assert np.all(np.isfinite(c.x_min))
            with pytest.raises(NotImplementedError):
                c.central_potential()
            c.add(PlummerPotential(1e9, 0.5, centre=_C0 + np.array([0.0, 1.0, 0.0])))
            with pytest.raises(NotImplementedError):
                c.central_potential()

    def test_add_updates_the_minimum(self):
        from roadrunner.physics.potentials import CompositeSphericalPotential
        p1 = PlummerPotential(1e10, 1.0, centre=_C0)
        p2 = PlummerPotential(5e9, 0.8, centre=_C0 + np.array([1.5, 0.5, 0.0]))
        p3 = HernquistPotential(3e9, 0.6, centre=_C0 + np.array([0.3, 1.2, -0.4]))
        c = CompositeSphericalPotential(p1)
        np.testing.assert_array_equal(c.x_min, _C0)
        c.add(p2)
        np.testing.assert_array_equal(c.x_min, CompositeSphericalPotential(p1, p2).x_min)
        assert not np.array_equal(c.x_min, _C0)
        d = CompositeSphericalPotential(p1)
        d.add(p2, p3)
        np.testing.assert_array_equal(d.x_min, CompositeSphericalPotential(p1, p2, p3).x_min)

import numpy as np

from roadrunner._mcf_types import PotentialModel
from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER
from roadrunner.physics.potentials import (
    KeplerPotential,
    NFWPotential,
    dynamical_time,
    potential,
    tidal_denominator,
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
        E = np.array([-100.0, -200.0])
        result = k.dynamical_time(E)
        assert np.all(result > 0)

    def test_tidal_denominator(self):
        M = 1e12
        k = KeplerPotential(M)
        assert k.tidal_denominator() == 3 * M


class TestNFWPotential:
    def test_potential_negative(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([1.0, 10.0, 100.0])
        result = nfw.potential(r)
        assert np.all(result < 0)

    def test_potential_monotonic(self):
        nfw = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        r = np.array([1.0, 10.0, 100.0])
        result = nfw.potential(r)
        assert result[0] < result[1] < result[2]

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


class TestFactoryFunctions:
    def test_potential_factory_kepler(self):
        M, G = 1e12, 4.3e-6
        r = np.array([100.0])
        factory_result = potential("kepler", r, M=M, G=G)
        direct_result = KeplerPotential(M, G=G).potential(r)
        assert np.isclose(factory_result, direct_result).all()

    def test_potential_factory_nfw(self):
        M, Rs, c, G = 1e12, 10.0, 10.0, 4.3e-6
        r = np.array([10.0])
        factory_result = potential("nfw", r, M=M, Rs=Rs, c=c, G=G)
        direct_result = NFWPotential(M, Rs, c, G=G).potential(r)
        assert np.isclose(factory_result, direct_result).all()

    def test_dynamical_time_factory_kepler(self):
        M, G = 1e12, 4.3e-6
        E = np.array([-100.0])
        factory_result = dynamical_time("kepler", E=E, M=M, G=G)
        direct_result = KeplerPotential(M, G=G).dynamical_time(E)
        assert np.isclose(factory_result, direct_result).all()

    def test_dynamical_time_factory_nfw(self):
        M, Rs, c, G = 1e12, 10.0, 10.0, 4.3e-6
        r = np.array([10.0])
        factory_result = dynamical_time("nfw", r=r, M=M, Rs=Rs, c=c, G=G)
        direct_result = NFWPotential(M, Rs, c, G=G).dynamical_time(r)
        assert np.isclose(factory_result, direct_result).all()

    def test_tidal_denominator_factory_kepler(self):
        result = tidal_denominator("kepler", M=1e12)
        assert result == 3 * 1e12

    def test_tidal_denominator_factory_nfw(self):
        result = tidal_denominator(
            "nfw", r=np.array([10.0]), M=1e12, Rs=10.0, c=10.0
        )
        assert result > 0

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown potential model"):
            potential("foo", np.array([1.0]), M=1e12)


class TestProtocol:
    def test_kepler_is_potential_model(self):
        assert isinstance(KeplerPotential(M=1e12), PotentialModel)

    def test_nfw_is_potential_model(self):
        assert isinstance(NFWPotential(M=1e12, Rs=10.0, c=10.0), PotentialModel)

    def test_mock_passes_protocol(self):
        class MockPot:
            def potential(self, r):
                return -r
            def dynamical_time(self, x):
                return x
            def tidal_denominator(self, r):
                return 3 * r
        assert isinstance(MockPot(), PotentialModel)


class TestGetPotential:
    def test_get_kepler(self):
        from roadrunner.physics.potentials import get_potential
        pot = get_potential("kepler", M=1e12)
        assert isinstance(pot, KeplerPotential)

    def test_get_nfw(self):
        from roadrunner.physics.potentials import get_potential
        pot = get_potential("nfw", M=1e12, Rs=10.0, c=10.0)
        assert isinstance(pot, NFWPotential)

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
        r = np.array([10.0])
        expected = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6).potential(r)
        assert np.isclose(pot.potential(r), expected).all()


class TestGetPotentialClass:
    def test_get_kepler_class_no_kwargs(self):
        from roadrunner.physics.potentials import get_potential
        cls = get_potential("kepler")
        assert cls is KeplerPotential
        assert issubclass(cls, PotentialModel)

    def test_get_nfw_class_no_kwargs(self):
        from roadrunner.physics.potentials import get_potential
        cls = get_potential("nfw")
        assert cls is NFWPotential
        assert issubclass(cls, PotentialModel)

    def test_reusable_kepler_class(self):
        from roadrunner.physics.potentials import get_potential
        cls = get_potential("kepler")
        instance_1 = cls(M=1e12, G=G_KM)
        instance_2 = cls(M=5e11, G=G_KM)
        assert instance_1 is not instance_2
        assert instance_1.M != instance_2.M
        assert np.isclose(instance_1.potential(np.array([100.0])),
                          -G_KM * 1e12 / 100.0, rtol=1e-3)

    def test_reusable_nfw_class(self):
        from roadrunner.physics.potentials import get_potential
        cls = get_potential("nfw")
        inst_a = cls(M=1e12, Rs=10.0, c=10.0, G=G_KM)
        inst_b = cls(M=5e11, Rs=5.0, c=8.0, G=G_KM)
        assert inst_a is not inst_b
        assert inst_a.M != inst_b.M

    def test_class_identity(self):
        from roadrunner.physics.potentials import get_potential
        assert get_potential("kepler") is get_potential("kepler")
        assert get_potential("nfw") is get_potential("nfw")

    def test_class_raises_on_unknown(self):
        from roadrunner.physics.potentials import get_potential
        import pytest
        with pytest.raises(ValueError, match="Unknown potential model"):
            get_potential("foo")

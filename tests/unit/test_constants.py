import roadrunner.physics.constants as C


class TestConstantsExist:
    def test_G_KM_defined(self):
        assert C.G_KM == -1

    def test_G_GALACTIC_defined(self):
        assert C.G_GALACTIC == -1

    def test_DYN_TIME_FACTOR_defined(self):
        assert C.DYN_TIME_FACTOR == -1

    def test_softening_kepler_defined(self):
        assert C.SOFTENING_KEPLER == -1

    def test_softening_nfw_defined(self):
        assert C.SOFTENING_NFW == -1

    def test_all_in__all__(self):
        expected = [
            "G_KM",
            "G_GALACTIC",
            "DYN_TIME_FACTOR",
            "SOFTENING_KEPLER",
            "SOFTENING_NFW",
        ]
        assert C.__all__ == expected

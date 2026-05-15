from math import sqrt

import roadrunner.physics.constants as C


class TestConstantsExist:
    def test_G_KM_positive(self):
        assert C.G_KM > 0

    def test_G_KM_value(self):
        assert C.G_KM == 4.300917270038E-6

    def test_G_GALACTIC_positive(self):
        assert C.G_GALACTIC > 0

    def test_G_GALACTIC_value(self):
        assert C.G_GALACTIC == 4.49850215E-6

    def test_DYN_TIME_FACTOR_positive(self):
        assert C.DYN_TIME_FACTOR > 0

    def test_DYN_TIME_FACTOR_self_consistent(self):
        expected = sqrt(C.G_GALACTIC / C.G_KM)
        assert C.DYN_TIME_FACTOR == expected

    def test_softening_kepler_positive(self):
        assert C.SOFTENING_KEPLER > 0

    def test_softening_nfw_positive(self):
        assert C.SOFTENING_NFW > 0

    def test_all_in__all__(self):
        expected = [
            "G_KM",
            "G_GALACTIC",
            "DYN_TIME_FACTOR",
            "SOFTENING_KEPLER",
            "SOFTENING_NFW",
        ]
        assert C.__all__ == expected

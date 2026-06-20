#############################################################################
#
# package:   roadrunner.physics
# file:      potentials.py
# brief:     <TODO>
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   13 may 2026 - Created
#            13 may 2026 - Last edit
#
#############################################################################

import numpy as np

from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER, SOFTENING_NFW

_2PI = 2 * np.pi
class KeplerPotential:
    def __init__(self, M, G=G_KM):
        self.M = M
        self.G = G

    def potential(self, r):
        r_safe = np.sqrt(r**2 + SOFTENING_KEPLER**2)
        return -self.G * self.M / r_safe

    def dynamical_time(self, r):
        with np.errstate(invalid="ignore"):
            return _2PI * np.sqrt(r**3 / (self.G * self.M))

    def tidal_denominator(self, r):
        return 3 * self.M


class NFWPotential:
    def __init__(self, M, Rs, c, G=G_KM):
        self.M = M
        self.Rs = Rs
        self.c = c
        self.G = G

    def potential(self, r):
        r_safe = np.sqrt(r**2 + SOFTENING_NFW**2)
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return -self.G * (self.M / r_safe) * np.log(1 + r_safe / self.Rs) / A

    def enclosed_mass(self, r):
        x = np.minimum(self.c, np.sqrt(r**2 + SOFTENING_NFW**2) / self.Rs)
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        B = np.log(1 + x) - x / (1 + x)
        return self.M * B / A

    def dynamical_time(self, r):
        with np.errstate(invalid="ignore"):
            menc = self.enclosed_mass(r)
            return _2PI * np.sqrt(r**3 / (self.G * menc))

    def tidal_denominator(self, r):
        def _f(x):
            return np.log(1 + x) - x / (1 + x)
        x = np.minimum(self.c, np.sqrt(r**2 + SOFTENING_NFW**2) / self.Rs)
        return (self.M / _f(self.c)) * (3 * _f(x) - x**2 / (1 + x)**2)


def potential(model, r, **kwargs):
    if model.lower() == "kepler":
        p = KeplerPotential(M=kwargs["M"], G=kwargs.get("G", G_KM))
        return p.potential(r)
    if model.lower() == "nfw":
        p = NFWPotential(
            M=kwargs["M"], Rs=kwargs["Rs"],
            c=kwargs["c"], G=kwargs.get("G", G_KM),
        )
        return p.potential(r)
    raise ValueError(f"Unknown potential model: {model}")


def dynamical_time(model, **kwargs):
    if model.lower() == "kepler":
        p = KeplerPotential(M=kwargs["M"], G=kwargs.get("G", G_KM))
        return p.dynamical_time(kwargs["r"])
    if model.lower() == "nfw":
        p = NFWPotential(
            M=kwargs["M"], Rs=kwargs["Rs"],
            c=kwargs["c"], G=kwargs.get("G", G_KM),
        )
        return p.dynamical_time(kwargs["r"])
    raise ValueError(f"Unknown potential model: {model}")


def tidal_denominator(model, **kwargs):
    if model.lower() == "kepler":
        return 3 * kwargs["M"]
    if model.lower() == "nfw":
        p = NFWPotential(
            M=kwargs["M"], Rs=kwargs["Rs"], c=kwargs["c"],
        )
        return p.tidal_denominator(kwargs["r"])
    raise ValueError(f"Unknown potential model: {model}")


_POTENTIAL_MODELS: dict[str, type[KeplerPotential | NFWPotential]] = {
    "kepler": KeplerPotential,
    "nfw": NFWPotential,
}


def get_potential(
    model: str, **kwargs
) -> type[KeplerPotential | NFWPotential] | KeplerPotential | NFWPotential:
    cls = _POTENTIAL_MODELS.get(model.lower())
    if cls is None:
        raise ValueError(
            f"Unknown potential model: {model}. "
            f"Available: {list(_POTENTIAL_MODELS)}"
        )
    if not kwargs:
        return cls
    return cls(**kwargs)

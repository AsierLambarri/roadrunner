"""Kepler and NFW gravitational potential implementations.

Provides :class:`KeplerPotential`, :class:`NFWPotential`, and
convenience functions (:func:`potential`, :func:`dynamical_time`,
:func:`tidal_denominator`, :func:`get_potential`) for dispatch
by model name.
"""

import numpy as np

from roadrunner.physics.constants import G_KM, SOFTENING_KEPLER, SOFTENING_NFW

_2PI = 2 * np.pi


class KeplerPotential:
    """Kepler (point-mass) gravitational potential.

    ``Φ(r) = -G M / sqrt(r² + ε²)`` with a softening length
    :const:`SOFTENING_KEPLER` to avoid the singularity at ``r = 0``.

    Parameters
    ----------
    M : float
        Enclosed mass.
    G : float, default=G_KM
        Gravitational constant.
    """

    def __init__(self, M, G=G_KM):
        self.M = M
        self.G = G

    def potential(self, r):
        """Evaluate the potential.

        Parameters
        ----------
        r : ndarray
            Radius values.

        Returns
        -------
        phi : ndarray
            ``-G M / sqrt(r² + ε²)``.
        """
        r_safe = np.sqrt(r**2 + SOFTENING_KEPLER**2)
        return -self.G * self.M / r_safe

    def dynamical_time(self, r):
        """Dynamical time ``2π sqrt(r³ / (G M))``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        tdyn : ndarray
        """
        with np.errstate(invalid="ignore"):
            return _2PI * np.sqrt(r**3 / (self.G * self.M))

    def tidal_denominator(self, r):
        """Tidal denominator for tidal-radius estimation.

        For a Keplerian profile the tidal denominator is ``3 M``.

        Parameters
        ----------
        r : ndarray (unused)

        Returns
        -------
        denom : float or ndarray
        """
        return 3 * self.M


class NFWPotential:
    """Navarro–Frenk–White gravitational potential.

    ``Φ(r) = -G M / r · ln(1 + r/Rs) / A`` where
    ``A = ln(1 + c) - c/(1 + c)``, with a softening length
    :const:`SOFTENING_NFW` to regularise the origin.

    Parameters
    ----------
    M : float
        Virial mass.
    Rs : float
        Scale radius.
    c : float
        Concentration ``c = Rvir / Rs``.
    G : float, default=G_KM
        Gravitational constant.
    """

    def __init__(self, M, Rs, c, G=G_KM):
        self.M = M
        self.Rs = Rs
        self.c = c
        self.G = G

    def potential(self, r):
        """Evaluate the NFW potential.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        phi : ndarray
        """
        r_safe = np.sqrt(r**2 + SOFTENING_NFW**2)
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        return -self.G * (self.M / r_safe) * np.log(1 + r_safe / self.Rs) / A

    def enclosed_mass(self, r):
        """Enclosed mass at radius ``r``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        menc : ndarray
        """
        x = np.minimum(self.c, np.sqrt(r**2 + SOFTENING_NFW**2) / self.Rs)
        A = np.log(1 + self.c) - self.c / (1 + self.c)
        B = np.log(1 + x) - x / (1 + x)
        return self.M * B / A

    def dynamical_time(self, r):
        """Dynamical time using the enclosed mass.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        tdyn : ndarray
        """
        with np.errstate(invalid="ignore"):
            menc = self.enclosed_mass(r)
            return _2PI * np.sqrt(r**3 / (self.G * menc))

    def tidal_denominator(self, r):
        """Tidal denominator for the NFW profile.

        ``(M / A(c)) · (3 A(x) − x²/(1 + x)²)`` where
        ``A(x) = ln(1 + x) − x/(1 + x)`` and ``x = r / Rs``.

        Parameters
        ----------
        r : ndarray

        Returns
        -------
        denom : ndarray
        """

        def _f(x):
            """``ln(1 + x) - x / (1 + x)``, the NFW profile shape function."""
            return np.log(1 + x) - x / (1 + x)

        x = np.minimum(self.c, np.sqrt(r**2 + SOFTENING_NFW**2) / self.Rs)
        return (self.M / _f(self.c)) * (3 * _f(x) - x**2 / (1 + x)**2)


def potential(model, r, **kwargs):
    """Evaluate the potential for a given model at radii ``r``.

    Parameters
    ----------
    model : str
        ``"kepler"`` or ``"nfw"``.
    r : ndarray
    **kwargs
        Passed to the potential constructor.

    Returns
    -------
    phi : ndarray
    """
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
    """Compute the dynamical time for a given model.

    Parameters
    ----------
    model : str
    **kwargs
        Passed to the potential constructor; must include ``"r"``.

    Returns
    -------
    tdyn : ndarray
    """
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
    """Compute the tidal denominator for a given model.

    Parameters
    ----------
    model : str
    **kwargs
        Passed to the potential constructor; must include ``"r"``.

    Returns
    -------
    denom : ndarray
    """
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
    """Return a potential class (no kwargs) or an instantiated object.

    Parameters
    ----------
    model : str
    **kwargs
        Constructor arguments.  If empty the class itself is returned.

    Returns
    -------
    potential : type or instance
    """
    cls = _POTENTIAL_MODELS.get(model.lower())
    if cls is None:
        raise ValueError(
            f"Unknown potential model: {model}. "
            f"Available: {list(_POTENTIAL_MODELS)}"
        )
    if not kwargs:
        return cls
    return cls(**kwargs)

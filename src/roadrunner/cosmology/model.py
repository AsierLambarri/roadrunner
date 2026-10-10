#############################################################################
#
# package:   roadrunner.cosmology
# file:      model.py
# brief:     Flat LCDM cosmology, scoped as the pipeline's active cosmology.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   10 Oct 2026 - Created
#
#############################################################################

"""Flat LCDM cosmology, scoped as the pipeline's active cosmology.

:class:`Cosmology` is an immutable bag of flat-LCDM parameters that lazily
builds its own Ishiyama et al. (2021) ``c_vir(M_vir, z)`` table on first use
(see :mod:`roadrunner.cosmology.ishiyama21`). A snapshot's cosmology is made
the active one for the duration of a ``with cosmology(...):`` block; code
that needs it anywhere inside that block (e.g. :func:`merger_tree.concentration`)
reads it back with :func:`current_cosmology`, rather than threading a
``Cosmology`` argument through every call in between.
"""

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property

from roadrunner._exceptions import ConfigurationError
from roadrunner.cosmology import ishiyama21

_active = contextvars.ContextVar("roadrunner_cosmology")   # no default: unset raises LookupError


@dataclass(frozen=True)
class Cosmology:
    """Flat LCDM cosmology, with its own lazily-built concentration table.

    Parameters
    ----------
    h : float
        Dimensionless Hubble constant, ``H0 / (100 km/s/Mpc)``.
    omega_m : float
        Matter density parameter at z = 0.
    omega_b : float
        Baryon density parameter at z = 0.
    sigma8 : float
        RMS matter fluctuation in 8 Mpc/h spheres at z = 0.
    n_s : float
        Scalar spectral index.

    Notes
    -----
    CMB temperature (``Tcmb0 = 2.7255`` K) and effective neutrino species
    (``Neff = 3.046``) are fixed module constants, not fields: they only
    enter the sub-percent radiation correction to the growth factor, and
    are never varied by any caller.
    """

    h: float
    omega_m: float
    omega_b: float
    sigma8: float
    n_s: float

    @cached_property
    def _ln_c(self):
        return ishiyama21.table(self)

    def concentration(self, m_vir, z):
        """Median Ishiyama et al. (2021) NFW concentration ``c_vir(M_vir, z)``.

        Parameters
        ----------
        m_vir : array_like
            Virial mass (Bryan & Norman 1998), Msun.
        z : array_like
            Redshift. Broadcasts against ``m_vir``.

        Returns
        -------
        c_vir : ndarray
            Shape of ``np.broadcast(m_vir, z)``.
        """
        return ishiyama21.lookup(self._ln_c, m_vir, z)


@contextmanager
def cosmology(c):
    """Make ``c`` the active cosmology for the duration of the ``with`` block (nestable).

    Parameters
    ----------
    c : Cosmology
    """
    token = _active.set(c)
    try:
        yield
    finally:
        _active.reset(token)


def current_cosmology():
    """The active cosmology.

    Returns
    -------
    cosmology : Cosmology

    Raises
    ------
    ConfigurationError
        Outside any ``with cosmology(...):`` block.
    """
    try:
        return _active.get()
    except LookupError:
        raise ConfigurationError("no active cosmology: wrap the call in `with cosmology(Cosmology(...)):`") from None

#############################################################################
#
# package:   roadrunner.scripts
# file:      make_concentration_reference.py
# brief:     Colossus Ishiyama+21 c_vir(M, z) reference table for the test suite.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   10 Oct 2026 - Created
#
#############################################################################

"""Colossus Ishiyama+21 ``c_vir(M, z)`` reference table for the test suite.

Run once, by hand, in the throwaway ``rr-colossus-diag`` conda environment
(``colossus`` pins incompatible dependency versions, so it is never a
``roadrunner`` dependency). Writes ``tests/data/ishiyama21_reference.csv``,
read back by ``tests/unit/test_cosmology.py`` to check
:mod:`roadrunner.cosmology.ishiyama21` against ``colossus`` itself. This
script never imports ``pandas`` or ``roadrunner``.
"""

import csv
import os
import shutil
import tempfile

_TMPDIR = tempfile.mkdtemp(prefix="rr_colossus_")

import colossus.settings as settings   # noqa: E402  (BASE_DIR must be set before the next imports)
settings.BASE_DIR = _TMPDIR
settings.PERSISTENCE = ""

import numpy as np   # noqa: E402
from colossus.cosmology import cosmology   # noqa: E402
from colossus.halo.concentration import concentration   # noqa: E402

_BUILTIN = ("planck13", "planck15", "planck18", "WMAP9")
_CUSTOM = {
    "VINTERGATAN": dict(h=0.702, omega_m=0.272, omega_b=0.045, sigma8=0.807, n_s=0.961),
    "VINTERGATAN-GM": dict(h=0.6727, omega_m=0.3139, omega_b=0.04916, sigma8=0.844, n_s=0.9645),
    "Auriga": dict(h=0.6777, omega_m=0.307, omega_b=0.048, sigma8=0.8288, n_s=0.9611),
}

_LOG10_M = np.arange(3.025, 16.0, 0.75)
_Z = (0.0, 0.3, 0.7, 1.2, 2.0, 3.0, 4.5, 5.5, 7.0, 9.0, 11.0, 14.0, 17.0, 19.5, 22.0, 26.0, 29.5)

_OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "ishiyama21_reference.csv")


def _rows(name, cosmo):
    """One CSV row per ``(log10_m, z)`` grid point, for an already-current ``cosmo``."""
    h, om, ob, s8, ns = cosmo.h, cosmo.Om0, cosmo.Ob0, cosmo.sigma8, cosmo.ns
    M_h = 10.0**_LOG10_M * h
    for z in _Z:
        c = concentration(M_h, "vir", z, model="ishiyama21", halo_sample="all", c_type="fit")
        for log10_m, c_i in zip(_LOG10_M, c):
            yield [name, h, om, ob, s8, ns, log10_m, z, c_i]


def main():
    try:
        os.makedirs(os.path.dirname(_OUT), exist_ok=True)
        with open(_OUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cosmology", "h", "omega_m", "omega_b", "sigma8", "n_s", "log10_m", "z", "c_colossus"])
            for name in _BUILTIN:
                cosmo = cosmology.setCosmology(name, persistence="")
                writer.writerows(_rows(name, cosmo))
            for name, p in _CUSTOM.items():
                cosmo = cosmology.Cosmology(name=name, flat=True, H0=100.0 * p["h"], Om0=p["omega_m"],
                                             Ob0=p["omega_b"], sigma8=p["sigma8"], ns=p["n_s"], persistence="")
                cosmology.setCurrent(cosmo)
                writer.writerows(_rows(name, cosmo))
        print(f"wrote {_OUT}")
    finally:
        shutil.rmtree(_TMPDIR, ignore_errors=True)


if __name__ == "__main__":
    main()

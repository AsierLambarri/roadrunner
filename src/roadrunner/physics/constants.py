from math import sqrt


G_KM = 4.300917270038E-6  # kpc * km^2 / s^2 / Msun

G_GALACTIC = 4.49850215E-6  # kpc^3 / (Msun * Gyr^2)

DYN_TIME_FACTOR = sqrt(G_GALACTIC/G_KM)  # converts km/s to kpc/Gyr

SOFTENING_KEPLER = 1E-3

SOFTENING_NFW = 1E-3

__all__ = [
    "G_KM",
    "G_GALACTIC",
    "DYN_TIME_FACTOR",
    "SOFTENING_KEPLER",
    "SOFTENING_NFW",
]

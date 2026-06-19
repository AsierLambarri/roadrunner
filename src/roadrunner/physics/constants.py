from math import sqrt


G_KM = 4.300917270038E-6  # kpc * km^2 / s^2 / Msun

G_GALACTIC = 4.49850215E-6  # kpc^3 / (Msun * Gyr^2)

DYN_TIME_FACTOR = sqrt(G_GALACTIC/G_KM)  # converts km/s to kpc/Gyr

SOFTENING_KEPLER = 1E-3

SOFTENING_NFW = 1E-3

RILEY_BOUND_THRESHOLD = 0.97
RILEY_SVM_SLOPE = 3.51
RILEY_SVM_INTERCEPT = 1.08
RILEY_WI = 4 # units of kpc·km^-1·s
# ── Timescales ─────────────────────────────────────────────────────
MAX_DYN_TIMESCALE = 0.4          # Gyr, upper clamp for particle dynamical time

# ── Duffy 2008 c(M,z) relation ─────────────────────────────────────
DUFFY_A = 7.85
DUFFY_B = -0.081
DUFFY_C = -0.71
DUFFY_PIVOT_MASS = 2e12          # Msun/h

# ── Merger tree geometry ───────────────────────────────────────────
MIN_DISTANCE = 1e-10              # kpc, floor to avoid division by zero

# ── StandardScaler ─────────────────────────────────────────────────
SCALER_RANGE = 10.0               # maps data range to interval of width 10

# ── Mixing analysis ────────────────────────────────────────────────
NN_FRACTION = 0.01                 # nearest-neighbour fraction for local sigma


__all__ = [
    "G_KM",
    "G_GALACTIC",
    "DYN_TIME_FACTOR",
    "SOFTENING_KEPLER",
    "SOFTENING_NFW",
    "RILEY_BOUND_THRESHOLD",
    "RILEY_SVM_SLOPE",
    "RILEY_SVM_INTERCEPT",
    "MAX_DYN_TIMESCALE",
    "DUFFY_A",
    "DUFFY_B",
    "DUFFY_C",
    "DUFFY_PIVOT_MASS",
    "MIN_DISTANCE",
    "SCALER_RANGE",
    "NN_FRACTION",
]

#############################################################################
#
# package:   roadrunner
# file:      _defaults.py
# brief:     Global algorithmic defaults and configuration constants.
#
# Constants are organised by subsystem (birth tracker, galaxy properties,
# centering, GMM assignment, BGMM priors, coresets, k-means, I/O).
# These values serve as module-level defaults and are imported where needed.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
# changes:   16 Jun 2026 - Created
#            18 Jun 2026 - Last edit
#
#############################################################################

"""Module-level default constants for the roadrunner package."""

import contextvars
from contextlib import contextmanager

import numpy as np

# ── Birth tracker ──────────────────────────────────────────────────
BIRTH_GAUSSIAN_WIDTH = 0.3        # width of the gaussian window function

# ── Galaxy properties ──────────────────────────────────────────────
MIN_PARTICLES_STRUCTURAL = 30      # min particles for rh/sigma/props
FRAGMENT_THRESHOLD = 10            # galaxies with fewer particles are fragments
MASS_FRACTION_R20 = 0.2
MASS_FRACTION_RH = 0.5
MASS_FRACTION_R80 = 0.8

# ── Centering ──────────────────────────────────────────────────────
SSC_NMIN = 30                      # default shrink-sphere minimum particles
SSC_ALPHA = 0.9                    # default shrink-sphere shrink factor

# ── GMM assignment ─────────────────────────────────────────────────
UNRESOLVED_GROUP_RATIO = 10        # N < RATIO * n_comp triggers flat assignment

# ── BGMM priors ────────────────────────────────────────────────────
PRIOR_DOF_OFFSET = 4               # degrees_of_freedom = n_features + offset
PRIOR_COVARIANCE_SCALE_SMALL = 1.0 # covariance_prior scale f(n) at n <= PRIOR_SCALE_N_SMALL
PRIOR_COVARIANCE_SCALE_LARGE = 0.7 # f(n) at n >= PRIOR_SCALE_N_LARGE, flat beyond
PRIOR_SCALE_N_SMALL = 10.0         # f(n) is linear in n between these two counts
PRIOR_SCALE_N_LARGE = 300.0
PRIOR_GROWTH_MIN = 0.05           # bounds on the covariance growth factor (M_t/M_t-1)^(2/3):
PRIOR_GROWTH_MAX = 2.0            # the stellar cloud is detached from the dark mass
PRIOR_WEIGHT_DIVISOR = 5.0         # nk / divisor → weight_concentration_prior
PRIOR_MEAN_DIVISOR = 20.0          # nk / divisor → mean_precision_prior

# ── Plausibility (latent prior α, "energy" model) ──────────────────
PLAUSIBILITY_PRIOR_WEIGHT = 0.2    # A: Errani prior's weight relative to the member histogram
PLAUSIBILITY_FLOOR = 1e-8          # κ: share of members that look like random halo particles
PLAUSIBILITY_MIN_BINS = 10         # adaptive histogram in t = -ln u: equal-weight bins,
PLAUSIBILITY_MAX_BINS = 100        # 2 n_eff^(1/3) of them (Rice rule), clipped to these
PLAUSIBILITY_MIN_MEMBERS = 100     # minimum effective member count for a refit (energy, phase)
PLAUSIBILITY_LIT_SCATTER_DEX = 0.4 # "phase": member radii around r_1/2: exp. disk 0.35 dex ⊕ Kravtsov 0.2 dex
PLAUSIBILITY_MIN_SCATTER_DEX = 0.2 # "phase": sigma* floor, Kravtsov's galaxy-size scatter (mapped per halo)
PLAUSIBILITY_SHRINK_COUNT = 100    # "kinematic": prior count shrinking each halo's member <u^2> to the pooled one

# ── Coresets ───────────────────────────────────────────────────────
CORESET_ALPHA_BASE = 16
CORESET_ALPHA_OFFSET = 2

# ── K-means ────────────────────────────────────────────────────────
KMEANS_MAX_ITER = 300
KMEANS_PP_MAX_ITER = 1

# ── IO ─────────────────────────────────────────────────────────────
ZSTD_COMPRESSION_LEVEL = 3
COL_WIDTH_RUNTIME = 12
COL_WIDTH_INT = 10
COL_WIDTH_FLOAT = 9

# ── Precision (single/double) ────────────────────────────────────
# User-facing names; "single" and "double" are the only valid values.
# Consumed via data_dtype()/math_dtype() or the precision() scope —
# never passed as method arguments.
_PRECISION_MAP = {"single": np.float32, "double": np.float64}

_data_var = contextvars.ContextVar("roadrunner_data_precision", default="single")
_math_var = contextvars.ContextVar("roadrunner_math_precision", default="single")


def _resolve_precision(name):
    """Map a user precision name to a numpy dtype (raises on unknown)."""
    try:
        return _PRECISION_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown precision {name!r}. Choose from {sorted(_PRECISION_MAP)}"
        ) from None


@contextmanager
def precision(data=None, math=None):
    """Scope the data/math precision for a run (nestable, thread-safe).

    Parameters
    ----------
    data, math : {"single", "double"} or None
        ``None`` leaves the current value untouched.
    """
    tokens = []
    if data is not None:
        _resolve_precision(data)
        tokens.append((_data_var, _data_var.set(data)))
    if math is not None:
        _resolve_precision(math)
        tokens.append((_math_var, _math_var.set(math)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def data_dtype():
    """Numpy dtype for loaded particle data (mass/position/velocity)."""
    return _resolve_precision(_data_var.get())


def math_dtype():
    """Numpy dtype for compute kernels (mixture/scaler/halos)."""
    return _resolve_precision(_math_var.get())


# ── Integer IDs ──────────────────────────────────────────────────
# SIM_ID: particle IDs from simulations (uint64, never narrowed/signed).
# LOCAL_IDX: local array positions (signed: -1 == missing).
# GALAXY_ID: galaxy/Sub_tree IDs (signed: -1 == unbound, load-bearing
#   sentinel in assembly, statistics, and properties — never uint).
# SNAP_ID: snapshot numbers (small).
SIM_ID = np.uint64
LOCAL_IDX = np.int64
GALAXY_ID = np.int64
SNAP_ID = np.int32
UNBOUND = MISSING = -1


__all__ = [
    "BIRTH_GAUSSIAN_WIDTH",
    "MIN_PARTICLES_STRUCTURAL",
    "FRAGMENT_THRESHOLD",
    "MASS_FRACTION_R20",
    "MASS_FRACTION_RH",
    "MASS_FRACTION_R80",
    "SSC_NMIN",
    "SSC_ALPHA",
    "UNRESOLVED_GROUP_RATIO",
    "PRIOR_DOF_OFFSET",
    "PRIOR_COVARIANCE_SCALE_SMALL",
    "PRIOR_COVARIANCE_SCALE_LARGE",
    "PRIOR_SCALE_N_SMALL",
    "PRIOR_SCALE_N_LARGE",
    "PRIOR_WEIGHT_DIVISOR",
    "PRIOR_MEAN_DIVISOR",
    "CORESET_ALPHA_BASE",
    "CORESET_ALPHA_OFFSET",
    "KMEANS_MAX_ITER",
    "KMEANS_PP_MAX_ITER",
    "ZSTD_COMPRESSION_LEVEL",
    "COL_WIDTH_RUNTIME",
    "COL_WIDTH_INT",
    "COL_WIDTH_FLOAT",
    "precision",
    "data_dtype",
    "math_dtype",
    "SIM_ID",
    "LOCAL_IDX",
    "GALAXY_ID",
    "SNAP_ID",
    "UNBOUND",
    "MISSING",
]
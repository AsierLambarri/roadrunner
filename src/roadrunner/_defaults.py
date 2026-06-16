# ── Birth tracker ──────────────────────────────────────────────────
BIRTH_GAUSSIAN_WIDTH = 0.3        # width of the gaussian window function

# ── Galaxy properties ──────────────────────────────────────────────
MIN_PARTICLES_STRUCTURAL = 30      # min particles for rh/sigma/props
FRAGMENT_THRESHOLD = 10            # galaxies with fewer particles are fragments
MASS_FRACTION_R20 = 0.2
MASS_FRACTION_RH = 0.5
MASS_FRACTION_R80 = 0.8
CENTER_QUANTILE = 0.95             # inner-region quantile for center finding
CENTER_SCALE = 0.5                 # inner-region scale factor

# ── Mixing analysis ────────────────────────────────────────────────
NN_FRACTION = 0.01                 # nearest-neighbour fraction for local sigma

# ── GMM assignment ─────────────────────────────────────────────────
UNRESOLVED_GROUP_RATIO = 10        # N < RATIO * n_comp triggers flat assignment

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


__all__ = [
    "BIRTH_GAUSSIAN_WIDTH",
    "MIN_PARTICLES_STRUCTURAL",
    "FRAGMENT_THRESHOLD",
    "MASS_FRACTION_R20",
    "MASS_FRACTION_RH",
    "MASS_FRACTION_R80",
    "CENTER_QUANTILE",
    "CENTER_SCALE",
    "NN_FRACTION",
    "UNRESOLVED_GROUP_RATIO",
    "CORESET_ALPHA_BASE",
    "CORESET_ALPHA_OFFSET",
    "KMEANS_MAX_ITER",
    "KMEANS_PP_MAX_ITER",
    "ZSTD_COMPRESSION_LEVEL",
    "COL_WIDTH_RUNTIME",
    "COL_WIDTH_INT",
    "COL_WIDTH_FLOAT",
]
#############################################################################
#
# package:   roadrunner.clustering.assignment
# file:      plausibility.py
# brief:     Plausibility models: the latent prior α of the assignment fits.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
#
#############################################################################

"""Plausibility models: the latent prior α of the assignment fits.

``alpha[n, k]`` is the plausibility of particle ``n``'s boundness under
"``n`` belongs to halo ``k``'s galaxy" (model B, soft-label evidence). Only
bound pairs carry a positive value. Every model is computed per halo,
before the groups are fit, and exposes the same interface:

- ``prepare(halos)``: α of every bound pair of this snapshot;
- ``column_values(column_id)``: α of the columns of a group's matrix;
- ``update(resp_map, weights)``: refit after the snapshot (between
  snapshots, never inside EM);
- ``state`` / ``set_state``: checkpoint round trip.

Models:

- ``"rank"``: ``log1p`` of the ordinal rank of the boundness within each
  halo.
- ``"energy"`` (NFW only): the likelihood ratio ``p*(u)`` of member stars
  over the halo's dark matter, on the dark-matter energy rank
  ``u = F_c(calE)`` (uniform for the dark matter itself). ``p*`` is a
  histogram of the previous snapshot's members in ``t = -ln u``, mixed with
  Errani et al. (2022)'s tagging ratio (exponential galaxies, with
  ``r_1/2 = 0.015 R_vir``, Kravtsov 2013) and floored:
  ``alpha = (1 - kappa) [(1 - lam) h(t) e^t + lam P*] + kappa``,
  ``lam = A / (1 + A)``. Without a histogram yet (first snapshot) ``lam = 1``.
"""

import functools

import numpy as np
from scipy.stats import rankdata

from roadrunner._defaults import (
    PLAUSIBILITY_FLOOR,
    PLAUSIBILITY_MAX_BINS,
    PLAUSIBILITY_MIN_BINS,
    PLAUSIBILITY_PRIOR_WEIGHT,
)
from roadrunner.physics.energy_distribution import (
    CAL_GRID,
    nfw_density_of_states,
    nfw_phi,
    trapezoid_weights,
)

ERRANI_SLOPE = 3.3          # dN*/dcalE ∝ calE^3.3 exp(-(calE / calE_s)^4)
ERRANI_SHARPNESS = 4.0
KRAVTSOV_RHALF = 0.015      # stellar half-mass radius / virial radius
_U_MIN = 1e-12              # u floor: t = -ln u <= 27.6
_LOG_ALPHA_MAX = 50.0       # alpha in [1e-8, e^50]: row sums finite and row ratios
                            # (>= 1e-8 / (K e^50) ~ 1e-30) normal in float32
_EMPTY = np.empty(0, dtype=np.float64)


def _log1p_rank(values):
    """``log1p`` of the ordinal ranks of ``values`` (float64; empty stays empty).

    Parameters
    ----------
    values : ndarray
        Boundness values of one halo.

    Returns
    -------
    alpha : ndarray of float64
    """
    if len(values) == 0:
        return np.asarray(values, dtype=np.float64)
    return np.log1p(rankdata(values, method="ordinal"))


def _errani_unnormalised(cal, es):
    """Errani's stellar energy distribution ``calE^a exp(-(calE / calE_s)^b)``, unnormalised."""
    return cal**ERRANI_SLOPE * np.exp(-(cal / es) ** ERRANI_SHARPNESS)


@functools.cache
def _errani_scales():
    """Errani ``calE_s`` against the potential depth at the stellar half-mass radius.

    For each trial scale the stellar distribution function is
    ``f* = n*(calE) / g(calE)``, its density
    ``rho*(x) ∝ int f* sqrt(2 (E - phi(x))) dE``, and ``r_1/2`` is read off
    the enclosed mass. The depth there, ``calE_1/2 = 1 - phi(r_1/2) / phi(0)``,
    increases with the scale, so ``calE_s`` is tabulated against it. A halo
    then needs only ``Phi(KRAVTSOV_RHALF R_vir) / Phi_0`` to pick its scale:
    no profile parameter is read.

    Returns
    -------
    cal_half : ndarray of shape (80,)
        Depth at the half-mass radius, increasing.
    es : ndarray of shape (80,)
        Matching ``calE_s``.
    """
    es_try = np.geomspace(3e-3, 0.95, 80)
    x = np.geomspace(1e-4, 10**2.5, 240)
    tw = trapezoid_weights(CAL_GRID)
    f_star = (_errani_unnormalised(CAL_GRID[None, :], es_try[:, None])
              / nfw_density_of_states(CAL_GRID, np.inf)[None, :])
    kin = np.sqrt(np.clip(2.0 * ((CAL_GRID - 1.0)[None, :] - nfw_phi(x)[:, None]), 0.0, None))
    rho = (f_star * tw) @ kin.T                                  # (trial, radius)
    seg = 0.5 * (rho[:, 1:] * x[1:] ** 2 + rho[:, :-1] * x[:-1] ** 2) * np.diff(x)
    mass = np.concatenate([np.zeros((len(es_try), 1)), np.cumsum(seg, axis=1)], axis=1)
    r_half = np.array([np.interp(0.5, m / m[-1], x) for m in mass])
    return 1.0 + nfw_phi(r_half), es_try


def errani_log_ratio(halo, eps):
    """Log Errani tagging ratio ``log P*`` of a halo's bound particles.

    ``P* = n*(calE; calE_s) / n_DM(calE)`` at ``calE = 1 - eps``, both
    normalised on ``[0, 1]``. ``calE_s`` places the stellar half-mass
    radius at ``KRAVTSOV_RHALF * R_vir`` (Kravtsov 2013); it is found from
    the halo's potential depth there.

    Parameters
    ----------
    halo : HaloModel
        Halo whose potential defines ``central_potential`` and
        ``log_energy_density`` (NFW).
    eps : ndarray
        Normalised boundness ``E / Phi_0``.

    Returns
    -------
    log_ratio : ndarray
    """
    r_half = np.array([KRAVTSOV_RHALF * halo.virial_radius])
    depth = 1.0 - float(halo.potential(r_half)[0]) / halo.central_potential()
    cal_half, es_try = _errani_scales()
    es = float(np.exp(np.interp(np.log(depth), np.log(cal_half), np.log(es_try))))
    cal = np.clip(1.0 - eps, CAL_GRID[0], CAL_GRID[-1])
    log_norm = np.log(trapezoid_weights(CAL_GRID) @ _errani_unnormalised(CAL_GRID, es))
    log_star = ERRANI_SLOPE * np.log(cal) - (cal / es) ** ERRANI_SHARPNESS - log_norm
    return log_star - halo.log_energy_density(eps)


class _Plausibility:
    """Shared interface: per-halo α, a no-op refit and no state."""

    name = ""

    def __init__(self):
        self._alpha: dict[int, np.ndarray] = {}

    def prepare(self, halos):
        """Compute α for every bound pair of this snapshot.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            Halos with boundness already computed.
        """
        raise NotImplementedError

    def column_values(self, column_id):
        """α of each column of a group's boundness matrix.

        Parameters
        ----------
        column_id : ndarray of int
            ``Sub_tree_id`` of each column.

        Returns
        -------
        values : list of ndarray of float64
            Parallel to the columns' (index-sorted) bound particles; empty
            for halos without boundness (empty columns).
        """
        return [self._alpha.get(int(s), _EMPTY) for s in column_id]

    def update(self, resp_map, weights=None):
        """Refit after a snapshot (no-op here).

        Parameters
        ----------
        resp_map : dict of {int: (ndarray, ndarray)}
            ``Sub_tree_id -> (particle rows, responsibilities)``.
        weights : ndarray of shape (n_particles,), optional
            Particle weights (masses under ``mass_weighting``).
        """
        return None

    @property
    def state(self):
        """Checkpoint state (``None``: nothing to keep)."""
        return None

    def set_state(self, state):
        """Restore a checkpoint state (no-op here).

        Parameters
        ----------
        state : object or None
        """
        return None


class RankPlausibility(_Plausibility):
    """``log1p`` of the ordinal boundness rank within each halo."""

    name = "rank"

    def prepare(self, halos):
        """Compute α for every bound pair of this snapshot.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
        """
        self._alpha = {int(h.sub_tree_id): _log1p_rank(h.get_boundness()[1])
                       for h in halos if h.has_boundness}


class EnergyPlausibility(_Plausibility):
    """Hybrid empirical ``p*(u)``: member histogram in ``t = -ln u`` + Errani prior, floored."""

    name = "energy"

    def __init__(self):
        super().__init__()
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._edges: np.ndarray | None = None
        self._log_density: np.ndarray | None = None

    def _log_pstar(self, t):
        """Log histogram ratio ``log h(t) + t``, held at the end bins beyond the data."""
        tc = np.clip(t, self._edges[0], self._edges[-1])
        b = np.clip(np.searchsorted(self._edges, tc, side="right") - 1, 0, self._log_density.size - 1)
        return self._log_density[b] + tc

    def prepare(self, halos):
        """Compute α for every bound pair of this snapshot.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            NFW halos with boundness ``E / Phi_0`` already computed.
        """
        lam = PLAUSIBILITY_PRIOR_WEIGHT / (1.0 + PLAUSIBILITY_PRIOR_WEIGHT)
        self._alpha, self._t = {}, {}
        for h in halos:
            if not h.has_boundness:
                continue
            rows, eps, _ = h.get_boundness()
            eps = eps.astype(np.float64)
            t = -np.log(np.clip(h.energy_fraction(eps), _U_MIN, 1.0))
            log_mix = errani_log_ratio(h, eps)
            if self._edges is not None:
                log_mix = np.logaddexp(np.log1p(-lam) + self._log_pstar(t), np.log(lam) + log_mix)
            log_alpha = np.logaddexp(np.log1p(-PLAUSIBILITY_FLOOR) + log_mix, np.log(PLAUSIBILITY_FLOOR))
            sid = int(h.sub_tree_id)
            self._alpha[sid] = np.exp(np.minimum(log_alpha, _LOG_ALPHA_MAX))
            self._t[sid] = (rows, t)

    def update(self, resp_map, weights=None):
        """Refit the member histogram from this snapshot's responsibilities.

        Pairs are the bound pairs with their responsibility ``r_nk`` (times
        the particle weight); single-halo groups give ``r = 1``. Bins are
        equal-weight quantiles of ``t``, ``2 n_eff^(1/3)`` of them clipped
        to ``[PLAUSIBILITY_MIN_BINS, PLAUSIBILITY_MAX_BINS]``. With
        ``n_eff < 10 PLAUSIBILITY_MIN_BINS`` the previous histogram is kept.

        Parameters
        ----------
        resp_map : dict of {int: (ndarray, ndarray)}
            ``Sub_tree_id -> (sorted particle rows, responsibilities)``.
        weights : ndarray of shape (n_particles,), optional
            Particle weights (masses under ``mass_weighting``).
        """
        ts, ws = [], []
        for sid, (rows_r, r) in resp_map.items():
            cached = self._t.get(int(sid))
            if cached is None:
                continue
            rows_b, t = cached
            pos = np.searchsorted(rows_b, rows_r)
            ok = pos < rows_b.size
            ok[ok] = rows_b[pos[ok]] == rows_r[ok]
            w = np.asarray(r, dtype=np.float64)[ok]
            if weights is not None:
                w = w * weights[rows_r[ok]]
            ts.append(t[pos[ok]])
            ws.append(w)
        if not ts:
            return
        t, w = np.concatenate(ts), np.concatenate(ws)
        keep = w > 0
        t, w = t[keep], w[keep]
        if w.size == 0:
            return
        n_eff = w.sum() ** 2 / np.dot(w, w)
        if n_eff < 10 * PLAUSIBILITY_MIN_BINS:
            return
        n_bins = int(np.clip(round(2.0 * n_eff ** (1.0 / 3.0)),
                             PLAUSIBILITY_MIN_BINS, PLAUSIBILITY_MAX_BINS))
        order = np.argsort(t)
        t, w = t[order], w[order]
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        edges = np.unique(np.interp(np.linspace(0.0, 1.0, n_bins + 1), cdf, t))
        if edges.size < 2:
            return
        mass, _ = np.histogram(t, bins=edges, weights=w)
        density = mass / (w.sum() * np.diff(edges))
        self._edges = edges
        self._log_density = np.log(np.maximum(density, np.finfo(np.float64).tiny))

    @property
    def state(self):
        """Histogram ``{"edges", "log_density"}``, or ``None`` before the first refit."""
        if self._edges is None:
            return None
        return {"edges": self._edges, "log_density": self._log_density}

    def set_state(self, state):
        """Restore the histogram (``None``: start from the Errani prior alone).

        Parameters
        ----------
        state : dict or None
        """
        if state is None:
            self._edges = self._log_density = None
        else:
            self._edges = np.asarray(state["edges"], dtype=np.float64)
            self._log_density = np.asarray(state["log_density"], dtype=np.float64)


_MODELS = {"rank": RankPlausibility, "energy": EnergyPlausibility}


def make_plausibility(name):
    """Instantiate a plausibility model by name.

    Parameters
    ----------
    name : str
        ``"rank"`` or ``"energy"``.

    Returns
    -------
    model : RankPlausibility or EnergyPlausibility
    """
    try:
        return _MODELS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown plausibility {name!r}. Choose from: {list(_MODELS)}")

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
- ``"phase"`` (any potential): the same ratio on the fraction ``w`` of the
  halo's bound phase space more bound than the particle (uniform for a
  smooth phase-space background, so ``t = -ln w`` is ``Exp(1)``), with a
  log-normal member model ``t ~ Normal(mu*, sigma*)``: the pooled moments of
  the previous snapshot's members mixed with each halo's literature model
  (circular orbits around ``r_1/2 = 0.015 R_vir``), and floored.
- ``"kinematic"`` (any potential): co-movement. ``u^2 = |v - v_k|^2 / v_esc,k^2(r)``
  at the particle's position, so ``nu = u^3`` is ``U(0, 1)`` for a uniform
  phase-space background there and the potential at that position carries
  no evidence. Members follow an isotropic equilibrium tracer
  ``f ∝ (-E)^q`` (``u^2 ~ Beta(3/2, q + 1)``, hot or cold), with each halo's
  ``<u^2>`` from its previous-snapshot members shrunk to the pooled value,
  mixed with the literature value ``v_c^2 / v_esc^2`` at ``r_1/2``
  (Wolf et al. 2010), and floored.
"""

import functools

import numpy as np
from scipy.special import betaln, xlog1py
from scipy.stats import rankdata

from roadrunner._defaults import (
    math_dtype,
    PLAUSIBILITY_FLOOR,
    PLAUSIBILITY_LIT_SCATTER_DEX,
    PLAUSIBILITY_MAX_BINS,
    PLAUSIBILITY_MIN_BINS,
    PLAUSIBILITY_MIN_MEMBERS,
    PLAUSIBILITY_MIN_SCATTER_DEX,
    PLAUSIBILITY_PRIOR_WEIGHT,
    PLAUSIBILITY_SHRINK_COUNT,
)
from roadrunner.physics.distribution import nfw_density_of_states, nfw_phi, trapezoid_weights

ERRANI_SLOPE = 3.3          # dN*/dcalE ∝ calE^3.3 exp(-(calE / calE_s)^4)
ERRANI_SHARPNESS = 4.0
KRAVTSOV_RHALF = 0.015      # stellar half-mass radius / virial radius
CAL_GRID = np.concatenate([np.geomspace(1e-4, 0.1, 200, endpoint=False), np.linspace(0.1, 1.0 - 1e-4, 200)])
                            # binding depths of Errani's integrals
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


def errani_log_ratio(halo, b):
    """Log Errani tagging ratio ``log P*`` of a halo's bound particles.

    ``P* = n*(calE; calE_s) / n_DM(calE)`` at ``calE = 1 - E / Φ₀``, both
    normalised on ``[0, 1]``. ``calE_s`` places the stellar half-mass
    radius at ``KRAVTSOV_RHALF * R_vir`` (Kravtsov 2013); it is found from
    the halo's potential depth there. The stored boundness gives
    ``E = -b s`` (``s`` the halo's energy scale), and the halo's
    ``ln dN/db`` becomes ``ln dN/dcalE`` with ``|db / dcalE| = |Φ₀| / s``.

    Parameters
    ----------
    halo : HaloModel
        Halo whose potential defines ``central_potential`` and
        ``log_energy_density`` (NFW).
    b : ndarray
        Stored boundness ``-E / halo.binding_energy_scale()``.

    Returns
    -------
    log_ratio : ndarray
    """
    phi0, s = halo.central_potential(), halo.binding_energy_scale()
    r_half = np.array([KRAVTSOV_RHALF * halo.virial_radius])
    depth = 1.0 - float(halo.potential(r_half)[0]) / phi0
    cal_half, es_try = _errani_scales()
    es = float(np.exp(np.interp(np.log(depth), np.log(cal_half), np.log(es_try))))
    cal = np.clip(1.0 + b * s / phi0, CAL_GRID[0], CAL_GRID[-1])        # calE = 1 - E / Φ₀, E = -b s
    log_norm = np.log(trapezoid_weights(CAL_GRID) @ _errani_unnormalised(CAL_GRID, es))
    log_star = ERRANI_SLOPE * np.log(cal) - (cal / es) ** ERRANI_SHARPNESS - log_norm
    # n_DM per unit calE = per unit b times |db / dcalE| = |Φ₀| / s
    return log_star - (halo.log_energy_density(b) + np.log(-phi0 / s))


class _Plausibility:
    """Shared interface: per-halo α, a no-op refit and no state."""

    name = ""

    def __init__(self):
        self._alpha: dict[int, np.ndarray] = {}

    def prepare(self, halos, coords=None):
        """Compute α for every bound pair of this snapshot.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            Halos with boundness already computed.
        coords : ndarray of shape (n_particles, 6), optional
            Phase-space coordinates (used by ``kinematic`` only).
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
        values : list of ndarray of math_dtype
            Parallel to the columns' (index-sorted) bound particles; empty
            for halos without boundness (empty columns). α is computed in
            float64, from the halos' float64 distributions, and cast here
            once: the boundary to ``math_dtype()``.
        """
        md = math_dtype()
        return [self._alpha.get(int(s), _EMPTY).astype(md, copy=False) for s in column_id]

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

    def _member_pairs(self, resp_map, weights=None):
        """Per halo: cached per-pair values and weights ``r_nk (x weight)`` of this snapshot's bound members.

        Parameters
        ----------
        resp_map : dict of {int: (ndarray, ndarray)}
            ``Sub_tree_id -> (sorted particle rows, responsibilities)``.
        weights : ndarray of shape (n_particles,), optional
            Particle weights (masses under ``mass_weighting``).

        Returns
        -------
        per : dict of {int: (ndarray, ndarray)}
            Positive weights only; halos without cached values are skipped.
        """
        per = {}
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
            keep = w > 0
            per[int(sid)] = (t[pos[ok]][keep], w[keep])
        return per

    def _member_depths(self, resp_map, weights=None):
        """Depths ``t`` and weights ``r_nk (x weight)`` of this snapshot's bound member pairs.

        Uses the per-halo ``(rows, t)`` that a refitting model's ``prepare``
        caches in ``self._t``.

        Parameters
        ----------
        resp_map : dict of {int: (ndarray, ndarray)}
            ``Sub_tree_id -> (sorted particle rows, responsibilities)``.
        weights : ndarray of shape (n_particles,), optional
            Particle weights (masses under ``mass_weighting``).

        Returns
        -------
        t, w : ndarray
            Positive weights only. ``None, None`` when the effective member
            count is below ``PLAUSIBILITY_MIN_MEMBERS`` (keep the previous fit).
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
            return None, None
        t, w = np.concatenate(ts), np.concatenate(ws)
        keep = w > 0
        t, w = t[keep], w[keep]
        if w.size == 0 or w.sum() ** 2 / np.dot(w, w) < PLAUSIBILITY_MIN_MEMBERS:
            return None, None
        return t, w

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

    def prepare(self, halos, coords=None):
        """Compute α for every bound pair of this snapshot.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
        coords : ndarray of shape (n_particles, 6), optional
            Unused here.
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

    def prepare(self, halos, coords=None):
        """Compute α for every bound pair of this snapshot.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            NFW halos with boundness already computed.
        coords : ndarray of shape (n_particles, 6), optional
            Unused here.
        """
        lam = PLAUSIBILITY_PRIOR_WEIGHT / (1.0 + PLAUSIBILITY_PRIOR_WEIGHT)
        self._alpha, self._t = {}, {}
        for h in halos:
            if not h.has_boundness:
                continue
            rows, b, _ = h.get_boundness()
            b = b.astype(np.float64)
            t = -np.log(np.clip(h.energy_fraction(b), _U_MIN, 1.0))
            log_mix = errani_log_ratio(h, b)
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
        ``n_eff < PLAUSIBILITY_MIN_MEMBERS`` the previous histogram is kept.

        Parameters
        ----------
        resp_map : dict of {int: (ndarray, ndarray)}
            ``Sub_tree_id -> (sorted particle rows, responsibilities)``.
        weights : ndarray of shape (n_particles,), optional
            Particle weights (masses under ``mass_weighting``).
        """
        t, w = self._member_depths(resp_map, weights)
        if t is None:
            return
        n_eff = w.sum() ** 2 / np.dot(w, w)
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


_LOG_2PI = np.log(2.0 * np.pi)
_FD_STEP = 1e-3                                   # central difference step in ln r


def _normal_log_pdf(t, mu, sd):
    """Log density of ``Normal(mu, sd)`` at ``t``."""
    return -0.5 * ((t - mu) / sd) ** 2 - np.log(sd) - 0.5 * _LOG_2PI


def literature_depth(halo):
    """Literature member model of a halo in ``t = -ln w``, and its width floor.

    Circular orbits at ``r_1/2 = KRAVTSOV_RHALF R_vir`` and at
    ``r_1/2 * 10^(±s)`` for ``s`` = ``PLAUSIBILITY_LIT_SCATTER_DEX`` and
    ``PLAUSIBILITY_MIN_SCATTER_DEX``: ``v_c^2 = dPhi / d ln r`` (central
    difference in ``ln r``, so independent of the comoving/physical
    scaling), ``E = Phi + v_c^2 / 2``. One potential call and one
    phase-space call per halo.

    Parameters
    ----------
    halo : HaloModel

    Returns
    -------
    mu : float
        Depth of the ``r_1/2`` circular orbit.
    sd : float
        Half the depth range over ``± PLAUSIBILITY_LIT_SCATTER_DEX`` dex.
    sd_min : float
        Half the depth range over ``± PLAUSIBILITY_MIN_SCATTER_DEX`` dex.
    """
    s1, s2 = PLAUSIBILITY_LIT_SCATTER_DEX, PLAUSIBILITY_MIN_SCATTER_DEX
    r = KRAVTSOV_RHALF * halo.virial_radius * 10.0 ** np.array([0.0, -s1, s1, -s2, s2])
    steps = np.exp(np.array([-_FD_STEP, 0.0, _FD_STEP]))
    phi = halo.potential((r[:, None] * steps).ravel()).reshape(r.size, 3)
    v2 = (phi[:, 2] - phi[:, 0]) / (2.0 * _FD_STEP)
    b = -(phi[:, 1] + 0.5 * v2) / halo.binding_energy_scale()
    t = -halo.log_phase_space_fraction(b)
    return float(t[0]), 0.5 * abs(t[1] - t[2]), 0.5 * abs(t[3] - t[4])


class PhaseSpacePlausibility(_Plausibility):
    """Log-normal member model on the phase-space fraction ``w`` (any potential).

    ``t = -ln w`` is ``Exp(1)`` for a uniform phase-space background, so
    ``alpha = p*(t) e^t``. ``p*`` mixes the pooled empirical
    ``Normal(mu*, sigma*)`` of the previous snapshot's members (weight
    ``1 - lam``) with each halo's literature Normal (weight ``lam``), and is
    floored like the energy model.
    """

    name = "phase"

    def __init__(self):
        super().__init__()
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._mu: float | None = None
        self._sd: float | None = None
        self._sd_min = 0.0

    def prepare(self, halos, coords=None):
        """Compute α for every bound pair of this snapshot.

        Also records the snapshot's width floor: the median over halos of
        the ``PLAUSIBILITY_MIN_SCATTER_DEX`` depth spread (so the floor
        follows the potential family).

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            Halos with boundness already computed (any potential).
        coords : ndarray of shape (n_particles, 6), optional
            Unused here.
        """
        lam = PLAUSIBILITY_PRIOR_WEIGHT / (1.0 + PLAUSIBILITY_PRIOR_WEIGHT)
        log_keep, log_floor = np.log1p(-PLAUSIBILITY_FLOOR), np.log(PLAUSIBILITY_FLOOR)
        self._alpha, self._t = {}, {}
        floors = []
        for h in halos:
            if not h.has_boundness:
                continue
            rows, b, _ = h.get_boundness()
            t = -h.log_phase_space_fraction(b.astype(np.float64))
            mu_l, sd_l, sd_min = literature_depth(h)
            floors.append(sd_min)
            log_mix = _normal_log_pdf(t, mu_l, sd_l)
            if self._mu is not None:
                log_mix = np.logaddexp(np.log1p(-lam) + _normal_log_pdf(t, self._mu, self._sd),
                                       np.log(lam) + log_mix)
            log_alpha = np.logaddexp(log_keep + log_mix + t, log_floor)
            sid = int(h.sub_tree_id)
            self._alpha[sid] = np.exp(np.minimum(log_alpha, _LOG_ALPHA_MAX))
            self._t[sid] = (rows, t)
        if floors:
            self._sd_min = float(np.median(floors))

    def update(self, resp_map, weights=None):
        """Refit ``(mu*, sigma*)`` as weighted moments of this snapshot's member depths.

        ``sigma*`` is floored at this snapshot's width floor (see
        :meth:`prepare`). With fewer than ``PLAUSIBILITY_MIN_MEMBERS``
        effective members the previous fit is kept.

        Parameters
        ----------
        resp_map : dict of {int: (ndarray, ndarray)}
            ``Sub_tree_id -> (sorted particle rows, responsibilities)``.
        weights : ndarray of shape (n_particles,), optional
            Particle weights (masses under ``mass_weighting``).
        """
        t, w = self._member_depths(resp_map, weights)
        if t is None:
            return
        mu = float(np.average(t, weights=w))
        sd = float(np.sqrt(np.average((t - mu) ** 2, weights=w)))
        self._mu, self._sd = mu, max(sd, self._sd_min)

    @property
    def state(self):
        """``{"mu", "sd"}``, or ``None`` before the first refit."""
        return None if self._mu is None else {"mu": self._mu, "sd": self._sd}

    def set_state(self, state):
        """Restore ``(mu*, sigma*)`` (``None``: literature model alone).

        Parameters
        ----------
        state : dict or None
        """
        self._mu = None if state is None else float(state["mu"])
        self._sd = None if state is None else float(state["sd"])


_LOG_2_3 = np.log(2.0 / 3.0)


def escape_speed_fraction(halo, rows, coords):
    """``u^2 = |v - v_k|^2 / (-2 Phi_k(r))`` of a halo's bound particles, in ``[0, 1]``.

    Same centre, frame (tree bulk velocity) and ``(1 + z)`` scaling as
    :func:`compute_halo_bound_particles`: ``E < 0`` there is ``u^2 < 1`` here
    (float rounding clipped). ``nu = u^3`` is ``U(0, 1)`` for a uniform
    phase-space background at fixed radius.

    Parameters
    ----------
    halo : HaloModel
    rows : ndarray of int
        Bound particle rows.
    coords : ndarray of shape (n_particles, 6)
        Positions and velocities.

    Returns
    -------
    u2 : ndarray
    """
    x = np.asarray(coords[rows], dtype=np.float64)
    v2 = np.sum((x[:, 3:6] - halo.velocity) ** 2, axis=1)
    return np.minimum(v2 / (-2.0 * halo.potential(x[:, :3])), 1.0)


def literature_speed(halo):
    """Literature member ``<u^2> = v_c^2 / v_esc^2`` at ``r_1/2 = KRAVTSOV_RHALF R_vir``.

    Wolf et al. (2010): ``<v^2> = v_c^2(r_1/2)`` for any isotropic equilibrium
    tracer, hot or cold; it matches Errani's NFW tracer to 5% for c = 3-30, and
    Kepler gives 1/2. ``v_c^2 = dPhi / d ln r`` by central difference.

    Parameters
    ----------
    halo : HaloModel

    Returns
    -------
    m : float
    """
    r = KRAVTSOV_RHALF * halo.virial_radius * np.exp(np.array([-_FD_STEP, 0.0, _FD_STEP]))
    phi = halo.potential(r)
    return float((phi[2] - phi[0]) / (2.0 * _FD_STEP) / (-2.0 * phi[1]))


def _speed_log_ratio(u2, m):
    """``log p*(nu)`` of an isotropic power-law tracer ``f ∝ (-E)^q`` against the uniform escape ball.

    ``u^2 ~ Beta(3/2, q + 1)`` at every radius in any potential, with
    ``q = 3 / (2 m) - 5/2`` (``m = <u^2>``); ``m >= 3/5`` is the background
    itself (``q = 0``, ratio 1). ``p*(nu) = 2 (1 - u^2)^q / (3 B(3/2, q + 1))``.

    Parameters
    ----------
    u2 : ndarray
    m : float

    Returns
    -------
    log_ratio : ndarray
    """
    q = max(1.5 / m - 2.5, 0.0)
    return _LOG_2_3 - betaln(1.5, q + 1.0) + xlog1py(q, -u2)


class KinematicPlausibility(_Plausibility):
    """Co-movement model on ``nu = (|v - v_k| / v_esc,k(r))^3`` (any potential).

    ``nu`` is ``U(0, 1)`` for a uniform phase-space background at the
    particle's position, so the potential there carries no evidence and
    ``alpha = p*(nu)``. Members follow an isotropic equilibrium tracer, hot or
    cold, whose ``<u^2>`` is each halo's previous-snapshot member mean, shrunk
    to the pooled one with ``PLAUSIBILITY_SHRINK_COUNT``; it is mixed with the
    halo's literature model (weight ``lam``) and floored, like the other models.
    """

    name = "kinematic"

    def __init__(self):
        super().__init__()
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._m_pool: float | None = None
        self._m: dict[int, float] = {}

    def prepare(self, halos, coords):
        """Compute α for every bound pair of this snapshot.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            Halos with boundness already computed (any potential).
        coords : ndarray of shape (n_particles, 6)
            Positions and velocities.
        """
        lam = PLAUSIBILITY_PRIOR_WEIGHT / (1.0 + PLAUSIBILITY_PRIOR_WEIGHT)
        log_keep, log_floor = np.log1p(-PLAUSIBILITY_FLOOR), np.log(PLAUSIBILITY_FLOOR)
        self._alpha, self._t = {}, {}
        for h in halos:
            if not h.has_boundness:
                continue
            rows = h.get_boundness()[0]
            u2 = escape_speed_fraction(h, rows, coords)
            sid = int(h.sub_tree_id)
            log_mix = _speed_log_ratio(u2, literature_speed(h))
            m = self._m.get(sid, self._m_pool)
            if m is not None:
                log_mix = np.logaddexp(np.log1p(-lam) + _speed_log_ratio(u2, m), np.log(lam) + log_mix)
            log_alpha = np.logaddexp(log_keep + log_mix, log_floor)
            self._alpha[sid] = np.exp(np.minimum(log_alpha, _LOG_ALPHA_MAX))
            self._t[sid] = (rows, u2)

    def update(self, resp_map, weights=None):
        """Refit the pooled and per-halo member ``<u^2>``.

        The pool is the weighted mean over all member pairs; each halo's value is
        its own weighted mean shrunk to the pool:
        ``m_k = (n_k m_hat_k + n0 m_pool) / (n_k + n0)``, ``n0 = PLAUSIBILITY_SHRINK_COUNT``.
        Below ``PLAUSIBILITY_MIN_MEMBERS`` pooled effective members the previous
        fit is kept.

        Parameters
        ----------
        resp_map : dict of {int: (ndarray, ndarray)}
            ``Sub_tree_id -> (sorted particle rows, responsibilities)``.
        weights : ndarray of shape (n_particles,), optional
            Particle weights (masses under ``mass_weighting``).
        """
        per = {sid: (u, w) for sid, (u, w) in self._member_pairs(resp_map, weights).items() if w.size}
        if not per:
            return
        w_all = np.concatenate([w for _, w in per.values()])
        if w_all.sum() ** 2 / np.dot(w_all, w_all) < PLAUSIBILITY_MIN_MEMBERS:
            return
        m_pool = float(np.average(np.concatenate([u for u, _ in per.values()]), weights=w_all))
        n0 = PLAUSIBILITY_SHRINK_COUNT
        m = {}
        for sid, (u, w) in per.items():
            n = w.sum() ** 2 / np.dot(w, w)
            m[sid] = float((n * np.average(u, weights=w) + n0 * m_pool) / (n + n0))
        self._m_pool, self._m = m_pool, m

    @property
    def state(self):
        """``{"m_pool", "sid", "m"}``, or ``None`` before the first refit."""
        if self._m_pool is None:
            return None
        sid = np.array(sorted(self._m), dtype=np.int64)
        return {"m_pool": self._m_pool, "sid": sid, "m": np.array([self._m[s] for s in sid.tolist()])}

    def set_state(self, state):
        """Restore the member model (``None``: literature model alone).

        Parameters
        ----------
        state : dict or None
        """
        self._m_pool = None if state is None else float(state["m_pool"])
        self._m = {} if state is None else dict(zip(np.asarray(state["sid"]).tolist(),
                                                     np.asarray(state["m"], dtype=float).tolist()))


_MODELS = {"rank": RankPlausibility, "energy": EnergyPlausibility, "phase": PhaseSpacePlausibility,
           "kinematic": KinematicPlausibility}


def make_plausibility(name):
    """Instantiate a plausibility model by name.

    Parameters
    ----------
    name : str
        ``"rank"``, ``"energy"``, ``"phase"`` or ``"kinematic"``.

    Returns
    -------
    model : RankPlausibility, EnergyPlausibility, PhaseSpacePlausibility or KinematicPlausibility
    """
    try:
        return _MODELS[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown plausibility {name!r}. Choose from: {list(_MODELS)}")

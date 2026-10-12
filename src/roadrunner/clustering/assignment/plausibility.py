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
"``n`` belongs to halo ``k``'s galaxy" (model B, soft-label evidence): the halo's
evidence, next to the galaxy's phase-space model the fits learn. Each model is a
ratio ``p*(s) / p_0(s)`` of a statistic ``s`` of the pair: members over interlopers,
spread uniformly in the halo's bound phase space (a smooth background, by
Liouville). Only bound pairs carry a positive value. Every model reads the physical energies
from :meth:`HaloModel.get_bound` (never the scaled boundness). Every model is
computed per halo, before the groups are fit, and exposes the same interface:

- ``prepare(halos, coords)``: α of every bound pair of this snapshot;
- ``column_values(column_id)``: α of the columns of a group's matrix;
- ``update(resp_map, weights)``: refit after the snapshot (between
  snapshots, never inside EM);
- ``state`` / ``set_state``: checkpoint round trip.

Models:

- ``"rank"``: ``log1p`` of the ordinal rank of the boundness within each
  halo.
- ``"energy"`` (any potential): the ratio ``p*(w)`` on the fraction ``w`` of the
  halo's bound phase space inside the search sphere more bound than the
  particle (uniform for the background, so ``t = -ln w`` is ``Exp(1)``).
  ``p*`` is a histogram of the previous snapshot's members in ``t``, mixed with
  the literature galaxy's ``P* = dN*/dw`` (:func:`tracer_log_ratio`), and floored:
  ``alpha = (1 - kappa) [(1 - lam) h(t) e^t + lam P*] + kappa``,
  ``lam = A / (1 + A)``. Without a histogram yet (first snapshot) ``lam = 1``.
- ``"phase"`` (any potential): the same statistic and literature galaxy, with a
  log-normal empirical member model ``t ~ Normal(mu*, sigma*)``: the pooled
  moments of the previous snapshot's members, mixed with ``P*`` and floored.
- ``"kinematic"`` (any potential): co-movement. ``u^2 = |v - v_k|^2 / v_esc,k^2(r)``
  at the particle's position, so ``nu = u^3`` is ``U(0, 1)`` for the background
  there and the potential at that position carries no evidence. Members follow
  an isotropic equilibrium tracer ``f ∝ (-E)^q`` (``u^2 ~ Beta(3/2, q + 1)``, hot
  or cold), with each halo's ``<u^2>`` from its previous-snapshot members shrunk
  to the pooled value, mixed with the literature galaxy's ``<u^2>``
  (:func:`literature_speed`), and floored.

The literature galaxy, one for every model: an isotropic Hernquist tracer of
half-mass radius ``r_1/2`` in equilibrium in the halo's whole potential, about
the bottom of its well, counted inside the search sphere (a density only, not
part of the potential).

``r_1/2``, the stellar half-mass radius every literature model reads, is chosen by
``literature_rhalf`` (:data:`LITERATURE_RHALF`): ``"kravtsov"``, ``KRAVTSOV_RHALF R_vir``
(``0.015 R_vir``, Kravtsov 2013); ``"stars"``, the half-mass radius of the halo's ``stars``
group (Kravtsov's for a halo without one).

The literature galaxy (its ``dN*/dw`` and ``<u^2>``) and the phase model's width floor read the
halo's spherical profile about the bottom of its well, the one the phase-space tables read; only
the kinematic ``u^2`` reads the potential at each particle's position.
"""

import numpy as np
from scipy.special import betaln, xlog1py
from scipy.stats import rankdata

from roadrunner._defaults import (
    math_dtype,
    PLAUSIBILITY_FLOOR,
    PLAUSIBILITY_MAX_BINS,
    PLAUSIBILITY_MIN_BINS,
    PLAUSIBILITY_MIN_MEMBERS,
    PLAUSIBILITY_MIN_SCATTER_DEX,
    PLAUSIBILITY_PRIOR_WEIGHT,
    PLAUSIBILITY_SHRINK_COUNT,
)
from roadrunner.physics.potentials import CompositeSphericalPotential, HernquistPotential

KRAVTSOV_RHALF = 0.015      # stellar half-mass radius / virial radius
_HALF_MASS_GRID = np.geomspace(1e-3, 1e3, 2401)   # radii of the stars' half-mass search, in their largest scale radius
_LOG_ALPHA_MAX = 50.0       # alpha in [1e-8, e^50]: row sums finite and row ratios
                            # (>= 1e-8 / (K e^50) ~ 1e-30) normal in float32
_EMPTY = np.empty(0, dtype=np.float64)

_LAM = PLAUSIBILITY_PRIOR_WEIGHT / (1.0 + PLAUSIBILITY_PRIOR_WEIGHT)   # literature weight
_LOG_KEEP, _LOG_FLOOR = np.log1p(-PLAUSIBILITY_FLOOR), np.log(PLAUSIBILITY_FLOOR)


def _log1p_rank(values):
    """``log1p`` of the ordinal ranks of ``values`` (float64; empty stays empty).

    Parameters
    ----------
    values : ndarray
        Values of one halo, ranked ascending (``-E``: the most bound ranks highest).

    Returns
    -------
    alpha : ndarray of float64
    """
    if len(values) == 0:
        return np.asarray(values, dtype=np.float64)
    return np.log1p(rankdata(values, method="ordinal"))


def _mixed_alpha(log_lit, log_emp):
    """``alpha = (1 - kappa) [(1 - lam) R_emp + lam R_lit] + kappa`` from log ratios to the null (float64).

    ``log_emp`` is ``None`` before the first refit: ``alpha = (1 - kappa) R_lit + kappa``.
    """
    log_mix = log_lit if log_emp is None else np.logaddexp(np.log1p(-_LAM) + log_emp, np.log(_LAM) + log_lit)
    return np.exp(np.minimum(np.logaddexp(_LOG_KEEP + log_mix, _LOG_FLOOR), _LOG_ALPHA_MAX))


def kravtsov_half_mass_radius(halo):
    """Kravtsov (2013)'s stellar half-mass radius ``KRAVTSOV_RHALF R_vir`` (catalogue units).

    Parameters
    ----------
    halo : HaloModel

    Returns
    -------
    r_half : float
    """
    return KRAVTSOV_RHALF * halo.virial_radius


def stars_half_mass_radius(halo):
    """Half-mass radius of the halo's ``stars`` group (catalogue units); Kravtsov's (:func:`kravtsov_half_mass_radius`) without one.

    The group as a composite of its own, about its own bottom of the well: the radius where its
    enclosed mass is half its mass, ``M(<r) = M / 2``, linear between the radii ``_HALF_MASS_GRID``
    times its components' largest scale radius (a Shell's is its half-mass radius, a Hernquist's or
    a Plummer's ``a``); physical, made catalogue with ``length_scale``.

    Parameters
    ----------
    halo : HaloModel

    Returns
    -------
    r_half : float
    """
    stars = halo.potential_model.groups.get("stars")
    if stars is None:
        return kravtsov_half_mass_radius(halo)
    group = CompositeSphericalPotential(stars=stars)
    r = max(p._scale_radius for p in stars) * _HALF_MASS_GRID
    return float(np.interp(0.5 * group.M, group._enclosed_mass(r), r)) / halo.length_scale


LITERATURE_RHALF = {"kravtsov": kravtsov_half_mass_radius, "stars": stars_half_mass_radius}   # literature_rhalf -> r_1/2 of a halo


def _tracer(halo, r_half):
    """The literature galaxy's density: a Hernquist model of half-mass radius ``r_1/2`` (``a = r_1/2 / (1 + √2)``; physical; its mass drops out)."""
    return HernquistPotential(1.0, r_half * halo.length_scale / (1.0 + np.sqrt(2.0)))


def tracer_log_ratio(halo, E, r_half):
    """Log literature ratio ``log P* = ln dN*/dw`` of the literature galaxy over a uniform phase-space background.

    ``N*`` is the energy distribution of a Hernquist density of half-mass radius ``r_1/2``
    (:func:`_tracer`) about the bottom of the well, in the halo's whole potential (Eddington's
    two-model ``f*``, negative values clipped with a warning): a density only, not part of the
    potential, so the boundness is untouched. Per unit fraction ``w`` of the bound phase space inside
    the search sphere, ``dN*/dw = f* q(0) / ∫ f* g dE``: ``f*`` over its mean there, the ratio to
    interlopers spread uniformly in that phase space (``w ~ U(0, 1]``), with ``∫ P* dw = 1``.

    Parameters
    ----------
    halo : HaloModel
    E : ndarray of float64
        Physical specific orbital energies.
    r_half : float
        Stellar half-mass radius (catalogue units, :data:`LITERATURE_RHALF`).

    Returns
    -------
    log_ratio : ndarray of float64
    """
    return halo.log_phase_space_density(E, density=_tracer(halo, r_half))


class _Plausibility:
    """Shared interface: per-halo α from a per-pair statistic, mixed and floored alike for every model; a no-op refit and no state.

    Parameters
    ----------
    literature_rhalf : {"kravtsov", "stars"}, default="kravtsov"
        The stellar half-mass radius ``r_1/2`` its literature prior reads (:data:`LITERATURE_RHALF`);
        unused by ``"rank"``.
    """

    name = ""

    def __init__(self, literature_rhalf="kravtsov"):
        self._half_mass_radius = LITERATURE_RHALF[literature_rhalf]
        self._alpha: dict[int, np.ndarray] = {}
        self._t: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def prepare(self, halos, coords):
        """Compute α for every bound pair of this snapshot.

        Each model reads the physical energies from :meth:`HaloModel.get_bound`
        (never the scaled boundness), so α does not depend on the energy scale.

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            Halos with boundness already computed.
        coords : ndarray of shape (n_particles, 6) or None
            Phase-space coordinates (catalogue units, km/s); only models that read them need them.
        """
        self._alpha, self._t = {}, {}
        for h in halos:
            if not h.has_bound:
                continue
            rows, E, _ = h.get_bound()
            stat, log_lit, log_emp = self._log_ratios(h, rows, E.astype(np.float64), coords)
            sid = int(h.sub_tree_id)
            self._alpha[sid] = _mixed_alpha(log_lit, log_emp)
            self._t[sid] = (rows, stat)
            h.release_tables()                      # U6: tables read above, once per halo; drop them now

    def _log_ratios(self, halo, rows, E, coords):
        """Per-pair statistic and its log ratios to the null: literature, empirical (``None`` before a refit).

        Parameters
        ----------
        halo : HaloModel
        rows : ndarray of int
            Bound particle rows.
        E : ndarray of float64
            Their physical energies.
        coords : ndarray of shape (n_particles, 6) or None
            The snapshot's phase-space coordinates.

        Returns
        -------
        stat, log_lit, log_emp : ndarray, ndarray, ndarray or None
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
        per = self._member_pairs(resp_map, weights)
        t = np.concatenate([t for t, _ in per.values()]) if per else _EMPTY
        w = np.concatenate([w for _, w in per.values()]) if per else _EMPTY
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

    def prepare(self, halos, coords):
        """Compute α for every bound pair of this snapshot (``coords`` unused)."""
        self._alpha = {int(h.sub_tree_id): _log1p_rank(-h.get_bound()[1])      # -E: same order as b
                       for h in halos if h.has_bound}


class EnergyPlausibility(_Plausibility):
    """Hybrid empirical ``p*(w)``: member histogram in ``t = -ln w`` + the tracer prior (:func:`tracer_log_ratio`), floored."""

    name = "energy"

    def __init__(self, literature_rhalf="kravtsov"):
        super().__init__(literature_rhalf)
        self._edges: np.ndarray | None = None
        self._log_density: np.ndarray | None = None

    def _log_pstar(self, t):
        """Log histogram ratio ``log h(t) + t``, held at the end bins beyond the data."""
        tc = np.clip(t, self._edges[0], self._edges[-1])
        b = np.clip(np.searchsorted(self._edges, tc, side="right") - 1, 0, self._log_density.size - 1)
        return self._log_density[b] + tc

    def _log_ratios(self, halo, rows, E, coords):
        """``t = -ln w``; literature: the tracer's ratio at ``r_1/2`` (:func:`tracer_log_ratio`); empirical: the member histogram's."""
        t = -halo.log_phase_space_fraction(E)
        log_emp = None if self._edges is None else self._log_pstar(t)
        return t, tracer_log_ratio(halo, E, self._half_mass_radius(halo)), log_emp

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
        """Restore the histogram (``None``: start from the tracer prior alone).

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


def _summary_potential(halo, r):
    """Potential (float64, physical) at catalogue radii ``r`` about the bottom of the well: the radial summaries' input.

    The halo's spherical profile about ``x_min`` (``potential_model._potential``: each component along
    the pathway its composite fixed, its own profile (concentric) or its shell average about ``x_min``
    (monopole)), the one its energy and phase tables read; for a single potential it is the true potential along any ray.

    Parameters
    ----------
    halo : HaloModel
    r : ndarray of float64
        Radii in catalogue units, made physical with ``length_scale``.

    Returns
    -------
    phi : ndarray of float64, shape of ``r``
    """
    return halo.potential_model._potential(r * halo.length_scale)


def _normal_log_pdf(t, mu, sd):
    """Log density of ``Normal(mu, sd)`` at ``t``."""
    return -0.5 * ((t - mu) / sd) ** 2 - np.log(sd) - 0.5 * _LOG_2PI


def size_scatter_depth(halo, r_half):
    """Width floor of the members' depth ``t = -ln w``: Kravtsov's galaxy-size scatter mapped to depth.

    Half the depth range of the circular orbits at ``r_1/2 * 10^(±s)``, ``s`` =
    ``PLAUSIBILITY_MIN_SCATTER_DEX``, about the bottom of the well in the halo's spherical profile
    (:func:`_summary_potential`): ``v_c^2 = dPhi / d ln r`` (central difference in ``ln r``),
    ``E = Phi + v_c^2 / 2``. How far the members' depth moves when the galaxy's size scatters
    at fixed halo. One potential call and one phase-space call per halo.

    Parameters
    ----------
    halo : HaloModel
    r_half : float
        Stellar half-mass radius (catalogue units, :data:`LITERATURE_RHALF`).

    Returns
    -------
    sd_min : float
    """
    r = r_half * 10.0 ** np.array([-PLAUSIBILITY_MIN_SCATTER_DEX, PLAUSIBILITY_MIN_SCATTER_DEX])
    steps = np.exp(np.array([-_FD_STEP, 0.0, _FD_STEP]))
    phi = _summary_potential(halo, (r[:, None] * steps).ravel()).reshape(r.size, 3)
    v2 = (phi[:, 2] - phi[:, 0]) / (2.0 * _FD_STEP)
    t = -halo.log_phase_space_fraction(phi[:, 1] + 0.5 * v2)
    return 0.5 * abs(t[0] - t[1])


class PhaseSpacePlausibility(_Plausibility):
    """Log-normal member model on the phase-space fraction ``w`` (any potential).

    ``t = -ln w`` is ``Exp(1)`` for a uniform phase-space background, so
    ``alpha = p*(t) e^t``. ``p*`` mixes the pooled empirical
    ``Normal(mu*, sigma*)`` of the previous snapshot's members (weight
    ``1 - lam``) with the literature galaxy's ``P*`` (:func:`tracer_log_ratio`,
    weight ``lam``), and is floored like the energy model.
    """

    name = "phase"

    def __init__(self, literature_rhalf="kravtsov"):
        super().__init__(literature_rhalf)
        self._mu: float | None = None
        self._sd: float | None = None
        self._sd_min = 0.0
        self._floors: list[float] = []

    def prepare(self, halos, coords):
        """Compute α; also records this snapshot's width floor (median over halos of
        :func:`size_scatter_depth`, so the floor follows the potential family).

        Parameters
        ----------
        halos : HaloEnsemble or iterable of HaloModel
            Halos with boundness already computed (any potential).
        coords : ndarray of shape (n_particles, 6) or None
            Unused here.
        """
        self._floors = []
        super().prepare(halos, coords)
        if self._floors:
            self._sd_min = float(np.median(self._floors))

    def _log_ratios(self, halo, rows, E, coords):
        t = -halo.log_phase_space_fraction(E)
        r_half = self._half_mass_radius(halo)
        self._floors.append(size_scatter_depth(halo, r_half))
        log_emp = None if self._mu is None else _normal_log_pdf(t, self._mu, self._sd) + t
        return t, tracer_log_ratio(halo, E, r_half), log_emp

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
_JEANS_NODES = 400                                # radii of the literature <u^2>, inside and outside the search sphere


def escape_speed_fraction(halo, rows, coords):
    """``u^2 = |v - v_k|^2 / (-2 Phi_k(r))`` of a halo's bound particles, in ``[0, 1]``.

    Same centre, frame (the halo's centre-of-mass velocity) and ``(1 + z)`` scaling as
    :func:`compute_halo_bound_particles`: ``E < 0`` there is ``u^2 < 1`` here
    (float rounding clipped). ``nu = u^3`` is ``U(0, 1)`` for a uniform
    phase-space background at fixed radius. ``Phi_k`` is the halo's true potential at the
    particle's position, as in the boundness, not the spherical profile of the literature
    summaries.

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
    v2 = np.sum((x[:, 3:6] - halo.com_velocity) ** 2, axis=1)
    return np.minimum(v2 / (-2.0 * halo.potential_model.potential(x[:, :3] * halo.length_scale)), 1.0)


def _trapezoid(y, x):
    """Trapezoid integral of ``y`` over ``x``."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def literature_speed(halo, r_half):
    """Literature member ``<u^2>``: the literature galaxy's mean ``v^2 / v_esc^2`` inside the search sphere.

    The tracer of :func:`tracer_log_ratio` (isotropic, in equilibrium in the halo's spherical
    profile about the bottom of its well, :func:`_summary_potential`). Jeans' equation,
    ``rho sigma^2 (r) = ∫_r^∞ rho v_c^2 d ln r'`` with ``v_c^2 = dPhi / d ln r`` (central
    difference), gives ``<v^2>(r) = 3 sigma^2``, so
    ``<u^2> = ∫ 3 rho sigma^2 / (-2 Phi) r^3 d ln r / ∫ rho r^3 d ln r`` over ``r < r_max``:
    trapezoids in ``ln r`` from ``1e-4 r_1/2``, the Jeans integral out to ``1e3`` times the larger of
    ``r_max`` and ``r_1/2``. One potential call per halo.

    Parameters
    ----------
    halo : HaloModel
    r_half : float
        Stellar half-mass radius (catalogue units, :data:`LITERATURE_RHALF`).

    Returns
    -------
    m : float
    """
    r_max = halo.search_factor * halo.virial_radius
    r = np.stack((np.geomspace(1e-4 * r_half, r_max, _JEANS_NODES),                  # inside the search sphere
                  np.geomspace(r_max, 1e3 * max(r_max, r_half), _JEANS_NODES)))     # outside it
    steps = np.exp(np.array([-_FD_STEP, 0.0, _FD_STEP]))
    phi = _summary_potential(halo, (r[..., None] * steps).ravel()).reshape(*r.shape, 3)
    rho = np.asarray(_tracer(halo, r_half)._density(r * halo.length_scale), dtype=np.float64)
    ln_r = np.log(r)
    jeans = rho * (phi[..., 2] - phi[..., 0]) / (2.0 * _FD_STEP)                    # rho v_c^2
    seg = 0.5 * (jeans[:, 1:] + jeans[:, :-1]) * np.diff(ln_r, axis=-1)
    pressure = np.concatenate((np.cumsum(seg[0, ::-1])[::-1], [0.0])) + seg[1].sum()   # rho sigma^2 inside: ∫_r^∞ rho v_c^2 d ln r
    return (_trapezoid(3.0 * pressure * r[0]**3 / (-2.0 * phi[0, :, 1]), ln_r[0])
            / _trapezoid(rho[0] * r[0]**3, ln_r[0]))


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

    def __init__(self, literature_rhalf="kravtsov"):
        super().__init__(literature_rhalf)
        self._m_pool: float | None = None
        self._m: dict[int, float] = {}

    def _log_ratios(self, halo, rows, E, coords):
        u2 = escape_speed_fraction(halo, rows, coords)
        m = self._m.get(int(halo.sub_tree_id), self._m_pool)
        return u2, _speed_log_ratio(u2, literature_speed(halo, self._half_mass_radius(halo))), None if m is None else _speed_log_ratio(u2, m)

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


PLAUSIBILITY_MODELS = {"rank": RankPlausibility, "energy": EnergyPlausibility, "phase": PhaseSpacePlausibility,
                        "kinematic": KinematicPlausibility}


def make_plausibility(name, literature_rhalf="kravtsov"):
    """Instantiate a plausibility model by name.

    Parameters
    ----------
    name : str
        ``"rank"``, ``"energy"``, ``"phase"`` or ``"kinematic"``.
    literature_rhalf : str, default="kravtsov"
        ``r_1/2`` of the literature priors (:data:`LITERATURE_RHALF`): ``"kravtsov"`` or ``"stars"``.

    Returns
    -------
    model : RankPlausibility, EnergyPlausibility, PhaseSpacePlausibility or KinematicPlausibility
    """
    try:
        model = PLAUSIBILITY_MODELS[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown plausibility {name!r}. Choose from: {list(PLAUSIBILITY_MODELS)}")
    if literature_rhalf not in LITERATURE_RHALF:
        raise ValueError(f"Unknown literature_rhalf {literature_rhalf!r}. Choose from: {list(LITERATURE_RHALF)}")
    return model(literature_rhalf)

#############################################################################
#
# package:   roadrunner.physics
# file:      distribution.py
# brief:     Energy distribution and phase-space volume of spherical potentials.
#
# copyright: GPLv3
# author:    Asier Lambarri Martinez
#
#############################################################################

"""Energy distribution and phase-space volume of spherical potentials.

Notation: ``energy`` (``E``) is always the physical specific orbital energy
``Φ + v²/2``, negative for bound particles. Scaled energies have their own
names: ``eps = E / Φ₀`` and the binding depth ``cal = 1 - E / Φ₀`` (0 at the
centre, 1 at ``E = 0``).

For a spherical, isotropic system with potential ``Φ(r)`` and density
``ρ(r)``, both untruncated, counted inside a radius ``R`` (in the pipeline
the boundness search sphere, ``search_factor`` virial radii):

- phase-space volume more bound than ``E``:
  ``q(E) = (16π²/3) ∫_0^{min(R, r_E)} r² [2 (E - Φ)]^{3/2} dr``, with
  ``Φ(r_E) = E`` the apocentre of a radial orbit;
- density of states ``g(E) = dq/dE = 16π² ∫_0^{min(R, r_E)} r² sqrt(2 (E - Φ)) dr``;
- Eddington distribution function of the untruncated profile
  ``f(E) = (1 / (√8 π²)) ∫_0^{ψ_E} (d²ρ/dψ²) dψ / sqrt(ψ_E - ψ)``,
  ``ψ = -Φ``, ``ψ_E = -E`` (no boundary term: ``ρ -> 0`` as ``ψ -> 0``);
- energy distribution of the mass inside ``R``: ``N(E) = f(E) g(E)``,
  normalised to 1 over the bound energies, so ``∫ N dE`` is the fraction of
  that mass more bound than ``E``;
- phase-space fraction ``w = q(E) / q(0)``: uniform on ``(0, 1]`` for
  particles spread uniformly in the bound phase space inside ``R``.

``f`` describes the equilibrium profile, which extends beyond ``R``; ``g``
and ``q`` count only what lies inside ``R``, the volume the pipeline looks
at. :class:`SphericalDistribution` computes all of them for any potential
from its primitives (potential, enclosed mass, density); potentials
override them where exact forms exist. Self-similar profiles (Plummer,
Hernquist) share one set of such tables per process, with rows over
``c = R / a`` (:func:`self_similar_tables`).

The NFW tables below are those exact forms for NFW, in scaled units
``G = Rs = 1``, ``phi(x) = -ln(1 + x) / x``. They take ``eps``, their
truncation radius ``c`` is ``R / Rs``, and they are built once per process,
one row per ``c`` of a grid. Each row has its own depth nodes, with the
depth where the radial apocentre reaches ``c`` as a node: ``g`` and ``q``
have a kink there, and rows are blended kink to kink.

Two-model inversion: :func:`eddington_distribution_function` takes a
``background`` (its ψ and dψ/dr set the equilibrium) and, optionally, a
``density`` in another potential's units (its own when ``density`` is
``None``, the self-consistent case above). The resulting ``f`` need not be
non-negative — a density without an isotropic equilibrium in that
background can want negative phase-space density at some energies; those
values are clipped to 0 with a warning, since no such orbit exists.

Precision: everything here is float64, inputs read as float64 and outputs
returned in it. Finite differences of ``ρ`` and deep quadratures need it,
and so do the plausibility models that exponentiate these quantities: a
float32 cast here shifts their float32 latent prior by tens of ulp. Those
models cast their own output to ``math_dtype()``. Table queries
(:func:`table_query`), the Eddington quadrature and, for potentials with a
``profile()`` (:func:`profile_phi`), the q and g quadratures are compiled
with numba and parallel.
"""

import functools
import warnings
from typing import NamedTuple

import numpy as np
from numba import njit, prange


C_GRID = np.geomspace(1.0, 100.0, 48)
_DEEPEST = 1e-6                                 # deepest node of every table: calE = 1e-6
_KINK_GAP = 1e-3                                # first node past the kink, as a fraction of the way to E = 0
_NFW_NODES = 400, 370                           # nodes deeper and shallower than the kink (see _depth_nodes)
_X = np.geomspace(1e-7, 1e10, 34000)           # radial grid of the inversion
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(96)   # Gauss–Legendre on [-1, 1]: the quadratures here and composite shell averages
_W, _WW = 0.5 * (GL_NODES + 1.0), 0.5 * GL_WEIGHTS  # Gauss-Legendre on [0, 1]
_TINY = 2.2250738585072014e-308                 # smallest normal float64
PROFILE_PLUMMER, PROFILE_HERNQUIST, PROFILE_NFW, PROFILE_SHELL = 0.0, 1.0, 2.0, 3.0   # kinds of profile rows (see profile_phi)


@njit(cache=True)
def _interp1(x, xp, fp):
    """``np.interp`` at one ``x``: linear between the increasing nodes ``xp``, held at the end nodes (NaN stays NaN)."""
    n = xp.size
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    return fp[lo] + (x - xp[lo]) / (xp[hi] - xp[lo]) * (fp[hi] - fp[lo])


@njit(parallel=True, cache=True)
def _table_query_kernel(eps, nodes, row, slope):
    """:func:`table_query` on a flat ``eps`` (parallel)."""
    out = np.empty(eps.size)
    last = nodes[nodes.size - 1]
    cal_last = 1.0 - last
    deep = slope == slope                                   # a finite slope switches the deep extension on
    for i in prange(eps.size):
        e = eps[i]
        if deep and e > last:
            out[i] = row[row.size - 1] + slope * np.log(max(1.0 - e, _TINY) / cal_last)
        else:
            out[i] = _interp1(e, nodes, row)
    return out


def table_query(eps, nodes, row, deep_slope=None):
    """Values of a table at scaled energies ``eps = E / Φ₀`` (compiled, parallel over ``eps``).

    Linear in ``eps`` between the increasing ``nodes``, held at the end nodes. With
    ``deep_slope``, energies deeper than the deepest node follow the power law
    ``row[-1] + deep_slope ln(calE / calE_last)`` instead (``calE = 1 - eps``, floored at the
    smallest normal float so energies at or below ``Φ₀`` stay finite).

    Parameters
    ----------
    eps : ndarray
    nodes, row : ndarray of shape (n,)
    deep_slope : float, optional

    Returns
    -------
    values : ndarray of float64, shape of ``eps``
    """
    eps = np.asarray(eps, dtype=np.float64)
    slope = np.nan if deep_slope is None else float(deep_slope)
    out = _table_query_kernel(np.ascontiguousarray(eps).ravel(), np.ascontiguousarray(nodes, dtype=np.float64),
                              np.ascontiguousarray(row, dtype=np.float64), slope)
    return out.reshape(eps.shape)


def _node_slope(nodes, row):
    """Slope of ``row`` against ``ln calE`` through the two deepest nodes (``calE = 1 - nodes``)."""
    cal = 1.0 - nodes[-2:]
    return (row[-1] - row[-2]) / np.log(cal[1] / cal[0])


@njit(cache=True)
def _cell(r, c, inv_d, K):
    """Cell of radius ``r`` in a grid of ``K`` log-radius cells starting at ``ln r = c``.

    Cell 0 lies inside the innermost node, cells ``1..K`` between nodes and
    cell ``K + 1`` beyond the outermost node.

    Parameters
    ----------
    r : float
    c : float
        Log radius of the innermost node.
    inv_d : float
        Inverse log-radius width of a cell.
    K : int
        Number of cells between nodes.

    Returns
    -------
    b : int
    """
    x = (np.log(r) - c) * inv_d
    if not x >= 0.0:
        return 0
    if x >= K:
        return K + 1
    return int(x) + 1


@njit(cache=True)
def _row_phi(profile, j, r):
    """Potential at radius ``r`` of profile row ``j`` by its own formula (no offset).

    ``PROFILE_SHELL``: the row's own tabulated cell at ``r`` (see :func:`profile_phi`); the cell
    0 (inside the innermost node) is the constant intercept, avoiding the ``1/r`` term at ``r = 0``.
    """
    kind, k, a = profile[j, 0], profile[j, 1], profile[j, 2]
    if kind == PROFILE_PLUMMER:
        return -k / np.sqrt(r * r + a * a)
    if kind == PROFILE_HERNQUIST:
        return -k / (r + a)
    if kind == PROFILE_SHELL:
        b = _cell(r, profile[j + 1, 0], profile[j + 1, 1], int(k) - 3)
        if b == 0:
            return profile[j + 2, 1]
        return profile[j + 2 + b, 0] / r + profile[j + 2 + b, 1]
    if r == 0.0:
        return -k / a
    return -k * np.log1p(r / a) / r


@njit(cache=True)
def _row_mass(profile, j, r):
    """``G M(<r)`` of profile row ``j`` by its own formula (no offset): ``k`` is ``G M`` (NFW: ``G M / A(c)``).

    ``PROFILE_SHELL``: exact, ``G M(<r) = -`` the row's cell slope (:class:`~roadrunner.physics.potentials.ShellPotential`'s
    ``Φ = slope / r + intercept``, so ``G M = r² dΦ/dr`` is ``-slope`` in each cell, 0 inside the innermost node).
    """
    kind, k, a = profile[j, 0], profile[j, 1], profile[j, 2]
    if kind == PROFILE_PLUMMER:
        return k * r**3 / (r * r + a * a) ** 1.5
    if kind == PROFILE_HERNQUIST:
        return k * r * r / ((r + a) * (r + a))
    if kind == PROFILE_SHELL:
        b = _cell(r, profile[j + 1, 0], profile[j + 1, 1], int(k) - 3)
        return -profile[j + 2 + b, 0]
    x = r / a
    return k * (np.log1p(x) - x / (1.0 + x))


@njit(cache=True)
def _row_rho(kind, k, a, r):
    """``G ρ`` of one profile row, own formula (no offset); infinite at ``r = 0`` in a cusp (Hernquist, NFW)."""
    if kind == PROFILE_PLUMMER:
        return 0.75 / np.pi * k / a**3 * (1.0 + (r / a) ** 2) ** -2.5
    if kind == PROFILE_HERNQUIST:
        return k * a / (2.0 * np.pi * r * (r + a) ** 3)
    return k / (4.0 * np.pi * a * a * r * (1.0 + r / a) ** 2)   # NFW: k = G M / A(c)


@njit(cache=True)
def profile_phi(profile, r):
    """Potential at radius ``r`` of a profile: the sum of its rows ``(kind, k, a, d)``.

    ``PROFILE_PLUMMER``: ``-k / sqrt(r² + a²)``; ``PROFILE_HERNQUIST``: ``-k / (r + a)``;
    ``PROFILE_NFW``: ``-k ln(1 + r/a) / r`` (``-k / a`` at ``r = 0``); ``PROFILE_SHELL``: a
    particle set's tabulated potential, ``k`` the number of table rows that follow (``K + 3``):
    ``(ln r₀, 1/Δ)``, then the ``K + 2`` cells' ``(slope, intercept)`` of ``Φ`` in ``1/r``
    (columns 0–1), skipped by this row's own entry in the loop. ``k`` is ``G M`` for the other
    kinds (NFW: ``G M / A(c)``), ``a`` the scale radius and ``d`` the row's offset from the
    profile's centre: 0 evaluates the row's own formula (:func:`_row_phi`); ``d > 0`` its shell
    average about the centre, ``Σ_k w_k s_k Φ(s_k) / (2 max(r, d))``,
    ``s_k = max(r, d) + min(r, d) x_k`` (Gauss–Legendre on [-1, 1]; the composite's monopole
    pathway). A composite potential stacks its components' rows.
    """
    phi = 0.0
    j = 0
    while j < profile.shape[0]:
        d = profile[j, 3]
        if d > 0.0:
            mid, half = max(r, d), min(r, d)
            acc = 0.0
            for m in range(GL_NODES.size):
                s = mid + half * GL_NODES[m]
                acc += GL_WEIGHTS[m] * s * _row_phi(profile, j, s)
            phi += acc / (2.0 * mid)
        else:
            phi += _row_phi(profile, j, r)
        j += (1 + int(profile[j, 1])) if profile[j, 0] == PROFILE_SHELL else 1
    return phi


@njit(parallel=True, cache=True)
def profile_potential(profile, r):
    """``profile_phi`` at every radius of ``r`` (compiled, parallel).

    Parameters
    ----------
    profile : ndarray of shape (k, 4)
    r : ndarray of shape (n,)

    Returns
    -------
    phi : ndarray of float64, shape (n,)
    """
    phi = np.empty(r.size)
    for i in prange(r.size):
        phi[i] = profile_phi(profile, r[i])
    return phi


@njit(parallel=True, cache=True)
def profile_mass(profile, r):
    """``G M(<r)`` of a profile at every radius of ``r`` (compiled, parallel): the sum of its rows.

    A concentric row (``d = 0``) adds its own :func:`_row_mass` (``PROFILE_SHELL`` exact, see
    there); an offset row (``d > 0``) adds the monopole's enclosed mass,
    :func:`~roadrunner.physics.potentials._shell_mass` on the same Gauss–Legendre nodes as
    :func:`profile_phi`'s shell average. An offset ``PROFILE_SHELL`` row never reaches this
    function: its density has no closed form for the monopole quadrature, so callers
    (:func:`_radial_mass`) route it to the numpy :func:`~roadrunner.physics.potentials._shell_mass`
    instead; its payload is still skipped here.

    Parameters
    ----------
    profile : ndarray of shape (k, 4)
    r : ndarray of shape (n,)

    Returns
    -------
    gm : ndarray of float64, shape (n,)
    """
    gm = np.zeros(r.size)
    for i in prange(r.size):
        ri = r[i]
        j = 0
        while j < profile.shape[0]:
            kind, k, a, d = profile[j, 0], profile[j, 1], profile[j, 2], profile[j, 3]
            if kind == PROFILE_SHELL:
                if d == 0.0:
                    gm[i] += _row_mass(profile, j, ri)
                j += 1 + int(k)
            else:
                if d > 0.0:
                    mid, half, inner = max(ri, d), min(ri, d), max(ri - d, 0.0)
                    acc = 0.0
                    for m in range(GL_NODES.size):
                        x = GL_NODES[m]
                        s = mid + half * x
                        acc += GL_WEIGHTS[m] * s * _row_rho(kind, k, a, s) * half * (1.0 - x) * (ri + inner + half * x)
                    gm[i] += _row_mass(profile, j, inner) + np.pi / d * half * acc
                else:
                    gm[i] += _row_mass(profile, j, ri)
                j += 1
    return gm


@njit(parallel=True, cache=True)
def profile_density(profile, r):
    """``G ρ(r)`` of a profile at every radius of ``r`` (compiled, parallel): the sum of its rows.

    A concentric row (``d = 0``) adds its own :func:`_row_rho`; an offset row (``d > 0``) adds its
    shell average, :func:`~roadrunner.physics.potentials._shell_mean` on the same nodes.
    ``PROFILE_SHELL`` has no density here (its windowed estimate is too noisy for an Eddington
    inversion): it contributes nothing, its payload only skipped. Reached only through
    :func:`~roadrunner.physics.potentials._GroupDensity._density`, whose rows never hold one
    (a Shell has no ``distribution_function``, so it is never in a ``has_distribution`` group).

    Parameters
    ----------
    profile : ndarray of shape (k, 4)
    r : ndarray of shape (n,)

    Returns
    -------
    rho : ndarray of float64, shape (n,)
    """
    rho = np.zeros(r.size)
    for i in prange(r.size):
        ri = r[i]
        j = 0
        while j < profile.shape[0]:
            kind, k, a, d = profile[j, 0], profile[j, 1], profile[j, 2], profile[j, 3]
            if kind != PROFILE_SHELL:
                if d > 0.0:
                    mid, half = max(ri, d), min(ri, d)
                    acc = 0.0
                    for m in range(GL_NODES.size):
                        x = GL_NODES[m]
                        s = mid + half * x
                        acc += GL_WEIGHTS[m] * s * _row_rho(kind, k, a, s)
                    rho[i] += acc / (2.0 * mid)
                else:
                    rho[i] += _row_rho(kind, k, a, ri)
            j += (1 + int(k)) if kind == PROFILE_SHELL else 1
    return rho


def _radial_potential(p, r):
    """``Φ(r)`` of a potential's radial profile: compiled (:func:`profile_potential`) when it has one, else :meth:`~roadrunner.physics.potentials.SphericalPotential._potential`.

    Parameters
    ----------
    p : SphericalPotential
    r : ndarray of shape (n,)
        Physical radii (float64, contiguous).

    Returns
    -------
    phi : ndarray of float64, shape (n,)
    """
    rows = p.profile()
    return np.asarray(p._potential(r), dtype=np.float64) if rows is None else profile_potential(rows, r)


def _has_offset_shell(rows):
    """Whether profile rows hold an offset (``d > 0``) ``PROFILE_SHELL`` header row, walking the header-by-header structure."""
    j = 0
    while j < len(rows):
        kind, k, _, d = rows[j]
        if kind == PROFILE_SHELL:
            if d > 0.0:
                return True
            j += 1 + int(k)
        else:
            j += 1
    return False


def _radial_mass(p, r):
    """``G M(<r)`` of a potential's radial profile: compiled (:func:`profile_mass`) when it has one, else ``G`` times :meth:`~roadrunner.physics.potentials.SphericalPotential._enclosed_mass`.

    An offset ``PROFILE_SHELL`` row (its density is too noisy for :func:`profile_mass`'s monopole
    quadrature) falls back to the same numpy route as ``profile() is None``, which reaches it
    through the exact numpy :func:`~roadrunner.physics.potentials._shell_mass` monopole instead.

    Parameters
    ----------
    p : SphericalPotential
    r : ndarray of shape (n,)
        Physical radii (float64, contiguous).

    Returns
    -------
    gm : ndarray of float64, shape (n,)
    """
    rows = p.profile()
    if rows is None or _has_offset_shell(rows):
        return p.G * np.asarray(p._enclosed_mass(r), dtype=np.float64)
    return profile_mass(rows, r)


@njit(cache=True)
def _offset_phi(x, ln_r1, inv_h, phi, dphi):
    """Φ̄ of a potential's offset rows at ``x``, tabulated on their ``ln r`` nodes (U2): a Hermite spline, slopes by finite difference.

    ``phi``/``dphi`` are the offset rows' shell-averaged potential and its ``d/d ln r`` at the
    nodes (:class:`SphericalDistribution`'s ``_r``, geometric from node 1 on); 0 with no offset
    rows (``phi`` empty). Below the first node, ``Φ̄`` is quadratic in ``r`` (it is even about
    ``r = 0``: the shell average of a point off-centre is unchanged by ``r -> -r``, so it has no
    linear term there, consistent with ``dphi[0] = 0``); beyond the last, the Hermite spline is
    held at the last segment (``j`` clipped).

    Parameters
    ----------
    x : float
    ln_r1 : float
        ``ln`` of the first (smallest) node.
    inv_h : float
        Inverse node spacing in ``ln r``.
    phi, dphi : ndarray of shape (n,)

    Returns
    -------
    phi_x : float
    """
    n = phi.size
    if n == 0:
        return 0.0
    t = (np.log(x) - ln_r1) * inv_h if x > 0.0 else -1.0
    if t < 0.0:
        return phi[0] + (phi[1] - phi[0]) * (x * np.exp(-ln_r1)) ** 2   # centre to first log node: even in r
    j = min(int(t), n - 3)
    u = t - j
    h = 1.0 / inv_h
    u2 = u * u
    u3 = u2 * u
    return ((2 * u3 - 3 * u2 + 1) * phi[j + 1] + (u3 - 2 * u2 + u) * h * dphi[j + 1]
            + (3 * u2 - 2 * u3) * phi[j + 2] + (u3 - u2) * h * dphi[j + 2])


@njit(parallel=True, cache=True)
def _q_and_g_kernel(energy, x_max, profile, w, ww, ln_r1, inv_h, phi_off, dphi_off):
    """Phase-space volume and density of states of a profile at flat ``energy`` (parallel; see :func:`q_and_g`).

    ``profile``'s concentric rows only (U2): the offset rows' Φ̄ comes from the tabulated
    :func:`_offset_phi` instead of a Gauss–Legendre sum per node.
    """
    q = np.empty(energy.size)
    g = np.empty(energy.size)
    for i in prange(energy.size):
        sq = 0.0
        sg = 0.0
        for k in range(w.size):
            x = x_max[i] * (1.0 - w[k] * w[k])
            phi = profile_phi(profile, x) + _offset_phi(x, ln_r1, inv_h, phi_off, dphi_off)
            kin2 = max(2.0 * (energy[i] - phi), 0.0)
            wt = 16.0 * np.pi**2 * ww[k] * 2.0 * x_max[i] * w[k] * x * x
            root = np.sqrt(kin2)
            sq += wt * kin2 * root
            sg += wt * root
        q[i] = sq / 3.0
        g[i] = sg
    return q, g


@njit(parallel=True, cache=True)
def _eddington_kernel(b, depth, psi, d2, w, ww):
    """:func:`_eddington_integral` on flat ``b`` and ``depth`` (parallel)."""
    f = np.empty(b.size)
    norm = np.sqrt(8.0) * np.pi**2
    for i in prange(b.size):
        bi = max(b[i], 0.0)
        di = max(depth[i], _TINY)
        t_max = np.arcsinh(np.sqrt(bi / di))
        root = np.sqrt(di)
        s = 0.0
        for k in range(w.size):
            t = t_max * w[k]
            sh = root * np.sinh(t)
            s += ww[k] * 2.0 * _interp1(bi - sh * sh, psi, d2) * root * np.cosh(t)
        f[i] = t_max * s / norm
    return f


def nfw_phi(x):
    """Scaled NFW potential ``-ln(1 + x) / x``.

    Parameters
    ----------
    x : ndarray
        Radius in units of the scale radius.

    Returns
    -------
    phi : ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    return -np.log1p(x) / x


def _radius_of_energy(e):
    """Radius where ``phi(x) = e``, for scaled energies ``e`` in ``(-1, 0)``."""
    return np.interp(e, nfw_phi(_X), _X)


def nfw_density_of_states(cal, c):
    """Density of states ``g = 16 pi^2 int_0^{min(c, r_E)} x^2 sqrt(2 (E - phi)) dx``.

    The substitution ``x = x_max (1 - w^2)`` removes the square-root
    endpoint at the radial apocentre ``r_E``.

    Parameters
    ----------
    cal : ndarray
        Binding depths ``calE`` (scaled energy ``E = calE - 1``).
    c : float or ndarray
        Truncation radius (concentration); ``np.inf`` for none.
        Broadcasts against ``cal``.

    Returns
    -------
    g : ndarray
        Broadcast shape of ``cal`` and ``c``.
    """
    e = np.asarray(cal, dtype=np.float64) - 1.0
    x_max = np.minimum(c, _radius_of_energy(e))
    x = x_max[..., None] * (1.0 - _W**2)
    kin = np.sqrt(np.clip(2.0 * (e[..., None] - nfw_phi(x)), 0.0, None))
    return 16.0 * np.pi**2 * np.sum(_WW * 2.0 * x_max[..., None] * _W * x**2 * kin, axis=-1)


def nfw_phase_volume(cal, c):
    """Phase-space volume more bound than ``E`` inside ``x <= c``.

    ``q = (16 pi^2 / 3) int_0^{min(c, r_E)} x^2 [2 (E - phi)]^{3/2} dx``, with
    the same ``x = x_max (1 - w^2)`` substitution as the density of states.

    Parameters
    ----------
    cal : ndarray
        Binding depths ``calE`` (scaled energy ``E = calE - 1``).
    c : float or ndarray
        Truncation radius (concentration). Broadcasts against ``cal``.

    Returns
    -------
    q : ndarray
    """
    e = np.asarray(cal, dtype=np.float64) - 1.0
    x_max = np.minimum(c, _radius_of_energy(e))
    x = x_max[..., None] * (1.0 - _W**2)
    kin = np.clip(2.0 * (e[..., None] - nfw_phi(x)), 0.0, None) ** 1.5
    return 16.0 * np.pi**2 / 3.0 * np.sum(_WW * 2.0 * x_max[..., None] * _W * x**2 * kin, axis=-1)


def _eddington_integral(b, depth, psi, d2):
    """Eddington's ``f(b) = (1 / (√8 π²)) ∫_0^b (d²ρ/dψ²) dψ / sqrt(b - ψ)`` from a tabulated ``d²ρ/dψ²``.

    With ``ψ = b - s²`` and ``s = sqrt(δ) sinh t``, where ``δ = ψ(0) - b``
    is the depth of ``b`` below the bottom of the well, the integral is
    ``∫_0^{asinh(sqrt(b / δ))} 2 (d²ρ/dψ²) sqrt(δ) cosh t dt``, taken by
    Gauss–Legendre in ``t``. Near the centre ``d²ρ/dψ²`` peaks like a power
    of ``δ + s²`` within ``s ~ sqrt(δ)`` of the endpoint; in ``t`` that peak
    is smooth and resolved at any depth.

    Parameters
    ----------
    b : ndarray
        Relative energies ``-E``; ``f = 0`` for ``b <= 0``.
    depth : ndarray
        ``ψ(0) - b``, broadcasting against ``b``; floored at the smallest
        normal float.
    psi, d2 : ndarray
        ``ψ`` (increasing) and ``d²ρ/dψ²`` there.

    Returns
    -------
    f : ndarray

    Compiled, parallel over ``b``.
    """
    b = np.asarray(b, dtype=np.float64)
    depth = np.broadcast_to(np.asarray(depth, dtype=np.float64), b.shape)
    f = _eddington_kernel(np.ascontiguousarray(b).ravel(), np.ascontiguousarray(depth).ravel(),
                          np.ascontiguousarray(psi, dtype=np.float64), np.ascontiguousarray(d2, dtype=np.float64), _W, _WW)
    return f.reshape(b.shape)


def _eddington(b):
    """Eddington distribution function of the isotropic NFW profile at relative energies ``b = -E = 1 - calE``.

    :func:`_eddington_integral` with ``ψ(0) = 1``. Accurate (to 1e-3) down
    to ``b ~ 1e-8``, set by where ``_X`` ends.
    """
    psi = -nfw_phi(_X)                                    # decreasing in x
    rho = 1.0 / (4.0 * np.pi * _X * (1.0 + _X) ** 2)
    d2 = np.gradient(np.gradient(rho, psi), psi)
    b = np.asarray(b, dtype=np.float64)
    return _eddington_integral(b, 1.0 - b, psi[::-1], d2[::-1])


def _depth_nodes(eps_kink, eps_last, n_deep, n_shallow):
    """Scaled energies ``eps = E / Φ₀`` of a table's nodes, increasing: from ``E = 0`` towards the centre.

    The kink ``eps_kink = Φ(R) / Φ₀``, where the radial apocentre reaches the
    counting radius ``R``, is a node: ``g`` and ``q`` stop growing like the
    untruncated ones there, and ``g`` has a ``(eps_kink - eps)^(3/2)`` term
    just past it.

    - Deeper, ``n_deep`` nodes are uniform in ``logit(calE)``,
      ``calE = 1 - eps``, from the kink to :data:`_DEEPEST`. They are
      geometric in ``calE`` towards the centre, where the distributions are
      power laws of ``calE``, and geometric in ``eps`` below a
      point-mass-like kink (``eps_kink << 1``), where they are power laws of
      ``eps``.
    - Shallower, ``n_shallow`` nodes are uniform in
      ``logit(1 - eps / eps_kink)`` from :data:`_KINK_GAP` to
      ``eps = eps_last``: geometric away from the kink, and towards
      ``E = 0``, where ``N`` is a power law of ``-E``.

    Linear interpolation in ``eps`` is then accurate at both ends and at the
    kink without a logarithm per query, and ``eps``, unlike ``calE``, keeps
    its relative precision as ``E -> 0``.

    Parameters
    ----------
    eps_kink : float or ndarray
        One row of nodes per element.
    eps_last : float or ndarray
        Shallowest node, below ``eps_kink (1 - _KINK_GAP)``; broadcasts
        against ``eps_kink``.
    n_deep, n_shallow : int

    Returns
    -------
    eps : ndarray of shape ``np.shape(eps_kink) + (n_shallow + 1 + n_deep,)``
    """
    k = np.asarray(eps_kink, dtype=np.float64)[..., None]
    r = np.asarray(eps_last, dtype=np.float64)[..., None] / k
    t_last, t_gap = np.log1p(-r) - np.log(r), np.log(_KINK_GAP) - np.log1p(-_KINK_GAP)
    shallow = k / (1.0 + np.exp(t_last + (t_gap - t_last) * np.linspace(0.0, 1.0, n_shallow)))
    t_kink, t_deep = np.log1p(-k) - np.log(k), np.log(_DEEPEST) - np.log1p(-_DEEPEST)   # logit(calE)
    deep = 1.0 / (1.0 + np.exp(t_kink + (t_deep - t_kink) * np.linspace(0.0, 1.0, n_deep + 1)[1:]))
    return np.concatenate([shallow, k, deep], axis=-1)


def _nfw_nodes(c):
    """Node energies ``eps`` of the table row at concentration ``c``: :func:`_depth_nodes` with the kink ``-phi(c)`` and the last node at ``1e-5`` of it.

    Parameters
    ----------
    c : float or ndarray
        Concentration(s); an array gives one row of nodes per element.

    Returns
    -------
    eps : ndarray of shape ``np.shape(c) + (n,)``
    """
    k = -nfw_phi(c)
    return _depth_nodes(k, 1e-5 * k, *_NFW_NODES)


class NFWEnergyTables(NamedTuple):
    """NFW energy-distribution tables: one row per ``C_GRID`` value, on that row's nodes ``_nfw_nodes(c)``."""

    log_density: np.ndarray          # log n(calE | c), normalised on [0, 1]
    cdf: np.ndarray                  # u = F_c(calE), the mass fraction more bound
    log_volume_fraction: np.ndarray  # log w = log q(calE | c) - log q(1 | c)


_NFW_CHUNK = 2     # C_GRID rows per chunk (U8): caps the (chunk, n, 96) Eddington/g intermediates


@functools.cache
def nfw_energy_tables():
    """Build the NFW energy-distribution tables (once per process), in chunks over ``C_GRID`` rows (U8).

    Every step is row-wise (no sum or normalisation crosses rows: the node grid, the Eddington
    and density-of-states quadratures, the cumulative mass and its per-row normalisation are all
    taken one ``c`` at a time), so a :data:`_NFW_CHUNK`-row chunk goes through exactly the same
    floating-point operations as the whole grid at once; the result is bit-identical to building
    all of :data:`C_GRID` in one pass. Chunking only caps the size of the transient
    ``(chunk, n, 96)`` Eddington/density-of-states arrays that otherwise peak at the whole grid's.

    Returns
    -------
    tables : NFWEnergyTables
        Arrays of shape ``(len(C_GRID), n)``, on the nodes
        ``_nfw_nodes(C_GRID)``.
    """
    n_nodes = sum(_NFW_NODES) + 1
    log_n = np.empty((len(C_GRID), n_nodes))
    cdf = np.empty((len(C_GRID), n_nodes))
    log_w = np.empty((len(C_GRID), n_nodes))
    for lo in range(0, len(C_GRID), _NFW_CHUNK):
        c = C_GRID[lo:lo + _NFW_CHUNK]
        eps = _nfw_nodes(c)
        cal = 1.0 - eps
        n = _eddington(eps) * nfw_density_of_states(cal, c[:, None])
        seg = 0.5 * (n[:, 1:] + n[:, :-1]) * np.diff(eps, axis=1)
        # mass more bound than each node: summed from the deepest, plus the power-law tail below it
        tail = _deep_tail(n[:, -2:], cal[:, -2:])
        row_cdf = np.concatenate([np.cumsum(seg[:, ::-1], axis=1)[:, ::-1], np.zeros((len(c), 1))], axis=1) + tail[:, None]
        n /= row_cdf[:, :1]
        row_cdf /= row_cdf[:, :1]
        log_q = np.log(nfw_phase_volume(cal, c[:, None]))
        log_q_zero = np.log(nfw_phase_volume(np.ones(1), c[:, None]))     # E = 0
        hi = lo + len(c)
        log_n[lo:hi], cdf[lo:hi], log_w[lo:hi] = np.log(n), row_cdf, log_q - log_q_zero
    return NFWEnergyTables(log_n, cdf, log_w)


def _deep_tail(n, cal):
    """Mass deeper than the deepest node, ``∫_0^{cal_0} n dcal = n_0 cal_0 / (p + 1)``.

    ``n`` is a power law ``cal^p`` there; ``p`` comes from the two deepest
    nodes. 0 when both nodes hold nothing (f clipped to 0 there: the 0/0 limit).

    Parameters
    ----------
    n, cal : ndarray of shape (..., 2)
        ``n`` (per unit ``cal``) and ``cal`` at the second-deepest and the
        deepest node.

    Returns
    -------
    tail : ndarray of shape (...)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.log(n[..., 1] / n[..., 0]) / np.log(cal[..., 1] / cal[..., 0])
        return np.nan_to_num(n[..., 1] * cal[..., 1] / (p + 1.0), nan=0.0, posinf=np.inf, neginf=-np.inf)


def _row(table, grid, c):
    """Row of ``table`` at ``c``: linear in log c between the rows of ``grid`` (node by node, kink to kink), clipped to the grid."""
    log_grid = np.log(grid)
    lc = np.log(np.clip(c, grid[0], grid[-1]))
    j = int(np.clip(np.searchsorted(log_grid, lc), 1, len(grid) - 1))
    w = (lc - log_grid[j - 1]) / (log_grid[j] - log_grid[j - 1])
    return (1.0 - w) * table[j - 1] + w * table[j]


def _lookup(table, eps, c):
    """``table`` at scaled energies ``eps`` and concentration ``c``.

    Linear in ``eps`` between the row's nodes (:func:`table_query`), held
    at the end nodes beyond them; ``c`` is clipped to the grid.
    """
    c = np.clip(c, C_GRID[0], C_GRID[-1])
    return table_query(eps, _nfw_nodes(c), _row(table, C_GRID, c))


def nfw_energy_fraction(eps, c):
    """Fraction ``u = F_c(calE)`` of the halo's mass more bound than ``eps``.

    Parameters
    ----------
    eps : ndarray
        Normalised energy ``eps = E / Phi_0`` (1 at the centre, 0 at ``E = 0``).
    c : float
        Concentration.

    Returns
    -------
    u : ndarray
        In ``[0, 1]``; uniform for the halo's own dark matter.
    """
    return _lookup(nfw_energy_tables().cdf, eps, c)


def nfw_log_energy_density(eps, c):
    """Log dark-matter energy distribution ``log dN/dcalE`` (normalised on [0, 1]).

    Parameters
    ----------
    eps : ndarray
        Normalised energy ``eps = E / Phi_0``.
    c : float
        Concentration.

    Returns
    -------
    log_n : ndarray
    """
    return _lookup(nfw_energy_tables().log_density, eps, c)


_CENTRAL_VOLUME_SLOPE = 4.5        # q ∝ calE^(9/2) near the NFW centre (phi ≈ -1 + x/2)


def nfw_log_phase_space_fraction(eps, c):
    """Log fraction ``ln w = ln q(E) - ln q(0)`` of the virial sphere's bound phase space more bound than ``eps``.

    ``w`` is uniform on ``(0, 1]`` for particles spread uniformly in the
    bound phase space of ``x <= c``. Deeper than the table, the exact
    central power law ``q ∝ calE^(9/2)`` extends it; working in logs keeps
    the deepest values (``w`` down to ~1e-20 and below) exact.

    Parameters
    ----------
    eps : ndarray
        Normalised energy ``eps = E / Phi_0``.
    c : float
        Concentration.

    Returns
    -------
    log_w : ndarray
        ``<= 0``.
    """
    c = np.clip(c, C_GRID[0], C_GRID[-1])
    return table_query(np.atleast_1d(np.asarray(eps, dtype=np.float64)), _nfw_nodes(c),
                       _row(nfw_energy_tables().log_volume_fraction, C_GRID, c), deep_slope=_CENTRAL_VOLUME_SLOPE)


_SHALLOWEST = 1e-12                            # shallowest node of the generic tables: -E = 1e-12 |Φ(0)|
_GENERIC_NODES = 640, 660                      # nodes deeper and shallower than the kink (see _depth_nodes)
_APO_NODES = 4000                              # radial nodes (0, then log-spaced 1e-6 R .. R) of the apocentre lookup
_EDDINGTON_R = np.geomspace(1e-7, 1e16, 23000)  # physical radii (kpc) of the generic Eddington inversion: f to -E ~ 1e-12 |Φ(0)|


@njit(cache=True)
def _eddington_preprocess(r, psi, rho, dpsi):
    """``d²ρ/dψ²`` through ``r`` and its reversed, finite, Ψ-decreasing selection, in one pass (U7).

    Two non-uniform-grid gradients (second order in the interior, one-sided at the ends, as
    ``np.gradient``) of ``rho`` against ``r``, each divided by ``dpsi``; then the points kept are
    those with a finite ``d²ρ/dψ²`` and (but for the first) a strictly smaller ``ψ`` than the one
    before, returned reversed (increasing ``ψ``) as :func:`_eddington_integral` needs them.

    Parameters
    ----------
    r : ndarray of shape (n,)
        Increasing physical radii.
    psi, rho, dpsi : ndarray of shape (n,)
        ``ψ = -Φ``, the density and ``dψ/dr`` there.

    Returns
    -------
    psi_kept, d2_kept : ndarray of float64
        Increasing ``ψ``.
    """
    n = r.size
    drho = np.empty(n)
    for i in range(n):
        if i == 0:
            drho[i] = (rho[1] - rho[0]) / (r[1] - r[0])
        elif i == n - 1:
            drho[i] = (rho[n - 1] - rho[n - 2]) / (r[n - 1] - r[n - 2])
        else:
            hs, hd = r[i] - r[i - 1], r[i + 1] - r[i]           # np.gradient's own non-uniform formula and grouping
            a, b, c = -hd / (hs * (hs + hd)), (hd - hs) / (hs * hd), hs / (hd * (hs + hd))
            drho[i] = a * rho[i - 1] + b * rho[i] + c * rho[i + 1]
        drho[i] /= dpsi[i]
    d2 = np.empty(n)
    for i in range(n):
        if i == 0:
            d2[i] = (drho[1] - drho[0]) / (r[1] - r[0])
        elif i == n - 1:
            d2[i] = (drho[n - 1] - drho[n - 2]) / (r[n - 1] - r[n - 2])
        else:
            hs, hd = r[i] - r[i - 1], r[i + 1] - r[i]
            a, b, c = -hd / (hs * (hs + hd)), (hd - hs) / (hs * hd), hs / (hd * (hs + hd))
            d2[i] = a * drho[i - 1] + b * drho[i] + c * drho[i + 1]
        d2[i] /= dpsi[i]
    keep = np.empty(n, dtype=np.bool_)
    count = 0
    for i in range(n):
        ok = np.isfinite(d2[i]) and (i == 0 or psi[i] < psi[i - 1])
        keep[i] = ok
        count += ok
    psi_kept, d2_kept = np.empty(count), np.empty(count)
    j = count - 1
    for i in range(n):
        if keep[i]:
            psi_kept[j], d2_kept[j] = psi[i], d2[i]
            j -= 1
    return psi_kept, d2_kept


def eddington_distribution_function(background, energy, density=None):
    """Isotropic distribution function ``f(E)`` of a density in a potential's equilibrium.

    Eddington's inversion for an isotropic system,
    ``f(E) = (1 / (√8 π²)) ∫_0^{ψ_E} (d²ρ/dψ²) dψ / sqrt(ψ_E - ψ)`` with
    ``ψ = -Φ`` and ``ψ_E = -E``, without the boundary term (``ρ -> 0`` as
    ``ψ -> 0``), taken by :func:`_eddington_integral`, resolved at any
    depth below ``ψ(0) = -Φ(0)``. ``background`` provides ``ψ = -_potential``,
    ``dψ/dr = -G _enclosed_mass / r²``, ``Φ(0)`` and ``G``; ``density``
    provides ``_density(r)`` (``background``'s own when ``None``: the
    self-consistent ``f``). ``ρ(ψ)`` is tabulated on :data:`_EDDINGTON_R`,
    and its derivatives go through the radius, ``dψ/dr``, which stays
    finite in cores where ``ψ`` barely changes. ``f`` is the phase-space
    density whose velocity integral gives ``density``'s mass density.

    ``f`` may be negative for a density in another potential (no isotropic
    equilibrium there, e.g. a cored density in a cusp); it is clipped to 0
    with a ``RuntimeWarning``.

    Parameters
    ----------
    background : PotentialModel
        Provides ``potential``, ``enclosed_mass`` and ``G``: the equilibrium ``f`` is built in.
    energy : ndarray
        Physical specific energies; ``f = 0`` for ``energy >= 0``.
    density : optional
        Provides ``_density(r)``; ``background``'s own when ``None``.

    Returns
    -------
    f : ndarray of float64
    """
    density = background if density is None else density
    r = _EDDINGTON_R
    psi = -_radial_potential(background, r)
    dpsi = -_radial_mass(background, r) / r**2
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.asarray(density._density(r), dtype=np.float64)
        psi, d2 = _eddington_preprocess(r, psi, rho, dpsi)   # ψ strictly decreasing in r; reversed, finite: increasing ψ
    energy = np.asarray(energy, dtype=np.float64)
    phi0 = float(np.asarray(background._potential(np.zeros(1)), dtype=np.float64)[0])
    f = _eddington_integral(-energy, energy - phi0, psi, d2)
    negative = f < 0.0
    if negative.any():
        eps = energy / phi0
        warnings.warn(
            f"Eddington's f of {density!r} in the potential of {background!r} is negative at "
            f"{np.count_nonzero(negative)} of {f.size} energies (E / Φ(0) from "
            f"{np.min(eps, where=negative, initial=np.inf):.3g} to {np.max(eps, where=negative, initial=-np.inf):.3g}, "
            f"min f / max |f| = {f.min() / np.abs(f).max():.2g}): this density has no isotropic equilibrium in this "
            "potential there; f is clipped to 0.", RuntimeWarning, stacklevel=2)
    return np.maximum(f, 0.0)


def _offset_table(potential, r):
    """Split ``potential.profile()`` into concentric and offset rows, and tabulate the offset rows' Φ̄ on ``r`` (U2).

    Called once by :class:`SphericalDistribution`, on its radial grid (``r[0] = 0``, then geometric). The
    offset rows' shell-averaged potential and its ``d / d ln r`` (finite differences) feed :func:`_offset_phi`'s
    Hermite spline, so the q/g quadratures no longer run a Gauss–Legendre sum per row per node for
    them; the concentric rows stay exact, read straight by :func:`profile_phi`.

    Parameters
    ----------
    potential : SphericalPotential
    r : ndarray of shape (n,)
        Increasing radii, ``r[0] = 0``, ``r[1:]`` geometric.

    Returns
    -------
    profile : ndarray of shape (k, 4) or None
        Concentric rows; ``None`` when ``potential.profile()`` is ``None`` (no compiled route).
    ln_r1, inv_h : float
    phi_off, dphi_off : ndarray of shape (n,)
        Empty when there are no offset rows.
    """
    rows = potential.profile()
    profile = None if rows is None else rows[rows[:, 3] == 0.0]
    offset = np.empty((0, 4)) if rows is None else rows[rows[:, 3] > 0.0]
    r_off = r if len(offset) else r[:0]
    phi_off = profile_potential(offset, r_off)
    ln_r = np.log(r[1:])
    ln_r1, inv_h = ln_r[0], (ln_r.size - 1) / (ln_r[-1] - ln_r[0])
    dphi_off = np.concatenate(([0.0], np.gradient(phi_off[1:], ln_r))) if len(offset) else r_off
    return profile, ln_r1, inv_h, phi_off, dphi_off


def q_and_g(potential, profile, energy, x_max, ln_r1, inv_h, phi_off, dphi_off):
    """Phase-space volume ``q(E)`` and density of states ``g(E)`` of the orbits out to ``x_max``.

    Both integrals run over ``r`` from 0 to ``x_max`` (the radial apocentre ``r_E``, capped at a
    counting radius when there is one), with ``r = x_max (1 - w²)``, which removes the
    square-root endpoint at the apocentre, and Gauss–Legendre in ``w``. They share their nodes
    and potential evaluations: compiled (parallel) when ``profile`` is not ``None``, otherwise
    from ``potential._potential`` (exact offsets included, no tabulation).

    Parameters
    ----------
    potential : SphericalPotential
    profile : ndarray of shape (k, 4) or None
        The concentric rows of ``potential.profile()`` (:func:`_offset_table`).
    energy : ndarray of shape (n,)
        Physical energies (float64).
    x_max : ndarray of shape (n,)
        Outer radii (float64, contiguous).
    ln_r1, inv_h, phi_off, dphi_off
        The offset rows' table (:func:`_offset_table`); unused in the numpy fallback.

    Returns
    -------
    q, g : ndarray of float64, shape (n,)
    """
    if profile is not None:
        return _q_and_g_kernel(np.ascontiguousarray(energy, dtype=np.float64), x_max, profile, _W, _WW,
                               ln_r1, inv_h, phi_off, dphi_off)
    x = x_max[:, None] * (1.0 - _W**2)
    kin2 = np.clip(2.0 * (energy[:, None] - np.asarray(potential._potential(x), dtype=np.float64)), 0.0, None)
    w = 16.0 * np.pi**2 * _WW * 2.0 * x_max[:, None] * _W * x**2
    return np.sum(w * kin2**1.5, axis=-1) / 3.0, np.sum(w * np.sqrt(kin2), axis=-1)


class SphericalDistribution:
    """Energy distribution and phase-space volume of a spherical potential inside ``r_max``.

    It counts the potential's untruncated, isotropic profile inside the
    sphere of radius ``r_max`` (see the module docstring for the
    definitions). It provides three quantities at any physical energy:

    - :meth:`energy_fraction`: the fraction of the mass inside ``r_max``
      that is more bound than ``E``. It is uniform on ``(0, 1)`` for that
      mass itself;
    - :meth:`log_energy_density`: ``ln N(E)``, per unit ``E``;
    - :meth:`log_phase_space_fraction`: ``ln w(E)``. ``w`` is uniform on
      ``(0, 1]`` for particles spread uniformly in the bound phase space
      inside ``r_max``.

    Everything is tabulated over ``eps = E / Φ(0)`` (every potential is
    finite at ``r = 0``) on :func:`_depth_nodes`, from ``calE = 1 - eps =
    1e-6`` to ``eps = 1e-12``, with the kink ``Φ(r_max) / Φ(0)`` as a
    node. That resolves a deep core (``calE -> 0``, where ``q`` is a power
    law of ``calE``), the kink, the approach to ``E = 0``, and a
    point-mass-like well, where most particles sit at ``eps ~ softening / r``.

    The tables are built on first use, in float64:
    - the phase-space fraction needs ``q`` only, so it works for any
      potential;
    - the energy pair also needs ``f`` (the potential's
      :meth:`distribution_function`, which raises for a potential without
      one) and ``g``.

    Instances are cached per potential and ``r_max`` by
    :class:`~roadrunner.physics.potentials.SphericalPotential`, and per
    group by the composite's group densities. Physical energies in;
    float64 out.

    Parameters
    ----------
    potential : SphericalPotential
        A :class:`~roadrunner.physics.potentials.SphericalPotential`; its ``profile()``, when not
        ``None``, makes the q and g quadratures compiled.
    r_max : float
        Counting radius (physical).
    distribution_function : callable, optional
        The f(E) whose N = f g is tabulated; the potential's own
        ``distribution_function`` when None. A composite's group passes
        Eddington's f of its density in the composite's potential, so N
        counts that group's mass inside r_max.
    """

    def __init__(self, potential, r_max, distribution_function=None):
        self.potential = potential
        self.r_max = float(r_max)
        self._distribution_function = potential.distribution_function if distribution_function is None else distribution_function
        self._r = self.r_max * np.concatenate(([0.0], np.geomspace(1e-6, 1.0, _APO_NODES - 1)))
        self._phi = _radial_potential(potential, self._r)
        self._eps = _depth_nodes(self._phi[-1] / self._phi[0], _SHALLOWEST, *_GENERIC_NODES)
        self._energies = self._phi[0] * self._eps                 # physical energies at the nodes
        self._log_w = None
        self._log_density = self._cdf = None
        # U2: the offset rows' Φ̄, tabulated on self._r with Hermite slopes; the concentric rows stay exact
        self._profile, self._ln_r1, self._inv_h, self._phi_off, self._dphi_off = _offset_table(potential, self._r)

    def _scaled(self, energy):
        """``eps = E / Φ(0)``: 0 at ``E = 0``, 1 at the bottom of the well.

        Parameters
        ----------
        energy : ndarray

        Returns
        -------
        eps : ndarray of float64
        """
        return np.asarray(energy, dtype=np.float64) / self._phi[0]

    def _q_and_g(self, energy):
        """Phase-space volume ``q(E)`` and density of states ``g(E)`` inside ``r_max``: :func:`q_and_g` out to the apocentre, capped at ``r_max``.

        Parameters
        ----------
        energy : ndarray of shape (n,)
            Physical energies (float64).

        Returns
        -------
        q, g : ndarray of float64, shape (n,)
        """
        return q_and_g(self.potential, self._profile, energy, np.interp(energy, self._phi, self._r),
                      self._ln_r1, self._inv_h, self._phi_off, self._dphi_off)

    def _phase_table(self):
        """``ln q(E) - ln q(0)`` at the depth nodes, built once.

        Returns
        -------
        log_w : ndarray of float64
        """
        if self._log_w is None:
            q, _ = self._q_and_g(self._energies)
            q0, _ = self._q_and_g(np.zeros(1))
            self._log_w = np.log(q) - np.log(q0[0])
        return self._log_w

    def _energy_table(self):
        """``ln N`` per unit ``E`` and its cumulative from the deepest node, built once.

        ``N = f g`` is normalised so that ``∫ N dE = 1``, with
        ``dE = |Φ(0)| d eps``: over the nodes, plus the power-law tail deeper
        than ``calE = 1e-6`` (:func:`_deep_tail`).

        Returns
        -------
        log_n, cdf : ndarray of float64
        """
        if self._log_density is None:
            f = np.asarray(self._distribution_function(self._energies), dtype=np.float64)
            _, g = self._q_and_g(self._energies)
            n = f * g
            seg = -self._phi[0] * 0.5 * (n[1:] + n[:-1]) * np.diff(self._eps)
            tail = -self._phi[0] * _deep_tail(n[-2:], 1.0 - self._eps[-2:])
            cdf = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0])) + tail   # mass more bound than each node
            with np.errstate(divide="ignore"):
                self._log_density = np.log(n / cdf[0])
            self._cdf = cdf / cdf[0]
        return self._log_density, self._cdf

    def energy_fraction(self, energy):
        """Fraction of the mass inside ``r_max`` more bound than ``E``.

        It is uniform on ``(0, 1)`` for that mass itself, which is the null
        hypothesis of the ``energy`` plausibility model. It is held at 0 and
        1 beyond the depth grid.

        Parameters
        ----------
        energy : ndarray
            Physical specific energies.

        Returns
        -------
        u : ndarray of float64
        """
        return table_query(self._scaled(energy), self._eps, self._energy_table()[1])

    def log_energy_density(self, energy):
        """``ln N(E)``: energy distribution of the mass inside ``r_max``, per unit ``E``.

        It is held at its end values beyond the depth grid.

        Parameters
        ----------
        energy : ndarray
            Physical specific energies.

        Returns
        -------
        log_n : ndarray of float64
        """
        return table_query(self._scaled(energy), self._eps, self._energy_table()[0])

    def log_phase_space_fraction(self, energy):
        """``ln w = ln q(E) - ln q(0)``: fraction of the bound phase space inside ``r_max`` more bound than ``E``.

        It is uniform on ``(0, 1]`` for particles spread uniformly in that
        phase space. Deeper than the table it follows the local power law of
        ``q`` in ``calE`` through the two deepest nodes; ``calE`` is floored
        at the smallest normal float, so energies at or below ``Φ(0)`` (a
        particle at rest at the centre, or below it by rounding) stay
        finite. It is held at its shallowest node (``≈ 0``) for ``E >= 0``.

        Parameters
        ----------
        energy : ndarray
            Physical specific energies.

        Returns
        -------
        log_w : ndarray of float64
            ``<= 0``.
        """
        row = self._phase_table()
        return table_query(np.atleast_1d(self._scaled(energy)), self._eps, row, deep_slope=_node_slope(self._eps, row))


SELF_SIMILAR_C = np.geomspace(0.1, 1e4, 121)   # counting radius / scale radius of the self-similar table rows


class SelfSimilarTables(NamedTuple):
    """Tables of a self-similar profile: one row per ``SELF_SIMILAR_C`` value, on that row's generic nodes.

    ``log_density`` is per unit ``eps = E / Φ₀`` (the reference has ``Φ₀ = -1``).
    """

    log_density: np.ndarray
    cdf: np.ndarray
    log_volume_fraction: np.ndarray


def self_similar_tables(reference):
    """Build the tables of a self-similar profile from a reference instance with ``G = M = a = 1``.

    Each row is the generic :class:`SphericalDistribution` of ``reference`` counted inside
    ``r_max = c`` (its scale radius is 1), so a halo of the same profile with scale radius ``a``
    reads them at ``c = r_max / a`` and ``eps = E / Φ₀``. Its ``Φ₀`` must be ``-1``.

    Parameters
    ----------
    reference : PotentialModel
        The profile with ``G = M = a = 1``.

    Returns
    -------
    tables : SelfSimilarTables
        Arrays of shape ``(len(SELF_SIMILAR_C), n)``.
    """
    rows = [SphericalDistribution(reference, c) for c in SELF_SIMILAR_C]
    energy = [d._energy_table() for d in rows]
    return SelfSimilarTables(np.array([e[0] for e in energy]), np.array([e[1] for e in energy]),
                             np.array([d._phase_table() for d in rows]))


def self_similar_lookup(table, eps, eps_kink, c, deep=False):
    """``table`` of :func:`self_similar_tables` at scaled energies ``eps`` and ``c = r_max / a``.

    Rows are blended linearly in ``log c`` node by node (kink to kink); the nodes are those of
    the generic route for the kink ``eps_kink = Φ(r_max) / Φ₀``, and the value is linear in
    ``eps`` between them (:func:`table_query`), held at the end nodes beyond them. With ``deep``
    (for ``ln w``) the deepest values follow the power law through the two deepest nodes instead.

    Parameters
    ----------
    table : ndarray of shape (len(SELF_SIMILAR_C), n)
    eps : ndarray
        Scaled energies ``E / Φ₀``.
    eps_kink : float
    c : float
        Inside ``SELF_SIMILAR_C``.
    deep : bool, default=False

    Returns
    -------
    values : ndarray of float64
    """
    nodes = _depth_nodes(eps_kink, _SHALLOWEST, *_GENERIC_NODES)
    row = _row(table, SELF_SIMILAR_C, c)
    return table_query(np.atleast_1d(np.asarray(eps, dtype=np.float64)), nodes, row,
                       deep_slope=_node_slope(nodes, row) if deep else None)

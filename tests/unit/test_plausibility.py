"""Plausibility models: rank bit-identity, energy model, refit and assigner wiring."""

import numpy as np
import pytest
from scipy.stats import rankdata

from roadrunner._defaults import PLAUSIBILITY_FLOOR, PLAUSIBILITY_MIN_MEMBERS, math_dtype, precision
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.clustering.assignment.plausibility import (
    EnergyPlausibility,
    KRAVTSOV_RHALF,
    PhaseSpacePlausibility,
    RankPlausibility,
    literature_depth,
    _errani_scales,
    errani_log_ratio,
    make_plausibility,
)
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.mixture._math import row_l1_normalize
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential
from roadrunner.physics.scaler import StandardScaler


def _old_rank_transform(values):
    """The removed ``gmm._rank_transform``, kept here as the reference."""
    if len(values) == 0:
        return values
    return np.log1p(rankdata(values, method="ordinal")).astype(math_dtype(), copy=False)


def _nfw_halo(xcen, vel, M, rvir, c, sid):
    inner = NFWPotential(M=M, Rs=rvir / c, c=c)
    return HaloModel(inner, np.asarray(xcen, float), np.asarray(vel, float), rvir,
                     sub_tree_id=sid, redshift=0.0, comoving=False)


def _two_nfw_halos(seed=3):
    """A host and a satellite whose bound sets overlap, with boundness computed."""
    rng = np.random.default_rng(seed)
    host = _nfw_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, 10.0, 1)
    sat = _nfw_halo([15, 0, 0], [0, 50, 0], 1e10, 25.0, 12.0, 2)
    coords = np.vstack([
        np.hstack([rng.normal(0, 5, (300, 3)), rng.normal(0, 30, (300, 3))]),
        np.hstack([rng.normal([15, 0, 0], 1.5, (200, 3)), rng.normal([0, 50, 0], 10, (200, 3))]),
    ])
    halos = HaloEnsemble([host, sat])
    compute_halo_bound_particles(halos, coords)
    return halos, coords


class TestRankPlausibility:
    @pytest.mark.parametrize("math", ["single", "double"])
    def test_latent_prior_bit_identical_to_old_rank_transform(self, math):
        rng = np.random.default_rng(1)
        h1 = HaloModel(KeplerPotential(M=1e12), np.zeros(3), np.zeros(3), 100.0, 1, 0.0)
        h2 = HaloModel(KeplerPotential(M=1e11), np.zeros(3), np.zeros(3), 50.0, 2, 0.0)
        h1.set_boundness(np.arange(0, 150), rng.uniform(0, 5, 150).astype(np.float32), np.ones(150))
        h2.set_boundness(np.arange(80, 200), rng.uniform(0, 5, 120).astype(np.float32), np.ones(120))
        ens = HaloEnsemble([h1, h2])
        csc_b, _ = ens.get_particles()
        a = XGMMAssigner(verbose=0)
        a.previous_resp = {}
        a.plausibility.prepare(ens)
        csc_a = SparseCSC(csc_b.column_indices, a.plausibility.column_values(csc_b.column_id),
                          column_id=csc_b.column_id)
        with precision(math=math):
            old = row_l1_normalize(csc_b.to_dense(col_func=_old_rank_transform)).astype(math_dtype())
            new, _ = a._initial_responsibilities(None, csc_b, csc_a, None, None)
        assert new.dtype == old.dtype
        np.testing.assert_array_equal(new, old)

    def test_registry(self):
        assert isinstance(make_plausibility("rank"), RankPlausibility)
        assert isinstance(make_plausibility("energy"), EnergyPlausibility)
        with pytest.raises(ValueError):
            make_plausibility("nope")


class TestEnergyPlausibility:
    def test_errani_scale_matches_kravtsov_calibration(self):
        # calE_s for which r_1/2 = 0.015 R_vir (reference from the scratch quad calibration).
        cal_half, es = _errani_scales()
        assert np.all(np.diff(cal_half) > 0)
        for c, ref in [(6.53, 0.0739), (21.31, 0.2023)]:
            h = _nfw_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, c, 1)
            r = np.array([KRAVTSOV_RHALF * h.virial_radius])
            depth = 1.0 - h.potential(r)[0] / h.central_potential()
            got = np.exp(np.interp(np.log(depth), np.log(cal_half), np.log(es)))
            np.testing.assert_allclose(got, ref, rtol=5e-3)

    def test_first_snapshot_is_floored_errani_and_positive(self):
        halos, _ = _two_nfw_halos()
        p = EnergyPlausibility()
        p.prepare(halos)
        for h in halos:
            rows, eps, _ = h.get_boundness()
            alpha = p.column_values([h.sub_tree_id])[0]
            assert alpha.shape == rows.shape
            assert np.all(np.isfinite(alpha)) and np.all(alpha >= PLAUSIBILITY_FLOOR * (1 - 1e-12))
            ref = (1 - PLAUSIBILITY_FLOOR) * np.exp(errani_log_ratio(h, eps.astype(float))) + PLAUSIBILITY_FLOOR
            np.testing.assert_allclose(alpha, np.minimum(ref, np.exp(50.0)), rtol=1e-10)

    def test_deep_satellite_particle_beats_shallow_host_pair(self):
        host = _nfw_halo([0, 0, 0], [0, 0, 0], 1e12, 200.0, 10.0, 1)
        sat = _nfw_halo([50, 0, 0], [0, 0, 0], 1e10, 30.0, 12.0, 2)
        host.set_boundness(np.array([0]), np.array([0.3]), np.ones(1))   # shallow in the host
        sat.set_boundness(np.array([0]), np.array([0.8]), np.ones(1))    # deep in the satellite
        p = EnergyPlausibility()
        p.prepare([host, sat])
        a_host, a_sat = (v[0] for v in p.column_values([1, 2]))
        assert a_sat / (a_sat + a_host) > 0.9

    def test_update_equal_weight_bins_skip_and_state(self):
        p = EnergyPlausibility()
        rng = np.random.default_rng(0)
        rows = np.arange(5000)
        p._t = {7: (rows, rng.exponential(2.0, rows.size))}
        p.update({7: (rows, np.ones(rows.size))})
        state = p.state
        widths = np.diff(state["edges"])
        mass = np.exp(state["log_density"]) * widths
        np.testing.assert_allclose(mass, 1.0 / widths.size, rtol=0.05)   # equal weight
        # Too few members: the previous histogram is kept.
        p._t = {7: (rows[:20], np.ones(20))}
        p.update({7: (rows[:20], np.ones(20))})
        assert p.state["edges"] is state["edges"]
        q = EnergyPlausibility()
        q.set_state(state)
        t = np.array([0.1, 1.0, 5.0])
        np.testing.assert_array_equal(q._log_pstar(t), p._log_pstar(t))
        q.set_state(None)
        assert q.state is None


class TestAssignerWithEnergyPlausibility:
    def test_assign_twice_keeps_every_bound_pair_and_refits(self):
        halos, coords = _two_nfw_halos()
        csc_b, _ = halos.get_particles()
        assert np.intersect1d(*csc_b.column_indices).size > 0          # contested particles
        a = XGMMAssigner(verbose=0, method="bgmm", plausibility="energy")
        assert a.plausibility.state is None
        r1 = a.assign(halos, coords, [[0, 1]], seed=1)
        state = a.plausibility.state
        assert state is not None
        # Every bound pair keeps a positive prior, so it stays in the responsibilities.
        for j, sid in enumerate(r1.responsibilities.column_id):
            k = list(csc_b.column_id).index(sid)
            np.testing.assert_array_equal(r1.responsibilities.column_indices[j], csc_b.column_indices[k])
        r2 = a.assign(halos, coords, [[0, 1]], seed=1, previous_resp=r1.responsibilities)
        assert (r2.particle_df["Sub_tree_id"] > 0).sum() == np.union1d(*csc_b.column_indices).size


def _kepler_halo(xcen, vel, M, rvir, sid):
    return HaloModel(KeplerPotential(M=M), np.asarray(xcen, float), np.asarray(vel, float), rvir,
                     sub_tree_id=sid, redshift=0.0, comoving=False)


def _two_kepler_halos(seed=3):
    """Kepler host and satellite with overlapping bound sets, boundness computed."""
    rng = np.random.default_rng(seed)
    host = _kepler_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, 1)
    sat = _kepler_halo([15, 0, 0], [0, 50, 0], 1e10, 25.0, 2)
    coords = np.vstack([
        np.hstack([rng.normal(0, 5, (300, 3)), rng.normal(0, 30, (300, 3))]),
        np.hstack([rng.normal([15, 0, 0], 1.5, (200, 3)), rng.normal([0, 50, 0], 10, (200, 3))]),
    ])
    halos = HaloEnsemble([host, sat])
    compute_halo_bound_particles(halos, coords)
    return halos, coords


class TestPhaseSpacePlausibility:
    def test_literature_depth_values(self):
        mu, sd, sd_min = literature_depth(_kepler_halo([0, 0, 0], [0, 0, 0], 1e10, 50.0, 1))
        np.testing.assert_allclose([mu, sd, sd_min], [6.482, 1.382, 0.691], rtol=2e-3)
        for c, ref in [(3.0, 17.44), (10.0, 14.88), (30.0, 12.80)]:
            mu, sd, sd_min = literature_depth(_nfw_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, c, 1))
            np.testing.assert_allclose(mu, ref, rtol=1e-2)
            assert 1.5 < sd_min < sd                       # NFW: ~1.8-2.0 floor, ~3.6-4.1 width

    @pytest.mark.parametrize("make", [_two_kepler_halos, _two_nfw_halos])
    def test_alpha_positive_and_literature_only_first(self, make):
        halos, _ = make()
        p = PhaseSpacePlausibility()
        p.prepare(halos)
        for h in halos:
            rows, b, _ = h.get_boundness()
            alpha = p.column_values([h.sub_tree_id])[0]
            assert alpha.shape == rows.shape
            assert np.all(np.isfinite(alpha)) and np.all(alpha >= PLAUSIBILITY_FLOOR * (1 - 1e-12))
            t = -h.log_phase_space_fraction(b.astype(float))
            mu, sd, _ = literature_depth(h)
            ref = (1 - PLAUSIBILITY_FLOOR) * np.exp(-0.5 * ((t - mu) / sd) ** 2 + t) / (sd * np.sqrt(2 * np.pi)) + PLAUSIBILITY_FLOOR
            np.testing.assert_allclose(alpha, ref, rtol=1e-10)

    def test_update_moments_floor_skip_and_state(self):
        rng = np.random.default_rng(0)
        rows = np.arange(5000)
        t = rng.normal(7.0, 2.0, rows.size)
        r = rng.uniform(0.1, 1.0, rows.size)
        p = PhaseSpacePlausibility()
        p._t, p._sd_min = {3: (rows, t)}, 0.5
        p.update({3: (rows, r)})
        mu = np.average(t, weights=r)
        np.testing.assert_allclose([p._mu, p._sd], [mu, np.sqrt(np.average((t - mu) ** 2, weights=r))])
        p._sd_min = 10.0                                     # floor binds
        p.update({3: (rows, r)})
        assert p._sd == 10.0
        state = p.state
        p.update({3: (rows[: PLAUSIBILITY_MIN_MEMBERS // 2], r[: PLAUSIBILITY_MIN_MEMBERS // 2])})
        assert p.state == state                              # too few members: kept
        q = PhaseSpacePlausibility()
        q.set_state(state)
        assert q.state == state
        q.set_state(None)
        assert q.state is None

    def test_deep_satellite_particle_beats_shallow_host_pair(self):
        host = _kepler_halo([0, 0, 0], [0, 0, 0], 1e12, 200.0, 1)
        sat = _kepler_halo([50, 0, 0], [0, 0, 0], 1e10, 30.0, 2)
        host.set_boundness(np.array([0]), np.array([1.0]), np.ones(1))    # shallow in the host
        sat.set_boundness(np.array([0]), np.array([30.0]), np.ones(1))    # deep in the satellite
        p = PhaseSpacePlausibility()
        p.prepare([host, sat])
        a_host, a_sat = (v[0] for v in p.column_values([1, 2]))
        assert a_sat / (a_sat + a_host) > 0.9

    def test_assigner_runs_twice_on_kepler_halos(self):
        halos, coords = _two_kepler_halos()
        csc_b, _ = halos.get_particles()
        assert np.intersect1d(*csc_b.column_indices).size > 0
        a = XGMMAssigner(verbose=0, method="bgmm", plausibility="phase")
        r1 = a.assign(halos, coords, [[0, 1]], seed=1)
        assert a.plausibility.state is not None
        for j, sid in enumerate(r1.responsibilities.column_id):
            k = list(csc_b.column_id).index(sid)
            np.testing.assert_array_equal(r1.responsibilities.column_indices[j], csc_b.column_indices[k])
        r2 = a.assign(halos, coords, [[0, 1]], seed=1, previous_resp=r1.responsibilities)
        assert (r2.particle_df["Sub_tree_id"] > 0).sum() == np.union1d(*csc_b.column_indices).size


# ---------------------------------------------------------------- kinematic
from scipy.integrate import quad
from scipy.stats import kstest

from roadrunner.clustering.assignment.plausibility import (
    KinematicPlausibility,
    _speed_log_ratio,
    escape_speed_fraction,
    literature_speed,
)


def _unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _kin_floored(u2, m):
    """Floored literature-only kinematic alpha (first snapshot)."""
    p = np.exp(_speed_log_ratio(u2, m))
    return np.minimum((1 - PLAUSIBILITY_FLOOR) * p + PLAUSIBILITY_FLOOR, np.exp(50.0))


def _members(rng, h, n, r_scale):
    """Isotropic members of halo h: positions ~ N(centre, r_scale), Wolf dispersion at r_1/2."""
    sigma = np.sqrt(literature_speed(h) * -2.0 * h.potential(np.array([KRAVTSOV_RHALF * h.virial_radius]))[0] / 3.0)
    pos = h.xcen + rng.normal(0.0, r_scale, (n, 3))
    vel = h.velocity + rng.normal(0.0, sigma, (n, 3))
    return np.hstack([pos, vel])


def _alpha_pairs(halos, coords):
    """Literature-only alpha of every particle for every halo it is bound to (u^2 < 1)."""
    for h in halos:
        u2 = escape_speed_fraction(h, np.arange(len(coords)), coords)
        rows = np.flatnonzero(u2 < 1.0)
        h.set_boundness(rows, np.ones(rows.size), np.ones(rows.size))
    p = KinematicPlausibility()
    p.prepare(halos, coords)
    out = {}
    for h in halos:
        a = np.zeros(len(coords))
        a[h.get_boundness()[0]] = p.column_values([h.sub_tree_id])[0]
        out[int(h.sub_tree_id)] = a
    return out


class TestKinematicPlausibility:
    @pytest.mark.parametrize("model", ["kepler", "nfw"])
    @pytest.mark.parametrize("z,comoving", [(0.0, False), (1.5, True)])
    def test_uniform_escape_ball_is_uniform_in_nu_and_matches_boundness(self, model, z, comoving):
        rng = np.random.default_rng(4)
        f = 1.0 / (1.0 + z) if comoving else 1.0
        inner = KeplerPotential(M=1e11) if model == "kepler" else NFWPotential(M=1e11, Rs=60.0 * f / 10.0, c=10.0)
        h = HaloModel(inner, np.array([100.0, 200.0, 300.0]), np.array([50.0, -20.0, 10.0]), 60.0,
                      sub_tree_id=1, redshift=z, comoving=comoving)
        n, r = 20000, 7.0
        v_esc = np.sqrt(-2.0 * h.potential(np.array([r]))[0])
        coords = np.hstack([h.xcen + r * _unit(rng, n),
                            h.velocity + v_esc * rng.uniform(size=(n, 1)) ** (1 / 3) * _unit(rng, n)])
        u2 = escape_speed_fraction(h, np.arange(n), coords)
        assert kstest(u2 ** 1.5, "uniform").pvalue > 1e-3
        ens = HaloEnsemble([h])
        compute_halo_bound_particles(ens, coords)
        rows, b, _ = h.get_boundness()
        e_over_phi = -b.astype(float) * h.binding_energy_scale() / h.potential(np.array([r]))[0]
        np.testing.assert_allclose(1.0 - u2[rows], e_over_phi, rtol=1e-4, atol=1e-5)

    def test_member_model_normalisation_moment_and_null(self):
        for m in (0.01, 0.1, 0.3, 0.59):
            p = lambda nu: np.exp(_speed_log_ratio(np.array([nu ** (2 / 3)]), m))[0]
            norm = quad(p, 0, 1, limit=400, points=[1e-6, 1e-3, 0.1])[0]
            mom = quad(lambda nu: nu ** (2 / 3) * p(nu), 0, 1, limit=400, points=[1e-6, 1e-3, 0.1])[0]
            np.testing.assert_allclose([norm, mom], [1.0, m], rtol=1e-6)
        np.testing.assert_allclose(_speed_log_ratio(np.linspace(0, 0.99, 5), 0.7), 0.0, atol=1e-12)

    def test_literature_speed(self):
        np.testing.assert_allclose(literature_speed(_kepler_halo([0, 0, 0], [0, 0, 0], 1e11, 100.0, 1)), 0.5, rtol=1e-5)
        for c, ref in [(3.0, 0.0108), (10.0, 0.0334), (30.0, 0.0824)]:
            np.testing.assert_allclose(literature_speed(_nfw_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, c, 1)), ref, rtol=1e-2)

    @pytest.mark.parametrize("make", [_two_kepler_halos, _two_nfw_halos])
    def test_first_snapshot_is_floored_literature(self, make):
        halos, coords = make()
        p = KinematicPlausibility()
        p.prepare(halos, coords)
        for h in halos:
            rows = h.get_boundness()[0]
            alpha = p.column_values([h.sub_tree_id])[0]
            assert alpha.shape == rows.shape and np.all(np.isfinite(alpha))
            assert np.all(alpha >= PLAUSIBILITY_FLOOR * (1 - 1e-12))
            ref = _kin_floored(escape_speed_fraction(h, rows, coords), literature_speed(h))
            np.testing.assert_allclose(alpha, ref, rtol=1e-10)

    def test_update_shrinkage_skip_and_state(self):
        rng = np.random.default_rng(0)
        p = KinematicPlausibility()
        big, small = np.arange(5000), np.arange(20)
        u_big, u_small = rng.uniform(0, 0.2, big.size), rng.uniform(0.3, 0.5, small.size)
        p._t = {1: (big, u_big), 2: (small, u_small)}
        w_big, w_small = rng.uniform(0.1, 1, big.size), np.ones(small.size)
        p.update({1: (big, w_big), 2: (small, w_small)})
        pool = np.average(np.r_[u_big, u_small], weights=np.r_[w_big, w_small])
        assert p._m_pool == pytest.approx(pool)
        n_big = w_big.sum() ** 2 / np.dot(w_big, w_big)
        assert p._m[1] == pytest.approx((n_big * np.average(u_big, weights=w_big) + 100 * pool) / (n_big + 100))
        assert p._m[2] == pytest.approx((20 * u_small.mean() + 100 * pool) / 120)
        state = p.state
        p.update({2: (small[:10], w_small[:10])})               # too few members: kept
        assert p.state["m_pool"] == state["m_pool"]
        q = KinematicPlausibility()
        q.set_state(state)
        assert q._m == p._m and q._m_pool == p._m_pool
        q.set_state(None)
        assert q.state is None and q._m == {}

    @staticmethod
    def _host_sat(bulk):
        """Host 1e12 (c 10) at the origin; satellite 1e10 (c 15) at 3 kpc moving with `bulk`."""
        host = _nfw_halo([0, 0, 0], [0, 0, 0], 1e12, 200.0, 10.0, 1)
        sat = _nfw_halo([3, 0, 0], bulk, 1e10, 40.0, 15.0, 2)
        return host, sat

    def test_infalling_satellite_stars_go_to_the_satellite(self):
        rng = np.random.default_rng(1)
        host, _ = self._host_sat([0, 0, 0])
        v_esc = np.sqrt(-2.0 * host.potential(np.array([3.0]))[0])
        host, sat = self._host_sat([-0.9 * v_esc, 0, 0])
        coords = _members(rng, sat, 3000, 0.3)
        a = _alpha_pairs([host, sat], coords)
        share = a[2] / (a[1] + a[2])
        assert np.median(share) > 0.99

    def test_sunk_satellite_is_neutral_and_host_stars_go_to_host(self):
        rng = np.random.default_rng(2)
        host, _ = self._host_sat([0, 0, 0])
        r = np.array([3.0 * np.exp(-1e-3), 3.0, 3.0 * np.exp(1e-3)])
        phi = host.potential(r)
        v_circ = np.sqrt((phi[2] - phi[0]) / 2e-3)
        host, sat = self._host_sat([0, v_circ, 0])
        sat_stars = _members(rng, sat, 3000, 0.3)
        sigma_h = v_circ / np.sqrt(3.0)
        host_stars = np.hstack([sat.xcen + rng.normal(0, 0.3, (3000, 3)), rng.normal(0, sigma_h, (3000, 3))])
        a = _alpha_pairs([host, sat], np.vstack([sat_stars, host_stars]))
        both = (a[1] > 0) & (a[2] > 0)
        s, h = both[:3000], both[3000:]
        assert abs(np.median(np.log(a[2][:3000][s] / a[1][:3000][s]))) < 1.0
        assert np.median(np.log(a[1][3000:][h] / a[2][3000:][h])) > 5.0

    def test_host_centre_newborn_beats_small_neighbour(self):
        host = _nfw_halo([0, 0, 0], [0, 0, 0], 1e12, 200.0, 10.0, 1)
        nb = _nfw_halo([5, 0, 0], [0, 0, 0], 1e9, 20.0, 15.0, 2)
        v_esc_nb = np.sqrt(-2.0 * nb.potential(np.array([5.0]))[0])
        nb = _nfw_halo([5, 0, 0], [0.5 * v_esc_nb, 0, 0], 1e9, 20.0, 15.0, 2)
        coords = np.array([[0.01, 0, 0, 0, 0, 0]], dtype=float)
        a = _alpha_pairs([host, nb], coords)
        assert a[2][0] > 0                                      # the newborn is bound to the neighbour too
        assert a[1][0] / (a[1][0] + a[2][0]) > 0.9

    @pytest.mark.parametrize("make", [_two_kepler_halos, _two_nfw_halos])
    def test_assigner_runs_twice(self, make):
        halos, coords = make()
        csc_b, _ = halos.get_particles()
        a = XGMMAssigner(verbose=0, method="bgmm", plausibility="kinematic")
        r1 = a.assign(halos, coords, [[0, 1]], seed=1)
        assert a.plausibility.state is not None
        for j, sid in enumerate(r1.responsibilities.column_id):
            k = list(csc_b.column_id).index(sid)
            np.testing.assert_array_equal(r1.responsibilities.column_indices[j], csc_b.column_indices[k])
        r2 = a.assign(halos, coords, [[0, 1]], seed=1, previous_resp=r1.responsibilities)
        assert (r2.particle_df["Sub_tree_id"] > 0).sum() == np.union1d(*csc_b.column_indices).size
        assert isinstance(make_plausibility("kinematic"), KinematicPlausibility)

"""Plausibility models: rank bit-identity, energy model, refit and assigner wiring."""

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import rankdata

from roadrunner._defaults import (
    PLAUSIBILITY_FLOOR,
    PLAUSIBILITY_LIT_SCATTER_DEX,
    PLAUSIBILITY_MIN_MEMBERS,
    PLAUSIBILITY_MIN_SCATTER_DEX,
    math_dtype,
    precision,
)
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.clustering.assignment.plausibility import (
    EnergyPlausibility,
    KRAVTSOV_RHALF,
    LITERATURE_RHALF,
    PLAUSIBILITY_MODELS,
    PhaseSpacePlausibility,
    RankPlausibility,
    _FD_STEP,
    _summary_potential,
    kravtsov_half_mass_radius,
    literature_depth,
    make_plausibility,
    stars_half_mass_radius,
    tracer_log_ratio,
)
from roadrunner._mcf_types import SnapshotData
from roadrunner.clustering.sparse import SparseCSC
from roadrunner.mixture._math import row_l1_normalize
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import (
    CompositeSphericalPotential,
    HernquistPotential,
    KeplerPotential,
    NFWPotential,
    PlummerPotential,
    ShellPotential,
)
from roadrunner.physics.scaler import StandardScaler
from roadrunner.pipeline.processing import ProcessingConfig


def _old_rank_transform(values):
    """The removed ``gmm._rank_transform``, kept here as the reference."""
    if len(values) == 0:
        return values
    return np.log1p(rankdata(values, method="ordinal")).astype(math_dtype(), copy=False)


_X = np.array([1.0, 0.0, 0.0])


def _nfw_halo(xcen, vel, M, rvir, c, sid):
    xcen = np.asarray(xcen, float)
    inner = NFWPotential(M=M, Rs=rvir / c, c=c, centre=xcen, velocity=np.asarray(vel, float))
    return HaloModel(inner, xcen, np.asarray(vel, float), rvir,
                     sub_tree_id=sid, redshift=0.0, comoving=False)


def _snap(coords):
    n = coords.shape[0]
    return SnapshotData(
        index=np.arange(n, dtype=np.uint64), mass=np.ones(n),
        position=coords[:, :3], velocity=coords[:, 3:6],
        redshift=0.0, time=0.0,
    )


def _nfw_row(xcen, vel, M, rvir, c, sid):
    return {
        "Sub_tree_id": sid,
        "position_x": xcen[0], "position_y": xcen[1], "position_z": xcen[2],
        "velocity_x": vel[0], "velocity_y": vel[1], "velocity_z": vel[2],
        "mass": M, "virial_radius": rvir, "scale_radius": rvir / c, "Redshift": 0.0,
    }


def _two_nfw_halos(seed=3, stars="none"):
    """A host and a satellite whose bound sets overlap, with boundness computed."""
    rng = np.random.default_rng(seed)
    rows = [
        _nfw_row([0, 0, 0], [0, 0, 0], 1e11, 60.0, 10.0, 1),
        _nfw_row([15, 0, 0], [0, 50, 0], 1e10, 25.0, 12.0, 2),
    ]
    coords = np.vstack([
        np.hstack([rng.normal(0, 5, (300, 3)), rng.normal(0, 30, (300, 3))]),
        np.hstack([rng.normal([15, 0, 0], 1.5, (200, 3)), rng.normal([0, 50, 0], 10, (200, 3))]),
    ])
    config = ProcessingConfig(halo_model="nfw", comoving=False, stars=stars)
    halos = HaloEnsemble(compute_halo_bound_particles(rows, _snap(coords), config))
    return halos, coords


class TestRankPlausibility:
    @pytest.mark.parametrize("math", ["single", "double"])
    def test_latent_prior_bit_identical_to_old_rank_transform(self, math):
        rng = np.random.default_rng(1)
        h1 = HaloModel(KeplerPotential(M=1e12), np.zeros(3), np.zeros(3), 100.0, 1, 0.0)
        h2 = HaloModel(KeplerPotential(M=1e11), np.zeros(3), np.zeros(3), 50.0, 2, 0.0)
        h1.set_bound(np.arange(0, 150), -rng.uniform(0, 5, 150).astype(np.float32), np.ones(150))
        h2.set_bound(np.arange(80, 200), -rng.uniform(0, 5, 120).astype(np.float32), np.ones(120))
        h1.inner_position = h2.inner_position = np.array([1.0, 0.0, 0.0])   # a point mass's scale needs its innermost particle
        ens = HaloEnsemble([h1, h2])
        csc_b, _ = ens.get_particles()
        csc_s, _ = ens.get_particles(scaled_energies=True)     # the old transform ranked b
        a = XGMMAssigner(verbose=0)
        a.previous_resp = {}
        a.plausibility.prepare(ens, None)
        with precision(math=math):        # α is cast to the run's math_dtype by column_values
            csc_a = SparseCSC(csc_b.column_indices, a.plausibility.column_values(csc_b.column_id),
                              column_id=csc_b.column_id)
            old = row_l1_normalize(csc_s.to_dense(col_func=_old_rank_transform)).astype(math_dtype())
            new, _ = a._initial_responsibilities(None, csc_b, csc_a, None, None)
        assert new.dtype == old.dtype
        np.testing.assert_array_equal(new, old)

    def test_registry(self):
        assert isinstance(make_plausibility("rank"), RankPlausibility)
        assert isinstance(make_plausibility("energy"), EnergyPlausibility)
        with pytest.raises(ValueError):
            make_plausibility("nope")
        for name in PLAUSIBILITY_MODELS:                 # every model takes the same constructor arguments
            assert make_plausibility(name)._half_mass_radius is kravtsov_half_mass_radius
            assert make_plausibility(name, "stars")._half_mass_radius is stars_half_mass_radius
        a = XGMMAssigner(verbose=0, plausibility="phase", literature_rhalf="stars", tempering=0.5)
        assert a.plausibility._half_mass_radius is stars_half_mass_radius
        assert a.plausibility.tempering == 0.5

    @pytest.mark.parametrize("name", list(PLAUSIBILITY_MODELS))
    def test_tempering_reads_alpha_to_the_beta(self, name):
        """column_values gives alpha**beta: beta = 1 exactly alpha, 0.5 its root, 0 ones."""
        alpha = np.array([1e-8, 0.3, 4.0, np.exp(50.0)])
        for beta, want in ((1.0, alpha), (0.5, np.sqrt(alpha)), (0.0, np.ones(4))):
            p = make_plausibility(name, tempering=beta)
            p._alpha = {7: alpha}
            got = p.column_values(np.array([7, 8]))
            np.testing.assert_allclose(got[0], want.astype(math_dtype()), rtol=1e-6)
            assert got[1].size == 0
        np.testing.assert_array_equal(make_plausibility(name).column_values(np.array([7]))[0].size, 0)
        p = make_plausibility(name)
        p._alpha = {7: alpha}
        np.testing.assert_array_equal(p.column_values(np.array([7]))[0], alpha.astype(math_dtype()))   # beta = 1: exact


class TestEnergyPlausibility:
    def test_first_snapshot_is_floored_tracer_and_positive(self):
        halos, coords = _two_nfw_halos()
        p = EnergyPlausibility()
        p.prepare(halos, coords)
        for h in halos:
            rows, E, _ = h.get_bound()
            alpha = p.column_values([h.sub_tree_id])[0]
            assert alpha.shape == rows.shape
            assert np.all(np.isfinite(alpha)) and np.all(alpha >= PLAUSIBILITY_FLOOR * (1 - 1e-6))
            ref = (1 - PLAUSIBILITY_FLOOR) * np.exp(tracer_log_ratio(h, E.astype(float), KRAVTSOV_RHALF * h.virial_radius)) + PLAUSIBILITY_FLOOR
            np.testing.assert_allclose(alpha, np.minimum(ref, np.exp(50.0)), rtol=1e-6)

    def test_tracer_like_the_halo_is_its_distribution_function(self):
        # a Hernquist halo and a Hernquist tracer of the same size: P* = f q(0) / M(<r_max), f the halo's own (generic
        # inversion vs the halo's own route); a unit or counting-radius slip in the tracer's r_1/2 would break it
        dm = HernquistPotential(1e11, 4.0)
        h = HaloModel(dm, np.zeros(3), np.zeros(3), 60.0, sub_tree_id=1, redshift=1.0, comoving=True)   # length_scale 1/2
        E = h.central_potential() * np.array([0.9, 0.5, 0.1, 0.01])
        r_half = (1.0 + np.sqrt(2.0)) * dm.a / h.length_scale                                       # catalogue units
        r_max = h._counting_radius()
        q0 = dm._distribution(r_max)._q_and_g(np.zeros(1))[0][0]
        ref = np.log(dm.distribution_function(E) * q0 / (dm.M * (r_max / (r_max + dm.a)) ** 2))
        np.testing.assert_allclose(tracer_log_ratio(h, E, r_half), ref, atol=1e-3)
        with pytest.raises(ValueError, match="literature_rhalf"):
            make_plausibility("energy", "nope")

    def test_tracer_prior_is_normalised_in_w(self):
        # ∫ P* dw = ∫ N* dE = 1 over the bound phase space; P* finite (so positive); without stars "stars" is Kravtsov's
        dm = NFWPotential(1e11, 6.0, 10.0)
        shell = ShellPotential(np.random.default_rng(0).normal(0.0, 0.5, (3000, 3)), 1e6)
        eps = np.unique(np.concatenate([np.geomspace(1e-12, 0.5, 4000), 1.0 - np.geomspace(1e-9, 0.5, 4000)]))
        for stars in ({}, {"stars": shell}):
            h = HaloModel(CompositeSphericalPotential(dark_matter=dm, **stars), np.zeros(3), np.zeros(3), 60.0,
                          sub_tree_id=1, redshift=1.0, comoving=True)
            E = h.central_potential() * eps
            log_ratio = tracer_log_ratio(h, E, LITERATURE_RHALF["stars"](h))
            assert np.all(np.isfinite(log_ratio))
            log_w = h.log_phase_space_fraction(E)
            np.testing.assert_allclose(abs(trapezoid(np.exp(log_ratio + log_w), log_w)), 1.0, rtol=1e-2)

    @pytest.mark.parametrize("stars", ["none", "free"])
    @pytest.mark.parametrize("literature_rhalf", ["kravtsov", "stars"])
    def test_end_to_end_finite_with_and_without_stars(self, stars, literature_rhalf):
        halos, coords = _two_nfw_halos(stars=stars)
        p = EnergyPlausibility(literature_rhalf)
        p.prepare(halos, coords)
        for h in halos:
            assert ("stars" in h.potential_model.groups) == (stars == "free")      # free: an NFW+Shell halo
            alpha = p.column_values([h.sub_tree_id])[0]
            assert alpha.shape == h.get_bound()[0].shape
            assert np.all(np.isfinite(alpha)) and np.all(alpha >= PLAUSIBILITY_FLOOR * (1 - 1e-6))

    def test_deep_satellite_particle_beats_shallow_host_pair(self):
        host = _nfw_halo([0, 0, 0], [0, 0, 0], 1e12, 200.0, 10.0, 1)
        sat = _nfw_halo([50, 0, 0], [0, 0, 0], 1e10, 30.0, 12.0, 2)
        # E / Phi_0 = 0.3 (shallow in the host) and 0.8 (deep in the satellite)
        host.set_bound(np.array([0]), np.array([0.3 * host.central_potential()]), np.ones(1))
        sat.set_bound(np.array([0]), np.array([0.8 * sat.central_potential()]), np.ones(1))
        p = EnergyPlausibility()
        p.prepare([host, sat], None)
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

    def test_energy_statistic_is_the_phase_space_depth(self):
        # the energy model reads t = -ln w, the phase model's statistic (Exp(1) for the uniform background), also with stars
        h = _nfw_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, 10.0, 1)
        E = np.array([0.9, 0.5, 0.1]) * h.central_potential()
        hs = HaloModel(CompositeSphericalPotential(dark_matter=h.potential_model[0], stars=HernquistPotential(5e9, 0.5)),
                       np.zeros(3), np.zeros(3), 60.0, sub_tree_id=2, redshift=0.0, comoving=False)
        for halo in (h, hs):
            t, _, _ = EnergyPlausibility()._log_ratios(halo, np.arange(3), E, None)
            np.testing.assert_array_equal(t, -halo.log_phase_space_fraction(E))


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
    xcen = np.asarray(xcen, float)
    return HaloModel(KeplerPotential(M=M, centre=xcen, velocity=np.asarray(vel, float)), xcen, np.asarray(vel, float), rvir,
                     sub_tree_id=sid, redshift=0.0, comoving=False)


def _kepler_row(xcen, vel, M, rvir, sid):
    return {
        "Sub_tree_id": sid,
        "position_x": xcen[0], "position_y": xcen[1], "position_z": xcen[2],
        "velocity_x": vel[0], "velocity_y": vel[1], "velocity_z": vel[2],
        "mass": M, "virial_radius": rvir, "scale_radius": rvir / 10.0, "Redshift": 0.0,
    }


def _two_kepler_halos(seed=3):
    """Kepler host and satellite with overlapping bound sets, boundness computed."""
    rng = np.random.default_rng(seed)
    rows = [
        _kepler_row([0, 0, 0], [0, 0, 0], 1e11, 60.0, 1),
        _kepler_row([15, 0, 0], [0, 50, 0], 1e10, 25.0, 2),
    ]
    coords = np.vstack([
        np.hstack([rng.normal(0, 5, (300, 3)), rng.normal(0, 30, (300, 3))]),
        np.hstack([rng.normal([15, 0, 0], 1.5, (200, 3)), rng.normal([0, 50, 0], 10, (200, 3))]),
    ])
    config = ProcessingConfig(halo_model="kepler", comoving=False, stars="none")
    halos = HaloEnsemble(compute_halo_bound_particles(rows, _snap(coords), config))
    return halos, coords


class TestPhaseSpacePlausibility:
    def test_literature_depth_values(self):
        h = _kepler_halo([0, 0, 0], [0, 0, 0], 1e10, 50.0, 1)
        mu, sd, sd_min = literature_depth(h, kravtsov_half_mass_radius(h))
        np.testing.assert_allclose([mu, sd, sd_min], [6.482, 1.382, 0.691], rtol=2e-3)
        for c, ref in [(3.0, 17.44), (10.0, 14.88), (30.0, 12.80)]:
            h = _nfw_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, c, 1)
            mu, sd, sd_min = literature_depth(h, kravtsov_half_mass_radius(h))
            np.testing.assert_allclose(mu, ref, rtol=1e-2)
            assert 1.5 < sd_min < sd                       # NFW: ~1.8-2.0 floor, ~3.6-4.1 width

    @pytest.mark.parametrize("make", [_two_kepler_halos, _two_nfw_halos])
    def test_alpha_positive_and_literature_only_first(self, make):
        halos, coords = make()
        p = PhaseSpacePlausibility()
        p.prepare(halos, coords)
        for h in halos:
            rows, E, _ = h.get_bound()
            alpha = p.column_values([h.sub_tree_id])[0]
            assert alpha.shape == rows.shape
            assert np.all(np.isfinite(alpha)) and np.all(alpha >= PLAUSIBILITY_FLOOR * (1 - 1e-6))
            t = -h.log_phase_space_fraction(E.astype(float))
            mu, sd, _ = literature_depth(h, kravtsov_half_mass_radius(h))
            ref = (1 - PLAUSIBILITY_FLOOR) * np.exp(-0.5 * ((t - mu) / sd) ** 2 + t) / (sd * np.sqrt(2 * np.pi)) + PLAUSIBILITY_FLOOR
            np.testing.assert_allclose(alpha, ref, rtol=1e-6)

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
        # E = -G M / r_vir in the host (shallow) and -30 G M / r_vir in the satellite (deep)
        for h, depth in ((host, 1.0), (sat, 30.0)):
            E = -depth * h.tree_mass * KeplerPotential(M=1.0).G / h.virial_radius
            h.set_bound(np.array([0]), np.array([E]), np.ones(1))
        p = PhaseSpacePlausibility()
        p.prepare([host, sat], None)
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
    r_half = np.array([KRAVTSOV_RHALF * h.virial_radius])
    sigma = np.sqrt(literature_speed(h, KRAVTSOV_RHALF * h.virial_radius) * -2.0 * h.potential(h.tree_centre + r_half[:, None] * _X)[0] / 3.0)
    pos = h.tree_centre + rng.normal(0.0, r_scale, (n, 3))
    vel = h.tree_velocity + rng.normal(0.0, sigma, (n, 3))
    return np.hstack([pos, vel])


def _alpha_pairs(halos, coords):
    """Literature-only alpha of every particle for every halo it is bound to (u^2 < 1)."""
    for h in halos:
        u2 = escape_speed_fraction(h, np.arange(len(coords)), coords)
        rows = np.flatnonzero(u2 < 1.0)
        h.set_bound(rows, -np.ones(rows.size), np.ones(rows.size))
    p = KinematicPlausibility()
    p.prepare(halos, coords)
    out = {}
    for h in halos:
        a = np.zeros(len(coords))
        a[h.get_bound()[0]] = p.column_values([h.sub_tree_id])[0]
        out[int(h.sub_tree_id)] = a
    return out


class TestKinematicPlausibility:
    @pytest.mark.parametrize("model", ["kepler", "nfw"])
    @pytest.mark.parametrize("z,comoving", [(0.0, False), (1.5, True)])
    def test_uniform_escape_ball_is_uniform_in_nu_and_matches_boundness(self, model, z, comoving):
        rng = np.random.default_rng(4)
        f = 1.0 / (1.0 + z) if comoving else 1.0
        xcen = np.array([100.0, 200.0, 300.0])
        vbulk = np.array([50.0, -20.0, 10.0])
        inner = (KeplerPotential(M=1e11, centre=xcen * f, velocity=vbulk) if model == "kepler"
                 else NFWPotential(M=1e11, Rs=60.0 * f / 10.0, c=10.0, centre=xcen * f, velocity=vbulk))
        h = HaloModel(inner, xcen, np.array([50.0, -20.0, 10.0]), 60.0,
                      sub_tree_id=1, redshift=z, comoving=comoving)
        n, r = 20000, 7.0
        v_esc = np.sqrt(-2.0 * h.potential(h.tree_centre + np.array([[r, 0.0, 0.0]]))[0])
        coords = np.hstack([h.tree_centre + r * _unit(rng, n),
                            h.tree_velocity + v_esc * rng.uniform(size=(n, 1)) ** (1 / 3) * _unit(rng, n)])
        u2 = escape_speed_fraction(h, np.arange(n), coords)
        assert kstest(u2 ** 1.5, "uniform").pvalue > 1e-3
        tree_row = {
            "Sub_tree_id": 1,
            "position_x": xcen[0], "position_y": xcen[1], "position_z": xcen[2],
            "velocity_x": vbulk[0], "velocity_y": vbulk[1], "velocity_z": vbulk[2],
            "mass": 1e11, "virial_radius": 60.0, "scale_radius": 6.0, "Redshift": z,
        }
        config = ProcessingConfig(halo_model=model, comoving=comoving, stars="none")
        halo = compute_halo_bound_particles([tree_row], _snap(coords), config)[0]
        idx, E, _ = halo.get_bound()
        e_over_phi = E.astype(float) / halo.potential(halo.tree_centre + np.array([[r, 0.0, 0.0]]))[0]
        np.testing.assert_allclose(1.0 - u2[idx], e_over_phi, rtol=1e-4, atol=1e-5)

    def test_member_model_normalisation_moment_and_null(self):
        for m in (0.01, 0.1, 0.3, 0.59):
            p = lambda nu: np.exp(_speed_log_ratio(np.array([nu ** (2 / 3)]), m))[0]
            norm = quad(p, 0, 1, limit=400, points=[1e-6, 1e-3, 0.1])[0]
            mom = quad(lambda nu: nu ** (2 / 3) * p(nu), 0, 1, limit=400, points=[1e-6, 1e-3, 0.1])[0]
            np.testing.assert_allclose([norm, mom], [1.0, m], rtol=1e-6)
        np.testing.assert_allclose(_speed_log_ratio(np.linspace(0, 0.99, 5), 0.7), 0.0, atol=1e-12)

    def test_literature_speed(self):
        h = _kepler_halo([0, 0, 0], [0, 0, 0], 1e11, 100.0, 1)
        np.testing.assert_allclose(literature_speed(h, kravtsov_half_mass_radius(h)), 0.5, rtol=1e-5)
        for c, ref in [(3.0, 0.0108), (10.0, 0.0334), (30.0, 0.0824)]:
            h = _nfw_halo([0, 0, 0], [0, 0, 0], 1e11, 60.0, c, 1)
            np.testing.assert_allclose(literature_speed(h, kravtsov_half_mass_radius(h)), ref, rtol=1e-2)

    @pytest.mark.parametrize("make", [_two_kepler_halos, _two_nfw_halos])
    def test_first_snapshot_is_floored_literature(self, make):
        halos, coords = make()
        p = KinematicPlausibility()
        p.prepare(halos, coords)
        for h in halos:
            rows = h.get_bound()[0]
            alpha = p.column_values([h.sub_tree_id])[0]
            assert alpha.shape == rows.shape and np.all(np.isfinite(alpha))
            assert np.all(alpha >= PLAUSIBILITY_FLOOR * (1 - 1e-6))
            ref = _kin_floored(escape_speed_fraction(h, rows, coords), literature_speed(h, kravtsov_half_mass_radius(h)))
            np.testing.assert_allclose(alpha, ref, rtol=1e-6)

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
        v_esc = np.sqrt(-2.0 * host.potential(host.tree_centre + np.array([[3.0, 0.0, 0.0]]))[0])
        host, sat = self._host_sat([-0.9 * v_esc, 0, 0])
        coords = _members(rng, sat, 3000, 0.3)
        a = _alpha_pairs([host, sat], coords)
        share = a[2] / (a[1] + a[2])
        assert np.median(share) > 0.99

    def test_sunk_satellite_is_neutral_and_host_stars_go_to_host(self):
        rng = np.random.default_rng(2)
        host, _ = self._host_sat([0, 0, 0])
        r = np.array([3.0 * np.exp(-1e-3), 3.0, 3.0 * np.exp(1e-3)])
        phi = host.potential(host.tree_centre + r[:, None] * _X)
        v_circ = np.sqrt((phi[2] - phi[0]) / 2e-3)
        host, sat = self._host_sat([0, v_circ, 0])
        sat_stars = _members(rng, sat, 3000, 0.3)
        sigma_h = v_circ / np.sqrt(3.0)
        host_stars = np.hstack([sat.tree_centre + rng.normal(0, 0.3, (3000, 3)), rng.normal(0, sigma_h, (3000, 3))])
        a = _alpha_pairs([host, sat], np.vstack([sat_stars, host_stars]))
        both = (a[1] > 0) & (a[2] > 0)
        s, h = both[:3000], both[3000:]
        assert abs(np.median(np.log(a[2][:3000][s] / a[1][:3000][s]))) < 1.0
        assert np.median(np.log(a[1][3000:][h] / a[2][3000:][h])) > 5.0

    def test_host_centre_newborn_beats_small_neighbour(self):
        host = _nfw_halo([0, 0, 0], [0, 0, 0], 1e12, 200.0, 10.0, 1)
        nb = _nfw_halo([5, 0, 0], [0, 0, 0], 1e9, 20.0, 15.0, 2)
        v_esc_nb = np.sqrt(-2.0 * nb.potential(nb.tree_centre + np.array([[5.0, 0.0, 0.0]]))[0])
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


def test_literature_summaries_read_the_spherical_profile_about_x_min():
    # offset stars: the summaries are the composite's spherical profile about x_min, with
    # v_c^2 = G M(<r) / r (Gauss on the sphere about x_min; FD error ~2e-7), whatever the offset's
    # direction (the true potential along x from the tree centre differed by ~15% between directions)
    s1, s2 = PLAUSIBILITY_LIT_SCATTER_DEX, PLAUSIBILITY_MIN_SCATTER_DEX
    got = []
    for d in ([0.3, 0.0, 0.0], [0.0, -0.3, 0.0], [0.3 / np.sqrt(3.0)] * 3):
        comp = CompositeSphericalPotential(dark_matter=NFWPotential(1e10, 3.0, 10.0),
                                           stars=PlummerPotential(2e7, 0.4, centre=np.array(d)))
        # comoving at z = 1: r_1/2 is physical (catalogue R_vir times length_scale = 1/2)
        h = HaloModel(comp, np.zeros(3), np.zeros(3), 30.0, sub_tree_id=1, redshift=1.0, comoving=True)
        pm = h.potential_model
        r = KRAVTSOV_RHALF * h.virial_radius * h.length_scale * 10.0 ** np.array([0.0, -s1, s1, -s2, s2])
        phi, vc2 = pm._potential(r), pm.G * pm._enclosed_mass(r) / r
        t = -h.log_phase_space_fraction(phi + 0.5 * vc2)
        depth, speed = literature_depth(h, kravtsov_half_mass_radius(h)), literature_speed(h, kravtsov_half_mass_radius(h))
        np.testing.assert_allclose(depth, [t[0], 0.5 * abs(t[1] - t[2]), 0.5 * abs(t[3] - t[4])], rtol=1e-6)
        np.testing.assert_allclose(speed, vc2[0] / (-2.0 * phi[0]), rtol=1e-6)
        got.append((*depth, speed))
    # x_min sits exactly on the NFW cusp (it beats the Plummer pull), so only |d| matters and the directions agree to rounding
    np.testing.assert_allclose(got[1:], [got[0]] * 2, rtol=1e-12)


def _head_literature_depth(halo):
    """``literature_depth`` at a55592e (``r_1/2 = KRAVTSOV_RHALF R_vir`` hard-wired), kept here as the reference."""
    s1, s2 = PLAUSIBILITY_LIT_SCATTER_DEX, PLAUSIBILITY_MIN_SCATTER_DEX
    r = KRAVTSOV_RHALF * halo.virial_radius * 10.0 ** np.array([0.0, -s1, s1, -s2, s2])
    steps = np.exp(np.array([-_FD_STEP, 0.0, _FD_STEP]))
    phi = _summary_potential(halo, (r[:, None] * steps).ravel()).reshape(r.size, 3)
    v2 = (phi[:, 2] - phi[:, 0]) / (2.0 * _FD_STEP)
    t = -halo.log_phase_space_fraction(phi[:, 1] + 0.5 * v2)
    return float(t[0]), 0.5 * abs(t[1] - t[2]), 0.5 * abs(t[3] - t[4])


def _head_literature_speed(halo):
    """``literature_speed`` at a55592e, kept here as the reference."""
    r = KRAVTSOV_RHALF * halo.virial_radius * np.exp(np.array([-_FD_STEP, 0.0, _FD_STEP]))
    phi = _summary_potential(halo, r)
    return float((phi[2] - phi[0]) / (2.0 * _FD_STEP) / (-2.0 * phi[1]))


class TestLiteratureRhalf:
    def test_kravtsov_reproduces_the_hard_wired_radius_bit_for_bit(self):
        offset = CompositeSphericalPotential(dark_matter=NFWPotential(1e10, 3.0, 10.0),
                                             stars=PlummerPotential(2e7, 0.4, centre=np.array([0.3, 0.0, 0.0])))
        halos = [*_two_kepler_halos()[0], *_two_nfw_halos()[0],
                 HaloModel(offset, np.zeros(3), np.zeros(3), 30.0, sub_tree_id=3, redshift=1.0, comoving=True)]
        for h in halos:
            r_half = LITERATURE_RHALF["kravtsov"](h)
            assert r_half == KRAVTSOV_RHALF * h.virial_radius
            assert literature_depth(h, r_half) == _head_literature_depth(h)
            assert literature_speed(h, r_half) == _head_literature_speed(h)

    def test_stars_reads_the_stars_group_and_falls_back_to_kravtsov(self):
        dm = NFWPotential(1e11, 6.0, 10.0)
        shell = ShellPotential(np.random.default_rng(0).normal(0.0, 0.5, (3000, 3)), 1e6)
        # comoving at z = 1: the radius comes back in catalogue units (physical / length_scale)
        for stars, ref, rtol in (({"stars": HernquistPotential(5e9, 0.5)}, (1.0 + np.sqrt(2.0)) * 0.5, 1e-5),
                                 ({"stars": shell}, shell._scale_radius, 1e-2)):    # a Shell's scale radius: its half-mass radius, to a node
            h = HaloModel(CompositeSphericalPotential(dark_matter=dm, **stars), np.zeros(3), np.zeros(3), 60.0,
                          sub_tree_id=1, redshift=1.0, comoving=True)
            np.testing.assert_allclose(stars_half_mass_radius(h) * h.length_scale, ref, rtol=rtol)
        h = HaloModel(dm, np.zeros(3), np.zeros(3), 60.0, sub_tree_id=1, redshift=1.0, comoving=True)
        assert stars_half_mass_radius(h) == kravtsov_half_mass_radius(h) == KRAVTSOV_RHALF * h.virial_radius

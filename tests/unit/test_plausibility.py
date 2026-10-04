"""Plausibility models: rank bit-identity, energy model, refit and assigner wiring."""

import numpy as np
import pytest
from scipy.stats import rankdata

from roadrunner._defaults import PLAUSIBILITY_FLOOR, math_dtype, precision
from roadrunner.clustering.assignment.gmm import XGMMAssigner
from roadrunner.clustering.assignment.plausibility import (
    EnergyPlausibility,
    KRAVTSOV_RHALF,
    RankPlausibility,
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

import numpy as np
import pandas as pd

from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential
from roadrunner.physics.timescales import (
    compute_particle_dynamical_timescales,
    compute_tidal_radius,
)

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


def _make_halo(xcen, mass=1e12, rvir=RVR, sub_tree_id=1, inner=None):
    if inner is None:
        inner = KeplerPotential(M=mass, G=4.3e-6)
    return HaloModel(
        inner, np.asarray(xcen, dtype=np.float64),
        VCENTER, rvir, sub_tree_id=sub_tree_id, redshift=0.0,
    )


def _make_ensemble(halos):
    # set boundness with tdyn values
    for i, h in enumerate(halos):
        h.set_boundness(
            np.array([i * 10 + j for j in range(10)], dtype=np.uint64),
            np.full(10, 0.5, dtype=np.float32),
            np.full(10, float(i + 1) * 0.05, dtype=np.float32),
        )
    return HaloEnsemble(halos)


class TestDynamicalTimescales:
    def test_returns_same_df(self):
        halos = [_make_halo([0.0, 0.0, 0.0], sub_tree_id=1)]
        ens = _make_ensemble(halos)
        df = pd.DataFrame({"array_index": np.arange(10, dtype=np.uint64), "Sub_tree_id": 1})
        result = compute_particle_dynamical_timescales(df, ens, [[0]])
        assert "timescale" in result.columns
        assert len(result) == 10

    def test_timescale_positive(self):
        halos = [_make_halo([0.0, 0.0, 0.0], sub_tree_id=1)]
        ens = _make_ensemble(halos)
        df = pd.DataFrame({"array_index": np.arange(10, dtype=np.uint64), "Sub_tree_id": 1})
        result = compute_particle_dynamical_timescales(df, ens, [[0]], td_factor=1.0)
        assert np.all(result["timescale"].values > 0)

    def test_timescale_capped(self):
        halos = [_make_halo([0.0, 0.0, 0.0], sub_tree_id=1)]
        big_tdyn = np.full(10, 10.0, dtype=np.float32)
        ens = HaloEnsemble([halos[0]])
        halos[0].set_boundness(
            np.arange(10, dtype=np.uint64),
            np.full(10, 0.5, dtype=np.float32),
            big_tdyn,
        )
        df = pd.DataFrame({"array_index": np.arange(10, dtype=np.uint64), "Sub_tree_id": 1})
        result = compute_particle_dynamical_timescales(df, ens, [[0]], td_factor=1.0)
        assert np.all(result["timescale"].values <= 0.4 + 1e-6)

    def test_single_group_max(self):
        h1 = _make_halo([0.0, 0.0, 0.0], sub_tree_id=1)
        h2 = _make_halo([1.0, 1.0, 1.0], sub_tree_id=2)
        # same particles for both
        h1.set_boundness(
            np.arange(10, dtype=np.uint64),
            np.full(10, 0.5, dtype=np.float32),
            np.full(10, 0.1, dtype=np.float32),
        )
        h2.set_boundness(
            np.arange(10, dtype=np.uint64),
            np.full(10, 0.5, dtype=np.float32),
            np.full(10, 0.3, dtype=np.float32),
        )
        ens = HaloEnsemble([h1, h2])
        df = pd.DataFrame({"array_index": np.arange(10, dtype=np.uint64), "Sub_tree_id": -1})
        df["Sub_tree_id"] = [1, 2] * 5
        result = compute_particle_dynamical_timescales(df, ens, [[0, 1]], td_factor=1.0)
        # each particle should get the max timescale = 0.3 (not capped)
        assert np.allclose(result["timescale"].values, 0.3)

    def test_no_groups(self):
        halos = [_make_halo([0.0, 0.0, 0.0], sub_tree_id=1)]
        ens = _make_ensemble(halos)
        df = pd.DataFrame({"array_index": np.arange(10, dtype=np.uint64), "Sub_tree_id": 1})
        result = compute_particle_dynamical_timescales(df, ens, [], td_factor=1.0)
        assert np.all(result["timescale"].values == 0.0)

    def test_td_factor_applied(self):
        halos = [_make_halo([0.0, 0.0, 0.0], sub_tree_id=1)]
        ens = _make_ensemble(halos)
        df = pd.DataFrame({"array_index": np.arange(10, dtype=np.uint64), "Sub_tree_id": 1})
        r1 = compute_particle_dynamical_timescales(df.copy(), ens, [[0]], td_factor=1.0)
        r2 = compute_particle_dynamical_timescales(df.copy(), ens, [[0]], td_factor=2.0)
        assert np.allclose(r2["timescale"].values, r1["timescale"].values * 2.0, atol=1e-6)


class TestTidalRadius:
    def test_kepler_analytic(self):
        M, Ms = 1e12, 1e10
        D = 100.0
        halo = _make_halo([0.0, 0.0, 0.0], mass=M)
        rt = compute_tidal_radius(halo, Ms, D)
        expected = D * (Ms / (3 * M)) ** (1.0 / 3.0)
        assert np.isclose(rt, expected)

    def test_nfw_finite(self):
        inner = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        halo = _make_halo([0.0, 0.0, 0.0], inner=inner)
        rt = compute_tidal_radius(halo, 1e10, 100.0)
        assert np.isfinite(rt)
        assert rt > 0

    def test_large_distance_larger_rt(self):
        M, Ms = 1e12, 1e10
        halo = _make_halo([0.0, 0.0, 0.0], mass=M)
        rt_near = compute_tidal_radius(halo, Ms, 50.0)
        rt_far = compute_tidal_radius(halo, Ms, 200.0)
        assert rt_far > rt_near

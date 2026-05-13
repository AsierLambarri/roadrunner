import numpy as np
import pandas as pd

from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


class TestConstructor:
    def test_basic_attributes(self):
        inner = KeplerPotential(M=1e12)
        xcen = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.0, 0.0, 0.0])
        halo = HaloModel(inner, xcen, vel, RVR, sub_tree_id=42, redshift=0.5)
        assert halo.sub_tree_id == 42
        assert halo.redshift == 0.5
        assert np.allclose(halo.xcen, xcen)
        assert np.allclose(halo.velocity, vel)
        assert halo.virial_radius == RVR
        assert halo.comoving is True

    def test_non_comoving(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.5, comoving=False)
        assert halo.comoving is False


class TestPotential:
    def test_kepler_matches_direct(self):
        M, G = 1e12, 4.3e-6
        inner = KeplerPotential(M, G=G)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        xyz = np.array([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
        result = halo.potential(xyz)
        expected = inner.potential(np.array([100.0, 200.0]))
        assert np.allclose(result, expected)

    def test_nfw_comoving(self):
        inner = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=3.0)
        xyz = np.array([[40.0, 0.0, 0.0]])
        result = halo.potential(xyz)
        r_phys = 40.0 / 4.0
        expected = inner.potential(np.array([r_phys]))
        assert np.allclose(result, expected)

    def test_multiple_particles(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        xyz = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        result = halo.potential(xyz)
        expected = inner.potential(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result, expected)

    def test_passthrough_radius(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        r = np.array([50.0, 100.0])
        result = halo.potential(r)
        expected = inner.potential(r)
        assert np.allclose(result, expected)

    def test_1d_virial_radius_mode(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.5, comoving=True)
        dist = np.array([10.0, 20.0])
        result = halo.potential(dist)
        expected = inner.potential(dist / 1.5)
        assert np.allclose(result, expected)


class TestBoundness:
    def test_no_boundness_by_default(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        assert not halo.has_boundness
        assert halo.get_boundness() is None

    def test_set_boundness(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        indices = np.array([0, 1, 2], dtype=np.uint64)
        energies = np.array([0.5, 0.8], dtype=np.float64)
        tdyns = np.array([1.0, 2.0], dtype=np.float64)
        halo.set_boundness(indices, energies, tdyns)
        assert halo.has_boundness
        result = halo.get_boundness()
        assert result is not None
        assert np.array_equal(result[0], indices)
        assert np.array_equal(result[1], energies)
        assert np.array_equal(result[2], tdyns)

    def test_set_boundness_overwrites(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        halo.set_boundness(np.array([0]), np.array([0.1]), np.array([1.0]))
        halo.set_boundness(np.array([5]), np.array([0.9]), np.array([2.0]))
        result = halo.get_boundness()
        assert result[0][0] == 5

    def test_get_boundness_returns_tuple(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        halo.set_boundness(np.array([0]), np.array([0.1]), np.array([1.0]))
        result = halo.get_boundness()
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestTidalDenominator:
    def test_kepler(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        xyz = np.array([[10.0, 0.0, 0.0]])
        result = halo.tidal_denominator(xyz)
        assert result == 3 * 1e12


class TestFromSnapshotRow:
    def test_basic_row(self):
        row = pd.Series({
            "Sub_tree_id": 42,
            "position_x": 1.0, "position_y": 2.0, "position_z": 3.0,
            "velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0,
            "mass": 1e12,
            "virial_radius": 100.0,
            "scale_radius": 10.0,
            "Redshift": 0.5,
            "Snapshot": 0,
        })
        halo = HaloModel.from_snapshot_row(row, model="kepler")
        assert halo.sub_tree_id == 42
        assert np.allclose(halo.xcen, [1.0, 2.0, 3.0])
        assert np.isclose(halo.redshift, 0.5)
        assert halo.comoving is True

    def test_nfw_row(self):
        row = pd.Series({
            "Sub_tree_id": 7,
            "position_x": 10.0, "position_y": 20.0, "position_z": 30.0,
            "velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0,
            "mass": 5e11,
            "virial_radius": 80.0,
            "scale_radius": 8.0,
            "Redshift": 1.0,
            "Snapshot": 0,
        })
        halo = HaloModel.from_snapshot_row(row, model="nfw")
        r = np.array([[10.0, 20.0, 30.0]])
        result = halo.potential(r)
        assert np.isscalar(result) or result.size == 1
        assert np.all(np.isfinite(result))

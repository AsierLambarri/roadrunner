"""Ishiyama+21 c_vir(M, z) table against colossus, scoping, dtype/broadcast, and the merger-tree drop-in."""

import csv
import os
import warnings
from collections import defaultdict
from unittest.mock import MagicMock

import numpy as np
import pytest

from roadrunner._exceptions import ConfigurationError
from roadrunner.cosmology import Cosmology, cosmology, current_cosmology
from roadrunner.physics import merger_tree
from roadrunner.pipeline.config import RunConfig
from roadrunner.pipeline.entry import _resolve_cosmology

_TOLERANCE = 6e-3   # measured overall max |c/c_col - 1| (5.209e-3, VINTERGATAN-GM), rounded up to 1 sig fig

_REFERENCE = os.path.join(os.path.dirname(__file__), "..", "data", "ishiyama21_reference.csv")


def _reference_cosmologies():
    rows = defaultdict(list)
    with open(_REFERENCE) as f:
        for row in csv.DictReader(f):
            rows[row["cosmology"]].append(row)
    return rows


@pytest.fixture
def planck18():
    return Cosmology(0.6766, 0.3111, 0.0490, 0.8102, 0.9665)


class TestMatchesColossus:
    @pytest.mark.parametrize("name", list(_reference_cosmologies().keys()))
    def test_matches_colossus(self, name):
        rows = _reference_cosmologies()[name]
        h, om, ob, s8, ns = (float(rows[0][k]) for k in ("h", "omega_m", "omega_b", "sigma8", "n_s"))
        cosmo = Cosmology(h, om, ob, s8, ns)

        log10_m = np.array([float(r["log10_m"]) for r in rows])
        z = np.array([float(r["z"]) for r in rows])
        c_colossus = np.array([float(r["c_colossus"]) for r in rows])

        c = cosmo.concentration(10.0**log10_m, z)
        assert np.max(np.abs(c / c_colossus - 1.0)) < _TOLERANCE


class TestExtrapolationWarns:
    def test_warns_outside_table(self, planck18):
        with pytest.warns(RuntimeWarning, match="2 of 3"):
            c = planck18.concentration(np.array([1e2, 1e12, 1e12]), np.array([1.0, 1.0, 40.0]))
        assert np.all(np.isfinite(c))

    def test_no_warning_inside_table(self, planck18):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            planck18.concentration(np.array([2e3, 1e12, 5e15]), np.array([0.0, 3.0, 29.0]))


class TestDtypeBroadcast:
    def test_scalar(self, planck18):
        c = planck18.concentration(1e10, 0.0)
        assert c.shape == ()
        assert c.dtype == np.float64

    def test_float32(self, planck18):
        c64 = planck18.concentration(1e10, 0.0)
        c32 = planck18.concentration(np.float32(1e10), np.float32(0.0))
        assert c32.dtype == np.float32
        np.testing.assert_allclose(c32, c64, rtol=1e-6)

    def test_int64(self, planck18):
        c = planck18.concentration(np.int64(10**10), 0.0)
        assert c.dtype == np.float64

    def test_broadcast(self, planck18):
        m = np.array([[1e9], [1e10], [1e11]])
        z = np.array([0.0, 1.0, 2.0, 3.0])
        c = planck18.concentration(m, z)
        assert c.shape == (3, 4)
        c_scalar = planck18.concentration(1e9, 1.0)
        np.testing.assert_array_equal(planck18.concentration(np.array([1e9, 1e9]), np.array([1.0, 1.0])),
                                       np.array([c_scalar, c_scalar]))


class TestScope:
    def test_raises_outside(self):
        with pytest.raises(ConfigurationError):
            current_cosmology()

    def test_nested(self, planck18):
        other = Cosmology(0.6932, 0.2865, 0.0463, 0.8200, 0.9608)
        with cosmology(planck18):
            assert current_cosmology() is planck18
            with cosmology(other):
                assert current_cosmology() is other
            assert current_cosmology() is planck18

    def test_raises_after(self, planck18):
        with cosmology(planck18):
            pass
        with pytest.raises(ConfigurationError):
            current_cosmology()


class TestMergerTreeDropIn:
    def test_matches_cosmology_method(self, planck18):
        m_vir = np.array([1e8, 1e10, 1e12])
        z = np.array([0.0, 1.0, 2.0])
        with cosmology(planck18):
            c_merger_tree = merger_tree.concentration(m_vir, z)
        c_cosmology = planck18.concentration(m_vir, z)
        np.testing.assert_array_equal(c_merger_tree, c_cosmology)


def _equiv(max_snapshot=0):
    eq = MagicMock()
    eq.max_snapshot = max_snapshot
    eq.snapshot_path = lambda sid: f"/snap/{sid}"
    return eq


def _reader(params):
    reader = MagicMock()
    reader.read_cosmology.return_value = params
    return reader


class TestResolveCosmology:
    def test_file_and_config_merge(self):
        config = RunConfig(cosmology=dict(omega_b=0.0490, sigma8=0.8102, n_s=0.9665))
        cosmo = _resolve_cosmology(config, _reader({"h": 0.6766, "omega_m": 0.3111}), _equiv())
        assert cosmo == Cosmology(0.6766, 0.3111, 0.0490, 0.8102, 0.9665)

    def test_clash_warns_and_config_wins(self):
        config = RunConfig(cosmology=dict(h=0.7, omega_b=0.0490, sigma8=0.8102, n_s=0.9665))
        with pytest.warns(UserWarning, match="overrides the snapshot's"):
            cosmo = _resolve_cosmology(config, _reader({"h": 0.6766, "omega_m": 0.3111}), _equiv())
        assert cosmo.h == 0.7

    def test_missing_parameter_raises(self):
        config = RunConfig()
        with pytest.raises(ConfigurationError, match="sigma8"):
            _resolve_cosmology(config, _reader({"h": 0.6766, "omega_m": 0.3111}), _equiv())


class TestRunConfigCosmologyKeys:
    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown keys"):
            RunConfig(cosmology={"Omega_m": 0.3})

"""Shared fixtures for the roadrunner test suite."""

import pytest

from roadrunner.cosmology import Cosmology, cosmology


@pytest.fixture
def cosmology_scope():
    """Scope a Planck18-like Cosmology for the duration of a test."""
    with cosmology(Cosmology(0.6766, 0.3111, 0.0490, 0.8102, 0.9665)):
        yield

import numpy as np

from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


def _make_halo(xcen, model="kepler", mass=1e12, rvir=RVR, redshift=0.0, vel=VCENTER):
    inner = KeplerPotential(M=mass, G=4.3e-6) if model == "kepler" else NFWPotential(M=mass, Rs=10.0, c=10.0, G=4.3e-6)
    return HaloModel(inner, np.asarray(xcen, dtype=np.float64), vel, rvir, sub_tree_id=1, redshift=redshift)


class TestFunction:
    def test_halo_far_from_particles(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(-50, 50, (100, 6)).astype(np.float64)
        halos = [_make_halo([1e6, 0.0, 0.0])]
        result = compute_halo_bound_particles(halos, coords)
        assert result is halos
        inds, vals, tdyns = result[0].get_boundness()
        assert len(inds) == 0
        assert len(vals) == 0
        assert len(tdyns) == 0

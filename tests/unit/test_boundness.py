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
    def test_v_vir_sq_respects_comoving_flag(self):
        # C08: comoving halo (virial_radius comoving, needs converting)
        # and an equivalent already-physical halo (same physical radius,
        # comoving=False) must give identical boundness energies for the
        # same particles -- search radii chosen so both pick up the same
        # small, well-inside-both-radii particle set.
        rng = np.random.default_rng(0)
        z = 1.0
        factor = 1 + z  # comoving = physical * factor
        positions_physical = rng.uniform(-2, 2, (20, 3))
        velocities = rng.uniform(-5, 5, (20, 3))
        coords_physical = np.column_stack([positions_physical, velocities]).astype(np.float64)
        coords_comoving = np.column_stack([positions_physical * factor, velocities]).astype(np.float64)

        inner_comoving = KeplerPotential(M=1e12, G=4.3e-6)
        inner_physical = KeplerPotential(M=1e12, G=4.3e-6)
        halo_comoving = HaloModel(inner_comoving, VCENTER, VCENTER, 100.0 * factor,
                                   sub_tree_id=1, redshift=z, comoving=True)
        halo_physical = HaloModel(inner_physical, VCENTER, VCENTER, 100.0,
                                   sub_tree_id=1, redshift=z, comoving=False)

        compute_halo_bound_particles([halo_comoving], coords_comoving, search_factor=0.2)
        compute_halo_bound_particles([halo_physical], coords_physical, search_factor=0.2)

        i_com, e_com, _ = halo_comoving.get_boundness()
        i_phy, e_phy, _ = halo_physical.get_boundness()
        assert i_com.size > 0
        np.testing.assert_array_equal(i_com, i_phy)
        np.testing.assert_allclose(e_com, e_phy, rtol=1e-6)

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

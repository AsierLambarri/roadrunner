import numpy as np

from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.potentials import KeplerPotential, NFWPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


def _make_halo(xcen, model="kepler", mass=1e12, rvir=RVR, redshift=0.0, vel=VCENTER):
    xcen = np.asarray(xcen, dtype=np.float64)
    centre = xcen / (1 + redshift)
    inner = (KeplerPotential(M=mass, G=4.3e-6, centre=centre) if model == "kepler"
             else NFWPotential(M=mass, Rs=10.0, c=10.0, G=4.3e-6, centre=centre))
    return HaloModel(inner, xcen, vel, rvir, sub_tree_id=1, redshift=redshift)


class TestFunction:
    def test_unbound_particles_raise_no_overflow(self):
        # I03: the Kepler period is evaluated for bound particles only.
        import warnings
        rng = np.random.default_rng(1)
        vel = np.r_[rng.uniform(-50, 50, (100, 3)), rng.uniform(-5e3, 5e3, (100, 3))]
        coords = np.column_stack([rng.uniform(-50, 50, (200, 3)), vel]).astype(np.float32)
        halo = _make_halo(VCENTER)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compute_halo_bound_particles([halo], coords)
        idx, _, tdyn = halo.get_boundness()
        assert idx.size and np.all(np.isfinite(tdyn)) and np.all(tdyn > 0)

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
                                   sub_tree_id=1, redshift=z, comoving=True, search_factor=0.2)
        halo_physical = HaloModel(inner_physical, VCENTER, VCENTER, 100.0,
                                   sub_tree_id=1, redshift=z, comoving=False, search_factor=0.2)

        compute_halo_bound_particles([halo_comoving], coords_comoving)
        compute_halo_bound_particles([halo_physical], coords_physical)

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

    def test_energy_scale_is_the_well_depth(self):
        # |Φ(0)| for NFW; for Kepler |Φ| at the innermost particle, set before the scale is formed.
        # Every particle lies at or outside the innermost one, so the stored boundness is at most 1.
        rng = np.random.default_rng(3)
        coords = np.column_stack([rng.uniform(-30, 30, (300, 3)), rng.uniform(-20, 20, (300, 3))])
        kepler, nfw = _make_halo(VCENTER), _make_halo(VCENTER, model="nfw")
        compute_halo_bound_particles([kepler, nfw], coords)
        i_min = np.argmin(np.linalg.norm(coords[:, :3], axis=1))
        np.testing.assert_array_equal(kepler.inner_position, coords[i_min, :3])
        np.testing.assert_allclose(kepler.binding_energy_scale(), -kepler.potential_model.potential(coords[i_min:i_min + 1, :3])[0])
        np.testing.assert_allclose(nfw.binding_energy_scale(), -nfw.central_potential())
        for h in (kepler, nfw):
            b = h.get_boundness()[1]
            assert b.size and np.all((b > 0) & (b <= 1))

import numpy as np
import pandas as pd
import pytest

from roadrunner.physics.halo_model import HaloModel, physical_length_factor, tree_potential
from roadrunner.physics.potentials import KeplerPotential, NFWPotential, PlummerPotential

VCENTER = np.array([0.0, 0.0, 0.0])
RVR = 100.0


class TestConstructor:
    def test_basic_attributes(self):
        xcen = np.array([1.0, 2.0, 3.0])
        inner = KeplerPotential(M=1e12, centre=xcen / 1.5)
        vel = np.array([0.0, 0.0, 0.0])
        halo = HaloModel(inner, xcen, vel, RVR, sub_tree_id=42, redshift=0.5)
        assert halo.sub_tree_id == 42
        assert halo.redshift == 0.5
        assert np.allclose(halo.tree_centre, xcen)
        assert np.allclose(halo.tree_velocity, vel)
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
        assert np.allclose(halo.potential(xyz), inner._potential(np.array([100.0, 200.0])))

    def test_nfw_comoving_and_offset_components(self):
        from roadrunner._defaults import precision
        from roadrunner.physics.potentials import CompositeSphericalPotential
        with precision(math="double"):
            z, ls = 3.0, 0.25
            xcen = np.array([100.0, 0.0, 0.0])
            inner = NFWPotential(M=1e12, Rs=10.0, c=10.0, G=4.3e-6, centre=xcen * ls)
            # a second component at its own centre, 36 snapshot units (9 kpc physical) from the particle
            star = PlummerPotential(M=1e9, a=0.5, centre=(xcen + np.array([4.0, 0.0, 0.0])) * ls, G=4.3e-6)
            comp = CompositeSphericalPotential(dark_matter=inner, stars=star)
            halo = HaloModel(comp, xcen, VCENTER, RVR, sub_tree_id=1, redshift=z)
            xyz = xcen + np.array([[40.0, 0.0, 0.0]])
            r_phys = 40.0 * ls
            expected = inner._potential(np.array([r_phys]))
            single = HaloModel(inner, xcen, VCENTER, RVR, sub_tree_id=1, redshift=z)
            np.testing.assert_allclose(single.potential(xyz), expected, rtol=1e-12)
            np.testing.assert_allclose(halo.potential(xyz), expected + star._potential(np.array([9.0])), rtol=1e-12)
            v = np.array([[0.0, 30.0, 0.0]])
            np.testing.assert_allclose(halo.compute_energy(xyz, v),
                                       expected + star._potential(np.array([9.0])) + 450.0, rtol=1e-12)
            t2 = 4.3e-6 * (inner._enclosed_mass(np.array([r_phys])) / r_phys**3
                           + star._enclosed_mass(np.array([9.0])) / 9.0**3) / (2 * np.pi) ** 2
            np.testing.assert_allclose(halo.orbital_time(np.array([-1.0]), xyz), 1.0 / np.sqrt(t2), rtol=1e-10)
            assert halo.orbital_time(np.array([-1.0]), xcen[None])[0] == 0   # the NFW cusp's limit at its centre
            np.testing.assert_allclose(halo.central_potential(), inner.central_potential() + star.potential(inner.centre[None])[0])
            assert halo.tree_mass == 1e12
            assert halo.binding_energy_scale() == -halo.central_potential()       # the true bottom of the summed well (D3)
            assert np.all(np.isfinite(halo.log_phase_space_density(np.array([-0.5]) * halo.binding_energy_scale())))
            # the halo holds its own frozen copy: adding to comp afterwards leaves it untouched
            comp.add("gas", PlummerPotential(M=1e7, a=0.2, centre=xcen * ls, G=4.3e-6))
            assert len(halo) == 2

    def test_release_tables_forwards_to_the_potential_model(self):
        # U6: a composite (not a single NFW, which reads the per-process tables and never caches)
        from roadrunner.physics.potentials import CompositeSphericalPotential
        inner = NFWPotential(M=1e12, Rs=10.0, c=10.0)
        star = PlummerPotential(M=1e9, a=0.5, centre=np.array([4.0, 0.0, 0.0]))
        comp = CompositeSphericalPotential(dark_matter=inner, stars=star)
        halo = HaloModel(comp, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        before = halo.log_phase_space_fraction(np.array([-0.5]) * halo.binding_energy_scale())
        assert halo.potential_model._distributions
        halo.release_tables()
        assert not halo.potential_model._distributions
        np.testing.assert_array_equal(halo.log_phase_space_fraction(np.array([-0.5]) * halo.binding_energy_scale()), before)


class TestBoundness:
    def test_no_bound_by_default(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        assert not halo.has_bound
        assert halo.get_bound() is None

    def test_set_bound(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        indices = np.array([0, 1, 2], dtype=np.uint64)
        energies = np.array([-0.5, -0.8], dtype=np.float64)
        tdyns = np.array([1.0, 2.0], dtype=np.float64)
        halo.set_bound(indices, energies, tdyns)
        assert halo.has_bound
        result = halo.get_bound()
        assert result is not None
        assert np.array_equal(result[0], indices)
        assert np.array_equal(result[1], energies)
        assert np.array_equal(result[2], tdyns)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_set_bound_overwrites(self):
        inner = KeplerPotential(M=1e12)
        halo = HaloModel(inner, np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        halo.set_bound(np.array([0]), np.array([0.1]), np.array([1.0]))
        halo.set_bound(np.array([5]), np.array([0.9]), np.array([2.0]))
        result = halo.get_bound()
        assert result[0][0] == 5

    def test_get_bound_scaled_energies(self):
        halo = HaloModel(NFWPotential(M=1e12, Rs=10.0, c=10.0), np.zeros(3), VCENTER, RVR, sub_tree_id=1, redshift=0.0)
        E = (-np.array([0.5, 0.8]) * halo.binding_energy_scale()).astype(np.float32)
        tdyns = np.array([1.0, 2.0], dtype=np.float32)
        halo.set_bound(np.array([3, 7]), E, tdyns)
        assert halo.get_bound()[1] is E                         # stored physical energies
        idx, b, t = halo.get_bound(scaled_energies=True)
        np.testing.assert_array_equal(idx, [3, 7])
        assert b.dtype == np.float32 and t is tdyns
        np.testing.assert_allclose(b, [0.5, 0.8], rtol=1e-6)


class TestTidalDenominator:
    def test_kepler(self):
        inner = KeplerPotential(M=1e12)
        assert inner.tidal_denominator(np.array([[10.0, 0.0, 0.0]]))[0] == 3 * 1e12


class TestTreePotential:
    def test_basic_row(self):
        row = pd.Series({
            "Sub_tree_id": 42,
            "position_x": 1.0, "position_y": 2.0, "position_z": 3.0,
            "velocity_x": 1.0, "velocity_y": -2.0, "velocity_z": 3.0,
            "mass": 1e12,
            "virial_radius": 100.0,
            "scale_radius": 10.0,
            "Redshift": 0.5,
            "Snapshot": 0,
        })
        ls = physical_length_factor(row["Redshift"], comoving=True)
        pot = tree_potential(row, "kepler", ls)
        assert isinstance(pot, KeplerPotential)
        assert pot.M == 1e12
        np.testing.assert_allclose(pot.centre, np.array([1.0, 2.0, 3.0]) * ls)
        np.testing.assert_allclose(pot.velocity, [1.0, -2.0, 3.0])    # the row's velocity, unscaled

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
        ls = physical_length_factor(row["Redshift"], comoving=True)
        pot = tree_potential(row, "nfw", ls)
        result = pot.potential(np.array([[10.0, 20.0, 30.0]]) * ls)
        assert np.isscalar(result) or result.size == 1
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(pot.centre, [5.0, 10.0, 15.0])
        assert np.isclose(tree_potential(row, "plummer", ls).a, 4.0)

    def test_nfw_row_rs_respects_comoving_flag(self):
        # C08: comoving=False must not also halve Rs -- same physical Rs
        # either way, whether the input is comoving (converted here) or
        # already physical (passed straight through).
        z = 1.0
        row_comoving = pd.Series({
            "Sub_tree_id": 1, "position_x": 0.0, "position_y": 0.0, "position_z": 0.0,
            "velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0,
            "mass": 1e12, "virial_radius": 100.0, "scale_radius": 20.0,
            "Redshift": z, "Snapshot": 0,
        })
        row_physical = row_comoving.copy()
        row_physical["scale_radius"] = 20.0 / (1 + z)  # already physical

        pot_comoving = tree_potential(row_comoving, "nfw", physical_length_factor(z, comoving=True))
        pot_physical = tree_potential(row_physical, "nfw", physical_length_factor(z, comoving=False))
        assert np.isclose(pot_comoving.Rs, pot_physical.Rs)


class TestWrappers:
    def test_catalogue_coordinates_delegate_to_the_potential_model(self):
        from roadrunner._defaults import precision
        from roadrunner.physics.potentials import CompositeSphericalPotential
        with precision(math="double"):
            z, ls = 1.0, 0.5
            xcen = np.array([200.0, -40.0, 10.0])
            inner = NFWPotential(M=1e12, Rs=10.0, c=10.0, centre=xcen * ls)
            halo = HaloModel(inner, xcen, VCENTER, RVR, sub_tree_id=1, redshift=z, search_factor=2.0)
            xyz = xcen + np.array([[30.0, 0.0, 0.0], [0.0, -5.0, 2.0]])
            phys = xyz * ls
            for name in ("potential", "density", "enclosed_mass", "tidal_denominator"):
                np.testing.assert_allclose(getattr(halo, name)(xyz), getattr(inner, name)(phys), rtol=1e-12)
            np.testing.assert_allclose(halo.orbital_time(-np.ones(2), xyz), inner.orbital_time(-np.ones(2), phys), rtol=1e-12)
            np.testing.assert_allclose(halo.central_potential(), inner.central_potential())
            E = np.array([0.9, 0.5]) * inner.central_potential()
            R = 2.0 * RVR * ls
            for name in ("log_phase_space_density", "log_phase_space_fraction"):
                np.testing.assert_allclose(getattr(halo, name)(E), getattr(inner, name)(E, R), rtol=1e-12)
            np.testing.assert_allclose(halo.distribution_function(E), inner.distribution_function(E), rtol=1e-12)
            np.testing.assert_array_equal(halo.log_phase_space_density(E, density="dark_matter"), inner.log_phase_space_density(E, R))
            np.testing.assert_array_equal(halo.distribution_function(E, density="dark_matter"), inner.distribution_function(E))
            assert len(halo) == 1 and halo[0] is inner and list(halo) == [inner]
            star = PlummerPotential(M=1e9, a=0.5, centre=xcen * ls)
            halo = HaloModel(CompositeSphericalPotential(dark_matter=inner, stars=star), xcen, VCENTER, RVR,
                             sub_tree_id=1, redshift=z, search_factor=2.0)
            assert len(halo) == 2 and halo[0] is inner and halo[1] is star


class TestCentreOfMass:
    def test_com_frame_follows_the_components(self):
        from roadrunner._defaults import precision
        from roadrunner.physics.potentials import CompositeSphericalPotential
        with precision(math="double"):
            z, ls = 1.0, 0.5
            pos, vel = np.array([200.0, -40.0, 10.0]), np.array([100.0, 0.0, -30.0])
            inner = NFWPotential(M=9e11, Rs=10.0, c=10.0, centre=pos * ls, velocity=vel)
            single = HaloModel(inner, pos, vel, RVR, sub_tree_id=1, redshift=z)
            np.testing.assert_allclose(single.com_centre, pos)
            np.testing.assert_allclose(single.com_velocity, vel)
            # a displaced, moving component: the centre of mass and the energy frame follow it
            star = PlummerPotential(M=1e11, a=0.5, centre=(pos + np.array([20.0, 0.0, 0.0])) * ls,
                                    velocity=vel + np.array([0.0, 50.0, 0.0]))
            comp = CompositeSphericalPotential(dark_matter=inner, stars=star)
            halo = HaloModel(comp, pos, vel, RVR, sub_tree_id=1, redshift=z)
            assert halo.M == 1e12 and halo.tree_mass == 9e11
            np.testing.assert_allclose(halo.com_centre, pos + np.array([2.0, 0.0, 0.0]))
            np.testing.assert_allclose(halo.com_velocity, vel + np.array([0.0, 5.0, 0.0]))
            np.testing.assert_array_equal(halo.tree_centre, pos)
            np.testing.assert_array_equal(halo.tree_velocity, vel)
            xyz, v = pos[None] + np.array([[5.0, 0.0, 0.0]]), vel[None] + np.array([[0.0, 5.0, 0.0]])
            np.testing.assert_allclose(halo.compute_energy(xyz, v), halo.potential(xyz), rtol=1e-12)

    def test_row_gives_the_potential_its_velocity(self):
        # TestTreePotential.test_basic_row checks the potential itself gets the row's
        # velocity; this checks HaloModel.com_velocity forwards it for a single component.
        row = pd.Series({
            "Sub_tree_id": 7, "position_x": 10.0, "position_y": 20.0, "position_z": 30.0,
            "velocity_x": 1.0, "velocity_y": -2.0, "velocity_z": 3.0,
            "mass": 5e11, "virial_radius": 80.0, "scale_radius": 8.0, "Redshift": 1.0, "Snapshot": 0,
        })
        ls = physical_length_factor(row["Redshift"], comoving=True)
        inner = tree_potential(row, "nfw", ls)
        tree_centre = np.array([row["position_x"], row["position_y"], row["position_z"]])
        tree_velocity = np.array([row["velocity_x"], row["velocity_y"], row["velocity_z"]])
        halo = HaloModel(inner, tree_centre, tree_velocity, row["virial_radius"],
                         sub_tree_id=int(row["Sub_tree_id"]), redshift=row["Redshift"])
        np.testing.assert_allclose(halo.com_velocity, [1.0, -2.0, 3.0])

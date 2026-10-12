import numpy as np

from roadrunner._defaults import STARS_ITERATIONS
from roadrunner._mcf_types import SnapshotData
from roadrunner.mixture._math import row_squared_norms
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_model import tree_potential
from roadrunner.physics.potentials import CompositeSphericalPotential, ShellPotential
from roadrunner.pipeline.processing import ProcessingConfig
from roadrunner.postprocessing.centering import ssc_center

VCENTER = [0.0, 0.0, 0.0]
RVR = 100.0


def _row(position, velocity, mass, rvir, rs, hid, redshift=0.0):
    return {
        "Sub_tree_id": hid,
        "position_x": position[0], "position_y": position[1], "position_z": position[2],
        "velocity_x": velocity[0], "velocity_y": velocity[1], "velocity_z": velocity[2],
        "mass": mass, "virial_radius": rvir, "scale_radius": rs, "Redshift": redshift,
    }


def _snap(coords, masses=None):
    n = coords.shape[0]
    return SnapshotData(
        index=np.arange(n, dtype=np.uint64),
        mass=np.ones(n) if masses is None else masses,
        position=coords[:, :3],
        velocity=coords[:, 3:6],
        redshift=0.0,
        time=0.0,
    )


class TestFunction:
    def test_unbound_particles_raise_no_overflow(self):
        # I03: the Kepler period is evaluated for bound particles only.
        import warnings
        rng = np.random.default_rng(1)
        vel = np.r_[rng.uniform(-50, 50, (100, 3)), rng.uniform(-5e3, 5e3, (100, 3))]
        coords = np.column_stack([rng.uniform(-50, 50, (200, 3)), vel]).astype(np.float32)
        row = _row(VCENTER, VCENTER, 1e12, RVR, 10.0, 1)
        snap_data = _snap(coords)
        config = ProcessingConfig(halo_model="kepler", stars="none")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            halo = compute_halo_bound_particles([row], snap_data, config)[0]
        idx, _, tdyn = halo.get_bound()
        assert idx.size and np.all(np.isfinite(tdyn)) and np.all(tdyn > 0)
        # The halo still carries the row's identity and the config's comoving flag.
        assert halo.sub_tree_id == 1
        assert halo.redshift == 0.0
        assert halo.comoving is True

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

        row_comoving = _row(VCENTER, VCENTER, 1e12, 100.0 * factor, 10.0, 1, redshift=z)
        row_physical = _row(VCENTER, VCENTER, 1e12, 100.0, 10.0, 1, redshift=z)
        config_comoving = ProcessingConfig(halo_model="kepler", comoving=True, search_factor=0.2, stars="none")
        config_physical = ProcessingConfig(halo_model="kepler", comoving=False, search_factor=0.2, stars="none")

        halo_comoving = compute_halo_bound_particles([row_comoving], _snap(coords_comoving), config_comoving)[0]
        halo_physical = compute_halo_bound_particles([row_physical], _snap(coords_physical), config_physical)[0]

        i_com, e_com, _ = halo_comoving.get_bound()
        i_phy, e_phy, _ = halo_physical.get_bound()
        assert i_com.size > 0
        np.testing.assert_array_equal(i_com, i_phy)
        np.testing.assert_allclose(e_com, e_phy, rtol=1e-6)

    def test_halo_far_from_particles(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(-50, 50, (100, 6)).astype(np.float64)
        row = _row([1e6, 0.0, 0.0], VCENTER, 1e12, RVR, 10.0, 1)
        config = ProcessingConfig(halo_model="kepler", stars="none")
        halo = compute_halo_bound_particles([row], _snap(coords), config)[0]
        inds, vals, tdyns = halo.get_bound()
        assert len(inds) == 0
        assert len(vals) == 0
        assert len(tdyns) == 0

    def test_energy_scale_is_the_well_depth(self):
        # |Φ(0)| for NFW; for Kepler the unsoftened G M / r at the innermost particle, set before the scale is formed.
        # Every particle lies at or outside the innermost one, so the stored boundness is at most 1.
        rng = np.random.default_rng(3)
        coords = np.column_stack([rng.uniform(-30, 30, (300, 3)), rng.uniform(-20, 20, (300, 3))])
        snap_data = _snap(coords)
        config_kepler = ProcessingConfig(halo_model="kepler", stars="none")
        config_nfw = ProcessingConfig(halo_model="nfw", stars="none")
        kepler = compute_halo_bound_particles([_row(VCENTER, VCENTER, 1e12, RVR, 10.0, 1)], snap_data, config_kepler)[0]
        nfw = compute_halo_bound_particles([_row(VCENTER, VCENTER, 1e12, RVR, 10.0, 2)], snap_data, config_nfw)[0]
        i_min = np.argmin(np.linalg.norm(coords[:, :3], axis=1))
        np.testing.assert_array_equal(kepler.inner_position, coords[i_min, :3])
        k = kepler.potential_model[0]                 # unsoftened point mass at the innermost particle
        np.testing.assert_allclose(kepler.binding_energy_scale(), k.G * k.M / np.linalg.norm(coords[i_min, :3] - k.centre))
        np.testing.assert_allclose(nfw.binding_energy_scale(), -nfw.central_potential())
        for h in (kepler, nfw):
            b = h.get_bound(scaled_energies=True)[1]
            assert b.size and np.all((b > 0) & (b <= 1))


def _clump(centre, offset=(0.5, 0.0, 0.0), seed=7, n=2000, vel_scale=110.0, pos_scale=0.05, mass=5e5):
    """A tight clump of ``n`` particles ``offset`` from ``centre`` (catalogue units).

    Tuned (NFW_ROW's well + this velocity scale and total clump mass, 1e9 Msun) so the bound
    fraction genuinely climbs iteration by iteration -- about 18% dark-matter-alone, 49%, 82%,
    93% by the final Shell -- instead of saturating at 100% after the first rebuild, so the
    self-consistent loop's intermediate bound sets actually differ from one another.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(centre) + np.asarray(offset) + rng.normal(scale=pos_scale, size=(n, 3))
    v = rng.normal(scale=vel_scale, size=(n, 3))
    coords = np.column_stack([x, v])
    return coords, np.full(n, mass)


class TestStars:
    """NFW row + a 2000-particle clump 0.5 kpc off, bound by its self-gravity (D2)."""

    ROW = _row(VCENTER, VCENTER, 1e10, 50.0, 5.0, 1)

    def test_none_matches_dark_matter_alone(self):
        coords, masses = _clump(VCENTER)
        config = ProcessingConfig(halo_model="nfw", comoving=False, stars="none")
        halo = compute_halo_bound_particles([self.ROW], _snap(coords, masses), config)[0]
        assert len(halo) == 1

        dm = tree_potential(self.ROW, "nfw", 1.0)
        x, v = coords[:, :3], coords[:, 3:6]
        E_ref = dm.potential(x) + 0.5 * row_squared_norms(v - dm.velocity)
        mask = E_ref < 0

        idx, E, _ = halo.get_bound()
        np.testing.assert_array_equal(idx, np.flatnonzero(mask))
        np.testing.assert_allclose(E, E_ref[mask], rtol=1e-6)

    def test_free_shell_centre_is_the_second_to_last_bound_set(self):
        coords, masses = _clump(VCENTER)
        x, v, m = coords[:, :3], coords[:, 3:6], masses
        config = ProcessingConfig(halo_model="nfw", comoving=False, stars="free")
        halo = compute_halo_bound_particles([self.ROW], _snap(coords, masses), config)[0]

        assert set(halo.potential_model.groups) == {"dark_matter", "stars"}
        stars_group = halo.potential_model.groups["stars"]
        assert len(stars_group) == 1

        # Replay the self-consistent loop up to the second-to-last bound set
        # (the one that builds the final, 3rd Shell).
        dm = tree_potential(self.ROW, "nfw", 1.0)
        energy = lambda p: p.potential(x) + 0.5 * row_squared_norms(v - p.com_velocity)
        potential = CompositeSphericalPotential(dark_matter=dm)
        E = energy(potential)
        b_dm = E < 0
        for _ in range(STARS_ITERATIONS - 1):
            b = E < 0
            c, w = ssc_center(x[b], v[b], m[b])
            shell = ShellPotential(x[b], m[b], G=dm.G, centre=c, velocity=w)
            potential = CompositeSphericalPotential(dark_matter=dm, stars=shell)
            E = energy(potential)
        second_to_last = E < 0
        assert 0 < second_to_last.sum() < len(second_to_last)        # not saturated: a real, partial bound set
        expected_centre, _ = ssc_center(x[second_to_last], v[second_to_last], m[second_to_last])
        np.testing.assert_allclose(stars_group[0].centre, expected_centre)

        # Dark matter alone binds strictly fewer particles than dark matter plus the stars it pulls in.
        idx, _, _ = halo.get_bound()
        assert np.all(np.isin(np.flatnonzero(b_dm), idx))
        assert idx.size > b_dm.sum()
        # the stored energies are the built halo's own (no recomputation, same formula and cast)
        _, E_stored, _ = halo.get_bound()
        np.testing.assert_array_equal(E_stored, halo.compute_energy(coords[idx, :3], coords[idx, 3:6]))

    def test_pinned_shell_sits_at_the_tree_centre(self):
        row = _row([10.0, 20.0, 30.0], VCENTER, 1e10, 50.0, 5.0, 1, redshift=1.0)
        coords, masses = _clump([10.0, 20.0, 30.0])
        config = ProcessingConfig(halo_model="nfw", comoving=True, stars="pinned")
        halo = compute_halo_bound_particles([row], _snap(coords, masses), config)[0]
        stars_group = halo.potential_model.groups["stars"]
        np.testing.assert_allclose(stars_group[0].centre, np.array([10.0, 20.0, 30.0]) * 0.5)

    def test_empty_search_sphere(self):
        rng = np.random.default_rng(5)
        coords = rng.uniform(-50, 50, (50, 6))
        row = _row([1e6, 0.0, 0.0], VCENTER, 1e10, 50.0, 5.0, 1)
        config = ProcessingConfig(halo_model="nfw", stars="free")
        halo = compute_halo_bound_particles([row], _snap(coords), config)[0]
        assert len(halo) == 1  # dark matter alone, nothing bound
        idx, E, tdyn = halo.get_bound()
        assert idx.size == 0 and E.size == 0 and tdyn.size == 0

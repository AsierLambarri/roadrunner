import numpy as np
import pandas as pd
import pytest

from roadrunner.postprocessing.properties import (
    random_lines_of_sight,
    rotation_matrix_from_los,
    find_center,
    enclosed_mass_radius,
    half_mass_radius,
    projected_half_mass_radius,
    velocity_dispersion,
    line_of_sight_velocity_dispersion,
    compute_galaxy_properties,
)


class TestRandomLinesOfSight:
    def test_returns_unit_vectors(self):
        N = 100
        vectors = random_lines_of_sight(N, seed=0)
        norms = np.linalg.norm(vectors, axis=1)
        assert norms.shape == (N,)
        assert np.allclose(norms, 1.0)

    def test_half_sphere_default(self):
        vectors = random_lines_of_sight(1000, seed=0)
        # all z should be >= 0 (cos_theta in [0, 1])
        assert np.all(vectors[:, 2] >= 0)

    def test_reproducible_seed(self):
        v1 = random_lines_of_sight(50, seed=42)
        v2 = random_lines_of_sight(50, seed=42)
        assert np.array_equal(v1, v2)

    def test_different_seeds_different(self):
        v1 = random_lines_of_sight(50, seed=0)
        v2 = random_lines_of_sight(50, seed=1)
        assert not np.array_equal(v1, v2)

    def test_N_zero_returns_empty(self):
        vectors = random_lines_of_sight(0, seed=0)
        assert vectors.shape == (0, 3)

    def test_N_one_returns_single_vector(self):
        vectors = random_lines_of_sight(1, seed=0)
        assert vectors.shape == (1, 3)
        assert np.isclose(np.linalg.norm(vectors[0]), 1.0)

    def test_coverage_uniform(self):
        vectors = random_lines_of_sight(5000, half_sphere=False, seed=0)
        # mean of unit vectors on sphere should be approx zero
        mean_pos = np.mean(vectors, axis=0)
        assert np.allclose(mean_pos, 0.0, atol=0.1)


class TestRotationMatrixFromLos:
    # NOTE: tolerances are 1e-6 (not 1e-10): rotation matrices follow the
    # ambient math precision, which is float32 unless scoped otherwise.
    def test_aligns_los_with_z(self):
        los = np.array([1.0, 2.0, 3.0])
        R = rotation_matrix_from_los(los)
        rot_los = R @ (los / np.linalg.norm(los))
        assert np.allclose(rot_los, [0, 0, 1], atol=1e-6)

    def test_orthogonal(self):
        los = np.array([0.5, -1.0, 0.3])
        R = rotation_matrix_from_los(los)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)

    def test_determinant_one(self):
        los = np.array([-2.0, 1.0, 0.5])
        R = rotation_matrix_from_los(los)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_negative_z(self):
        los = np.array([0.0, 0.0, -1.0])
        R = rotation_matrix_from_los(los)
        rot_los = R @ (los / np.linalg.norm(los))
        assert np.allclose(rot_los, [0, 0, 1], atol=1e-6)

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="3-element"):
            rotation_matrix_from_los(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="3-element"):
            rotation_matrix_from_los(np.array([[1.0, 2.0, 3.0]]))

class TestCentering:
    def test_known_center(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(loc=[10.0, 20.0, 30.0], scale=2.0, size=(200, 3))
        vel = rng.normal(loc=[0.0, 0.0, 0.0], scale=100, size=(200, 3))
        masses = np.ones(200)
        cpos, cvel = find_center(pos, vel, masses)
        assert np.allclose(cpos, [10.0, 20.0, 30.0], atol=1.0)
        assert np.allclose(cvel, [0.0, 0.0, 0.0], atol=10.0)

    def test_mass_weighted_shift(self):
        rng = np.random.default_rng(42)
        # all particles near origin; massive ones are inside the 0.5×quantile cut
        n_light, n_heavy = 200, 50
        pos = np.vstack([
            rng.normal(0, 1, (n_light, 3)),       # light, at origin
            rng.normal(0.5, 0.3, (n_heavy, 3)),   # heavy, slightly offset
        ])
        masses = np.zeros(n_light + n_heavy)
        masses[:n_light] = 1.0
        masses[n_light:] = 10.0
        vel = rng.normal(0, 100, (n_light + n_heavy, 3))
        cpos, _ = find_center(pos, vel, masses)
        # unweighted center would be near origin; mass weighting pulls toward heavy clump
        unweighted = np.average(pos, axis=0)
        assert np.linalg.norm(cpos - unweighted) > 0.01

    def test_single_particle(self):
        pos = np.array([[5.0, 6.0, 7.0]])
        vel = np.array([[10.0, 20.0, 30.0]])
        masses = np.array([1.0])
        cpos, cvel = find_center(pos, vel, masses)
        assert np.allclose(cpos, [5.0, 6.0, 7.0])
        assert np.allclose(cvel, [10.0, 20.0, 30.0])


class TestEnclosedMassRadius:
    def test_known_radial_profile(self):
        # 100 particles at radii 0..99, equal masses → half-mass at r=50
        radii = np.arange(100, dtype=np.float64)
        masses = np.ones(100)
        r50 = enclosed_mass_radius(radii, masses, 0.5)
        assert np.isclose(r50, 50.0, atol=1.0)

    def test_mass_fraction_0_returns_min_radius(self):
        radii = np.array([1.0, 5.0, 10.0])
        masses = np.ones(3)
        assert enclosed_mass_radius(radii, masses, 0.0) == 1.0

    def test_mass_fraction_1_returns_max_radius(self):
        radii = np.array([1.0, 5.0, 10.0])
        masses = np.ones(3)
        assert enclosed_mass_radius(radii, masses, 1.0) == 10.0

    def test_zero_total_mass(self):
        radii = np.array([1.0, 2.0, 3.0])
        masses = np.zeros(3)
        assert enclosed_mass_radius(radii, masses, 0.5) == 0.0

    def test_invalid_mass_fraction_raises(self):
        radii = np.array([1.0, 2.0])
        masses = np.ones(2)
        with pytest.raises(ValueError, match="mass_fraction"):
            enclosed_mass_radius(radii, masses, 1.5)
        with pytest.raises(ValueError, match="mass_fraction"):
            enclosed_mass_radius(radii, masses, -0.1)

    def test_single_particle(self):
        assert enclosed_mass_radius(np.array([5.0]), np.array([1.0]), 0.5) == 5.0


class TestHalfMassRadius:
    def test_two_equal_masses(self):
        pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        masses = np.array([1.0, 1.0])
        r50 = half_mass_radius(pos, masses, center=np.array([0.0, 0.0, 0.0]))
        # radii = [0, 10], cum_mass = [0.5, 1.0], searchsorted(0.5) → 1,
        # interpolation: r = 0 + (0.5-0.5)/(1.0-0.5)*(10-0) = 0
        assert r50 == 0.0  # half-mass is at the inner particle


class TestProjectedHalfMassRadius:
    def test_sphere_vs_projected(self):
        # for a spherical distribution, projected r50 < 3D r50
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 3.0, (500, 3))
        masses = np.ones(500)
        center = np.zeros(3)
        los = np.eye(3)  # identity rotation
        r50_3d = half_mass_radius(pos, masses, center)
        r50_2d = projected_half_mass_radius(pos, masses, center, los[np.newaxis])
        assert r50_2d < r50_3d

    def test_single_los_returns_array(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 3.0, (100, 3))
        masses = np.ones(100)
        center = np.zeros(3)
        los = np.eye(3)[np.newaxis]
        result = projected_half_mass_radius(pos, masses, center, los)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_different_los_different_radii(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 3.0, (500, 3))
        masses = np.ones(500)
        center = np.zeros(3)
        # two different rotations
        R1 = np.eye(3)
        R2 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # permute axes
        los_matrices = np.array([R1, R2])
        result = projected_half_mass_radius(pos, masses, center, los_matrices)
        assert not np.isclose(result[0], result[1], atol=0.01)
        assert result.shape == (2,)


class TestVelocityDispersion:
    def test_known_velocities(self):
        vel = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        # σ_x = 1.414, σ_y = 0, σ_z = 0 → σ = 1.414
        sigma = velocity_dispersion(vel)
        assert np.isclose(sigma, np.sqrt(2), atol=1e-6)

    def test_single_particle(self):
        vel = np.array([[10.0, 20.0, 30.0]])
        sigma = velocity_dispersion(vel)
        # std of single point is 0 with ddof=1 → nan
        assert np.isnan(sigma) or sigma == 0.0

    def test_all_identical(self):
        vel = np.tile([100.0, 200.0, 300.0], (10, 1))
        sigma = velocity_dispersion(vel)
        assert sigma == 0.0


class TestLineOfSightVelocityDispersion:
    def test_single_particle_in_aperture(self):
        pos = np.array([[0.0, 0.0, 0.0]])
        vel = np.array([[100.0, 0.0, 0.0]])
        los = np.eye(3)[np.newaxis]
        apertures = np.array([10.0])
        sigma = line_of_sight_velocity_dispersion(pos, vel, los, apertures)
        # std of single point with ddof=1 → nan
        assert np.isnan(sigma[0])

    def test_multiple_particles_in_aperture(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        vel = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 20.0], [0.0, 0.0, 30.0]])
        los = np.eye(3)[np.newaxis]
        apertures = np.array([5.0])
        sigma = line_of_sight_velocity_dispersion(pos, vel, los, apertures)
        assert np.isclose(sigma[0], np.std([10.0, 20.0, 30.0], ddof=1))

    def test_no_particles_in_aperture(self):
        pos = np.array([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
        vel = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        los = np.eye(3)[np.newaxis]
        apertures = np.array([1.0])
        sigma = line_of_sight_velocity_dispersion(pos, vel, los, apertures)
        assert np.isnan(sigma[0])

    def test_projection_removes_xy_velocity(self):
        # particle with pure xy velocity → v_los = 0 after rot
        pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        vel = np.array([[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]])
        los = np.eye(3)[np.newaxis]
        apertures = np.array([1.0])
        sigma = line_of_sight_velocity_dispersion(pos, vel, los, apertures)
        # the std of [0, 0] with ddof=1 → 0
        assert np.isclose(sigma[0], 0.0)

    def test_multiple_los(self):
        rng = np.random.default_rng(42)
        pos = rng.normal(0, 1, (100, 3))
        vel = rng.normal(0, 100, (100, 3))
        los_matrices = np.array([np.eye(3), np.eye(3)])
        apertures = np.array([5.0, 5.0])
        sigma = line_of_sight_velocity_dispersion(pos, vel, los_matrices, apertures)
        assert sigma.shape == (2,)

    def test_mismatched_shapes_raises(self):
        pos = np.zeros((10, 3))
        vel = np.zeros((10, 3))
        los = np.eye(3)[np.newaxis]
        apertures = np.array([1.0, 2.0])  # 2 apertures, 1 LOS
        with pytest.raises(ValueError, match="match"):
            line_of_sight_velocity_dispersion(pos, vel, los, apertures)


class TestComputeGalaxyProperties:
    def test_output_columns(self):
        rng = np.random.default_rng(42)
        N = 200
        masses = rng.uniform(0.1, 1.0, N)
        coords = np.column_stack([
            rng.normal(0, 10, N), rng.normal(0, 10, N), rng.normal(0, 10, N),
            rng.normal(0, 50, N), rng.normal(0, 50, N), rng.normal(0, 50, N),
        ])
        galaxy_particles = {10: np.arange(0, 100), 20: np.arange(100, 200)}
        galaxy_table = pd.DataFrame({
            "host_id": [1, 1],
            "mass": [5e10, 1e10],
            "distance_to_acc_id": [50.0, 120.0],
        }, index=pd.Index([10, 20], name="Sub_tree_id"))
        host_props = pd.Series({
            "mass": 1e12, "scale_radius": 10.0,
            "virial_radius": 200.0, "Redshift": 0.1,
        })
        result = compute_galaxy_properties(
            accretion_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_bound=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model="kepler",
            n_los=3,
        )
        expected_cols = [
            "Sub_tree_id", "mb_host_id",
            "position_x", "position_y", "position_z",
            "velocity_x", "velocity_y", "velocity_z",
            "Mtot", "r20", "rh", "r80", "Rhp", "sigma", "sigma_los", "r_t",
        ]
        assert list(result.columns) == expected_cols
        assert len(result) == 2
        assert set(result["Sub_tree_id"]) == {10, 20}

        row10 = result[result["Sub_tree_id"] == 10].iloc[0]
        assert row10["Mtot"] == pytest.approx(masses[:100].sum())
        assert row10["r_t"] > 0
        assert not np.isnan(row10["rh"])
        assert not np.isnan(row10["sigma"])
        assert not np.isnan(row10["sigma_los"])

    def test_tidal_radius_uses_consistent_physical_frame(self):
        # C08: host_Rs was already converted to physical, but distance
        # wasn't, so they were combined in the wrong frame. Compute the
        # expected r_t directly (correctly, by hand) and compare.
        from roadrunner.physics.potentials import get_potential
        from roadrunner.physics.timescales import compute_tidal_radius

        z = 1.0
        distance_comoving = 100.0
        host_scale_radius_comoving = 20.0
        host_virial_radius = 200.0
        host_mass = 1e12
        sat_mass = 1e9

        factor = 1.0 / (1.0 + z)
        host_Rs_physical = host_scale_radius_comoving * factor
        host_c = host_virial_radius / host_scale_radius_comoving
        physical_potential = get_potential("nfw", M=host_mass, Rs=host_Rs_physical, c=host_c)
        expected_rt = compute_tidal_radius(
            physical_potential, sat_mass, distance_comoving * factor
        ) / factor

        galaxy_particles = {10: np.arange(50)}
        galaxy_table = pd.DataFrame({
            "host_id": [1], "mass": [sat_mass],
            "distance_to_acc_id": [distance_comoving],
        }, index=pd.Index([10], name="Sub_tree_id"))
        host_props = pd.Series({
            "mass": host_mass, "scale_radius": host_scale_radius_comoving,
            "virial_radius": host_virial_radius, "Redshift": z,
        })
        rng = np.random.default_rng(1)
        coords = np.column_stack([
            rng.normal(0, 1, (50, 3)), rng.normal(0, 10, (50, 3)),
        ])
        masses = np.ones(50)

        result = compute_galaxy_properties(
            accretion_id=1, particle_masses=masses, particle_coords=coords,
            galaxy_bound=galaxy_particles, galaxy_table=galaxy_table,
            host_props=host_props, halo_model="nfw", n_los=3,
        )
        assert result.iloc[0]["r_t"] == pytest.approx(expected_rt, rel=1e-6)

    def test_n_los_one_and_large_host_id_exact(self):
        rng = np.random.default_rng(42)
        N = 200
        masses = rng.uniform(0.1, 1.0, N)
        coords = np.column_stack([
            rng.normal(0, 10, N), rng.normal(0, 10, N), rng.normal(0, 10, N),
            rng.normal(0, 50, N), rng.normal(0, 50, N), rng.normal(0, 50, N),
        ])
        big_host_id = 2**53 + 1
        galaxy_particles = {10: np.arange(0, 100), 20: np.arange(100, 200)}
        galaxy_table = pd.DataFrame({
            "host_id": pd.array([1, big_host_id], dtype="int64"),
            "mass": pd.array([5e10, 1e10], dtype="float64"),
            "distance_to_acc_id": pd.array([50.0, 120.0], dtype="float64"),
        }, index=pd.Index([10, 20], name="Sub_tree_id"))
        host_props = pd.Series({
            "mass": 1e12, "scale_radius": 10.0,
            "virial_radius": 200.0, "Redshift": 0.1,
        })
        result = compute_galaxy_properties(
            accretion_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_bound=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model="kepler",
            n_los=1,
        )
        result = result.set_index("Sub_tree_id")
        assert np.isfinite(result.at[10, "Rhp"])
        assert np.isfinite(result.at[10, "sigma_los"])
        # read the mb_host_id column directly: a `.loc[row]` slice would
        # upcast to a single (float) dtype across the row's other float
        # columns, masking the very precision loss this test checks for.
        assert int(result.at[20, "mb_host_id"]) == big_host_id

    def test_few_particles_returns_nan_properties(self):
        masses = np.array([1.0, 1.0, 1.0])
        coords = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]])
        galaxy_particles = {7: np.array([0, 1, 2])}
        galaxy_table = pd.DataFrame({
            "host_id": [1], "mass": [1e10],
            "distance_to_acc_id": [100.0],
        }, index=pd.Index([7], name="Sub_tree_id"))
        host_props = pd.Series({
            "mass": 1e12, "scale_radius": 10.0,
            "virial_radius": 200.0, "Redshift": 0.1,
        })
        result = compute_galaxy_properties(
            accretion_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_bound=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model="kepler",
            n_los=3,
        )
        row = result.iloc[0]
        assert row["rh"] is np.nan or np.isnan(row["rh"])
        assert row["sigma"] is np.nan or np.isnan(row["sigma"])

    def test_very_few_particles_returns_nan_all(self):
        masses = np.array([1.0, 1.0])
        coords = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        galaxy_particles = {7: np.array([0, 1])}
        galaxy_table = pd.DataFrame({
            "host_id": [1], "mass": [1e10],
            "distance_to_acc_id": [100.0],
        }, index=pd.Index([7], name="Sub_tree_id"))
        host_props = pd.Series({
            "mass": 1e12, "scale_radius": 10.0,
            "virial_radius": 200.0, "Redshift": 0.1,
        })
        result = compute_galaxy_properties(
            accretion_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_bound=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model="kepler",
            n_los=3,
        )
        row = result.iloc[0]
        assert np.isnan(row["position_x"])
        assert np.isnan(row["sigma"])

    def test_empty_galaxy_skipped(self):
        galaxy_particles = {10: np.array([], dtype=int), 20: np.arange(30)}
        masses = np.ones(30)
        coords = np.column_stack([np.zeros(30), np.zeros(30), np.zeros(30),
                                  np.zeros(30), np.zeros(30), np.zeros(30)])
        galaxy_table = pd.DataFrame({
            "host_id": [1, 1], "mass": [1e10, 2e10],
            "distance_to_acc_id": [50.0, 100.0],
        }, index=pd.Index([10, 20], name="Sub_tree_id"))
        host_props = pd.Series({
            "mass": 1e12, "scale_radius": 10.0,
            "virial_radius": 200.0, "Redshift": 0.1,
        })
        result = compute_galaxy_properties(
            accretion_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_bound=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model="kepler",
            n_los=3,
        )
        assert len(result) == 1
        assert result.iloc[0]["Sub_tree_id"] == 20

    def test_minus_one_removed(self):
        galaxy_particles = {-1: np.arange(20), 5: np.arange(20, 50)}
        masses = np.ones(50)
        coords = np.column_stack([np.zeros(50), np.zeros(50), np.zeros(50),
                                  np.zeros(50), np.zeros(50), np.zeros(50)])
        galaxy_table = pd.DataFrame({
            "host_id": [1], "mass": [1e10],
            "distance_to_acc_id": [100.0],
        }, index=pd.Index([5], name="Sub_tree_id"))
        host_props = pd.Series({
            "mass": 1e12, "scale_radius": 10.0,
            "virial_radius": 200.0, "Redshift": 0.1,
        })
        result = compute_galaxy_properties(
            accretion_id=1,
            particle_masses=masses,
            particle_coords=coords,
            galaxy_bound=galaxy_particles,
            galaxy_table=galaxy_table,
            host_props=host_props,
            halo_model="kepler",
            n_los=3,
        )
        assert len(result) == 1
        assert result.iloc[0]["Sub_tree_id"] == 5

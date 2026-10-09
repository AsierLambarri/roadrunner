"""Compare old and new boundness implementations.

Both should produce identical bound-particle indices and energies.
"""

import numpy as np
from scipy.spatial import KDTree

from roadrunner.physics.constants import G_KM
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.physics.halo_model import HaloModel


def old_boundness(halos, particle_coordinates, search_factor=1.0):
    """Original boundness: vel_mags <= v_esc, tdyn from instantaneous r."""
    positions = particle_coordinates[:, :3]
    velocities = particle_coordinates[:, 3:6]
    tree = KDTree(positions)
    empty_idx = np.array([], dtype=np.uint64)
    empty_val = np.array([], dtype=np.float32)

    for halo in halos:
        local = np.asarray(
            tree.query_ball_point(
                halo.tree_position, r=search_factor * halo.virial_radius, workers=-1
            )
        )
        if local.size == 0:
            halo.set_boundness(empty_idx, empty_val, empty_val)
            continue

        pos, vel = positions[local], velocities[local]
        rel_vel = vel - halo.com_velocity
        vel_mags = np.linalg.norm(rel_vel, axis=1)

        halo.inner_position = pos[np.argmin(np.linalg.norm(pos - halo.tree_position, axis=1))]
        v_vir_sq = halo.binding_energy_scale()                   # the halo-owned scale
        phi = halo.potential(pos)
        v_esc = np.sqrt(2 * np.abs(phi))
        boundness = 0.5 * (v_esc**2 - vel_mags**2) / v_vir_sq
        tdyns = halo.orbital_time(phi + 0.5 * vel_mags**2, pos)

        bound = vel_mags <= v_esc
        valid = local[bound]
        if valid.size == 0:
            halo.set_boundness(empty_idx, empty_val, empty_val)
        else:
            halo.set_boundness(
                valid.astype(np.uint64),
                boundness[bound].astype(np.float32),
                tdyns[bound].astype(np.float32),
            )
    return halos


def _make_halo(position, velocity, mass, rvir, rs, hid, redshift=0.1,
               model="kepler", search_factor=1.0):
    kwargs = {"M": mass, "G": G_KM}
    if model == "nfw":
        kwargs["Rs"] = rs / (1 + redshift)
        kwargs["c"] = rvir / rs
    kwargs["centre"] = np.asarray(position, dtype=np.float64) / (1 + redshift)
    kwargs["velocity"] = np.asarray(velocity, dtype=np.float64)
    from roadrunner.physics.potentials import get_potential
    inner = get_potential(model, **kwargs)
    return HaloModel(
        inner=inner, tree_position=position, tree_velocity=velocity,
        virial_radius=rvir, sub_tree_id=hid, redshift=redshift,
        comoving=True, search_factor=search_factor,
    )


class TestBoundnessRegression:
    def test_kepler_same_indices_and_energies(self):
        rng = np.random.default_rng(42)
        n_particles = 5000
        coords = np.column_stack([
            rng.uniform(-50, 50, (n_particles, 3)),
            rng.uniform(-100, 100, (n_particles, 3)),
        ])

        halos_old = [
            _make_halo(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
                       1e10, 50.0, 5.0, 1),
            _make_halo(np.array([20.0, 10.0, 5.0]), np.array([30.0, 0.0, 0.0]),
                       5e9, 30.0, 3.0, 2),
        ]
        halos_new = [
            _make_halo(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
                       1e10, 50.0, 5.0, 1, search_factor=2.0),
            _make_halo(np.array([20.0, 10.0, 5.0]), np.array([30.0, 0.0, 0.0]),
                       5e9, 30.0, 3.0, 2, search_factor=2.0),
        ]

        old_result = old_boundness(halos_old, coords, search_factor=2.0)
        new_result = compute_halo_bound_particles(halos_new, coords)

        for h_old, h_new in zip(old_result, new_result):
            i_old, e_old, t_old = h_old.get_boundness()
            i_new, e_new, t_new = h_new.get_boundness()
            # `new` is sorted by particle index (C06); `old` (this file's
            # frozen local reference) is not -- sort both the same way so
            # this compares the underlying sets/values, not incidental order.
            order = np.argsort(i_old)
            i_old, e_old = i_old[order], e_old[order]
            np.testing.assert_array_equal(i_old, i_new,
                                          err_msg="Bound particle indices differ (Kepler)")
            np.testing.assert_allclose(e_old, e_new, atol=1e-5,
                                       err_msg="Boundness energies differ (Kepler)")
            # tdyn differs: old uses instantaneous r, new uses semi-major axis.
            # Both should be finite for bound particles.
            if i_new.size > 0:
                assert np.all(np.isfinite(t_new)), \
                    "tdyn has non-finite values for bound particles (Kepler)"
                assert np.all(t_new > 0), \
                    "Bound particles should have positive tdyn (Kepler)"

    def test_nfw_same_indices_and_energies(self):
        rng = np.random.default_rng(123)
        n_particles = 3000
        coords = np.column_stack([
            rng.uniform(-60, 60, (n_particles, 3)),
            rng.uniform(-200, 200, (n_particles, 3)),
        ])

        halos_old = [
            _make_halo(np.array([5.0, -3.0, 2.0]), np.array([10.0, 0.0, 5.0]),
                       2e10, 60.0, 8.0, 1, model="nfw"),
            _make_halo(np.array([-10.0, 15.0, 0.0]), np.array([-5.0, 10.0, 0.0]),
                       8e9, 40.0, 6.0, 2, model="nfw"),
        ]
        halos_new = [
            _make_halo(np.array([5.0, -3.0, 2.0]), np.array([10.0, 0.0, 5.0]),
                       2e10, 60.0, 8.0, 1, model="nfw"),
            _make_halo(np.array([-10.0, 15.0, 0.0]), np.array([-5.0, 10.0, 0.0]),
                       8e9, 40.0, 6.0, 2, model="nfw"),
        ]

        old_result = old_boundness(halos_old, coords)
        new_result = compute_halo_bound_particles(halos_new, coords)

        for h_old, h_new in zip(old_result, new_result):
            i_old, e_old, t_old = h_old.get_boundness()
            i_new, e_new, t_new = h_new.get_boundness()
            # `new` is sorted by particle index (C06); `old` (this file's
            # frozen local reference) is not -- sort both the same way so
            # this compares the underlying sets/values, not incidental order.
            order = np.argsort(i_old)
            i_old, e_old, t_old = i_old[order], e_old[order], t_old[order]
            np.testing.assert_array_equal(i_old, i_new,
                                          err_msg="Bound particle indices differ (NFW)")
            np.testing.assert_allclose(e_old, e_new, atol=1e-5,
                                       err_msg="Boundness energies differ (NFW)")
            # NFW: both use instantaneous radius → tdyn should match
            np.testing.assert_allclose(t_old, t_new, atol=1e-5,
                                       err_msg="tdyns differ (NFW)")

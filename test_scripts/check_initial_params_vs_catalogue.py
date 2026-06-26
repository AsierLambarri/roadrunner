import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from roadrunner.physics.halo_ensemble import HaloEnsemble
from roadrunner.physics.halo_model import HaloModel
from roadrunner.physics.boundness import compute_halo_bound_particles
from roadrunner.clustering.assignment.gmm import XGMMAssigner

data_dir = sys.argv[1] if len(sys.argv) > 1 else "test_data/mock_simulation"
mt = pd.read_csv(os.path.join(data_dir, "merger_tree.csv"))

for snap_k in sorted(mt["Snapshot"].unique()):
    p = np.load(os.path.join(data_dir, f"particles_{snap_k:03d}.npz"))
    coords = p["coords"]
    snap_mt = mt[mt["Snapshot"] == snap_k]

    halos = [HaloModel.from_snapshot_row(row, model="kepler", comoving=True)
             for _, row in snap_mt.iterrows()]
    ensemble = HaloEnsemble(halos)
    compute_halo_bound_particles(ensemble, coords, search_factor=1.0)
    csc_b, _ = ensemble.get_particles()

    cat_pos = {int(s): ensemble.positions[i]
               for i, s in enumerate(ensemble.sub_tree_ids)}
    cat_vel = {int(s): ensemble.velocities[i]
               for i, s in enumerate(ensemble.sub_tree_ids)}

    assigner = XGMMAssigner(method="gmm", verbose=0)
    assigner.ensemble = ensemble
    assigner.particle_coords = coords
    assigner.newborn_indices = np.array([], dtype=np.int64)
    assigner.previous_resp = {}

    pop_idx = ensemble.populated_indices()
    print(f"\nSnap {snap_k}: {len(pop_idx)} populated groups")

    from roadrunner.physics.scaler import StandardScaler

    all_dpos, all_dvel = [], []
    for gi in pop_idx:
        sub_ens = ensemble.select([gi])
        sub_csc_b, _ = sub_ens.get_particles()
        gp_idx = np.unique(np.concatenate(sub_csc_b.column_indices))
        sid = int(sub_ens.sub_tree_ids[0])

        scaler = StandardScaler()
        group_coords = scaler.fit_transform(
            coords[gp_idx].astype(np.float64, copy=False))

        prior, nk, means_init, covs_init, cov_t = assigner._estimate_initial_params(
            group_coords, sub_csc_b)

        data_mean_natural = means_init[0] * scaler.scale_ + scaler.mean_
        cat_val = np.concatenate([cat_pos[sid], cat_vel[sid]])

        dpos = np.linalg.norm(data_mean_natural[:3] - cat_val[:3])
        dvel = np.linalg.norm(data_mean_natural[3:] - cat_val[3:])
        all_dpos.append(dpos)
        all_dvel.append(dvel)

    all_dpos = np.array(all_dpos)
    all_dvel = np.array(all_dvel)
    print(f"  |Δpos|: median={np.median(all_dpos):.3f}  p95={np.percentile(all_dpos,95):.3f}  max={all_dpos.max():.3f}")
    print(f"  |Δvel|: median={np.median(all_dvel):.3f}  p95={np.percentile(all_dvel,95):.3f}  max={all_dvel.max():.3f}")

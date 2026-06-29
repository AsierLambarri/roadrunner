"""Compare galaxy masses: loss (buggy) vs loss_fixed (correct)."""

import h5py
import numpy as np

# Expected mass-loss patterns (from generate_mock_mass_loss.py)
PATTERN_I  = [1.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
PATTERN_II = [1.0, 0.82, 0.64, 0.46, 0.28, 0.1, 0.23, 0.37, 0.5]
PATTERN_III = np.array(PATTERN_II) # approximate

mass_loss_ids = {2, 3, 4, 5, 6, 7, 8, 9, 10}

for label, path in [("LOSS (buggy)", "loss"), ("LOSS (FIXED)", "loss_fixed")]:
    p = f"/home/asier/roadrunner/mass_loss_test/{path}/catalogue.hdf5"
    with h5py.File(p, "r") as f:
        snaps = sorted(f["snapshots"], key=int)
        print(f"\n=== {label} ({len(snaps)} snapshots) ===")

        gal_masses = {}
        for snap in snaps:
            if "galaxy_properties" not in f[f"snapshots/{snap}"]:
                continue
            props = f[f"snapshots/{snap}/galaxy_properties"][()]
            for row in props:
                sid = row["Sub_tree_id"]
                mtot = row["Mtot"]
                if sid not in gal_masses:
                    gal_masses[sid] = []
                gal_masses[sid].append(mtot)

        loss_mono = 0
        loss_drop = 0
        for sid in sorted(gal_masses):
            if sid not in mass_loss_ids:
                continue
            masses = gal_masses[sid]
            is_mono = all(masses[i] <= masses[i+1] for i in range(len(masses)-1))
            if is_mono:
                loss_mono += 1
            else:
                loss_drop += 1
            first_mass = masses[0]
            rel_masses = [m / first_mass for m in masses]
            print(f"  Gal {sid:3d}: relative masses = "
                  f"{' → '.join(f'{rm:.3f}' for rm in rel_masses)}")

        print(f"\n  Loss-target galaxies monotonic: {loss_mono}/{len(mass_loss_ids)}")
        print(f"  Loss-target galaxies DROPPING:  {loss_drop}/{len(mass_loss_ids)}")

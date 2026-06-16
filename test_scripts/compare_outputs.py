#!/usr/bin/env python3
"""Compare ACCME vs roadrunner output.

Produces three files:
  comparison_boundness.txt
  comparison_assignment.txt
  comparison_properties.txt

Usage:
  python test_scripts/compare_outputs.py
"""

import os
import sys

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR  # files are in the repo root (same as script for now)

AC_CAT = os.path.join(SCRIPT_DIR, "..", "accme_catalogue.hdf5")
RR_CAT = os.path.join(SCRIPT_DIR, "..", "roadrunner_catalogue.hdf5")
RR_AS  = os.path.join(SCRIPT_DIR, "..", "roadrunner_assignment.hfd5")

SNAPSHOTS = list(range(20))


# ── Galaxy selection ───────────────────────────────────────────────

def _select_target_galaxies(rr_assign_path, snap=19, n_top=10, n_mid=5, n_low=5):
    with h5py.File(rr_assign_path, "r") as f:
        ha = f[f"snapshots/{snap}/hard_assignment"][:]
    gids = ha["Sub_tree_id"]
    mask = gids != -1
    unique, counts = np.unique(gids[mask], return_counts=True)
    order = np.argsort(-counts)
    sorted_gids = unique[order]
    n = len(sorted_gids)
    top = list(sorted_gids[:n_top])
    mid_start = n // 2 - n_mid // 2
    mid = list(sorted_gids[mid_start:mid_start + n_mid])
    low_start = max(0, n - n_low)
    low = list(sorted_gids[low_start:low_start + n_low])
    all_selected = top + mid + low
    return all_selected, {gid: int(c) for gid, c in zip(sorted_gids, counts)}


# ── Boundness check ────────────────────────────────────────────────

def _check_boundness(rr_assign_path, ac_cat_path, targets, snaps):
    lines = []
    lines.append("=" * 70)
    lines.append("BOUNDNESS CHECK — Particles per galaxy per snapshot")
    lines.append(f"Target galaxies (selected from RR snap 19): {targets}")
    lines.append("=" * 70)

    ac_f = h5py.File(ac_cat_path, "r")
    rr_f = h5py.File(rr_assign_path, "r")

    per_gal_lines = []
    for gid in targets:
        gal_lines = [f"\nGalaxy {gid}:"]
        gal_lines.append(f"  {'snap':>4}  {'N_rr':>8}  {'N_ac':>8}  {'diff':>8}  {'rel_err':>10}")
        gal_errors = []
        for s in snaps:
            ha_rr = rr_f[f"snapshots/{s}/hard_assignment"][:]
            n_rr = int((ha_rr["Sub_tree_id"] == gid).sum())

            s_str = str(s)
            if s_str in ac_f["catalogue"]:
                ha_ac = ac_f[f"catalogue/{s_str}"][:]
                n_ac = int((ha_ac["Sub_tree_id"] == gid).sum())
            else:
                n_ac = 0

            diff = abs(n_rr - n_ac)
            denom = max(n_rr, n_ac, 1)
            rel = diff / denom
            if denom > 10:
                gal_errors.append(rel)
            gal_lines.append(f"  {s:>4}  {n_rr:>8}  {n_ac:>8}  {diff:>8}  {rel:>10.6f}")
        if gal_errors:
            median_e = np.median(gal_errors)
            max_e = max(gal_errors)
            gal_lines.append(f"  -> median_rel={median_e:.6f}  max_rel={max_e:.6f}")
        per_gal_lines.extend(gal_lines)

    # Aggregate
    totals_rr = np.zeros(len(snaps), dtype=int)
    totals_ac = np.zeros(len(snaps), dtype=int)
    for i, s in enumerate(snaps):
        ha_rr = rr_f[f"snapshots/{s}/hard_assignment"][:]
        totals_rr[i] = int((ha_rr["Sub_tree_id"] != -1).sum())
        s_str = str(s)
        if s_str in ac_f["catalogue"]:
            ha_ac = ac_f[f"catalogue/{s_str}"][:]
            totals_ac[i] = int((ha_ac["Sub_tree_id"] != -1).sum())
    lines.append("\n\nPer-snapshot totals (all assigned particles):")
    lines.append(f"  {'snap':>4}  {'RR_total':>8}  {'AC_total':>8}  {'diff':>8}  {'rel_err':>10}")
    for i, s in enumerate(snaps):
        diff = abs(int(totals_rr[i]) - int(totals_ac[i]))
        denom = max(totals_rr[i], totals_ac[i], 1)
        rel = diff / denom
        lines.append(f"  {s:>4}  {totals_rr[i]:>8}  {totals_ac[i]:>8}  {diff:>8}  {rel:>10.6f}")

    lines.append("\n\nPer-galaxy detail (target galaxies):")
    lines.extend(per_gal_lines)

    rr_f.close()
    ac_f.close()
    return "\n".join(lines)


# ── Assignment check ───────────────────────────────────────────────

def _check_assignment(rr_assign_path, ac_cat_path, targets, snaps):
    lines = []
    lines.append("=" * 70)
    lines.append("ASSIGNMENT CHECK — Per-particle Sub_tree_id agreement")
    lines.append(f"Target galaxies: {targets}")
    lines.append("=" * 70)

    ac_f = h5py.File(ac_cat_path, "r")
    rr_f = h5py.File(rr_assign_path, "r")

    lines.append(f"\n{'snap':>4}  {'N_total':>8}  {'N_match':>8}  {'accuracy':>10}  {'N_rr_only':>10}  {'N_ac_only':>10}")
    all_acc = []
    for s in snaps:
        ha_rr = rr_f[f"snapshots/{s}/hard_assignment"][:]
        s_str = str(s)
        if s_str not in ac_f["catalogue"]:
            continue
        ha_ac = ac_f[f"catalogue/{s_str}"][:]

        rr_dict = {int(pid): int(sid) for pid, sid in zip(ha_rr["particle_index"], ha_rr["Sub_tree_id"])}
        ac_dict = {int(pid): int(sid) for pid, sid in zip(ha_ac["particle_index"], ha_ac["Sub_tree_id"])}

        rr_set = set(rr_dict.keys())
        ac_set = set(ac_dict.keys())
        common = rr_set & ac_set
        n_match = sum(1 for pid in common if rr_dict[pid] == ac_dict[pid])
        n_total = len(common)
        acc = n_match / n_total if n_total > 0 else 0.0
        all_acc.append(acc)
        lines.append(f"  {s:>4}  {n_total:>8}  {n_match:>8}  {acc:>10.6f}  {len(rr_set - ac_set):>10}  {len(ac_set - rr_set):>10}")

    if all_acc:
        lines.append(f"\nAccuracy across snapshots: median={np.median(all_acc):.6f}  min={min(all_acc):.6f}")

    # Per-galaxy accuracy for target galaxies
    lines.append(f"\n\nPer-galaxy assignment accuracy (target galaxies):")
    header = f"  {'galaxy':>8}"
    for s in snaps:
        header += f"  {'s'+str(s):>7}"
    lines.append(header)
    for gid in targets:
        row = f"{gid:>8}"
        for s in snaps:
            ha_rr = rr_f[f"snapshots/{s}/hard_assignment"][:]
            s_str = str(s)
            if s_str not in ac_f["catalogue"]:
                row += f"  {'--':>7}"
                continue
            ha_ac = ac_f[f"catalogue/{s_str}"][:]
            rr_mask = ha_rr["Sub_tree_id"] == gid
            ac_mask = ha_ac["Sub_tree_id"] == gid
            rr_pids = set(ha_rr["particle_index"][rr_mask])
            ac_pids = set(ha_ac["particle_index"][ac_mask])
            common = rr_pids & ac_pids
            n_match = len(common)
            n_total = max(len(rr_pids), len(ac_pids), 1)
            row += f"  {n_match / n_total:>7.4f}"
        lines.append(row)

    rr_f.close()
    ac_f.close()
    return "\n".join(lines)


# ── Properties check ───────────────────────────────────────────────

def _check_properties(rr_cat_path, ac_cat_path, targets, snaps):
    lines = []
    lines.append("=" * 70)
    lines.append("PROPERTIES CHECK — Galaxy properties comparison")
    lines.append(f"Target galaxies: {targets}")
    lines.append("=" * 70)

    ac_f = h5py.File(ac_cat_path, "r")
    rr_f = h5py.File(rr_cat_path, "r")

    ac_gp = ac_f["properties/galaxy_properties"][:]

    props = ["Mtot", "rh", "sigma"]
    per_prop_errors = {p: [] for p in props}

    for gid in targets:
        gal_lines = [f"\nGalaxy {gid}:"]
        header = f"  {'snap':>4}"
        for p in props:
            header += f"  {p}_rr    {p}_ac     diff    rel"
        gal_lines.append(header)

        for s in snaps:
            s_str = str(s)

            # Roadrunner properties for this snapshot
            snap_group = rr_f.get(f"snapshots/{s_str}")
            if snap_group is None or "galaxy_properties" not in snap_group:
                continue
            rr_gp = snap_group["galaxy_properties"][:]
            rr_idx = np.where(rr_gp["Sub_tree_id"] == gid)[0]
            if len(rr_idx) == 0:
                continue
            rr_row = rr_gp[rr_idx[0]]

            # ACCME properties for this snapshot + galaxy
            ac_mask = (ac_gp["Snapshot"] == s) & (ac_gp["Sub_tree_id"] == gid)
            ac_idx = np.where(ac_mask)[0]
            if len(ac_idx) == 0:
                continue
            ac_row = ac_gp[ac_idx[0]]

            row = f"  {s:>4}"
            valid = True
            for p in props:
                v_rr = float(rr_row[p])
                v_ac = float(ac_row[p])
                if np.isnan(v_rr) or np.isnan(v_ac):
                    valid = False
                    break
                diff = abs(v_rr - v_ac)
                denom = max(abs(v_rr), abs(v_ac), 1e-30)
                rel = diff / denom
                per_prop_errors[p].append(rel)
                row += f"  {v_rr:>8.2f}  {v_ac:>8.2f}  {diff:>6.2f}  {rel:>7.4f}"
            if valid:
                gal_lines.append(row)

        lines.extend(gal_lines)

    # Aggregate summary per property
    lines.append("\n\n" + "=" * 50)
    lines.append("AGGREGATE PROPERTIES SUMMARY")
    lines.append("=" * 50)
    lines.append(f"  {'property':>12}  {'median_rel':>10}  {'max_rel':>10}  {'N_samples':>10}")
    for p in props:
        errs = per_prop_errors[p]
        if errs:
            lines.append(f"  {p:>12}  {np.median(errs):>10.6f}  {max(errs):>10.6f}  {len(errs):>10}")
        else:
            lines.append(f"  {p:>12}  {'--':>10}  {'--':>10}  {0:>10}")

    ac_f.close()
    rr_f.close()
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────

def main():
    targets, snap19_counts = _select_target_galaxies(RR_AS, snap=19)
    print(f"Selected {len(targets)} target galaxies from RR snap 19:")
    print(f"  {targets}")
    print()

    out_dir = os.path.join(SCRIPT_DIR, "..")
    bound_path = os.path.join(out_dir, "comparison_boundness.txt")
    assign_path = os.path.join(out_dir, "comparison_assignment.txt")
    prop_path = os.path.join(out_dir, "comparison_properties.txt")

    report = _check_boundness(RR_AS, AC_CAT, targets, SNAPSHOTS)
    with open(bound_path, "w") as f:
        f.write(report)
    print(report[:500] + "\n...\n")
    print(f"Saved to {bound_path}")

    report = _check_assignment(RR_AS, AC_CAT, targets, SNAPSHOTS)
    with open(assign_path, "w") as f:
        f.write(report)
    print(report[:500] + "\n...\n")
    print(f"Saved to {assign_path}")

    report = _check_properties(RR_CAT, AC_CAT, targets, SNAPSHOTS)
    with open(prop_path, "w") as f:
        f.write(report)
    print(report[:500] + "\n...\n")
    print(f"Saved to {prop_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

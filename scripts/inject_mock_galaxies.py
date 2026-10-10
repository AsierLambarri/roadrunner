#!/usr/bin/env python3
"""Inject limepy galaxies into a mock snapshot or a mock simulation.

Galaxies are built as in ``generate_mock.py``: a ``limepy`` model
(``phi0``, ``g``, mass ``M``, half-mass radius ``rh``) sampled with
``limepy.sample`` (``5 n`` draws, ``n`` kept at random), star particles
of ``star_mass``, and a merger-tree row with ``mass = M``,
``virial_radius = rt`` and ``scale_radius = rh``.

The input kind is detected from the directory:

- a mock snapshot (``particles.npz``, written by ``generate_mock.py``):
  the galaxies are added to it; evolve it with
  ``generate_mock_simulation.py``.
- a mock simulation (``particles_XXX.npz`` series, written by
  ``generate_mock_simulation.py``): galaxies appear at any snapshot and
  move with the simulation's systematic drift ``--v-sys`` (the value that
  was given to ``generate_mock_simulation.py``), with its conventions:
  physical displacement ``v * dt`` (km/s x Gyr taken as kpc), comoving
  storage (physical x (1 + z)), stored velocities including ``v_sys``.

Injected stars get new particle IDs above every ID in the input and are
appended after the existing particles, so the existing ones are
untouched. The truth (``galaxy_id``, ``born_snap``, ``assignment.csv``)
is written for them too.

Usage:
  python scripts/inject_mock_galaxies.py \\
      --input-dir test_data/mock_simulation_tight \\
      --output-dir test_data/mock_simulation_events \\
      --scenario scripts/inject_mock_galaxies.yaml.example \\
      --v-sys 1000 1000 1000

Scenario (YAML), see ``scripts/inject_mock_galaxies.yaml.example``:
  defaults:                     # optional; any galaxy can override them
    phi0: 0.01
    g: 2.0                      # what the existing mocks were built with
    mass: 1.0e10                # limepy M = tree mass (Msun)
    rh_pc: 3000                 # limepy half-mass radius (pc)
    star_mass: 1.0e4            # Msun per star particle
  galaxies:
    - id: 101                   # optional: default is the next free Sub_tree_id
      birth_snapshot: 3         # simulation only: first snapshot with it
      n_stars: 1500             # 0: a dark halo (tree rows only)
      n_new_per_snapshot: 50    # simulation only: stars formed at later snapshots
      anchor: 5                 # optional: placed relative to this Sub_tree_id
      offset_kpc: [20, 0, 0]    # with anchor (or offset_rvir, in anchor Rvir)
      position_kpc: [0, 0, 0]   # without anchor: physical position at birth
      velocity: [0, 0, 0]       # without anchor: peculiar velocity (km/s)
      v_rel: [-30, 0, 0]        # km/s: extra motion, displaces it and adds to its velocity
      end_snapshot: 8           # simulation only: last snapshot with tree rows
      merge_into: 5             # with end_snapshot: its stars then join this halo
"""

import argparse
import dataclasses
import os
import shutil
import sys

import limepy
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
from roadrunner._defaults import COSMOLOGY_NPZ_PREFIX
from roadrunner.cosmology import Cosmology

_PC_TO_KPC = 1.0 / 1000.0
_DEFAULTS = dict(phi0=0.01, g=2.0, mass=1.0e10, rh_pc=3000.0, star_mass=1.0e4)
_KNOWN_KEYS = ({"indices", "masses", "coords", "galaxy_id", "born_snap", "metallicity"}
              | {f"{COSMOLOGY_NPZ_PREFIX}{f.name}" for f in dataclasses.fields(Cosmology)})
_SIM_ONLY = ("birth_snapshot", "end_snapshot", "n_new_per_snapshot", "merge_into")


def _stars(model, n, seed, rng):
    """``n`` stars of a limepy model about its centre, as in ``generate_mock.py``.

    Returns positions (kpc) and velocities (km/s) relative to the centre.
    """
    if n <= 0:
        return np.empty((0, 3)), np.empty((0, 3))
    sample = limepy.sample(model, N=5 * n, seed=seed)
    pick = rng.choice(sample.N, size=n, replace=False)
    pos = np.column_stack([sample.x, sample.y, sample.z])[pick] * _PC_TO_KPC
    vel = np.column_stack([sample.vx, sample.vy, sample.vz])[pick]
    return pos, vel


def _load_input(input_dir):
    """Snapshots, redshifts, times, particle files and tree of a mock directory."""
    tree = pd.read_csv(os.path.join(input_dir, "merger_tree.csv"))
    equiv = pd.read_csv(os.path.join(input_dir, "equivalence.csv"))
    if os.path.exists(os.path.join(input_dir, "particles.npz")):
        snaps = [int(tree["Snapshot"].iloc[0])]
        z = {snaps[0]: float(tree["Redshift"].iloc[0])}
        return False, snaps, z, {snaps[0]: 0.0}, {snaps[0]: "particles.npz"}, tree, equiv
    snaps = [int(s) for s in equiv["snapshot"]]
    z = dict(zip(snaps, equiv["redshift"].astype(float)))
    time = dict(zip(snaps, equiv["time"].astype(float)))
    files = dict(zip(snaps, equiv["snapname"]))
    return True, snaps, z, time, files, tree, equiv


def _row(tree, sid, snap):
    """The tree row of ``sid`` at ``snap``, or None."""
    rows = tree[(tree["Sub_tree_id"] == sid) & (tree["Snapshot"] == snap)]
    return None if rows.empty else rows.iloc[0]


def _phys_state(tree, sid, snap, z):
    """Physical position (kpc), velocity (km/s) and Rvir (kpc) of a tree halo."""
    r = _row(tree, sid, snap)
    if r is None:
        return None
    a = 1.0 + z[snap]
    pos = np.array([r["position_x"], r["position_y"], r["position_z"]], float) / a
    vel = np.array([r["velocity_x"], r["velocity_y"], r["velocity_z"]], float)
    return pos, vel, float(r["virial_radius"]) / a


def main():
    ap = argparse.ArgumentParser(description="Inject limepy galaxies into a mock snapshot or simulation")
    ap.add_argument("--input-dir", required=True, help="Mock snapshot or simulation directory")
    ap.add_argument("--output-dir", required=True, help="Output directory (same format)")
    ap.add_argument("--scenario", required=True, help="Scenario YAML")
    ap.add_argument("--v-sys", type=float, nargs=3, default=None,
                    help="Systematic drift of the simulation [vx, vy, vz] (physical km/s), "
                         "as given to generate_mock_simulation.py (simulations only)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("--output-dir must differ from --input-dir")
    is_sim, snaps, z, time, files, tree, equiv = _load_input(args.input_dir)
    if is_sim and args.v_sys is None:
        raise ValueError("--v-sys is required for a simulation (the drift given to generate_mock_simulation.py)")
    v_sys = np.zeros(3) if not is_sim else np.asarray(args.v_sys, float)
    with open(args.scenario) as f:
        scenario = yaml.safe_load(f)
    defaults = {**_DEFAULTS, **(scenario.get("defaults") or {})}
    rng = np.random.default_rng(args.seed)

    next_sid = int(tree["Sub_tree_id"].max()) + 1
    next_pid = 1 + max(int(np.load(os.path.join(args.input_dir, files[s]))["indices"].max())
                       for s in snaps)
    snap_pos = {s: i for i, s in enumerate(snaps)}
    new_rows, blocks, merges = [], [], []   # blocks: stars of one galaxy born at one snapshot

    for e, cfg in enumerate(scenario["galaxies"]):
        p = {**defaults, **cfg}
        # PyYAML reads unsigned exponents (1.0e10) as strings.
        for key in ("phi0", "g", "mass", "rh_pc", "star_mass"):
            p[key] = float(p[key])
        if not is_sim and any(k in cfg for k in _SIM_ONLY):
            raise ValueError(f"galaxy {e}: {', '.join(k for k in _SIM_ONLY if k in cfg)} "
                             "only apply to simulations")
        sid = int(cfg.get("id", next_sid))
        if sid in set(tree["Sub_tree_id"]) or any(r["Sub_tree_id"] == sid for r in new_rows):
            raise ValueError(f"galaxy {e}: Sub_tree_id {sid} already exists")
        next_sid = max(next_sid, sid + 1)
        birth = int(cfg.get("birth_snapshot", snaps[0]))
        end = int(cfg.get("end_snapshot", snaps[-1]))
        model = limepy.limepy(phi0=p["phi0"], g=p["g"], M=p["mass"], rh=p["rh_pc"])
        rt, rh = model.rt * _PC_TO_KPC, model.rh * _PC_TO_KPC
        v_rel = np.asarray(cfg.get("v_rel", [0, 0, 0]), float)

        # Centre at birth (physical) and base velocity per snapshot.
        anchor = cfg.get("anchor")
        if anchor is not None:
            state = _phys_state(tree, int(anchor), birth, z)
            if state is None:
                raise ValueError(f"galaxy {e}: anchor {anchor} has no tree row at snapshot {birth}")
            a_pos, a_vel, a_rvir = state
            offset = (np.asarray(cfg["offset_rvir"], float) * a_rvir if "offset_rvir" in cfg
                      else np.asarray(cfg.get("offset_kpc", [0, 0, 0]), float))
            centre_birth = a_pos + offset
        else:
            centre_birth = np.asarray(cfg["position_kpc"], float)
        base_vel = np.asarray(cfg.get("velocity", [0, 0, 0]), float) + v_sys

        # Bound per galaxy: the stars use them again when the snapshots are written.
        def centre(k, c0=centre_birth, t0=time[birth], v=v_sys + v_rel):
            return c0 + v * (time[k] - t0)

        def velocity(k, anc=anchor, v0=base_vel, vr=v_rel, last=[None]):
            if anc is not None:
                st = _phys_state(tree, int(anc), k, z)
                if st is not None:
                    last[0] = st[1]
                return last[0] + vr
            return v0 + vr

        state = {}
        for k in snaps[snap_pos[birth]:snap_pos[end] + 1]:
            c, v = centre(k), velocity(k)
            state[k] = (c, v)
            a = 1.0 + z[k]
            row = {"Snapshot": k, "Sub_tree_id": sid, "mass": p["mass"], "virial_radius": rt * a,
                   "scale_radius": rh, "position_x": c[0] * a, "position_y": c[1] * a,
                   "position_z": c[2] * a, "velocity_x": v[0], "velocity_y": v[1],
                   "velocity_z": v[2], "Redshift": z[k]}
            if "host_id" in tree.columns:
                row["host_id"] = -1
            if "distance_to_acc_id" in tree.columns:
                row["distance_to_acc_id"] = 0.0
            new_rows.append(row)

        # Stars: the initial population and those formed at later snapshots.
        target = cfg.get("merge_into")
        born = [(birth, int(p.get("n_stars", 0)))]
        born += [(k, int(cfg.get("n_new_per_snapshot", 0)))
                 for k in snaps[snap_pos[birth] + 1:snap_pos[end] + 1]]
        for j, n in born:
            if n <= 0:
                continue
            pos, vel = _stars(model, n, seed=args.seed + 1000 * e + j, rng=rng)
            blocks.append(dict(sid=sid, born=j, end=end, target=target, pos=pos, vel=vel,
                               ids=np.arange(next_pid, next_pid + n, dtype=np.uint64),
                               mass=p["star_mass"], state=state, centre=centre, velocity=velocity))
            next_pid += n

        if target is not None:
            merges.append((int(target), end, p))
        print(f"galaxy {sid}: M={p['mass']:.3g} rh={rh:.2f} kpc rt={rt:.2f} kpc, snapshots "
              f"{birth}-{end}, {sum(n for _, n in born)} stars" + (f", merges into {target}" if target else ""))

    tree_out = pd.concat([tree, pd.DataFrame(new_rows)], ignore_index=True)
    tree_out = tree_out.sort_values(["Snapshot", "Sub_tree_id"], kind="mergesort").reset_index(drop=True)
    # Mergers: after the end snapshot the target halo carries the summed mass,
    # and its Rvir is that of the limepy model of that mass (same rh).
    for target, end, p in merges:
        later = (tree_out["Sub_tree_id"] == target) & (tree_out["Snapshot"] > end)
        if later.any():
            t_rh = float(tree_out.loc[later, "scale_radius"].iloc[0])
            t_mass = float(tree_out.loc[later, "mass"].iloc[0]) + p["mass"]
            t_rt = limepy.limepy(phi0=p["phi0"], g=p["g"], M=t_mass, rh=t_rh / _PC_TO_KPC).rt * _PC_TO_KPC
            tree_out.loc[later, "mass"] = t_mass
            tree_out.loc[later, "virial_radius"] = t_rt * (1.0 + tree_out.loc[later, "Snapshot"].map(z))
    os.makedirs(args.output_dir, exist_ok=True)

    final = None
    for k in snaps:
        raw = np.load(os.path.join(args.input_dir, files[k]))
        unknown = set(raw.files) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"{files[k]}: unknown particle fields {sorted(unknown)}")
        out = {key: raw[key] for key in raw.files}
        a = 1.0 + z[k]
        add = {key: [] for key in ("indices", "masses", "coords", "galaxy_id", "born_snap")}
        for b in blocks:
            if b["born"] > k:
                continue
            if k <= b["end"]:
                c, v = b["state"][k]
                pos, vel, label = c + b["pos"], v + b["vel"], b["sid"]
            elif b["target"] is not None:
                # Joined the target: carried at the offset it had at the end snapshot.
                t_end = _phys_state(tree_out, b["target"], b["end"], z)
                t_now = _phys_state(tree_out, b["target"], k, z)
                c, v = b["state"][b["end"]]
                pos = t_now[0] + (c + b["pos"] - t_end[0])
                vel = t_now[1] + (v + b["vel"] - t_end[1])
                label = b["target"]
            else:
                pos, vel, label = b["centre"](k) + b["pos"], b["velocity"](k) + b["vel"], b["sid"]
            add["indices"].append(b["ids"])
            add["masses"].append(np.full(len(b["ids"]), b["mass"]))
            add["coords"].append(np.column_stack([pos * a, vel]))
            add["galaxy_id"].append(np.full(len(b["ids"]), label, dtype=np.int64))
            add["born_snap"].append(np.full(len(b["ids"]), b["born"], dtype=np.int32))
        if add["indices"]:
            for key in ("indices", "masses", "coords", "galaxy_id", "born_snap"):
                if key in out:
                    out[key] = np.concatenate([out[key], np.concatenate(add[key]).astype(out[key].dtype)])
            if "metallicity" in out:
                n_add = sum(len(i) for i in add["indices"])
                fill = np.full(n_add, np.median(raw["metallicity"]), dtype=raw["metallicity"].dtype)
                out["metallicity"] = np.concatenate([out["metallicity"], fill])
        np.savez_compressed(os.path.join(args.output_dir, files[k]), **out)
        final = out

    tree_out.to_csv(os.path.join(args.output_dir, "merger_tree.csv"), index=False)
    equiv.to_csv(os.path.join(args.output_dir, "equivalence.csv"), index=False)
    pd.DataFrame({"array_index": np.arange(len(final["galaxy_id"])),
                  "Sub_tree_id": final["galaxy_id"]}).to_csv(
        os.path.join(args.output_dir, "assignment.csv"), index=False)
    shutil.copy(args.scenario, os.path.join(args.output_dir, "scenario.yaml"))
    print(f"\nDone. {len(blocks)} star blocks, {len(new_rows)} tree rows added. Output in {args.output_dir}")


if __name__ == "__main__":
    main()

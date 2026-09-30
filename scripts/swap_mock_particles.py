#!/usr/bin/env python3
"""Move particles between halos of a mock simulation, or out of all of them.

Two operations, applied to the snapshots ``--from-snapshot`` to
``--until-snapshot`` (default: to the last one):

- swap: two particles in different halos exchange their phase-space
  coordinates, so each takes over the other's trajectory while keeping
  its own ID, mass and ``born_snap``. Seen by the pipeline, each particle
  moves to the other halo: it stops being bound to its old host and
  becomes bound to the new one.
- eject: a particle is moved far outside every halo (beyond the
  bounding box of all halos by 100 of the largest virial radii), so it
  is bound to none; after ``--until-snapshot`` it returns to its own
  trajectory.

The truth follows the coordinates: ``galaxy_id`` is swapped with them,
and is -1 while a particle is ejected. ``assignment.csv`` is rewritten
from the final snapshot, and ``moves.csv`` lists every move (particle,
kind, partner, snapshots, host before and during the move) for scoring.

Swap partners are drawn among the particles present at
``--from-snapshot``: from any two different halos (``--scope any``),
from halos of the same overlap group (``same-group``: virial spheres
connected by overlap, as the pipeline groups them), from different
groups (``cross-group``), or between two given halos (``--halos A B``).

Usage:
  python scripts/swap_mock_particles.py \\
      --input-dir  test_data/mock_simulation_tight \\
      --output-dir test_data/mock_simulation_tight_swaps \\
      --from-snapshot 4 --until-snapshot 6 --n-pairs 200 --scope same-group \\
      --n-eject 100 --seed 1

Operations compose: run the script again on its own output (``moves.csv``
is then extended).
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _groups(tree_k):
    """Overlap-group label of each halo (virial spheres connected by overlap)."""
    x = tree_k[["position_x", "position_y", "position_z"]].values
    r = tree_k["virial_radius"].values
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    _, labels = connected_components(csr_matrix(d <= r[:, None] + r[None, :]), directed=False)
    return dict(zip(tree_k["Sub_tree_id"].values, labels))


def _pick_pairs(ids, hosts, n, scope, group, halos, rng):
    """``n`` pairs of distinct particles in different halos, per ``scope``."""
    pairs = []
    if halos is not None:
        pool_a = np.flatnonzero(hosts == halos[0])
        pool_b = np.flatnonzero(hosts == halos[1])
        a_all = rng.permutation(pool_a)[:n]
        b_all = rng.permutation(pool_b)[:n]
        return list(zip(ids[a_all], ids[b_all]))
    host_group = np.array([group.get(h, -1) for h in hosts])
    free = hosts >= 0
    for i in rng.permutation(len(ids)):
        if len(pairs) == n:
            break
        if not free[i]:
            continue
        if scope == "same-group":
            cand = free & (hosts != hosts[i]) & (host_group == host_group[i])
        elif scope == "cross-group":
            cand = free & (host_group != host_group[i])
        else:
            cand = free & (hosts != hosts[i])
        js = np.flatnonzero(cand)
        if js.size == 0:
            continue
        j = rng.choice(js)
        pairs.append((ids[i], ids[j]))
        free[i] = free[j] = False
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Swap or eject particles in a mock simulation")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--from-snapshot", type=int, required=True)
    ap.add_argument("--until-snapshot", type=int, default=None,
                    help="Last snapshot of the moves (default: the last snapshot)")
    ap.add_argument("--n-pairs", type=int, default=0, help="Particle pairs to swap")
    ap.add_argument("--scope", choices=("any", "same-group", "cross-group"), default="any")
    ap.add_argument("--halos", type=int, nargs=2, default=None,
                    help="Swap only between these two Sub_tree_ids")
    ap.add_argument("--n-eject", type=int, default=0, help="Particles to eject")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("--output-dir must differ from --input-dir")

    equiv = pd.read_csv(os.path.join(args.input_dir, "equivalence.csv"))
    tree = pd.read_csv(os.path.join(args.input_dir, "merger_tree.csv"))
    snaps = [int(s) for s in equiv["snapshot"]]
    files = dict(zip(snaps, equiv["snapname"]))
    s0 = args.from_snapshot
    s1 = snaps[-1] if args.until_snapshot is None else args.until_snapshot
    rng = np.random.default_rng(args.seed)

    base = np.load(os.path.join(args.input_dir, files[s0]))
    ids, hosts = base["indices"].astype(np.int64), base["galaxy_id"].astype(np.int64)
    group = _groups(tree[tree["Snapshot"] == s0])
    pairs = _pick_pairs(ids, hosts, args.n_pairs, args.scope, group, args.halos, rng)
    taken = {p for pair in pairs for p in pair}
    free = np.flatnonzero(~np.isin(ids, list(taken)) & (hosts >= 0))
    eject = ids[rng.choice(free, size=min(args.n_eject, free.size), replace=False)]
    host_of = dict(zip(ids.tolist(), hosts.tolist()))

    # Ejection point: outside the bounding box of all halos (physical), per direction.
    t0 = tree[tree["Snapshot"] == s0]
    a0 = 1.0 + float(equiv.loc[equiv["snapshot"] == s0, "redshift"].iloc[0])
    xyz = t0[["position_x", "position_y", "position_z"]].values / a0
    span = np.linalg.norm(xyz.max(0) - xyz.min(0)) + 100 * t0["virial_radius"].max() / a0
    dirs = rng.normal(size=(len(eject), 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    kick = dirs * span                      # physical kpc, added to each ejected particle

    os.makedirs(args.output_dir, exist_ok=True)
    final = None
    for k in snaps:
        raw = np.load(os.path.join(args.input_dir, files[k]))
        out = {key: raw[key].copy() for key in raw.files}
        if s0 <= k <= s1:
            pos = {int(p): i for i, p in enumerate(out["indices"].tolist())}
            for a, b in pairs:
                ia, ib = pos[a], pos[b]
                out["coords"][[ia, ib]] = out["coords"][[ib, ia]]
                out["galaxy_id"][[ia, ib]] = out["galaxy_id"][[ib, ia]]
            ie = np.array([pos[int(p)] for p in eject], dtype=np.int64)
            if ie.size:
                a = 1.0 + float(equiv.loc[equiv["snapshot"] == k, "redshift"].iloc[0])
                out["coords"][ie, :3] += kick * a            # comoving storage
                out["galaxy_id"][ie] = -1
        np.savez_compressed(os.path.join(args.output_dir, files[k]), **out)
        final = out

    tree.to_csv(os.path.join(args.output_dir, "merger_tree.csv"), index=False)
    equiv.to_csv(os.path.join(args.output_dir, "equivalence.csv"), index=False)
    pd.DataFrame({"array_index": np.arange(len(final["galaxy_id"])),
                  "Sub_tree_id": final["galaxy_id"]}).to_csv(
        os.path.join(args.output_dir, "assignment.csv"), index=False)

    moves = [dict(particle=a, kind="swap", partner=b, from_snapshot=s0, until_snapshot=s1,
                  host_before=host_of[a], host_during=host_of[b]) for p in pairs for a, b in (p, p[::-1])]
    moves += [dict(particle=int(p), kind="eject", partner=-1, from_snapshot=s0, until_snapshot=s1,
                   host_before=host_of[int(p)], host_during=-1) for p in eject]
    moves = pd.DataFrame(moves, columns=["particle", "kind", "partner", "from_snapshot",
                                         "until_snapshot", "host_before", "host_during"])
    prev = os.path.join(args.input_dir, "moves.csv")
    if os.path.exists(prev):
        moves = pd.concat([pd.read_csv(prev), moves], ignore_index=True)
    moves.to_csv(os.path.join(args.output_dir, "moves.csv"), index=False)
    for extra in ("scenario.yaml",):
        src = os.path.join(args.input_dir, extra)
        if os.path.exists(src):
            with open(src) as f, open(os.path.join(args.output_dir, extra), "w") as g:
                g.write(f.read())
    print(f"{len(pairs)} pairs swapped ({args.scope}), {len(eject)} particles ejected, "
          f"snapshots {s0}-{s1}. Output in {args.output_dir}")


if __name__ == "__main__":
    main()

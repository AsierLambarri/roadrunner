"""Point 1: the run is technically sound.

    python3 evaluate/soundness.py --path RUN [--particles DIR] [--workers N] [--snaps A:B]

1.1 completion: progress.json finished at the last snapshot; run.log ends with "--- Run complete ---";
    error.log absent or empty; every snapshot has its assignment file, catalogue group and run.log row.
    Any miss is a FAIL.
1.2 consistency in each snapshot (FAIL):
    a. particle IDs unique;
    b. hard labels in the fitted galaxies of the assignment file, or -1;
    c. catalogue: Mtot finite and > 0; position and velocity all finite or all NaN (NaN for galaxies
       of <= 2 particles; rh, r20, r80, sigma may be NaN alone, for small ones); galaxies are tree
       halos of that snapshot;
    d. run.log UNASSIGNED = number of -1 labels;
    e. the particle IDs are those of the particle data (when given);
    f. |sum_k r - 1| < 1e-3 for every particle with responsibilities, log r <= 1e-6;
    g. argmax_k log r = hard label for >= 99.9% of them; -1 <=> no responsibilities;
    h. Gaussians: finite, symmetric to 1e-5 of max|Sigma|, smallest eigenvalue >= -1e-4 x the largest
       (the storage tolerance; 2-particle fits sit at ~0);
    i. satellite relations: hosts and satellites are tree halos of that snapshot, and the
       host -> satellite graph has no cycle (none can form while a host must be more massive than its
       satellites; kept as a guard). A satellite may have several hosts; the main host as a satellite
       is a FLAG;
    j. Gaussians no wider than their halo: the largest eigenvalue of the position block <= 15 R_vir^2
       and of the velocity block <= 15 v_max^2 (tree R_vir, comoving like the positions, and v_max of
       that galaxy's halo at s; v_max^2 is the largest G M(<r) / r). FAIL for a component backed by
       data (fitted count >= 1), FLAG for a prior-only one. Data alone cannot break it: bound
       particles lie inside the search sphere and below the escape speed, so it catches a covariance
       the prior or the regularisation has inflated (a runaway component). SCORE: the largest ratios
       over data-backed components.
1.3 run.log trends, snapshots with N_s >= 1e5 only (FLAG, never FAIL): see LOG_RULES. AVG_CONF,
    AVG_ENTROPY, LOG_COND, MED_RET, LOAD or PROCESS more than 5 MADs (floored at 1% of the median) off
    its rolling median over 9 snapshots is a spike (EMPTY and UNASSIGNED are ratios of small counts:
    thresholds only).
1.4 warnings.log (FLAG): > 1 "retrying with double" in a snapshot; any storage precision warning; a
    negative-f summary above 1e-3 of a density's mass; any "best lower bound >5% better than final".

Every FAIL and FLAG above is a symptom at its snapshot, with the galaxies it names; symptoms sharing a
galaxy, or found by the same check without galaxies, within 100 Myr or in adjacent snapshots, are one
episode, reported once (report.py). SCORE: episodes per cause (the check that found them).

Writes evaluation/soundness.txt and .log, plots/run/soundness_runlog.pdf (the run.log columns against
time), and plots/g<id>/soundness_gaussian.pdf (the 1.2j ratios against time) for the plotted galaxies.
"""
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):                   # run as a file: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import cli, plots
from evaluate.load import UNASSIGNED, Run, pool
from evaluate.report import Report

MIN_PARTICLES_LOG = 1e5                 # 1.3 applies to snapshots with at least this many particles
RESP_SUM_TOL, LOGR_TOL, ARGMAX_AGREE = 1e-3, 1e-6, 0.999
SYM_TOL, EIG_TOL = 1e-5, 1e-4
K_HALO = 15.0                           # 1.2j: Gaussian eigenvalues <= K_HALO x R_vir^2, x v_max^2
EMPTY_COUNT = 1.0                       # fitted count below which a component is prior-only (statistics.py)
NEG_F_MASS = 1e-3
SPIKE_WINDOW, SPIKE_MADS, MAD_FLOOR = 9, 5.0, 0.01

# column: (flag when "<" or ">" the threshold, the threshold, how it is scored)
LOG_RULES = {
    "AVG_CONF": ("<", 0.95, "median"),
    "AVG_ENTROPY": (">", 0.15, "median"),
    "LOG_COND": (">", 5.0, "median"),
    "BAD_COND": (">", 0.0, "sum"),
    "EMPTY_FRAC": (">", 0.2, "median"),
    "UNASSIGNED_FRAC": (">", 0.02, "median"),
    "MED_RET": ("<", 0.9, "median"),
    "FRAGS_FRAC": (None, None, "median"),
}
# averages and times; EMPTY and UNASSIGNED are ratios of small counts, judged by thresholds only
SPIKE_COLUMNS = ["AVG_CONF", "AVG_ENTROPY", "LOG_COND", "MED_RET", "LOAD", "PROCESS"]

WARNING = re.compile(r"^\[(?P<ts>[^\]]+)\] Snapshot (?P<snap>-?\d+): (?P<cat>\w+): (?P<msg>.*?)"
                     r"(?: \((?P<src>[^()]*:\d+)\))?$", re.S)
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")
NEG_F = re.compile(r"at most ([-+0-9.e]+) \(median")
CENTRE = ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z"]


def read_runlog(path):
    """The run.log table, one row per snapshot (the last for a resumed one), ``--`` as NaN."""
    lines = path.read_text().splitlines()
    head = next(line.split() for line in lines if line.split()[:2] == ["RUNTIME", "SNAP"])
    rows = [line.split() for line in lines
            if len(line.split()) == len(head) and line.split()[1].isdigit()]
    df = pd.DataFrame(rows, columns=head).replace("--", np.nan)
    df = df.astype({c: float for c in head[1:]})
    df["SNAP"] = df["SNAP"].astype(int)
    return df.drop_duplicates("SNAP", keep="last").set_index("SNAP").sort_index()


def spikes(x, window=SPIKE_WINDOW, n_mad=SPIKE_MADS):
    """Points more than ``n_mad`` robust sigmas (MAD floored at 1% of the median) off the rolling median."""
    med = x.rolling(window, center=True, min_periods=3).median()
    dev = (x - med).abs()
    mad = dev.rolling(window, center=True, min_periods=3).median()
    return dev > n_mad * 1.4826 * np.maximum(mad, MAD_FLOOR * med.abs())


def read_warnings(path):
    """warnings.log entries: snapshot, category, message, and its type (numbers replaced by #)."""
    if not path.exists():
        return pd.DataFrame(columns=["snap", "cat", "msg", "type"])
    entries = [WARNING.match(e.strip()) for e in path.read_text().split("\n---") if e.strip()]
    df = pd.DataFrame([(int(m["snap"]), m["cat"], m["msg"]) for m in entries if m],
                      columns=["snap", "cat", "msg"])
    df["type"] = df["cat"] + ": " + df["msg"].map(lambda s: NUMBER.sub("#", s)[:90])
    return df


def completion(run, rep, log):
    """1.1."""
    prog = run.path / "progress.json"
    if not prog.exists():
        rep.symptom("1.1 progress.json", "FAIL", "missing")
    else:
        p = json.loads(prog.read_text())
        last = int(run.planned.max())
        if not p.get("is_finished") or p.get("last_completed_snapshot") != last:
            rep.symptom("1.1 progress.json", "FAIL",
                        f"is_finished={p.get('is_finished')}, last={p.get('last_completed_snapshot')}")
    if "--- Run complete ---" not in (run.path / "run.log").read_text():
        rep.symptom("1.1 run.log", "FAIL", "no '--- Run complete ---'")
    err = run.path / "error.log"
    if err.exists() and err.stat().st_size:
        rep.symptom("1.1 error.log", "FAIL", f"{err.read_text().count(chr(10) + '---')} entries")
    for s in run.planned:                               # a crashed run's unwritten snapshots fail here
        for check, missing in (("1.1 assignment file missing", not run.assignment_file(s).exists()),
                               ("1.1 catalogue snapshot missing", f"snapshots/{s}" not in run.cat),
                               ("1.1 run.log row missing", s not in log.index)):
            if missing:
                rep.symptom(check, "FAIL", f"snapshot {s}", s0=int(s), t0=run.time.get(int(s)))


def has_cycle(rel):
    """Whether the host -> satellite graph has a cycle (Kahn: some node is never left without hosts)."""
    hosts_left = {}
    for h, sats in rel.items():
        hosts_left.setdefault(h, 0)
        for g in sats:
            hosts_left[g] = hosts_left.get(g, 0) + 1
    free = [n for n, k in hosts_left.items() if k == 0]
    peeled = 0
    while free:
        n = free.pop()
        peeled += 1
        for g in rel.get(n, ()):
            hosts_left[g] -= 1
            if hosts_left[g] == 0:
                free.append(g)
    return peeled < len(hosts_left)


def halo_ratios(cov, mean, r_vir, v_max):
    """Largest eigenvalue of the position and of the velocity block over R_vir^2 and v_max^2 (1.2j)."""
    cov = np.diag(np.broadcast_to(cov, mean.shape)) if np.ndim(cov) < 2 else cov
    return (np.linalg.eigvalsh(cov[:3, :3]).max() / r_vir ** 2,
            np.linalg.eigvalsh(cov[3:, 3:]).max() / v_max ** 2)


def snapshot_checks(run, s, log, halos, rep):
    """1.2 at snapshot ``s`` (``halos``: the tree's halos there, by ``Sub_tree_id``).

    Returns the counts for 1.3, the particle count of every label, and the 1.2j ratios of every
    Gaussian: ``{galaxy: (lambda_x / R_vir^2, lambda_v / v_max^2, prior-only)}``.
    """
    ids, lab = run.row_labels(s)
    p = run.pairs(s)
    rows, gid, logr = p["row"].astype(np.int64), p["galaxy"].astype(np.int64), p["log_resp"]
    with h5py.File(run.assignment_file(s), "r") as f:
        fitted = np.array([int(k) for k in f["galaxies"]], np.int64)
    gal = run.galaxies(s)

    def found(check, text, galaxies=(), severity="FAIL"):
        rep.symptom(check, severity, f"snapshot {s}: {text}", s0=int(s), t0=run.time[s], galaxies=galaxies)

    # a, b: particle IDs and labels
    if (dup := ids.size - np.unique(ids).size):
        found("1.2a duplicate particle IDs", f"{dup}")
    unknown = ~np.isin(lab, np.append(fitted, UNASSIGNED))
    if unknown.any():
        g = np.unique(lab[unknown]).tolist()
        found("1.2b labels outside the fitted galaxies", f"{unknown.sum()} particles, galaxies {g}", g)

    # c: the catalogue
    bad_m = gal.index[~np.isfinite(gal["Mtot"]) | (gal["Mtot"] <= 0)].to_list()
    if bad_m:
        found("1.2c catalogue Mtot not finite or <= 0", f"galaxies {bad_m}", bad_m)
    n_nan = gal[CENTRE].isna().sum(axis=1)
    partly = gal.index[(n_nan > 0) & (n_nan < len(CENTRE))].to_list()
    if partly:
        found("1.2c catalogue centre partly NaN", f"galaxies {partly}", partly)
    alive = halos.index.to_numpy(np.int64)
    off_tree = gal.index[~np.isin(gal.index.to_numpy(), alive)].to_list()
    if off_tree:
        found("1.2c catalogue galaxies not in the tree", f"galaxies {off_tree}", off_tree)

    # d, e: run.log and particle data
    n_unassigned = int((lab == UNASSIGNED).sum())
    logged = log.at[s, "UNASSIGNED"] if s in log.index else np.nan
    if np.isfinite(logged) and int(logged) != n_unassigned:
        found("1.2d run.log UNASSIGNED != -1 labels", f"{int(logged)} vs {n_unassigned}")
    if run.pdata is not None:
        missing = int(np.isnan(run.masses(s)).sum())
        if missing:
            found("1.2e particles missing from the particle data", f"{missing}")

    # f, g: responsibilities
    n = np.bincount(rows, minlength=ids.size)
    tot = np.bincount(rows, weights=np.exp(logr), minlength=ids.size)
    if (bad := int((np.abs(tot[n > 0] - 1.0) > RESP_SUM_TOL).sum())):
        found("1.2f responsibilities do not sum to 1",
              f"{bad} particles, worst {np.abs(tot[n > 0] - 1).max():.2e}")
    if (pos := int((logr > LOGR_TOL).sum())):
        found("1.2f log r > 0", f"{pos} entries, max {logr.max():.2e}")
    if rows.size:
        o = np.lexsort((-logr, rows))
        best = o[np.r_[True, rows[o][1:] != rows[o][:-1]]]
        agree = float((gid[best] == lab[rows[best]]).mean())
        if agree < ARGMAX_AGREE:
            found("1.2g argmax r != hard label", f"agreement {agree:.5f}")
    with_r = int((lab[n > 0] == UNASSIGNED).sum())
    without_r = int((lab[n == 0] != UNASSIGNED).sum())
    if with_r or without_r:
        found("1.2g -1 label <=> no responsibilities",
              f"{with_r} unassigned with r, {without_r} assigned without")

    # h, j: the Gaussians
    ratios = {}
    for g, (mean, cov, _, count) in run.gaussians(s).items():
        finite = np.isfinite(mean).all() and np.isfinite(cov).all()
        full = np.atleast_2d(cov)
        eig = np.linalg.eigvalsh(0.5 * (full + full.T)) if finite else np.array([np.nan])
        symmetric = finite and np.abs(full - full.T).max() <= SYM_TOL * np.abs(full).max()
        if not (symmetric and eig.min() >= -EIG_TOL * eig.max()):
            found("1.2h Gaussian not finite / symmetric / positive definite", f"galaxy {g}", [g])
        elif g in halos.index:
            rx, rv = halo_ratios(cov, mean, halos.at[g, "virial_radius"], halos.at[g, "vmax"])
            ratios[g] = (rx, rv, count < EMPTY_COUNT)
            if max(rx, rv) > K_HALO:
                found(f"1.2j Gaussian wider than its halo, > {K_HALO:g} R_vir^2 or v_max^2 "
                      "(data-backed FAIL, prior-only FLAG)",
                      f"galaxy {g}: lambda_x / R_vir^2 = {rx:.3g}, lambda_v / v_max^2 = {rv:.3g}, "
                      f"count {count:.3g}",
                      [g], "FAIL" if count >= EMPTY_COUNT else "FLAG")

    # i: satellite relations
    rel = run.satellites(s)
    off = (set(rel) | {g for v in rel.values() for g in v}) - set(alive.tolist())
    if off:
        found("1.2i relation galaxies not tree halos", f"galaxies {sorted(off)}", off)
    if has_cycle(rel):
        found("1.2i satellite relations have a cycle", "the host -> satellite graph")
    if (h := sorted(h for h, v in rel.items() if run.host in v)):
        found("1.2i main host is a satellite", f"of {h}", [run.host], "FLAG")

    data_backed = [(rx, rv) for rx, rv, prior in ratios.values() if not prior]
    counts = {
        "N_s": ids.size, "n_fit": fitted.size, "n_gal": len(gal),
        "nan_centre": int((n_nan == len(CENTRE)).sum()), "nan_rh": int(gal["rh"].isna().sum()),
        "widest_x": max((x for x, _ in data_backed), default=0.0),
        "widest_v": max((v for _, v in data_backed), default=0.0),
    }
    g, n_lab = np.unique(lab[lab != UNASSIGNED], return_counts=True)
    return counts, dict(zip(g.tolist(), n_lab.tolist())), ratios


def log_trends(run, log, counts, rep, pdf):
    """1.3, and the panel of the run.log columns against time."""
    df = log.join(counts, how="inner")
    df["EMPTY_FRAC"] = df["EMPTY"] / df["n_fit"]
    df["UNASSIGNED_FRAC"] = df["UNASSIGNED"] / df["N_s"]
    df["FRAGS_FRAC"] = df["FRAGS"] / df["n_fit"]
    df = df[df["N_s"] >= MIN_PARTICLES_LOG]
    t = np.array([run.time[s] for s in df.index])

    flagged = {}
    for col, (op, thr, how) in LOG_RULES.items():
        x = df[col]
        rep.score(f"1.3 {col} ({how})", float(x.sum() if how == "sum" else x.median()))
        if op is not None:
            bad = (x < thr if op == "<" else x > thr).to_numpy()
            flagged[col] = bad
            for s in df.index[bad]:
                rep.symptom(f"1.3 {col} {op} {thr:g}", "FLAG", f"snapshot {s}: {df.at[s, col]:.4g}",
                            s0=int(s), t0=run.time[s])
    for col in SPIKE_COLUMNS:
        sp = spikes(df[col]).to_numpy()
        flagged[col] = flagged.get(col, np.zeros(len(df), bool)) | sp
        for s in df.index[sp]:
            rep.symptom(f"1.3 {col} spike", "FLAG", f"snapshot {s}: {df.at[s, col]:.4g}",
                        s0=int(s), t0=run.time[s])
    rep.score("1.3 snapshots in scope (N_s >= 1e5)", len(df))

    fig, axes = plt.subplots(5, 2, figsize=(11, 13), sharex=True)
    for ax, col in zip(axes.flat, list(LOG_RULES) + ["LOAD", "PROCESS"]):
        ax.plot(t, df[col], lw=0.8, color="0.3")
        f = flagged.get(col, np.zeros(len(df), bool))
        ax.plot(t[f], df[col].to_numpy()[f], "o", ms=3, color="C3")
        op, thr, _ = LOG_RULES.get(col, (None, None, None))
        if op is not None:
            ax.axhline(thr, ls="--", lw=0.7, color="C3")
        ax.set_ylabel(col, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t [Gyr]")
    fig.suptitle(f"{run.name}: run.log (red: flagged)")
    plots.save(fig, pdf)


def warning_checks(run, rep):
    """1.4."""
    w = read_warnings(run.path / "warnings.log")
    if w.empty:
        rep.table("1.4 warnings.log", "  (none)")
        return

    def found(check, s, text):
        rep.symptom(check, "FLAG", f"snapshot {s}: {text}", s0=int(s), t0=run.time.get(int(s)))

    retry = w[w["msg"].str.contains("retrying with double")].groupby("snap").size()
    for s, k in retry[retry > 1].items():
        found("1.4 > 1 retry with double in a snapshot", s, f"{k}")
    for _, e in w[w["msg"].str.contains("cannot hold|non-finite value")].iterrows():
        found("1.4 storage precision", e.snap, e.msg[:100])
    for _, e in w[w["msg"].str.contains("negative and clipped")].iterrows():
        m = NEG_F.search(e.msg)
        if m and float(m.group(1)) > NEG_F_MASS:
            found(f"1.4 negative f above {NEG_F_MASS:g} of a density's mass", e.snap,
                  f"{float(m.group(1)):.1e}")
    for _, e in w[w["msg"].str.contains("better than final")].iterrows():
        found("1.4 best lower bound better than final", e.snap, "")

    tab = w.groupby("type")["snap"].agg(["size", "nunique", "min", "max"])
    tab = tab.sort_values("size", ascending=False)
    tab.columns = ["count", "snapshots", "first", "last"]
    rep.table("1.4 warnings.log by type", tab.to_string())


def _draw_gaussian(item):
    """A galaxy's 1.2j ratios against time (one PDF)."""
    path, g, t, rx, rv, prior = item
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(t, rx, "o-", ms=3, lw=0.8, label="position: lambda_max / R_vir^2")
    ax.semilogy(t, rv, "s-", ms=3, lw=0.8, label="velocity: lambda_max / v_max^2")
    ax.plot(t[prior], np.maximum(rx, rv)[prior], "x", color="C3", ms=6, label="prior-only fit")
    ax.axhline(K_HALO, ls="--", lw=0.7, color="C3")
    ax.set_xlabel("t [Gyr]")
    ax.set_ylabel("Gaussian width / halo scale")
    ax.set_title(f"galaxy {g}: 1.2j (dashed: the bound)")
    ax.legend(fontsize=8)
    plots.save(fig, path)


_W = {}                                 # a worker's run, run.log table and tree halos
_NO_HALOS = pd.DataFrame(columns=["virial_radius", "vmax"],
                         index=pd.Index([], dtype=np.int64, name="Sub_tree_id"))


def _init(path, source, snaps, log, halos):
    run = Run(path, particles=source)
    run.snaps = snaps
    _W.update(run=run, log=log, halos=halos)


def _snapshot(s):
    """1.2 at ``s`` in a worker: its own report part, counts, label counts and 1.2j ratios."""
    rep = Report("", "")
    found = snapshot_checks(_W["run"], s, _W["log"], _W["halos"].get(s, _NO_HALOS), rep)
    return (rep, *found)


def evaluate(run, out, workers=1):
    rep = Report("Point 1, technically sound", run.name, point="1")
    log = read_runlog(run.path / "run.log")
    completion(run, rep, log)

    halos = {int(s): g.set_index("Sub_tree_id")[["virial_radius", "vmax"]]
             for s, g in run.tree.groupby("Snapshot")}
    todo = [int(s) for s in run.snaps if run.assignment_file(s).exists() and f"snapshots/{s}" in run.cat]
    counts, peak, widths = {}, {}, []
    parts = pool(_snapshot, todo, workers, _init, (run.path, run.source, run.snaps, log, halos),
                 desc="1.2 snapshots")
    for s, (part, c, n_lab, ratios) in zip(todo, parts):
        rep.merge(part)
        counts[s] = c
        for g, n in n_lab.items():
            peak[g] = max(peak.get(g, 0), n)
        widths += [(g, s, run.time[s], rx, rv, prior) for g, (rx, rv, prior) in ratios.items()]
    counts = pd.DataFrame.from_dict(counts, orient="index")

    rep.score("1.2c catalogue galaxies with NaN centre (<= 2 particles), share",
              float(counts["nan_centre"].sum() / counts["n_gal"].sum()))
    rep.score("1.2c catalogue galaxies with NaN rh, share",
              float(counts["nan_rh"].sum() / counts["n_gal"].sum()))
    rep.score("1.2j largest lambda_x / R_vir^2 (data-backed components)", float(counts["widest_x"].max()))
    rep.score("1.2j largest lambda_v / v_max^2 (data-backed components)", float(counts["widest_v"].max()))
    log_trends(run, log, counts, rep, plots.run_plot(out, "soundness_runlog"))
    warning_checks(run, rep)

    named = rep.episodes()
    widths = pd.DataFrame(widths, columns=["g", "snap", "t", "rx", "rv", "prior"])
    items = []
    for g in plots.chosen(run.host, peak, named):
        w = widths[widths["g"] == g].sort_values("t")
        if not w.empty:
            items.append((plots.galaxy_plot(out, g, "soundness_gaussian"), g, w["t"].to_numpy(),
                          w["rx"].to_numpy(), w["rv"].to_numpy(), w["prior"].to_numpy(bool)))
    plots.per_galaxy(_draw_gaussian, items, workers, desc="1 galaxy plots")
    rep.write(out / "soundness.txt")
    return rep


def main():
    args = cli.arguments(__doc__)
    run, out = cli.start("soundness", args)
    evaluate(run, out, workers=args.workers)


if __name__ == "__main__":                      # spawned workers import this file, never run it
    main()

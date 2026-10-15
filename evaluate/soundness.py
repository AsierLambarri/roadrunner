"""Point 1: the run is technically sound.

1.1 completion: progress.json finished at the last snapshot; run.log ends with
    "--- Run complete ---"; error.log absent or empty; every snapshot has its
    assignment file, catalogue group and run.log row. Any miss is a FAIL.
1.2 consistency in each snapshot (FAIL):
    a. particle IDs unique;
    b. hard labels in the fitted galaxies of the assignment file, or -1;
    c. catalogue: Mtot finite and > 0; position and velocity all finite or all
       NaN (NaN for galaxies of <= 2 particles; rh, r20, r80, sigma may be NaN
       alone, for small ones); galaxies are tree halos of that snapshot;
    d. run.log UNASSIGNED = number of -1 labels;
    e. the particle IDs are those of the particle data (when given);
    f. |sum_k r - 1| < 1e-3 for every particle with responsibilities, log r <= 1e-6;
    g. argmax_k log r = hard label for >= 99.9% of them; -1 <=> no responsibilities;
    h. Gaussians: finite, symmetric to 1e-5 of max|Sigma|, smallest eigenvalue
       >= -1e-4 x the largest (the storage tolerance; 2-particle fits sit at ~0);
    i. satellite relations: hosts and satellites are tree halos of that snapshot
       (a satellite may have several hosts; the main host as a satellite is a FLAG).
1.3 run.log trends, snapshots with N_s >= 1e5 only (FLAG, never FAIL): see LOG_RULES;
    AVG_CONF, AVG_ENTROPY, LOG_COND, MED_RET, LOAD or PROCESS more than 5 MADs
    (floored at 1% of the median) off its rolling median over 9 snapshots is a
    spike (EMPTY and UNASSIGNED are ratios of small counts: thresholds only).
1.4 warnings.log (FLAG): > 1 "retrying with double" in a snapshot; any storage
    precision warning; a negative-f summary above 1e-3 of a density's mass; any
    "best lower bound >5% better than final".
"""
import json
import re

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate.load import UNASSIGNED
from evaluate.report import Report

MIN_PARTICLES_LOG = 1e5                 # 1.3 applies to snapshots with at least this many particles
RESP_SUM_TOL, LOGR_TOL, ARGMAX_AGREE = 1e-3, 1e-6, 0.999
SYM_TOL, EIG_TOL = 1e-5, 1e-4
NEG_F_MASS = 1e-3
SPIKE_WINDOW, SPIKE_MADS, MAD_FLOOR = 9, 5.0, 0.01

# column: (flag when "<" or ">" the threshold, threshold, score)
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
SPIKE_COLUMNS = ["AVG_CONF", "AVG_ENTROPY", "LOG_COND", "MED_RET", "LOAD", "PROCESS"]   # averages and times, not small-count ratios

WARNING = re.compile(r"^\[(?P<ts>[^\]]+)\] Snapshot (?P<snap>-?\d+): (?P<cat>\w+): (?P<msg>.*?)(?: \((?P<src>[^()]*:\d+)\))?$", re.S)
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")
NEG_F = re.compile(r"at most ([-+0-9.e]+) \(median")


def read_runlog(path):
    """The run.log table, one row per snapshot (the last for a resumed one), ``--`` as NaN."""
    lines = path.read_text().splitlines()
    head = next(l.split() for l in lines if l.split()[:2] == ["RUNTIME", "SNAP"])
    rows = [l.split() for l in lines if len(l.split()) == len(head) and l.split()[1].isdigit()]
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
    df = pd.DataFrame([(int(m["snap"]), m["cat"], m["msg"]) for m in entries if m], columns=["snap", "cat", "msg"])
    df["type"] = df["cat"] + ": " + df["msg"].map(lambda s: NUMBER.sub("#", s)[:90])
    return df


def completion(run, rep, log):
    """1.1."""
    prog = run.path / "progress.json"
    if not prog.exists():
        rep.fail("1.1 progress.json", "missing")
    else:
        p = json.loads(prog.read_text())
        if not p.get("is_finished") or p.get("last_completed_snapshot") != int(run.cat["header/snapshots"][:].max()):
            rep.fail("1.1 progress.json", f"is_finished={p.get('is_finished')}, last={p.get('last_completed_snapshot')}")
    if "--- Run complete ---" not in (run.path / "run.log").read_text():
        rep.fail("1.1 run.log", "no '--- Run complete ---'")
    err = run.path / "error.log"
    if err.exists() and err.stat().st_size:
        rep.fail("1.1 error.log", f"{err.read_text().count(chr(10) + '---')} entries")
    for s in run.snaps:
        if not run.assignment_file(s).exists():
            rep.fail("1.1 assignment file", f"snapshot {s}")
        if f"snapshots/{s}" not in run.cat:
            rep.fail("1.1 catalogue snapshot", f"snapshot {s}")
        if s not in log.index:
            rep.fail("1.1 run.log row", f"snapshot {s}")


def snapshot_checks(run, s, log, tree_at, rep):
    """1.2 at snapshot ``s``; returns its counts for 1.3."""
    ids, lab = run.row_labels(s)
    p = run.pairs(s)
    rows, gid, logr = p["row"].astype(np.int64), p["galaxy"].astype(np.int64), p["log_resp"]
    with h5py.File(run.assignment_file(s), "r") as f:
        fitted = np.array([int(k) for k in f["galaxies"]], np.int64)
    gal = run.galaxies(s)

    if (dup := ids.size - np.unique(ids).size):
        rep.fail("1.2a duplicate particle IDs", f"snapshot {s}: {dup}")
    unknown = ~np.isin(lab, np.append(fitted, UNASSIGNED))
    if unknown.any():
        rep.fail("1.2b labels outside the fitted galaxies", f"snapshot {s}: {unknown.sum()} particles, galaxies {np.unique(lab[unknown])[:5]}")

    bad_m = ~np.isfinite(gal["Mtot"]) | (gal["Mtot"] <= 0)
    if bad_m.any():
        rep.fail("1.2c catalogue Mtot not finite or <= 0", f"snapshot {s}: galaxies {gal.index[bad_m].to_list()[:5]}")
    centre = ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z"]
    n_nan = gal[centre].isna().sum(axis=1)
    if ((n_nan > 0) & (n_nan < len(centre))).any():
        rep.fail("1.2c catalogue centre partly NaN", f"snapshot {s}: galaxies {gal.index[(n_nan > 0) & (n_nan < len(centre))].to_list()[:5]}")
    alive = tree_at.get(s, np.array([], np.int64))
    off_tree = ~np.isin(gal.index.to_numpy(), alive)
    if off_tree.any():
        rep.fail("1.2c catalogue galaxies not in the tree", f"snapshot {s}: {gal.index[off_tree].to_list()[:5]}")

    n_unassigned = int((lab == UNASSIGNED).sum())
    if s in log.index and np.isfinite(log.at[s, "UNASSIGNED"]) and int(log.at[s, "UNASSIGNED"]) != n_unassigned:
        rep.fail("1.2d run.log UNASSIGNED != -1 labels", f"snapshot {s}: {int(log.at[s, 'UNASSIGNED'])} vs {n_unassigned}")

    if run.pdata is not None:
        missing = int(np.isnan(run.masses(s)).sum())
        if missing:
            rep.fail("1.2e particles missing from the particle data", f"snapshot {s}: {missing}")

    n = np.bincount(rows, minlength=ids.size)
    tot = np.bincount(rows, weights=np.exp(logr), minlength=ids.size)
    if (bad := int((np.abs(tot[n > 0] - 1.0) > RESP_SUM_TOL).sum())):
        rep.fail("1.2f responsibilities do not sum to 1", f"snapshot {s}: {bad} particles, worst {np.abs(tot[n > 0] - 1).max():.2e}")
    if (pos := int((logr > LOGR_TOL).sum())):
        rep.fail("1.2f log r > 0", f"snapshot {s}: {pos} entries, max {logr.max():.2e}")
    if rows.size:
        o = np.lexsort((-logr, rows))
        best = o[np.r_[True, rows[o][1:] != rows[o][:-1]]]
        agree = float((gid[best] == lab[rows[best]]).mean())
        if agree < ARGMAX_AGREE:
            rep.fail("1.2g argmax r != hard label", f"snapshot {s}: agreement {agree:.5f}")
    with_r, without_r = int((lab[n > 0] == UNASSIGNED).sum()), int((lab[n == 0] != UNASSIGNED).sum())
    if with_r or without_r:
        rep.fail("1.2g -1 label <=> no responsibilities", f"snapshot {s}: {with_r} unassigned with r, {without_r} assigned without")

    for g, (mean, cov, _) in run.gaussians(s).items():
        cov = np.atleast_2d(cov)
        finite = np.isfinite(mean).all() and np.isfinite(cov).all()
        eig = np.linalg.eigvalsh(0.5 * (cov + cov.T)) if finite else np.array([np.nan])
        if not (finite and np.abs(cov - cov.T).max() <= SYM_TOL * np.abs(cov).max() and eig.min() >= -EIG_TOL * eig.max()):
            rep.fail("1.2h Gaussian not finite / symmetric / positive definite", f"snapshot {s}: galaxy {g}")

    rel = run.satellites(s)
    off = (set(rel) | {g for v in rel.values() for g in v}) - set(alive.tolist())
    if off:
        rep.fail("1.2i relation galaxies not tree halos", f"snapshot {s}: {sorted(off)[:5]}")
    if (h := sorted(h for h, v in rel.items() if run.host in v)):
        rep.flag("1.2i main host is a satellite", f"snapshot {s}: of {h}")

    return {"N_s": ids.size, "n_fit": fitted.size, "n_gal": len(gal),
            "nan_centre": int((n_nan == len(centre)).sum()), "nan_rh": int(gal["rh"].isna().sum())}


def log_trends(run, log, counts, rep, png):
    """1.3, and the trend panel."""
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
            bad = x < thr if op == "<" else x > thr
            flagged[col] = bad.to_numpy()
            for s in df.index[bad.to_numpy()]:
                rep.flag(f"1.3 {col} {op} {thr:g}", f"snapshot {s}: {df.at[s, col]:.4g}")
    for col in SPIKE_COLUMNS:
        sp = spikes(df[col]).to_numpy()
        flagged[col] = flagged.get(col, np.zeros(len(df), bool)) | sp
        for s in df.index[sp]:
            rep.flag(f"1.3 {col} spike", f"snapshot {s}: {df.at[s, col]:.4g}")
    rep.score("1.3 snapshots in scope (N_s >= 1e5)", len(df))

    cols = list(LOG_RULES) + ["LOAD", "PROCESS"]
    fig, axes = plt.subplots(5, 2, figsize=(11, 13), sharex=True)
    for ax, col in zip(axes.flat, cols):
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
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def warning_checks(run, rep):
    """1.4."""
    w = read_warnings(run.path / "warnings.log")
    if w.empty:
        rep.table("1.4 warnings.log", "  (none)")
        return
    retry = w[w["msg"].str.contains("retrying with double")].groupby("snap").size()
    for s, k in retry[retry > 1].items():
        rep.flag("1.4 > 1 retry with double in a snapshot", f"snapshot {s}: {k}")
    for _, e in w[w["msg"].str.contains("cannot hold|non-finite value")].iterrows():
        rep.flag("1.4 storage precision", f"snapshot {e.snap}: {e.msg[:100]}")
    for _, e in w[w["msg"].str.contains("negative and clipped")].iterrows():
        m = NEG_F.search(e.msg)
        if m and float(m.group(1)) > NEG_F_MASS:
            rep.flag(f"1.4 negative f above {NEG_F_MASS:g} of a density's mass", f"snapshot {e.snap}: {float(m.group(1)):.1e}")
    for _, e in w[w["msg"].str.contains("better than final")].iterrows():
        rep.flag("1.4 best lower bound better than final", f"snapshot {e.snap}")
    tab = w.groupby("type")["snap"].agg(["size", "nunique", "min", "max"]).sort_values("size", ascending=False)
    tab.columns = ["count", "snapshots", "first", "last"]
    rep.table("1.4 warnings.log by type", tab.to_string())


def evaluate(run, out):
    rep = Report("Point 1, technically sound", run.name)
    log = read_runlog(run.path / "run.log")
    completion(run, rep, log)
    tree_at = {int(s): g.to_numpy(np.int64) for s, g in run.tree.groupby("Snapshot")["Sub_tree_id"]}
    counts = {}
    for s in run.snaps:
        if run.assignment_file(s).exists() and f"snapshots/{s}" in run.cat:
            counts[s] = snapshot_checks(run, s, log, tree_at, rep)
    counts = pd.DataFrame.from_dict(counts, orient="index")
    rep.score("1.2c catalogue galaxies with NaN centre (<= 2 particles), share",
              float(counts["nan_centre"].sum() / counts["n_gal"].sum()))
    rep.score("1.2c catalogue galaxies with NaN rh, share", float(counts["nan_rh"].sum() / counts["n_gal"].sum()))
    log_trends(run, log, counts, rep, out / "soundness_runlog.png")
    warning_checks(run, rep)
    rep.write(out / "soundness.txt")
    return rep

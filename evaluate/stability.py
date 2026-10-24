"""Point 2: the assignments are stable in time.

    python3 evaluate/stability.py --path RUN [--particles DIR] [--workers N] [--snaps A:B]

Masses are hard-label masses (the assignment itself), not the catalogue's Mtot (infall list x bound
set). A galaxy is present at s when it labels >= 1 particle, resolved with >= 100, major with >= 1000;
it is bound at s when it has bound stars (stored pairs, i.e. it was fitted). A star is old at s when it
is present at the last snapshot >= 1 Gyr before, a newborn at s when absent from the previous snapshot:
its first appearance, as stars persist once formed (ages count from the run's first snapshot; with
--snaps, from the first one evaluated). Windows are times, not snapshot counts: "within T after s"
holds every snapshot with t_s < t <= t_s + T, and always at least the next one; a change itself is
between consecutive snapshots (s-1 -> s). The exchange at s is the mass of the particles labelled a at
s-1 and b != a at s, per (a, b).

Symptoms (each a candidate finding):
2.1 gaps: snapshots where a galaxy is absent between two present ones while the tree has its halo
    (where the tree lacks it too, the gap is not ours and is left out), of any length, for galaxies
    with N >= 10 particles just before (the pipeline's fragment size). A label gap keeps bound stars at
    every missing snapshot (the mixture gave them to others): FLAG, "absorbed by a neighbour" when one
    galaxy took >= half of its members, else "label gap". A bound gap has no bound star at some
    missing snapshot: the members' centroid there is compared with the tree halo's centre; beyond
    R_vir at any of them, "the tree lost the stars": FLAG; within R_vir at all of them, "bound gap with
    members in place": FAIL when N >= 100 and >= 2 snapshots are unbound, FLAG otherwise; without
    particle data to place them, "bound gap with members not located": FLAG.
2.2 mass histories (resolved at both ends of a step): a step is |log10 M(s) / M(s-1)| > log10 2; a
    spike is a step that comes back within 25% of M(s-1) within 300 Myr (its return legs are neither
    spikes nor steps). A persistent step is explained, in this order, by "tree incomplete" (the
    galaxy's own tree rows at s-1 or s have no uid), star formation (the newborns it labels at s weigh
    >= half the gain), a merger (a tree merger into it within +-150 Myr whose progenitor's last mass is
    >= half the gain), "merging" (a drop, its own branch ending within 300 Myr) or "pericentre" (a drop
    of a satellite of any host at its smallest distance to its most massive host within +-200 Myr,
    with snapshots on both sides); the last four override the first. Spikes and unexplained steps are
    symptoms: "exchange between interacting galaxies" when other galaxies (not -1) give or take >= half
    of the change in the exchange at s, else "unexplained". A spike of a major galaxy, or an
    unexplained step of the main host: FAIL if unexplained, FLAG if an exchange; any other: FLAG.
2.3 old-star flicker: L(s-1) = a, L(s) = b != a, back to a within 300 Myr (-1 is a label).
2.4 permanent transfers of old stars: L(s-1) = a, L(s) = b != a, b throughout the next 300 Myr.
    A change a -> b is: merger (a's branch ends at s-1 or s); unbound / rebound (b / a is -1);
    stripping (b among a's hosts) or capture (a among b's hosts), together "related"; swap
    ("unrelated"). Related and unrelated changes are split by boundness: "contested" when the star was
    bound to both galaxies at s-1 or at s (a choice of the mixture), "forced" when only to one (its
    boundness changed). A satellite's hosts: every more massive halo it is bound to inside its R_vir
    (the tree's relations).
    SCORE: median and max, over snapshots with >= 1e4 old stars, of the share of the old mass that
    flickers / transfers, in all and per class and cell; the capture share of the hosts' old mass.
    Symptom (FLAG): a snapshot whose flicker share, capture share, or unrelated flicker or transfer
    share is above max(5 x its median, 1e-4); it involves the pair (a, b) carrying > half of it.
2.5 newborns: kept when labelled as at s throughout the next 150 Myr. SCORE: kept mass share, overall,
    main host, satellites, other centrals.
A snapshot without the previous one, or without a full 300 Myr after it, gets no 2.3-2.5 window.
2.6 label-mass noise (measurement only, no symptoms): white noise sigma on y = log10 Mres, bracketed
    around any smooth history. At a resolved snapshot i between two resolved neighbours, the
    pseudo-residual e_i = a_i y_{i-1} + b_i y_{i+1} - y_i (a_i, b_i: the linear-interpolation weights
    from the times, so a linear trend cancels exactly at any spacing) has noise variance c_i^2 sigma^2,
    c_i^2 = a_i^2 + b_i^2 + 1, and neighbouring ones share two noise terms: covariance
    -(a_{i+1} + b_i) sigma^2. A curved history adds kappa_i^2 to the first and kappa_i kappa_{i+1} to the
    second (kappa: the history's own interpolation error), so sigma_hi^2 = sum e_i^2 / sum c_i^2 is an
    upper bound and sigma_lo^2 = max(-sum e_i e_{i+1} / sum (a_{i+1} + b_i), 0) a lower bound while the
    curvature keeps its sign over four snapshots; for pure noise both equal sigma (both are bounds in
    expectation: from few triples the two estimates can cross). Kept triples have no 2.2 step
    (|log10 ratio| > log10 2: 2.2 reports those, so this is the jitter below them) and no snapshot
    within MERGER_GYR of a tree merger whose progenitor's last label mass is >= MERGER_SHARE of the
    galaxy's (starless mergers change nothing); galaxies need >= MIN_NOISE_PAIRS adjacent kept triples.
    Noise correlated over several snapshots (slow flicker) counts as history, not noise. SCORE:
    galaxies measured, median sigma_lo and sigma_hi, the main host's; a table by the galaxy's median
    resolved mass.

Episodes (report.py): one finding per physical cause. Two symptoms are linked when they are within
100 Myr or in adjacent snapshots and share a galaxy (its own, or a partner that gave or took >= half of
the symptom's mass), or, both without galaxies, come from the same check; the linked symptoms form one
episode, reported once with all its symptoms, as FAIL if any of them is a FAIL, else FLAG, under its
cause (that of a FAIL symptom, else the first present of: the tree lost the stars, bound gap with
members in place, bound gap with members not located, absorbed by a neighbour, label gap, exchange
between interacting galaxies, unexplained, share spike). SCORE: episodes by cause.

Writes evaluation/stability.txt and .log; plots/run/: stability_masses.pdf (host and largest
galaxies), stability_shares.pdf (2.3-2.5 against time), stability_episodes.pdf (episodes by cause
against time), stability_noise.pdf (2.6 noise bounds by mass and in time); plots/g<id>/stability_mass.pdf
(its mass history and symptoms) for the plotted galaxies.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):                   # run as a file: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluate import cli, plots
from evaluate.load import (ABSENT, MAJOR, OLD_GYR, RESOLVED, UNASSIGNED, Run, member, mergers, newborn,
                           on, pool, seen_before)
from evaluate.report import Report

RETURN_GYR, STAY_GYR, KEEP_GYR = 0.3, 0.3, 0.15   # 2.2 spikes and 2.3 flicker, 2.4 transfers, 2.5 newborns
MERGER_GYR, MERGING_GYR, PERI_GYR = 0.15, 0.3, 0.2  # 2.2 explanations: +-, after, +-
WINDOW_GYR = max(RETURN_GYR, STAY_GYR, KEEP_GYR)   # 2.3-2.5 read this far ahead
LOG_JUMP, LOG_BACK = np.log10(2.0), np.log10(1.25)
GAP_FAIL, GAP_FLAG = 100, 10                        # 2.1: particles just before a gap
PARTNER = 0.5                                       # share of a symptom's mass that makes a partner
MIN_OLD = 1e4
MIN_NOISE_PAIRS, MERGER_SHARE = 10, 0.01            # 2.6: adjacent kept triples per galaxy; merger size
MASS_BINS = [0.0, 1e7, 1e8, 1e9, np.inf]
MASS_LABELS = ["<1e7", "1e7-1e8", "1e8-1e9", ">1e9"]
SPIKE_FACTOR, SHARE_FLOOR = 5.0, 1e-4
POS = ["position_x", "position_y", "position_z"]
TRANSFERS = ["merger", "unbound", "rebound", "stripping", "capture", "swap"]
CELLS = ["contested related", "forced related", "contested unrelated", "forced unrelated"]
NEWBORN = ["all", "main host", "satellites", "other centrals"]
SHARES = {
    "flicker": "2.3 flicker",
    "flicker unrelated": "2.3 unrelated flicker",
    "transfer unrelated": "2.4 unrelated transfer",
    "capture of host old mass": "2.4 capture",
}
CAUSES = [
    "the tree lost the stars", "bound gap with members in place", "bound gap with members not located",
    "absorbed by a neighbour", "label gap", "exchange between interacting galaxies", "unexplained",
    "share spike",
]
COLOR = dict(zip(CAUSES, ["C0", "C1", "C2", "C4", "C5", "C6", "C8", "C7"]))   # no red: red is kept for FAIL


# --- label changes ------------------------------------------------------------------------------

def relation(a, b, hosts, ends):
    """Relation of a label change a -> b at a snapshot (see 2.4); ``hosts``: satellite -> its hosts."""
    if a in ends:
        return "merger"
    if b == UNASSIGNED:
        return "unbound"
    if a == UNASSIGNED:
        return "rebound"
    if b in hosts.get(a, ()):
        return "stripping"
    if a in hosts.get(b, ()):
        return "capture"
    return "swap"


def classify(a, b, hosts, ends):
    """:func:`relation` of every pair, computed once per distinct pair."""
    if a.size == 0:
        return np.array([], dtype=object)
    pairs, inv = np.unique(np.stack([a, b]), axis=1, return_inverse=True)
    rel = np.array([relation(int(x), int(y), hosts, ends) for x, y in pairs.T], dtype=object)
    return rel[inv.ravel()]


def cells(cls, contested):
    """2.4 cell of each change: contested / forced x related / unrelated ('' for merger, unbound, rebound)."""
    related, unrelated = np.isin(cls, ["stripping", "capture"]), cls == "swap"
    conditions = [related & contested, related & ~contested, unrelated & contested, unrelated & ~contested]
    return np.select(conditions, CELLS, "")


def ahead(t, k, T):
    """Indices of the snapshots within ``T`` after the k-th (``t_k < t <= t_k + T``), at least the next one."""
    end = np.searchsorted(t, t[k] + T, side="right")
    return np.arange(k + 1, min(max(end, k + 2), t.size))


def within(A, dt, T):
    """The rows of ``A`` (window snapshots, times ``dt`` after s, ascending) within ``T``; at least the first."""
    return A[:max(1, int((dt <= T).sum()))]


def candidates(run, s, factor):
    """Sorted keys ``id x factor + galaxy`` of the (particle, galaxy) pairs bound at ``s`` (stored pairs)."""
    p = run.pairs(s)
    ids = run.row_labels(s)[0][p["row"].astype(np.int64)]
    if ids.size and int(ids.max()) >= np.iinfo(np.int64).max // factor:
        raise OverflowError(f"particle IDs up to {ids.max()} x {factor} galaxy slots overflow int64 keys")
    return np.sort(ids * factor + p["galaxy"].astype(np.int64))


def exchange(ids, lab, m, prev):
    """Mass per (a, b), a != b, of the particles labelled a at the previous snapshot and b now."""
    a = on(ids, *prev)
    moved = (a != ABSENT) & (a != lab)
    return pd.DataFrame({"a": a[moved], "b": lab[moved], "m": m[moved]}).groupby(["a", "b"])["m"].sum()


def partners(flow, G, change):
    """Galaxies (not -1) exchanging with ``G`` in ``flow``, toward the sign of ``change``: their total
    mass, and the galaxies carrying >= PARTNER of |change|."""
    if flow is None or flow.empty:
        return 0.0, set()
    df = flow.reset_index()
    if change > 0:
        side = df[(df["b"] == G) & (df["a"] >= 0)]
        other = side["a"]
    else:
        side = df[(df["a"] == G) & (df["b"] >= 0)]
        other = side["b"]
    big = set(other[side["m"] >= PARTNER * abs(change)].astype(int).tolist())
    return float(side["m"].sum()), big


# --- verdicts -----------------------------------------------------------------------------------

def steps(M, t):
    """Spikes and persistent steps of a (galaxy x snapshot) mass table, NaN where not resolved; ``t`` the
    snapshots' times. A spike's return legs (its steps up to the return) are neither spikes nor steps."""
    lm = np.log10(M.to_numpy(np.float64))
    step = np.zeros(lm.shape, bool)
    step[:, 1:] = np.abs(lm[:, 1:] - lm[:, :-1]) > LOG_JUMP
    spike, leg = np.zeros_like(step), np.zeros_like(step)
    for c in range(1, lm.shape[1]):
        open_ = step[:, c] & ~leg[:, c]
        for j in ahead(t, c, RETURN_GYR):
            back = open_ & (np.abs(lm[:, j] - lm[:, c - 1]) < LOG_BACK)
            spike[:, c] |= back
            leg[:, c + 1:j + 1] |= back[:, None]
            open_ &= ~back
    return spike, step & ~spike & ~leg


def gap_verdict(n0, bound_missing, place, partner_share):
    """2.1 severity and cause of a gap.

    ``bound_missing`` is the number of snapshots without bound stars; ``place`` where the members were
    then: "in" (within R_vir of the tree centre at all of them), "out" (beyond at some) or None (not
    located); ``partner_share`` the largest share one galaxy took of the members.
    """
    if bound_missing == 0:
        return "FLAG", "absorbed by a neighbour" if partner_share >= PARTNER else "label gap"
    if place is None:
        return "FLAG", "bound gap with members not located"
    if place == "out":
        return "FLAG", "the tree lost the stars"
    severity = "FAIL" if n0 >= GAP_FAIL and bound_missing >= 2 else "FLAG"
    return severity, "bound gap with members in place"


# --- one 2.3-2.5 window ---------------------------------------------------------------------------

def window(run, k, t, labs, cands, H, ends, pair_mass, factor):
    """2.3-2.5 at the k-th snapshot of the run (``H``: its satellite -> hosts): one row of shares, with
    the pair dominating each share; adds the flickering pairs to ``pair_mass``."""
    s = run.snaps[k]
    ids, b = labs[k]
    nxt = ahead(t, k, WINDOW_GYR)
    dt = t[nxt] - t[k]
    a = on(ids, *labs[k - 1])
    A = np.stack([on(ids, *labs[j]) for j in nxt])
    m = np.nan_to_num(run.masses(s))

    present = (a != ABSENT) & (A != ABSENT).all(0)
    old = present & seen_before(run, s, ids, OLD_GYR)
    moved = old & (b != a)
    flick = moved & (within(A, dt, RETURN_GYR) == a).any(0)
    stay = moved & (within(A, dt, STAY_GYR) == b).all(0)
    contested = member(cands[k - 1], ids * factor + b) | member(cands[k], ids * factor + a)
    m_old = max(m[old].sum(), 1e-300)            # no old stars: shares 0, and the snapshot is out of scope
    row = {"snap": s, "t": t[k], "n_old": int(old.sum())}

    def share(col, sel, norm):
        """The share of ``sel`` in ``norm``, and the pair carrying most of it with its part."""
        row[col] = m[sel].sum() / norm
        pm = pd.DataFrame({"a": a[sel], "b": b[sel], "m": m[sel]}).groupby(["a", "b"])["m"].sum()
        row[f"{col} pair"] = tuple(int(x) for x in pm.idxmax()) if pm.size else None
        row[f"{col} pair part"] = float(pm.max() / pm.sum()) if pm.size else 0.0

    for name, sel in (("flicker", flick), ("transfer", stay)):
        cls = np.full(ids.size, "", dtype=object)
        cls[sel] = classify(a[sel], b[sel], H, ends)
        cell = np.full(ids.size, "", dtype=object)
        cell[sel] = cells(cls[sel], contested[sel])
        row[name] = m[sel].sum() / m_old
        row.update({f"{name} {c}": m[sel & (cls == c)].sum() / m_old for c in TRANSFERS})
        row.update({f"{name} {c}": m[sel & (cell == c)].sum() / m_old for c in CELLS})
        share(f"{name} unrelated", sel & (cls == "swap"), m_old)
        if name == "flicker":
            share("flicker", sel, m_old)
        else:
            hosting = old & np.isin(a, np.array(sorted(set().union(*H.values())), np.int64))
            share("capture of host old mass", sel & (cls == "capture"), max(m[hosting].sum(), 1e-300))

    fl = pd.DataFrame({"a": a[flick], "b": b[flick], "m": m[flick], "mc": m[flick] * contested[flick]})
    for (x, y), v in fl.groupby(["a", "b"])[["m", "mc"]].agg(["sum", "size"]).iterrows():
        acc = pair_mass.setdefault((int(x), int(y)), [0.0, 0.0, 0, 0, relation(int(x), int(y), H, ends)])
        acc[0] += v[("m", "sum")]
        acc[1] += v[("mc", "sum")]
        acc[2] += int(v[("m", "size")])
        acc[3] += 1

    keep = within(A, dt, KEEP_GYR)
    nb = ~member(labs[k - 1][0], ids) & (keep != ABSENT).all(0) & (b != UNASSIGNED)
    kept = nb & (keep == b).all(0)
    sat = np.isin(b, np.array(list(H), np.int64))
    kinds = (nb, nb & (b == run.host), nb & sat, nb & (b != run.host) & ~sat)
    for name, sel in zip(NEWBORN, kinds):
        row[f"newborn {name}"], row[f"newborn kept {name}"] = m[sel].sum(), m[kept & sel].sum()
    return row


# --- the workers ----------------------------------------------------------------------------------

_W = {}                                 # a worker's run and what its tasks share


def _init(path, source, snaps, factor, tree_last):
    run = Run(path, particles=source)
    run.snaps = snaps
    _W.update(run=run, t=np.array([run.time[s] for s in snaps]), factor=factor, tree_last=tree_last)


def _snapshot(k):
    """Phase A in a worker, at the k-th snapshot: per-galaxy counts, masses and newborn masses, the
    catalogue masses and positions, hosts, the fitted galaxies, and the exchange from the previous one."""
    run = _W["run"]
    s = run.snaps[k]
    ids, lab = run.labels(s)
    m = np.nan_to_num(run.masses(s))
    g, inv = np.unique(lab, return_inverse=True)
    inv = inv.ravel()
    gal = run.galaxies(s)
    flow = exchange(ids, lab, m, run.labels(run.snaps[k - 1])) if k else None
    return (
        pd.Series(np.bincount(inv), index=g),
        pd.Series(np.bincount(inv, weights=m), index=g),
        pd.Series(np.bincount(inv, weights=m * newborn(run, s, ids)), index=g),
        gal["Mtot"], gal[POS], run.hosts(s), run.fitted(s), flow,
    )


def _windows(ks):
    """Phase B in a worker: the 2.3-2.5 windows of a run of consecutive snapshots, reading each once."""
    run, t, factor, tree_last = _W["run"], _W["t"], _W["factor"], _W["tree_last"]
    sn = run.snaps
    labs, cands, rows, pair_mass = {}, {}, [], {}
    for k in ks:
        for j in [k - 1, k, *ahead(t, k, WINDOW_GYR)]:
            if j not in labs:
                labs[j] = run.labels(sn[j])
        for j in (k - 1, k):
            if j not in cands:
                cands[j] = candidates(run, sn[j], factor)
        for j in [j for j in labs if j < k - 1]:
            del labs[j]
        for j in [j for j in cands if j < k - 1]:
            del cands[j]
        ends = set(tree_last.index[tree_last.isin([sn[k - 1], sn[k]])].astype(int))
        rows.append(window(run, k, t, labs, cands, run.hosts(sn[k]), ends, pair_mass, factor))
    return rows, pair_mass


# --- the point ------------------------------------------------------------------------------------

def evaluate(run, out, workers=1):
    rep = Report("Point 2, stable in time", run.name, point="2", causes=CAUSES)
    sn = run.snaps
    t = np.array([run.time[s] for s in sn])
    tree = run.tree.set_index(["Snapshot", "Sub_tree_id"])
    tree_last = run.tree.groupby("Sub_tree_id")["Snapshot"].max()
    tree_has = set(tree.index.tolist())
    no_uid = set(zip(run.tree.loc[run.tree["uid"].isna(), "Snapshot"],
                     run.tree.loc[run.tree["uid"].isna(), "Sub_tree_id"].astype(int)))
    merge = mergers(run.tree)
    merge = merge.assign(t=merge["snapshot"].map(run.time))
    factor = int(run.tree["Sub_tree_id"].max()) + 3   # key slots: galaxies 0 .. max; labels -1, -2 hit none

    # phase A: per snapshot
    init = (run.path, run.source, sn, factor, tree_last)
    N, M, NB, Mcat, pos, hosts, bound, exch = {}, {}, {}, {}, {}, {}, {}, {}
    for k, part in enumerate(pool(_snapshot, range(sn.size), workers, _init, init, desc="2 snapshots")):
        s = sn[k]
        N[s], M[s], NB[s], Mcat[s], pos[s], hosts[s], bound[s], exch[k] = part

    # phase B: the 2.3-2.5 windows, in chunks of consecutive snapshots (each re-reads ~6 snapshots)
    ks = [k for k in range(1, sn.size) if t[-1] >= t[k] + WINDOW_GYR]
    chunks = [c for c in np.array_split(ks, 2 * max(workers, 1)) if c.size]
    rows, pair_mass = [], {}
    for part_rows, part_pairs in pool(_windows, chunks, workers, _init, init, desc="2.3-2.5 window chunks"):
        rows += part_rows
        for key, v in part_pairs.items():           # chunks in time order: the first relation met is kept
            acc = pair_mass.setdefault(key, [0.0, 0.0, 0, 0, v[4]])
            for i in range(4):
                acc[i] += v[i]

    N = pd.DataFrame(N).drop(index=UNASSIGNED, errors="ignore").reindex(columns=sn)
    M = pd.DataFrame(M).drop(index=UNASSIGNED, errors="ignore").reindex(columns=sn)
    NB = pd.DataFrame(NB).reindex(index=M.index, columns=sn).fillna(0.0)
    Mcat = pd.DataFrame(Mcat).reindex(columns=sn)
    W = pd.DataFrame(rows).set_index("snap")

    # phase C: the verdicts
    gaps(run, t, N, tree, tree_has, bound, exch, rep)
    Mres = M.where(N >= RESOLVED)
    events = mass_steps(run, t, Mres, M, N, NB, merge, tree, tree_last, no_uid, pos, hosts, exch, rep)
    cat_spikes = steps(Mcat.reindex(Mres.index).where(N >= RESOLVED), t)[0].sum()
    rep.score("2.2 catalogue Mtot spikes per 100 resolved histories",
              100.0 * cat_spikes / max(n_histories(Mres), 1))
    label_shares(W, pair_mass, rep)
    noise, noise_terms = label_noise(Mres, M, t, merge)
    noise_report(noise, run.host, rep)

    # plots
    named = rep.episodes()
    marks = {}
    for t0, t1, cause, fail, galaxies in rep.history:
        for g in galaxies:
            marks.setdefault(g, []).append((t0, t1, cause, fail))
    plot_masses(run, Mres, Mcat, events, plots.run_plot(out, "stability_masses"))
    plot_shares(run, W, plots.run_plot(out, "stability_shares"))
    plot_episodes(run, rep.history, plots.run_plot(out, "stability_episodes"))
    plot_noise(run, noise, noise_terms, plots.run_plot(out, "stability_noise"))
    items = [(plots.galaxy_plot(out, g, "stability_mass"), g, t, M.loc[g].to_numpy(),
              Mcat.loc[g].to_numpy() if g in Mcat.index else np.full(t.size, np.nan), marks.get(g, []),
              tuple(noise.loc[g, ["sigma_lo", "sigma_hi"]]) if g in noise.index else None)
             for g in plots.chosen(run.host, N.max(axis=1).dropna().to_dict(), named) if g in M.index]
    plots.per_galaxy(_draw_mass, items, workers, desc="2 galaxy plots")
    rep.write(out / "stability.txt")
    return rep


def n_histories(Mres):
    """Galaxies resolved at two consecutive snapshots at least."""
    r = Mres.notna().to_numpy()
    return int((r[:, 1:] & r[:, :-1]).any(axis=1).sum())


def centroid(run, s, ids, m):
    """Mass-weighted centre of the particles ``ids`` at ``s`` (NaN without particle data or when none
    is there)."""
    if run.pdata is None:
        return np.full(3, np.nan)
    pid, X, _ = run.particles_by_id(s)
    pos = np.minimum(np.searchsorted(pid, ids), pid.size - 1)
    here = pid[pos] == ids
    if not here.any():
        return np.full(3, np.nan)
    return np.average(X[pos[here], :3], axis=0, weights=m[here])


def gaps(run, t, N, tree, tree_has, bound, exch, rep):
    """2.1 symptoms."""
    sn = run.snaps
    n = N.fillna(0).to_numpy()
    for i, G in enumerate(tqdm(N.index.astype(int), desc="2.1 gaps", disable=not sys.stderr.isatty())):
        p = np.flatnonzero(n[i] >= 1)
        for k0, k1 in zip(p[:-1], p[1:]):
            missing = [k for k in range(k0 + 1, k1) if (int(sn[k]), G) in tree_has]
            n0 = n[i, k0]
            if not missing or n0 < GAP_FLAG:
                continue

            ids0, lab0 = run.labels(sn[k0])
            mine = ids0[lab0 == G]
            m0 = run.masses(sn[k0])[lab0 == G]
            flow = exch.get(missing[0])
            if flow is not None and G in flow.index.get_level_values("a"):
                took = flow.xs(G, level="a")
                took = took[took.index >= 0]
            else:
                took = pd.Series(dtype=float)
            partner, part = (int(took.idxmax()), float(took.max() / np.nansum(m0))) if took.size else (None, 0.0)

            unbound = [k for k in missing if G not in bound[sn[k]]]
            d = np.array([
                np.linalg.norm(centroid(run, sn[k], mine, m0) - tree.loc[(int(sn[k]), G)][POS].to_numpy(float))
                / tree.loc[(int(sn[k]), G)]["virial_radius"]
                for k in unbound
            ])
            place = None if not np.isfinite(d).all() else "in" if (d <= 1.0).all() else "out"
            severity, cause = gap_verdict(n0, len(unbound), place, part)
            text = (f"gap of {G} at {[int(sn[k]) for k in missing]} ({1e3 * (t[k1] - t[k0]):.0f} Myr "
                    f"between its present snapshots; {int(n0)} particles before; {len(unbound)} snapshots "
                    f"without bound stars; largest taker {partner} with {part:.0%})")
            rep.symptom("2.1 gap", severity, text, s0=int(sn[k0]), s1=int(sn[k1]), t0=t[k0], t1=t[k1],
                        galaxies={G} | ({partner} if part >= PARTNER else set()), cause=cause)


def mass_steps(run, t, Mres, M, N, NB, merge, tree, tree_last, no_uid, pos, hosts, exch, rep):
    """2.2 scores and symptoms; returns the events for the overview plot."""
    sn = list(run.snaps)
    spike, persist = steps(Mres, t)
    events = []

    def change(G, k, kind, major):
        gi = Mres.index.get_loc(G)
        m0, m1 = Mres.iat[gi, k - 1], Mres.iat[gi, k]
        total, big = partners(exch.get(k), G, m1 - m0)
        exchanged = total >= PARTNER * abs(m1 - m0)
        severity = "FAIL" if major and not exchanged else "FLAG"
        cause = "exchange between interacting galaxies" if exchanged else "unexplained"
        text = (f"{kind} of {G} at {sn[k]}: {m0:.3g} -> {m1:.3g} "
                f"({total / abs(m1 - m0):.0%} from / to other galaxies)")
        rep.symptom(f"2.2 {kind}", severity, text, s0=sn[k - 1], s1=sn[k], t0=t[k - 1], t1=t[k],
                    galaxies={G} | big, cause=cause)

    for gi, k in zip(*np.nonzero(spike)):
        G = int(Mres.index[gi])
        events.append((G, sn[k], "spike"))
        change(G, k, "spike", N.at[G, sn[k - 1]] >= MAJOR)

    reasons = {}
    for gi, k in zip(*np.nonzero(persist)):
        G, s = int(Mres.index[gi]), sn[k]
        m0, m1 = Mres.iat[gi, k - 1], Mres.iat[gi, k]
        why = "unexplained"
        if (s, G) in no_uid or (sn[k - 1], G) in no_uid:
            why = "tree incomplete"
        if m1 > m0 and NB.at[G, s] >= 0.5 * (m1 - m0):
            why = "star formation"
        elif m1 > m0:
            near = merge[(merge["into"] == G) & ((merge["t"] - t[k]).abs() <= MERGER_GYR)]
            gain = [M.loc[p, :s].dropna().iloc[-1] for p in near["progenitor"]
                    if p in M.index and M.loc[p, :s].notna().any()]
            if any(x >= 0.5 * (m1 - m0) for x in gain):
                why = "merger"
        elif G in tree_last.index and run.time.get(int(tree_last[G]), np.inf) <= t[k] + MERGING_GYR:
            why = "merging"
        elif hosts[s].get(G):
            host = max(hosts[s][G], key=lambda h: tree["mass"].get((s, h), 0.0))
            js = np.flatnonzero(np.abs(t - t[k]) <= PERI_GYR)
            d = [np.linalg.norm(pos[sn[j]].loc[G].to_numpy(float) - pos[sn[j]].loc[host].to_numpy(float))
                 if G in pos[sn[j]].index and host in pos[sn[j]].index else np.nan
                 for j in js]
            if js[0] < k < js[-1] and np.isfinite(d).all() and js[np.argmin(d)] == k:
                why = "pericentre"
        reasons[why] = reasons.get(why, 0) + 1
        if why == "unexplained":
            events.append((G, s, "step"))
            change(G, k, "unexplained step", G == run.host)

    nh = max(n_histories(Mres), 1)
    total = np.nansum(Mres.to_numpy())
    prev = Mres.shift(1, axis=1)
    step_mass = sum(Mres.at[G, sn[sn.index(s) - 1]] for G, s, e in events if e == "step")
    rep.score("2.2 spikes per 100 resolved histories", 100.0 * spike.sum() / nh)
    rep.score("2.2 unexplained steps per 100 resolved histories", 100.0 * reasons.get("unexplained", 0) / nh)
    rep.score("2.2 spikes, mass-weighted", float(np.nansum(prev.to_numpy()[spike]) / total))
    rep.score("2.2 unexplained steps, mass-weighted", float(step_mass / total))
    rep.table("2.2 persistent steps by explanation",
              "\n".join(f"  {k:16s} {v}" for k, v in sorted(reasons.items())) or "  (none)")
    return events


def label_shares(W, pair_mass, rep):
    """2.3-2.5 scores and tables, and the share spikes as symptoms."""
    inscope = W[W["n_old"] >= MIN_OLD]
    for col, check in SHARES.items():
        x = inscope[col]
        med = float(x.median())
        rep.score(f"{check} share, median", med)
        rep.score(f"{check} share, max", float(x.max()))
        for s in x.index[x > max(SPIKE_FACTOR * med, SHARE_FLOOR)]:
            pair, part = inscope.at[s, f"{col} pair"], inscope.at[s, f"{col} pair part"]
            text = f"{check} share {x[s]:.2e} at {s} (median {med:.2e}); pair {pair} carries {part:.0%}"
            rep.symptom(f"{check} share spike", "FLAG", text, s0=int(s), t0=inscope.at[s, "t"],
                        galaxies=set(pair) - {UNASSIGNED} if part > PARTNER else set(), cause="share spike")

    rep.score("2.4 swap share of old mass, median", float(inscope["transfer swap"].median()))
    for c in NEWBORN:
        kept = W[f"newborn kept {c}"].sum() / max(W[f"newborn {c}"].sum(), 1e-300)
        rep.score(f"2.5 newborn mass kept, {c}", float(kept))
    for name, title in (("flicker", "2.3 flicker"), ("transfer", "2.4 permanent transfers")):
        rep.table(f"{title}, share of old mass (median / max over snapshots)",
                  "\n".join(f"  {c:20s} {inscope[f'{name} {c}'].median():.2e} / "
                            f"{inscope[f'{name} {c}'].max():.2e}" for c in TRANSFERS + CELLS))
    top = sorted(pair_mass.items(), key=lambda kv: -kv[1][0])[:10]
    rep.table("2.3 top flickering pairs (a -> b -> a): mass, contested share, particles, snapshots, relation",
              "\n".join(f"  {a:>8d} -> {b:>8d}  {v[0]:10.3g} {v[1] / v[0]:6.3f} {v[2]:9d} {v[3]:5d}  {v[4]}"
                        for (a, b), v in top) or "  (none)")


def label_noise(Mres, M, t, merge):
    """2.6 label-mass noise: per galaxy, the lower and upper bounds on the white noise of log10 Mres
    (see the module docstring); and the kept terms with their middle snapshot's time, to pool in time.

    Returns a (galaxy-indexed) table of sigma_lo, sigma_hi, n_pairs, median_mass, and a table of terms
    (t; e2, c2: a kept triple's e_i^2 and c_i^2; ee, ab: e_i e_{i+1} and a_{i+1} + b_i when the next
    triple is kept too, else 0) of the measured galaxies.
    """
    G, sn = Mres.index.to_numpy(), Mres.columns.to_numpy()
    y = np.log10(Mres.to_numpy(np.float64))
    h = np.diff(t)
    a, b = h[1:] / (h[:-1] + h[1:]), h[:-1] / (h[:-1] + h[1:])    # triple j: snapshots j, j+1, j+2
    e = a * y[:, :-2] + b * y[:, 2:] - y[:, 1:-1]
    c2 = a ** 2 + b ** 2 + 1.0
    d = np.abs(np.diff(y, axis=1))

    Mf = M.ffill(axis=1).to_numpy(np.float64)                     # a progenitor's last label mass
    k = np.searchsorted(sn, merge["snapshot"].to_numpy()) - 1     # the snapshot before the merger
    ip, ig = M.index.get_indexer(merge["progenitor"]), M.index.get_indexer(merge["into"])
    use = (k >= 0) & (ip >= 0) & (ig >= 0)
    m_p, m_g = Mf[ip[use], k[use]], np.nan_to_num(Mf[ig[use], k[use]])
    big = m_p >= MERGER_SHARE * m_g                               # NaN (never labelled before): False
    near = np.zeros(y.shape, bool)
    for i, tm in zip(ig[use][big], merge["t"].to_numpy()[use][big]):
        near[i] |= np.abs(t - tm) <= MERGER_GYR

    kept = (np.isfinite(e) & (d[:, :-1] <= LOG_JUMP) & (d[:, 1:] <= LOG_JUMP)
            & ~(near[:, :-2] | near[:, 1:-1] | near[:, 2:]))
    pair = kept[:, :-1] & kept[:, 1:]
    e0 = np.where(kept, e, 0.0)
    e2, ee = e0 ** 2, np.where(pair, e0[:, :-1] * e0[:, 1:], 0.0)
    ab = np.where(pair, a[1:] + b[:-1], 0.0)
    cc = np.where(kept, c2, 0.0)

    n_pairs = pair.sum(axis=1)
    on = n_pairs >= MIN_NOISE_PAIRS
    noise = pd.DataFrame({
        "sigma_lo": np.sqrt(np.maximum(-ee[on].sum(axis=1) / ab[on].sum(axis=1), 0.0)),
        "sigma_hi": np.sqrt(e2[on].sum(axis=1) / cc[on].sum(axis=1)),
        "n_pairs": n_pairs[on],
        "median_mass": np.nanmedian(Mres.to_numpy(np.float64)[on], axis=1),
    }, index=pd.Index(G[on], name="galaxy"))
    gi, j = np.nonzero(kept & on[:, None])
    terms = pd.DataFrame({"t": t[j + 1], "e2": e2[gi, j], "c2": cc[gi, j],
                          "ee": np.pad(ee, ((0, 0), (0, 1)))[gi, j], "ab": np.pad(ab, ((0, 0), (0, 1)))[gi, j]})
    return noise, terms


def noise_report(noise, host, rep):
    """2.6 scores and the table by median resolved mass (measurement only: no symptoms)."""
    rep.score("2.6 label-mass noise: galaxies measured", int(noise.shape[0]))
    if noise.empty:
        return
    rep.score("2.6 median sigma_noise lower bound (dex)", float(noise["sigma_lo"].median()))
    rep.score("2.6 median sigma_noise upper bound (dex)", float(noise["sigma_hi"].median()))
    if host in noise.index:
        rep.score("2.6 main host sigma_noise lower bound (dex)", float(noise.at[host, "sigma_lo"]))
        rep.score("2.6 main host sigma_noise upper bound (dex)", float(noise.at[host, "sigma_hi"]))
    bin_ = pd.cut(noise["median_mass"], MASS_BINS, labels=MASS_LABELS, right=False)
    lines = []
    for label in MASS_LABELS:
        sub = noise[bin_ == label]
        lines.append(f"  {label:10s} n={sub.shape[0]:4d}  lower {sub['sigma_lo'].median():8.3g}  "
                     f"upper {sub['sigma_hi'].median():8.3g}  upper p90 {sub['sigma_hi'].quantile(0.9):8.3g}"
                     if sub.shape[0] else f"  {label:10s} n=   0")
    rep.table("2.6 label-mass noise by median stellar mass (medians over galaxies, dex)", "\n".join(lines))


# --- plots ----------------------------------------------------------------------------------------

def plot_masses(run, Mres, Mcat, events, pdf):
    """The main host and the 8 largest galaxies: label mass, catalogue Mtot, spikes and unexplained steps."""
    peak = Mres.max(axis=1).drop(index=run.host, errors="ignore").sort_values(ascending=False)
    show = [run.host] * (run.host in Mres.index) + list(peak.index[:8])
    t = np.array([run.time[s] for s in Mres.columns])
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, G in enumerate(show):
        c = f"C{i % 10}"
        ax.semilogy(t, Mres.loc[G], color=c, lw=1.0, label=str(G))
        if G in Mcat.index:
            ax.semilogy(t, Mcat.loc[G], color=c, lw=0.6, ls="--")
        for G2, s, e in events:
            if G2 == G:
                ax.plot(run.time[s], Mres.at[G, s], "x" if e == "spike" else "o", color="C3", ms=6, mfc="none")
    ax.set_xlabel("t [Gyr]")
    ax.set_ylabel("M [hard labels]; dashed: catalogue Mtot")
    ax.set_title(f"{run.name}: main host + 8 largest (red x spike, o unexplained step)")
    ax.legend(fontsize=7, ncol=3)
    plots.save(fig, pdf)


def plot_shares(run, W, pdf):
    """2.3-2.5 against time: shares of the old mass that change label, and newborns kept."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for c in ["flicker", "flicker unrelated", "capture of host old mass", "transfer stripping",
              "transfer unrelated"]:
        a1.semilogy(W["t"], W[c].clip(lower=1e-7), lw=0.9, label=c)
    a1.set_ylabel("share of old mass")
    a1.legend(fontsize=8)
    for c in NEWBORN[:3]:
        a2.plot(W["t"], W[f"newborn kept {c}"] / W[f"newborn {c}"], lw=0.9, label=c)
    a2.set_ylabel(f"newborn mass kept {1e3 * KEEP_GYR:.0f} Myr")
    a2.set_xlabel("t [Gyr]")
    a2.legend(fontsize=8)
    fig.suptitle(f"{run.name}: label stability")
    plots.save(fig, pdf)


def plot_episodes(run, history, pdf):
    """Episodes by cause against time (stacked, per 0.5 Gyr); FAIL episodes as red triangles."""
    df = pd.DataFrame([h[:4] for h in history], columns=["t", "t1", "cause", "fail"]).dropna(subset=["t"])
    edges = np.arange(0.0, max(run.time.values()) + 0.5, 0.5)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(edges.size - 1)
    for cause in CAUSES:
        n = np.histogram(df.loc[df["cause"] == cause, "t"], bins=edges)[0]
        if n.any():
            ax.bar(edges[:-1], n, width=0.5, bottom=bottom, align="edge", color=COLOR[cause], label=cause)
            bottom += n
    ax.plot(df.loc[df["fail"], "t"], np.zeros(int(df["fail"].sum())), "^", color="C3", mec="k", ms=10,
            clip_on=False, zorder=5, label="FAIL episode")
    ax.set_xlabel("t [Gyr]")
    ax.set_ylabel("episodes per 0.5 Gyr")
    ax.set_title(f"{run.name}: point 2 episodes by cause")
    ax.legend(fontsize=7)
    plots.save(fig, pdf)


def plot_noise(run, noise, terms, pdf):
    """2.6: each galaxy's noise bracket against its median mass; the bounds pooled over galaxies per Gyr."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if not noise.empty:
        a1.vlines(noise["median_mass"], noise["sigma_lo"], noise["sigma_hi"], color="0.6", lw=0.8)
        a1.plot(noise["median_mass"], noise["sigma_lo"], "v", color="C0", ms=3, label="lower bound")
        a1.plot(noise["median_mass"], noise["sigma_hi"], "^", color="C1", ms=3, label="upper bound")
        a1.legend(fontsize=8)
    a1.set_xscale("log")
    a1.set_xlabel("galaxy's median resolved mass")
    a1.set_ylabel("sigma_noise [dex]")
    if not terms.empty:
        s = terms.groupby(np.floor(terms["t"]).astype(int))[["e2", "c2", "ee", "ab"]].sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            a2.plot(s.index + 0.5, np.sqrt((-s["ee"] / s["ab"]).clip(lower=0.0)), "v-", color="C0", ms=3,
                    lw=1.0, label="lower bound")
            a2.plot(s.index + 0.5, np.sqrt(s["e2"] / s["c2"]), "^-", color="C1", ms=3, lw=1.0,
                    label="upper bound")
        a2.legend(fontsize=8)
    a2.set_xlabel("t [Gyr]")
    a2.set_ylabel("pooled sigma_noise(t) [dex]")
    fig.suptitle(f"{run.name}: 2.6 label-mass noise")
    plots.save(fig, pdf)


def _draw_mass(item):
    """A galaxy's label mass and catalogue Mtot against time, with its episodes (one PDF).

    Each episode is one shaded span (a line when it lasts one instant), coloured by its cause as in
    stability_episodes.pdf; FAIL episodes are hatched and edged in red.
    """
    path, g, t, M, Mcat, marks, bracket = item
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.semilogy(t, np.where(M > 0, M, np.nan), "o-", color="k", ms=2.5, lw=0.9, label="hard-label mass")
    ax.semilogy(t, np.where(Mcat > 0, Mcat, np.nan), "--", color="0.5", lw=0.8, label="catalogue Mtot")
    shown = set()
    for t0, t1, cause, fail in marks:
        color = COLOR.get(cause, "C9")
        label = None if cause in shown else cause
        shown.add(cause)
        if t1 > t0:
            ax.axvspan(t0, t1, fc=color, alpha=0.3, label=label,
                       ec="C3" if fail else "none", lw=1.5, hatch="//" if fail else None)
        else:
            ax.axvline(t0, color="C3" if fail else color, lw=2.0 if fail else 1.0, label=label)
    ax.set_xlabel("t [Gyr]")
    ax.set_ylabel("stellar mass")
    n_fail = sum(m[3] for m in marks)
    noise_txt = f", label noise {bracket[0]:.2g}-{bracket[1]:.2g} dex" if bracket is not None else ""
    ax.set_title(f"galaxy {g}: point 2, {len(marks)} episodes ({n_fail} FAIL, hatched red){noise_txt}")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plots.save(fig, path)


def main():
    args = cli.arguments(__doc__)
    run, out = cli.start("stability", args)
    evaluate(run, out, workers=args.workers)


if __name__ == "__main__":                      # spawned workers import this file, never run it
    main()

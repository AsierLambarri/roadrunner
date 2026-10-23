"""Goal 6: where do catalogues differ? Measures and reports, for two or more catalogues, each against
the first (the reference).

    python3 evaluate/compare.py --path REF --vs OTHER [...] [--particles DIR] [--workers N] [--snaps A:B]
                                [--every N] [--galaxies ID [ID ...]]

6.1 agreement: mass share of particles with the reference's hard label: all, old (present >= 1 Gyr
    before), newborn (absent from the previous snapshot), contested (>= 2 candidates in the first
    roadrunner run's stored pairs); median over snapshots, and the curve against time, on the particles
    every catalogue holds, every --every-th common snapshot with >= 1e5 of them. Ages come from the
    particle data: stars persist once formed, so presence is first appearance.
6.2 mass movers: mass the reference labels a and the other labels b != a, summed over the measured
    snapshots; the ten largest (a, b) with their relation in the reference at the first snapshot met
    (stripping: b among a's hosts; capture: a among b's hosts; unbound / rebound: b / a is -1; swap).
6.3 where they disagree: disagreeing share of the mass in bins of distance to the main host (units of
    its rh in the reference catalogue), inside / outside the host's disc, the reference label's kind
    (main host, satellite, other central, unassigned) and hard-label stellar mass, newborn / old. The
    disc: |z| < 3 h_z and R < R_90 in the frame of the angular momentum of the host's stars within 3 rh;
    h_z the median |z|, R_90 the 90th percentile of R of its young stars (absent 0.5 Gyr before)
    within 3 rh, or of all of them there when fewer than 100 are young.
6.4 galaxy properties, from the property tables alone (no particles, read at every common snapshot):
    for galaxies with Mtot >= 1e6 in both the reference and the other catalogue, log10 ratio
    (other / reference) of Mtot, rh and sigma; the position offset |x_other - x_ref| / rh_ref and the
    velocity offset |v_other - v_ref| / sigma_ref. The hard-label mass log ratio stays at the --every
    snapshots (it needs particles). Per galaxy over its matched snapshots (>= 3): median, 90th percentile
    and max |log10 ratio| of Mtot, rh, sigma; median position and velocity offsets; the time of its
    largest |log10 ratio| of Mtot. SCOREs per other catalogue: the medians over galaxies of those
    per-galaxy medians; and, over every (galaxy, snapshot) row, the signed median log10 ratio (bias) and
    the median / 90th percentile of |log10 ratio| (Mtot, rh, sigma, label mass); a table of the 20 most
    massive matched galaxies (by peak Mtot in the reference). The grid plot's selection, on the
    reference: present at its last two common snapshots, existing before 3 Gyr, peak Mtot > 5e6 and in
    every catalogue; largest peak first, --galaxies forced in first.

Writes evaluation/compare.txt and .log; plots/run/: compare_agreement.pdf (6.1 against time, one panel
per subset), compare_where.pdf (6.3's disagreeing share, one panel per breakdown), compare_properties.pdf
(6.4's 16-galaxy grid, one column of 3 stacked rows per galaxy against time: R/Rvir from the tree of the
particles run with a dashed line at 1, Mtot, rh with sigma on a twin axis; one colour per catalogue);
plots/g<id>/compare_properties.pdf (the same three rows) for the reference's largest galaxies (by
particle count at the measured snapshots) and the grid's galaxies.
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

from evaluate import cli, plots
from evaluate.load import (OLD_GYR, Run, UNASSIGNED, aligned, common_snaps, label_mass, log_ratio, newborn,
                           open_catalogue, pool, seen_before)
from evaluate.report import Report
from evaluate.stability import relation

MIN_PARTICLES, MIN_MASS, YOUNG_GYR, MIN_YOUNG = 1e5, 1e6, 0.5, 100
MIN_SNAPS = 3                            # 6.4: a galaxy needs this many matched snapshots for its summary
GRID_MIN_PEAK, EARLY_GYR, GRID_SIZE = 5e6, 3.0, 16   # 6.4's grid selection
POS, VEL = ["position_x", "position_y", "position_z"], ["velocity_x", "velocity_y", "velocity_z"]
R_BINS = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0, np.inf])
M_BINS = np.array([0.0, 1e6, 1e7, 1e8, 1e9, np.inf])
BREAKDOWNS = {
    "r / rh_host": ["<1", "1-2", "2-5", "5-10", "10-30", ">30", "no host"],
    "host disc": ["in disc", "out of disc"],
    "reference label": ["main host", "satellite", "other central", "unassigned"],
    "reference galaxy mass": ["<1e6", "1e6-1e7", "1e7-1e8", "1e8-1e9", ">1e9", "unassigned"],
    "age": ["newborn", "between", "old"],
}
SUBSETS = ("all", "old", "newborn", "contested")


def disc_tag(X, m, is_host, young, centre, velocity, rh):
    """Inside the host's disc (see 6.3)."""
    d, v = X[:, :3] - centre, X[:, 3:] - velocity
    inner = is_host & (np.linalg.norm(d, axis=1) < 3.0 * rh)
    J = (m[inner, None] * np.cross(d[inner], v[inner])).sum(axis=0)
    jhat = J / np.linalg.norm(J)
    z = d @ jhat
    R = np.linalg.norm(d - z[:, None] * jhat, axis=1)
    ref = inner & young if (inner & young).sum() >= MIN_YOUNG else inner
    return (np.abs(z) < 3.0 * np.median(np.abs(z[ref]))) & (R < np.percentile(R[ref], 90))


def bins(cat, s, X, m, lab, young, nb, old):
    """Bin index of every particle in each breakdown of 6.3 (from the reference catalogue)."""
    out = {}
    gal = cat.galaxies(s)
    host = cat.host
    has_host = host in gal.index and np.isfinite(gal.loc[host, POS + VEL + ["rh"]].to_numpy(np.float64)).all()
    if has_host:
        centre, velocity, rh = (gal.loc[host, POS].to_numpy(np.float64), gal.loc[host, VEL].to_numpy(np.float64),
                                float(gal.at[host, "rh"]))
        r = np.linalg.norm(X[:, :3] - centre, axis=1) / rh
        out["r / rh_host"] = np.digitize(r, R_BINS) - 1
        out["host disc"] = np.where(disc_tag(X, m, lab == host, young, centre, velocity, rh), 0, 1)
    else:
        out["r / rh_host"] = np.full(lab.size, 6)
        out["host disc"] = np.ones(lab.size, int)
    sat = np.isin(lab, np.array(list(cat.hosts(s)), np.int64))
    out["reference label"] = np.select([lab == host, sat, lab == UNASSIGNED], [0, 1, 3], 2)
    M = label_mass(lab, m)
    out["reference galaxy mass"] = np.where(lab == UNASSIGNED, 5, np.digitize(M.reindex(lab).to_numpy(), M_BINS) - 1)
    out["age"] = np.select([nb, old], [0, 2], 1)
    return out


def normalized_offset(a, b, scale):
    """``|a - b| / scale``, row-wise (6.4's position and velocity offsets)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.linalg.norm(a - b, axis=1) / scale


def galaxy_summary(df):
    """6.4 per galaxy, from its matched-snapshot rows (one other catalogue's): galaxies with >= MIN_SNAPS
    of them, their median, 90th percentile and max |log10 ratio| of Mtot, rh, sigma; median position and
    velocity offsets; the time of the largest |log10 ratio| of Mtot; and the peak reference Mtot seen."""
    out = []
    for g, d in (df.groupby("galaxy") if not df.empty else []):
        if len(d) < MIN_SNAPS:
            continue
        row = {"galaxy": int(g), "n": len(d), "peak Mtot ref": float(d["Mtot ref"].max())}
        for q in ("Mtot", "rh", "sigma"):
            a = d[q].abs()
            row[f"{q} median |log ratio|"] = float(a.median())
            row[f"{q} 90th |log ratio|"] = float(a.quantile(0.9))
            row[f"{q} max |log ratio|"] = float(a.max())
        row["position offset median"] = float(d["pos_offset"].median())
        row["velocity offset median"] = float(d["vel_offset"].median())
        row["t of largest Mtot |log ratio|"] = float(d.loc[d["Mtot"].abs().idxmax(), "t"])
        out.append(row)
    return pd.DataFrame(out)


def select_grid(ref_peak, early, last2, seen, galaxies, n=GRID_SIZE):
    """6.4's grid selection: present in every catalogue, at the reference's last two common snapshots and
    before EARLY_GYR, with peak Mtot above GRID_MIN_PEAK; largest peak first, ``galaxies`` forced in first."""
    present_all = set.intersection(*seen.values()) if seen else set()
    candidates = sorted((early & last2 & present_all) & {g for g, pk in ref_peak.items() if pk > GRID_MIN_PEAK},
                        key=lambda g: -ref_peak[g])
    return list(dict.fromkeys([int(g) for g in galaxies] + [int(g) for g in candidates]))[:n]


def rvir_series(particles, g):
    """``(t, R/Rvir)`` of galaxy ``g`` in the particles run's own tree."""
    d = particles.tree.loc[particles.tree["Sub_tree_id"] == g, ["Snapshot", "R/Rvir"]].dropna()
    return [(particles.time[int(s)], float(r)) for s, r in zip(d["Snapshot"], d["R/Rvir"])]


_W = {}                                 # a worker's catalogues and particle run


def _init(paths, path, source):
    _W.update(cats=[open_catalogue(p) for p in paths], particles=Run(path, particles=source))


def _snapshot(s):
    """6.1-6.3 at ``s`` in a worker, per other catalogue: agreement row, mass movers, where-sums and the
    hard-label mass log ratio (6.4, this snapshot only); None with too few common particles. Also the
    share of the particle data held by all, and the reference's particle count per label (for its own
    plots)."""
    cats, particles = _W["cats"], _W["particles"]
    ref, others = cats[0], cats[1:]
    ids, X, m, L, share = aligned(cats, particles, s)
    if ids.size < MIN_PARTICLES:
        return None
    nb, old = newborn(particles, s, ids), seen_before(particles, s, ids, OLD_GYR)
    sub = {"all": np.ones(ids.size, bool), "old": old, "newborn": nb}
    rr = next((c for c in cats if isinstance(c, Run)), None)
    if rr is not None:
        n = np.bincount(rr.pairs(s)["row"].astype(np.int64))
        many = np.sort(rr.row_labels(s)[0][np.flatnonzero(n >= 2)])
        sub["contested"] = np.isin(ids, many, assume_unique=True)
    B = bins(ref, s, X, m, L[0], ~seen_before(particles, s, ids, YOUNG_GYR), nb, old)
    H = ref.hosts(s)
    Mref = label_mass(L[0], m)
    G = ref.galaxies(s)
    found = {}
    for c, lab in zip(others, L[1:]):
        same = lab == L[0]
        agree = {"snap": s, "t": particles.time[s],
                 **{k: float(m[v & same].sum() / m[v].sum()) if v.any() else np.nan for k, v in sub.items()}}
        dis = ~same
        mv = pd.DataFrame({"a": L[0][dis], "b": lab[dis], "m": m[dis]}).groupby(["a", "b"])["m"].sum()
        movers = {(int(a), int(b)): [mass, relation(int(a), int(b), H, set())] for (a, b), mass in mv.items()}
        where = {k: np.stack([np.bincount(idx, weights=m, minlength=len(BREAKDOWNS[k])),
                              np.bincount(idx, weights=m * dis, minlength=len(BREAKDOWNS[k]))]) for k, idx in B.items()}
        Gc = c.galaxies(s)
        Mc = label_mass(lab, m)
        both = G.index[G["Mtot"] >= MIN_MASS].intersection(Gc.index[Gc["Mtot"] >= MIN_MASS])
        label_rows = [{"snap": s, "t": particles.time[s], "galaxy": int(g),
                       "label mass": float(log_ratio(np.array(Mc.get(g, 0.0)), np.array(Mref.get(g, 0.0))))}
                      for g in both]
        found[c.name] = (agree, movers, where, label_rows)
    g, n_lab = np.unique(L[0][L[0] != UNASSIGNED], return_counts=True)
    return share, found, dict(zip(g.tolist(), n_lab.tolist()))


def properties(cats, particles, snaps_all):
    """6.4's cheap pass, every common snapshot, no particles: per other catalogue the matched-snapshot
    rows (galaxies with Mtot >= MIN_MASS in both), every catalogue's raw history of every galaxy it holds
    (for the plots), and the reference's peak Mtot, its presence before EARLY_GYR, at its last two
    snapshots, and in every catalogue (for the grid selection)."""
    ref, others = cats[0], cats[1:]
    props = {c.name: [] for c in others}
    hist = {c.name: {} for c in cats}
    seen = {c.name: set() for c in cats}
    ref_peak, early, last2 = {}, set(), set()
    last2_snaps = set(snaps_all[-2:])
    for s in snaps_all:
        t = particles.time[s]
        tables = {c: c.galaxies(s) for c in cats}
        Gref = tables[ref]
        for g, row in Gref.iterrows():
            ref_peak[g] = max(ref_peak.get(g, 0.0), float(row["Mtot"]))
            if t < EARLY_GYR:
                early.add(g)
            if s in last2_snaps:
                last2.add(g)
        for c in cats:
            Gc = tables[c]
            seen[c.name].update(Gc.index.tolist())
            for g, row in Gc.iterrows():
                hist[c.name].setdefault(int(g), []).append(
                    (t, float(row["Mtot"]), float(row["rh"]), float(row["sigma"])))
        for c in others:
            Gc = tables[c]
            both = Gref.index[Gref["Mtot"] >= MIN_MASS].intersection(Gc.index[Gc["Mtot"] >= MIN_MASS])
            if not len(both):
                continue
            dx = normalized_offset(Gc.loc[both, POS].to_numpy(np.float64), Gref.loc[both, POS].to_numpy(np.float64),
                                   Gref.loc[both, "rh"].to_numpy(np.float64))
            dv = normalized_offset(Gc.loc[both, VEL].to_numpy(np.float64), Gref.loc[both, VEL].to_numpy(np.float64),
                                   Gref.loc[both, "sigma"].to_numpy(np.float64))
            props[c.name].append(pd.DataFrame({
                "snap": s, "t": t, "galaxy": both.to_numpy(np.int64),
                "Mtot": log_ratio(Gc.loc[both, "Mtot"].to_numpy(), Gref.loc[both, "Mtot"].to_numpy()),
                "rh": log_ratio(Gc.loc[both, "rh"].to_numpy(), Gref.loc[both, "rh"].to_numpy()),
                "sigma": log_ratio(Gc.loc[both, "sigma"].to_numpy(), Gref.loc[both, "sigma"].to_numpy()),
                "pos_offset": dx, "vel_offset": dv, "Mtot ref": Gref.loc[both, "Mtot"].to_numpy(np.float64),
            }))
    props = {name: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame() for name, rows in props.items()}
    return props, hist, seen, ref_peak, early, last2


def evaluate(cats, particles, out, every=10, workers=1, galaxies=()):
    ref, others = cats[0], cats[1:]
    rep = Report("Goal 6, catalogue comparison", " / ".join(c.name for c in cats), verdict=False)
    snaps_all = common_snaps(cats, particles)
    snaps = [int(s) for s in snaps_all[::every]]
    agree = {c.name: [] for c in others}
    movers = {c.name: {} for c in others}
    where = {c.name: {k: np.zeros((2, len(v))) for k, v in BREAKDOWNS.items()} for c in others}
    label_rows = {c.name: [] for c in others}
    peak, measured = {}, []
    for s, part in zip(snaps, pool(_snapshot, snaps, workers, _init,
                                   ([c.path for c in cats], particles.path, particles.source), desc="6 snapshots")):
        if part is None:
            continue
        share, found, counts = part
        measured.append((s, share))
        for g, n in counts.items():
            peak[g] = max(peak.get(g, 0), n)
        for name, (a, mv, w, lr) in found.items():       # snapshots in time order: the first relation met is kept
            agree[name].append(a)
            for key, (mass, rel) in mv.items():
                movers[name].setdefault(key, [0.0, rel])[0] += mass
            for k, v in w.items():
                where[name][k] += v
            label_rows[name] += lr

    rep.score("snapshots measured", len(measured))
    rep.score("median share of the particle data held by every catalogue",
              float(np.median([sh for _, sh in measured])) if measured else np.nan)

    props, hist, seen, ref_peak, early, last2 = properties(cats, particles, snaps_all)
    report(rep, ref, others, agree, movers, where, label_rows, props)

    plots.save(plot_agreement(ref, others, agree), plots.run_plot(out, "compare_agreement"))
    plots.save(plot_where(others, where), plots.run_plot(out, "compare_where"))
    chosen = select_grid(ref_peak, early, last2, seen, galaxies)
    plot_properties(cats, particles, chosen, hist, plots.run_plot(out, "compare_properties"))
    own = sorted(set(plots.chosen(ref.host, peak, set())) | set(chosen))
    items = [(plots.galaxy_plot(out, g, "compare_properties"), g, [c.name for c in cats], rvir_series(particles, g),
             {c.name: hist[c.name].get(g, []) for c in cats}) for g in own]
    plots.per_galaxy(_draw_properties, items, workers, desc="6 galaxy plots")

    rep.write(out / "compare.txt")
    return rep


def report(rep, ref, others, agree, movers, where, label_rows, props):
    for c in others:
        A = pd.DataFrame(agree[c.name])
        for k in [x for x in SUBSETS if x in A]:
            rep.score(f"{c.name}: 6.1 agreement with {ref.name}, {k} (median)", float(A[k].median()))
        top = sorted(movers[c.name].items(), key=lambda kv: -kv[1][0])[:10]
        rep.table(f"6.2 {c.name} vs {ref.name}: largest mass movers (summed over snapshots)",
                  "\n".join(f"  {a:>8d} -> {b:>8d}  {v[0]:10.3e}  {v[1]}" for (a, b), v in top) or "  (none)")
        lines = []
        for k, labels in BREAKDOWNS.items():
            tot, dis = where[c.name][k]
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = dis / tot
            lines.append(f"  {k}")
            lines += [f"    {lab:14s} mass {t:10.3e}   disagreeing share {f:.4f}" for lab, t, f in zip(labels, tot, frac) if t > 0]
        rep.table(f"6.3 {c.name} vs {ref.name}: where they disagree", "\n".join(lines))

        Pr, Lr = props[c.name], pd.DataFrame(label_rows[c.name])
        for q, df in (("Mtot", Pr), ("rh", Pr), ("sigma", Pr), ("label mass", Lr)):
            if df.empty or q not in df:
                continue
            x = df[q].dropna()
            rep.score(f"{c.name}: 6.4 {q} median log10 ratio to {ref.name} (bias)", float(x.median()))
            rep.score(f"{c.name}: 6.4 {q} median |log10 ratio|", float(x.abs().median()))
            rep.score(f"{c.name}: 6.4 {q} 90th percentile |log10 ratio|", float(x.abs().quantile(0.9)))

        S = galaxy_summary(Pr)
        if not S.empty:
            for q in ("Mtot", "rh", "sigma"):
                rep.score(f"{c.name}: 6.4 {q} median of per-galaxy median |log10 ratio|",
                          float(S[f"{q} median |log ratio|"].median()))
            rep.score(f"{c.name}: 6.4 median of per-galaxy median position offset", float(S["position offset median"].median()))
            rep.score(f"{c.name}: 6.4 median of per-galaxy median velocity offset", float(S["velocity offset median"].median()))
            top20 = S.sort_values("peak Mtot ref", ascending=False).head(20)
            rep.table(f"6.4 {c.name} vs {ref.name}: 20 most massive matched galaxies",
                      top20.to_string(index=False, float_format=lambda v: f"{v:.3e}"))


def plot_agreement(ref, others, agree):
    """6.1 against time, one panel per subset, one colour per other catalogue."""
    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(4.2 * len(SUBSETS), 4), sharey=True)
    for ax, k in zip(axes, SUBSETS):
        for i, c in enumerate(others):
            A = pd.DataFrame(agree[c.name])
            if k in A:
                ax.plot(A["t"], A[k], "-", color=f"C{i}", label=c.name)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("t [Gyr]")
    axes[0].set_ylabel(f"6.1 mass share labelled as {ref.name}")
    axes[0].legend(fontsize=7)
    return fig


def plot_where(others, where):
    """6.3's disagreeing share per bin, one panel per breakdown, one colour per other catalogue."""
    fig, axes = plt.subplots(1, len(BREAKDOWNS), figsize=(4.2 * len(BREAKDOWNS), 4))
    for ax, (k, labels) in zip(axes, BREAKDOWNS.items()):
        for i, c in enumerate(others):
            tot, dis = where[c.name][k]
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = dis / tot
            ax.plot(range(len(labels)), frac, "o-", color=f"C{i}", label=c.name)
        ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=30)
        ax.set_title(k, fontsize=9)
    axes[0].set_ylabel("disagreeing share")
    axes[0].legend(fontsize=7)
    return fig


def _properties_axes(a0, a1, a2, rv, cats, Mtot_rh_sigma):
    """The three stacked rows (R/Rvir already plotted on ``a0``), one line per catalogue. Each row is
    log-scaled only where it holds positive data (R/Rvir is exactly 0 for the host itself)."""
    a3 = a2.twinx()
    if rv and any(v > 0 for _, v in rv):
        a0.set_yscale("log")
    a0.axhline(1.0, ls="--", color="0.5", lw=0.8)
    Mtot_all, rh_all, sigma_all = [], [], []
    for i, name in enumerate(cats):
        d = Mtot_rh_sigma.get(name)
        if not d:
            continue
        tt, Mtot, rh, sigma = zip(*d)
        a1.plot(tt, Mtot, "-", color=f"C{i}", label=name)
        a2.plot(tt, rh, "-", color=f"C{i}")
        a3.plot(tt, sigma, "-.", color=f"C{i}")
        Mtot_all += Mtot
        rh_all += rh
        sigma_all += sigma
    if any(v > 0 for v in Mtot_all):
        a1.set_yscale("log")
    if any(v > 0 for v in rh_all):
        a2.set_yscale("log")
    if any(v > 0 for v in sigma_all):
        a3.set_yscale("log")
    return a3


def plot_properties(cats, particles, chosen, hist, path):
    """6.4's grid: up to GRID_SIZE galaxies, 4 columns of 3 stacked rows (R/Rvir, Mtot, rh + twin sigma)
    on a shared 0-14 Gyr time axis."""
    names = [c.name for c in cats]
    fig, axes = plt.subplots(3 * ((GRID_SIZE + 3) // 4), 4, figsize=(16, 4.5 * ((GRID_SIZE + 3) // 4)), sharex=True)
    for p, g in enumerate(chosen):
        row0, col = 3 * (p // 4), p % 4
        a0, a1, a2 = axes[row0, col], axes[row0 + 1, col], axes[row0 + 2, col]
        rv = rvir_series(particles, g)
        if rv:
            a0.plot(*zip(*rv), "-", color="k", lw=1.2)
        _properties_axes(a0, a1, a2, rv, names, {name: hist[name].get(g, []) for name in names})
        a0.text(0.95, 0.85, f"g{g}", transform=a0.transAxes, ha="right", va="top", fontsize=9)
        if col == 0:
            a0.set_ylabel("R/Rvir")
            a1.set_ylabel("Mtot")
            a2.set_ylabel("rh")
    for col in range(4):
        axes[-1, col].set_xlabel("t [Gyr]")
    axes[0, 0].set_xlim(0, 14)
    axes[1, 0].legend(fontsize=8, loc="lower right")       # one legend, inside the figure
    plots.save(fig, path)


def _draw_properties(item):
    """A galaxy's R/Rvir, Mtot and rh + twin sigma against time, one colour per catalogue."""
    path, g, cats, rv, hist = item
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    if rv:
        a0.plot(*zip(*rv), "-", color="k", lw=1.2)
    _properties_axes(a0, a1, a2, rv, cats, hist)
    a0.set_ylabel("R/Rvir")
    a1.set_ylabel("Mtot")
    a2.set_ylabel("rh")
    a2.set_xlabel("t [Gyr]")
    a0.set_xlim(0, 14)
    a1.legend(fontsize=7)
    fig.suptitle(f"galaxy {g}: goal 6, properties")
    plots.save(fig, path)


def main():
    args = cli.arguments(__doc__, every=10, compare=True, options={
        "galaxies": dict(type=int, nargs="+", default=[], help="6.4: galaxies forced into the property grid"),
    })
    cats, particles, out = cli.start_compare("compare", args)
    evaluate(cats, particles, out, every=args.every, workers=args.workers, galaxies=args.galaxies)


if __name__ == "__main__":                      # spawned workers import this file, never run it
    main()

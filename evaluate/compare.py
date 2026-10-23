"""Goal 6: where do catalogues differ? Measures and reports, for two or more catalogues, each against
the first (the reference), on the particles every catalogue holds, every --every-th common snapshot
with >= 1e5 of them.

6.1 agreement: mass share of particles with the reference's hard label: all, old (present >= 1 Gyr
    before), newborn (absent from the previous snapshot), contested (>= 2 candidates in the first
    roadrunner run's stored pairs); median over snapshots, and the curve against time. Ages come from the
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
6.4 galaxy properties: galaxies in both catalogues with >= 1e6 in both: median log10 ratio (other /
    reference), median and 90th percentile of |log10 ratio| of the catalogue's Mtot, rh and sigma, and of
    the hard-label mass.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate.load import (OLD_GYR, Run, UNASSIGNED, aligned, common_snaps, label_mass, log_ratio, newborn,
                           open_catalogue, pool, seen_before)
from evaluate.report import Report
from evaluate.stability import relation

MIN_PARTICLES, MIN_MASS, YOUNG_GYR, MIN_YOUNG = 1e5, 1e6, 0.5, 100
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


_W = {}                                 # a worker's catalogues and particle run


def _init(paths, path, source):
    _W.update(cats=[open_catalogue(p) for p in paths], particles=Run(path, particles=source))


def _snapshot(s):
    """Goal 6 at ``s`` in a worker, per other catalogue: agreement row, mass movers, where-sums and property
    rows; None with too few common particles. Also the share of the particle data held by all."""
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
        props = [{q: float(log_ratio(np.array(Gc.at[g, q]), np.array(G.at[g, q]))) for q in ("Mtot", "rh", "sigma")}
                 | {"label mass": float(log_ratio(np.array(Mc.get(g, 0.0)), np.array(Mref.get(g, 0.0)))),
                    "snap": s, "galaxy": int(g), "Mtot ref": float(G.at[g, "Mtot"]), "Mtot other": float(Gc.at[g, "Mtot"])}
                 for g in G.index.intersection(Gc.index) if G.at[g, "Mtot"] >= MIN_MASS and Gc.at[g, "Mtot"] >= MIN_MASS]
        found[c.name] = (agree, movers, where, props)
    return share, found


def evaluate(cats, particles, out, every=10, workers=1):
    ref, others = cats[0], cats[1:]
    rep = Report("Goal 6, catalogue comparison", " / ".join(c.name for c in cats), verdict=False)
    snaps = [int(s) for s in common_snaps(cats, particles)[::every]]
    agree = {c.name: [] for c in others}
    movers = {c.name: {} for c in others}
    where = {c.name: {k: np.zeros((2, len(v))) for k, v in BREAKDOWNS.items()} for c in others}
    props = {c.name: [] for c in others}
    measured = []
    for s, part in zip(snaps, pool(_snapshot, snaps, workers, _init,
                                   ([c.path for c in cats], particles.path, particles.source), desc="6 snapshots")):
        if part is None:
            continue
        share, found = part
        measured.append((s, share))
        for name, (a, mv, w, pr) in found.items():          # snapshots in time order: the first relation met is kept
            agree[name].append(a)
            for key, (mass, rel) in mv.items():
                movers[name].setdefault(key, [0.0, rel])[0] += mass
            for k, v in w.items():
                where[name][k] += v
            props[name] += pr

    rep.score("snapshots measured", len(measured))
    rep.score("median share of the particle data held by every catalogue",
              float(np.median([sh for _, sh in measured])) if measured else np.nan)
    report(rep, ref, others, agree, movers, where, props)
    plot(ref, others, agree, where, props, out / "compare.png")
    rep.write(out / "compare.txt")
    return rep


def report(rep, ref, others, agree, movers, where, props):
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
        Pr = pd.DataFrame(props[c.name])
        for q in ("Mtot", "rh", "sigma", "label mass"):
            if not Pr.empty:
                x = Pr[q].dropna()
                rep.score(f"{c.name}: 6.4 {q} median log10 ratio to {ref.name}", float(x.median()))
                rep.score(f"{c.name}: 6.4 {q} median |log10 ratio|", float(x.abs().median()))
                rep.score(f"{c.name}: 6.4 {q} 90th percentile |log10 ratio|", float(x.abs().quantile(0.9)))


def plot(ref, others, agree, where, props, png):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for i, c in enumerate(others):
        A = pd.DataFrame(agree[c.name])
        if A.empty:
            continue
        ax[0].plot(A["t"], A["all"], "-", color=f"C{i}", label=f"{c.name}, all")
        ax[0].plot(A["t"], A["newborn"], "--", color=f"C{i}", label=f"{c.name}, newborn")
        tot, dis = where[c.name]["r / rh_host"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ax[1].plot(np.arange(len(tot)), dis / tot, "o-", color=f"C{i}", label=c.name)
        Pr = pd.DataFrame(props[c.name])
        if not Pr.empty:
            last = Pr[Pr["snap"] == Pr["snap"].max()]
            ax[2].loglog(last["Mtot ref"], last["Mtot other"], "o", ms=3, color=f"C{i}", label=c.name)
    ax[0].set_xlabel("t [Gyr]")
    ax[0].set_ylabel(f"6.1 mass share labelled as {ref.name}")
    ax[0].legend(fontsize=7)
    ax[1].set_xticks(range(len(BREAKDOWNS["r / rh_host"])), BREAKDOWNS["r / rh_host"], fontsize=7)
    ax[1].set_xlabel("6.3 r / rh_host")
    ax[1].set_ylabel("disagreeing share")
    ax[1].legend(fontsize=7)
    lo, hi = ax[2].get_xlim() if others and not pd.DataFrame(props[others[0].name]).empty else (1e6, 1e11)
    ax[2].plot([lo, hi], [lo, hi], ":", color="0.5")
    ax[2].set_xlabel(f"6.4 Mtot, {ref.name} (last measured snapshot)")
    ax[2].set_ylabel("Mtot, other")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)

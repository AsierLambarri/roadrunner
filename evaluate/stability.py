"""Point 2: the assignments are stable in time.

Masses are hard-label masses (the assignment itself), not the catalogue's Mtot
(infall list x bound set). A galaxy is present at s when it labels >= 1 particle,
resolved with >= 100, major with >= 1000. A star is old at s when it was first
seen >= 1 Gyr before; a newborn at s when first seen at s. Snapshot steps are
consecutive entries of the run's snapshot list; ages count from its first
snapshot (with --snaps, from the first one evaluated).

2.1 continuity: a gap is 1-3 snapshots without the galaxy between two present
    ones, "dropped" when the tree has the galaxy there (else "tree gap", counted
    only); a premature end is a last present snapshot more than one before the
    tree branch's last. FAIL for a major galaxy before the gap / at the end,
    FLAG for a resolved one. SCORE: dropped gaps per 1000 resolved galaxy-snapshots.
2.2 mass histories (resolved at both ends of a step): a step is |log10 M(s)/M(s-1)|
    > log10 2; a spike is a step that returns within 25% of M(s-1) at s+1..s+3.
    A persistent step is explained by star formation (the newborns it labels at
    s weigh >= half the gain) or by a tree merger into the galaxy at s-1..s+1
    whose progenitor's last mass is >= half the gain (growth), by the galaxy's
    own branch ending within 3 snapshots (drop), or labelled "pericentre" (drop
    of a main-host satellite at a local minimum of its distance to the main
    host within +-2 snapshots) or "tree incomplete" (its own tree rows at s-1, s
    have no uid). A spike's return legs are neither spikes nor steps. FAIL: a spike in a major
    galaxy, an unexplained step of the main host; FLAG: any other spike or
    unexplained step. SCORE: per 100 resolved histories, and mass-weighted.
2.3 old-star flicker: L(s-1) = a, L(s) != a, L(s+j) = a for some j in 1..3 (-1 is a
    label). SCORE: median and max over snapshots with >= 1e4 old stars of the
    flickering share of the old mass; FLAG a snapshot above max(5 x median, 1e-4).
2.4 permanent transfers of old stars: L(s-1) = a, L(s) = b != a, L(s+1..s+3) = b;
    classes merger (a's branch ends at s-1 or s), unbound (b = -1), rebound
    (a = -1), stripping (b among a's hosts), capture (a among b's hosts), swap.
    A satellite's hosts: every more massive halo it is bound to inside its R_vir.
    SCORE: median capture share of the old mass of hosts, median swap share;
    FLAG a snapshot whose capture share is above max(5 x median, 1e-4).
2.5 newborns: kept when L(s+1) = L(s+2) = L(s). SCORE: kept mass share,
    overall, main host, satellites, other centrals.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate.load import ABSENT, MAJOR, RESOLVED, UNASSIGNED, mergers, on
from evaluate.report import Report

OLD_GYR = 1.0
LOG_JUMP, LOG_BACK, AHEAD = np.log10(2.0), np.log10(1.25), 3
MIN_OLD = 1e4
SPIKE_FACTOR, SHARE_FLOOR = 5.0, 1e-4
POS = ["position_x", "position_y", "position_z"]
FLICKER_CLASS = {"stripping": "host<->satellite", "capture": "host<->satellite", "unbound": "<->unbound",
                 "rebound": "<->unbound", "swap": "other", "merger": "merger"}
TRANSFERS = ["merger", "unbound", "rebound", "stripping", "capture", "swap"]
NEWBORN = ["all", "main host", "satellites", "other centrals"]


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


def steps(M):
    """Spikes and persistent steps of a (galaxy x snapshot) mass table, NaN where not resolved.

    A spike's return legs (its steps at s+1..s+j, back by s+j) are neither spikes nor steps.
    """
    lm = np.log10(M.to_numpy(np.float64))
    S = lm.shape[1]
    step = np.zeros(lm.shape, bool)
    step[:, 1:] = np.abs(lm[:, 1:] - lm[:, :-1]) > LOG_JUMP
    spike, leg = np.zeros_like(step), np.zeros_like(step)
    for j in range(1, AHEAD + 1):
        back = np.zeros_like(step)
        back[:, 1:S - j] = np.abs(lm[:, 1 + j:] - lm[:, :S - 1 - j]) < LOG_BACK     # M(s+j) vs M(s-1)
        new = step & back & ~spike
        spike |= new
        for i in range(1, j + 1):
            leg[:, i:] |= new[:, :S - i]
    return spike, step & ~spike & ~leg


def by_size(rep, n, check, what):
    """FAIL for a major galaxy, FLAG for a resolved one, nothing below."""
    if n >= MAJOR:
        rep.fail(check, what)
    elif n >= RESOLVED:
        rep.flag(check, what)


def window(run, k, first, t_first, hosts, ends, pair_mass):
    """2.3-2.5 at the k-th snapshot of the run: one row of shares."""
    sn = run.snaps
    s = sn[k]
    ids, lab = run.labels(s)
    L = np.stack([on(ids, *run.labels(sn[k - 1]))] + [lab]
                 + [on(ids, *run.labels(sn[k + j])) for j in range(1, AHEAD + 1)])
    m = np.nan_to_num(run.masses(s))
    present = (L != ABSENT).all(0)
    old = present & (run.time[s] - t_first[ids] >= OLD_GYR)
    a, b, later = L[0], L[1], L[2:]
    moved = old & (b != a)
    flick = moved & (later == a).any(0)
    stay = moved & (later == b).all(0)
    H = hosts[s]
    m_old = max(m[old].sum(), 1e-300)                 # no old stars: shares 0, and the snapshot is out of scope (n_old)
    row = {"snap": s, "t": run.time[s], "n_old": int(old.sum()), "flicker": m[flick].sum() / m_old}

    fc = pd.Series(classify(a[flick], b[flick], H, ends), dtype=object).map(FLICKER_CLASS).to_numpy()
    for c in set(FLICKER_CLASS.values()):
        row[f"flicker {c}"] = m[flick][fc == c].sum() / m_old
    pairs = pd.DataFrame({"a": a[flick], "b": b[flick], "m": m[flick]}).groupby(["a", "b"])["m"].agg(["sum", "size"])
    for (x, y), v in pairs.iterrows():
        acc = pair_mass.setdefault((int(x), int(y)), [0.0, 0, 0, FLICKER_CLASS[relation(int(x), int(y), H, ends)]])
        acc[0] += v["sum"]; acc[1] += int(v["size"]); acc[2] += 1

    tc = classify(a[stay], b[stay], H, ends)
    for c in TRANSFERS:
        row[c] = m[stay][tc == c].sum() / m_old
    hosting = old & np.isin(a, np.array(sorted(set().union(*H.values())), np.int64))
    row["capture of host old mass"] = m[stay][tc == "capture"].sum() / max(m[hosting].sum(), 1e-300)

    nb = (first[ids] == s) & (L[1:4] != ABSENT).all(0) & (b != UNASSIGNED)
    kept = nb & (later[0] == b) & (later[1] == b)
    sat = np.isin(b, np.array(list(H), np.int64))
    for name, sel in zip(NEWBORN, (nb, nb & (b == run.host), nb & sat, nb & (b != run.host) & ~sat)):
        row[f"newborn {name}"], row[f"newborn kept {name}"] = m[sel].sum(), m[kept & sel].sum()
    return row


def evaluate(run, out):
    rep = Report("Point 2, stable in time", run.name)
    sn = run.snaps
    tree_last = run.tree.groupby("Sub_tree_id")["Snapshot"].max()
    tree_has = set(zip(run.tree["Sub_tree_id"].astype(int), run.tree["Snapshot"]))
    no_uid = set(zip(run.tree.loc[run.tree["uid"].isna(), "Sub_tree_id"].astype(int),
                     run.tree.loc[run.tree["uid"].isna(), "Snapshot"]))
    merge = mergers(run.tree)

    first = np.full(0, -1, np.int64)                    # first-seen snapshot by particle ID
    time_of = np.array([run.time.get(s, np.nan) for s in range(int(sn.max()) + 1)])
    N, M, NB, Mcat, pos, hosts, rows, pair_mass = {}, {}, {}, {}, {}, {}, [], {}
    for k, s in enumerate(sn):
        ids, lab = run.labels(s)
        if ids.max() >= first.size:
            first = np.concatenate([first, np.full(int(ids.max()) + 1 - first.size, -1, np.int64)])
        first[ids] = np.where(first[ids] < 0, s, first[ids])
        m = np.nan_to_num(run.masses(s))
        g, inv = np.unique(lab, return_inverse=True)
        N[s] = pd.Series(np.bincount(inv), index=g)
        M[s] = pd.Series(np.bincount(inv, weights=m), index=g)
        NB[s] = pd.Series(np.bincount(inv, weights=m * (first[ids] == s)), index=g)
        gal = run.galaxies(s)
        Mcat[s], pos[s], hosts[s] = gal["Mtot"], gal[POS], run.hosts(s)
        w = k - AHEAD                                   # the window at sn[w] needs sn[w-1 .. w+3]: all read by now
        if w >= 1:
            ends = set(tree_last.index[tree_last.isin([sn[w - 1], sn[w]])].astype(int))
            rows.append(window(run, w, first, time_of[np.maximum(first, 0)], hosts, ends, pair_mass))
    N = pd.DataFrame(N).drop(index=UNASSIGNED, errors="ignore").reindex(columns=sn)
    M = pd.DataFrame(M).drop(index=UNASSIGNED, errors="ignore").reindex(columns=sn)
    NB = pd.DataFrame(NB).reindex(index=M.index, columns=sn).fillna(0.0)
    Mcat = pd.DataFrame(Mcat).reindex(columns=sn)
    W = pd.DataFrame(rows).set_index("snap")

    continuity(run, N, tree_has, tree_last, rep)
    Mres = M.where(N >= RESOLVED)
    events = mass_steps(run, Mres, M, N, NB, merge, tree_last, no_uid, pos, hosts, rep)
    rep.score("2.2 catalogue Mtot spikes per 100 resolved histories",
              100.0 * steps(Mcat.reindex(Mres.index).where(N >= RESOLVED))[0].sum() / max(n_histories(Mres), 1))
    label_shares(W, pair_mass, rep)
    plot_masses(run, Mres, Mcat, events, out / "stability_masses.png")
    plot_shares(run, W, out / "stability_labels.png")
    rep.write(out / "stability.txt")
    return rep


def n_histories(Mres):
    """Galaxies resolved at two consecutive snapshots at least."""
    r = Mres.notna().to_numpy()
    return int((r[:, 1:] & r[:, :-1]).any(axis=1).sum())


def continuity(run, N, tree_has, tree_last, rep):
    """2.1."""
    sn = run.snaps
    n = N.fillna(0).to_numpy()
    dropped = 0
    for i, G in enumerate(N.index.astype(int)):
        p = np.flatnonzero(n[i] >= 1)
        for k0, k1 in zip(p[:-1], p[1:]):
            gap = sn[k0 + 1:k1]
            if 2 <= k1 - k0 <= AHEAD + 1 and any((G, int(s)) in tree_has for s in gap):
                dropped += n[i, k0] >= RESOLVED
                by_size(rep, n[i, k0], "2.1 dropped gap (major FAIL, resolved FLAG)",
                        f"galaxy {G}: absent at {list(map(int, gap))}, {int(n[i, k0])} particles before")
        if p.size and p[-1] < sn.size - 1 and G in tree_last.index and sn[p[-1]] < tree_last[G] - 1:
            by_size(rep, n[i, p[-1]], "2.1 premature end (major FAIL, resolved FLAG)",
                    f"galaxy {G}: last at {int(sn[p[-1]])} with {int(n[i, p[-1]])} particles, tree to {int(tree_last[G])}")
    rep.score("2.1 dropped gaps per 1000 resolved galaxy-snapshots", 1000.0 * dropped / max(int((n >= RESOLVED).sum()), 1))


def mass_steps(run, Mres, M, N, NB, merge, tree_last, no_uid, pos, hosts, rep):
    """2.2; returns the events for the plot."""
    sn = list(run.snaps)
    spike, persist = steps(Mres)
    events = []
    for gi, k in zip(*np.nonzero(spike)):
        G = int(Mres.index[gi])
        events.append((G, sn[k], "spike"))
        (rep.fail if N.at[G, sn[k - 1]] >= MAJOR else rep.flag)(
            "2.2 spike (major FAIL)", f"galaxy {G} at {sn[k]}: {Mres.iat[gi, k - 1]:.3g} -> {Mres.iat[gi, k]:.3g}")
    reasons = {}
    for gi, k in zip(*np.nonzero(persist)):
        G, s = int(Mres.index[gi]), sn[k]
        m0, m1 = Mres.iat[gi, k - 1], Mres.iat[gi, k]
        why = "unexplained"
        if (G, s) in no_uid or (G, sn[k - 1]) in no_uid:
            why = "tree incomplete"
        if m1 > m0 and NB.at[G, s] >= 0.5 * (m1 - m0):
            why = "star formation"
        elif m1 > m0:
            near = merge[(merge["into"] == G) & merge["snapshot"].between(sn[k - 1], sn[min(k + 1, len(sn) - 1)])]
            gain = [M.loc[p, :s].dropna().iloc[-1] for p in near["progenitor"] if p in M.index and M.loc[p, :s].notna().any()]
            if any(x >= 0.5 * (m1 - m0) for x in gain):
                why = "merger"
        elif G in tree_last.index and tree_last[G] <= sn[min(k + AHEAD, len(sn) - 1)]:
            why = "merging"
        elif run.host in hosts[s].get(G, ()):
            d = []
            for j in range(max(k - 2, 0), min(k + 3, len(sn))):
                p = pos[sn[j]]
                d.append(np.linalg.norm(p.loc[G].to_numpy(float) - p.loc[run.host].to_numpy(float))
                         if G in p.index and run.host in p.index else np.nan)
            if np.isfinite(d).all() and np.argmin(d) == k - max(k - 2, 0):
                why = "pericentre"
        reasons[why] = reasons.get(why, 0) + 1
        if why == "unexplained":
            events.append((G, s, "step"))
            (rep.fail if G == run.host else rep.flag)("2.2 unexplained step (main host FAIL)", f"galaxy {G} at {s}: {m0:.3g} -> {m1:.3g}")
    nh = max(n_histories(Mres), 1)
    total = np.nansum(Mres.to_numpy())
    prev = Mres.shift(1, axis=1)
    rep.score("2.2 spikes per 100 resolved histories", 100.0 * spike.sum() / nh)
    rep.score("2.2 unexplained steps per 100 resolved histories", 100.0 * reasons.get("unexplained", 0) / nh)
    rep.score("2.2 spikes, mass-weighted", float(np.nansum(prev.to_numpy()[spike]) / total))
    rep.score("2.2 unexplained steps, mass-weighted",
              float(sum(Mres.at[G, sn[sn.index(s) - 1]] for G, s, e in events if e == "step") / total))
    rep.table("2.2 persistent steps by explanation", "\n".join(f"  {k:16s} {v}" for k, v in sorted(reasons.items())) or "  (none)")
    return events


def label_shares(W, pair_mass, rep):
    """2.3-2.5 scores and flags."""
    inscope = W[W["n_old"] >= MIN_OLD]
    for col, check in (("flicker", "2.3 flicker"), ("capture of host old mass", "2.4 capture")):
        x = inscope[col]
        med = float(x.median())
        rep.score(f"{check} share, median", med)
        rep.score(f"{check} share, max", float(x.max()))
        thr = max(SPIKE_FACTOR * med, SHARE_FLOOR)
        for s in x.index[x > thr]:
            rep.flag(f"{check} share above max(5 x median, 1e-4)", f"snapshot {s}: {x[s]:.2e}")
    rep.score("2.4 swap share of old mass, median", float(inscope["swap"].median()))
    for c in NEWBORN:
        rep.score(f"2.5 newborn mass kept, {c}", float(W[f"newborn kept {c}"].sum() / max(W[f"newborn {c}"].sum(), 1e-300)))
    rep.table("2.3 flicker share of old mass by relation (median over snapshots)",
              "\n".join(f"  {c:20s} {inscope[f'flicker {c}'].median():.2e}" for c in sorted(set(FLICKER_CLASS.values()))))
    rep.table("2.4 permanent transfers, share of old mass (median / max over snapshots)",
              "\n".join(f"  {c:12s} {inscope[c].median():.2e} / {inscope[c].max():.2e}" for c in TRANSFERS))
    top = sorted(pair_mass.items(), key=lambda kv: -kv[1][0])[:10]
    rep.table("2.3 top flickering pairs (a -> b -> a): mass, particles, snapshots, relation",
              "\n".join(f"  {a:>8d} -> {b:>8d}  {v[0]:10.3g} {v[1]:9d} {v[2]:5d}  {v[3]}" for (a, b), v in top) or "  (none)")


def plot_masses(run, Mres, Mcat, events, png):
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
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def plot_shares(run, W, png):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for c in ["flicker", "capture of host old mass", "stripping", "swap", "unbound"]:
        a1.semilogy(W["t"], W[c].clip(lower=1e-7), lw=0.9, label=c)
    a1.set_ylabel("share of old mass")
    a1.legend(fontsize=8)
    for c in NEWBORN[:3]:
        a2.plot(W["t"], W[f"newborn kept {c}"] / W[f"newborn {c}"], lw=0.9, label=c)
    a2.set_ylabel("newborn mass kept 2 snapshots")
    a2.set_xlabel("t [Gyr]")
    a2.legend(fontsize=8)
    fig.suptitle(f"{run.name}: label stability")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)

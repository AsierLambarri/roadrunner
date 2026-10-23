"""Goal 5: are the births and the ex-situ particles right? Measures and reports, for the reference
catalogue (--path) and each other (--vs), on the particles every catalogue holds (``load.aligned``).
Births are each catalogue's own table, ``on(ids, *c.births(), fill=ABSENT)`` (negative: unknown).
Galaxy ids are Sub_tree_ids, comparable across catalogues. Every number is a mass share or a
normalised overlap, so galaxies of any mass compare on one scale. Scores are prefixed with the other
catalogue's name.

5.1 births, on the particles held at the last common snapshot (stars persist, so it holds every star
    formed), with their masses there: the mass share with the same, known birth galaxy; the unknown
    share in each; the known-in-both-but-different share. Per birth galaxy g with >= 1e6 born mass in
    either: mA, mB, their overlap m_both; recall = m_both / mA, purity = m_both / mB, Jaccard
    J = m_both / (mA + mB - m_both); the largest destination of A's g-born mass in B (a galaxy, or
    "unknown") other than g, and its share of mA. SCOREs: same-birth share; unknown shares; (mA + mB)-
    weighted medians of J, recall, purity over birth galaxies; birth galaxies with J < 0.9; a table of
    the 15 largest by mA.
5.2 ex-situ, galaxy by galaxy, every --every-th common snapshot plus the last (parallel per snapshot,
    ``pool``): for every galaxy h with >= 1e6 hard-label mass in either catalogue at s,
    X_c(h) = {n: label_c(n) = h, birth_c(n) known, birth_c(n) != h}: recall, purity, J of X_A(h) and
    X_B(h); each catalogue's ex-situ share m(X_c(h)) / M_c(h); the unknown-birth share of h's members
    in each. The disagreement, as shares of m(X_A(h) | X_B(h)): membership (h in only one catalogue)
    and birth (h in both, ex-situ status differs) -- the two partition the symmetric difference. Per
    progenitor p (the union, over catalogues, of those carrying >= 1% of that catalogue's m(X_c(h))):
    recall, purity, J of C_A(h, p) = {label_A = h, birth_A = p} and C_B(h, p). SCOREs at the last
    snapshot: the host's ex-situ shares, J, recall, purity, disagreement shares; over galaxies >= 1e7
    there, the mass-weighted median J and the disagreement shares of their summed m(X_A | X_B); tables
    of the last snapshot's galaxies >= 1e7 (with the unknown-birth shares) and of the host's
    progenitors (union).
5.3 the host's main progenitors' histories: the union of the 10 largest contributors to its ex-situ
    mass at the last snapshot, in either catalogue. Per catalogue: hard-label mass at every measured
    snapshot; its peak and peak time; infall (the first measured snapshot the catalogue lists it as a
    satellite of the host, ``c.hosts(s)``); disruption (the first measured snapshot after the peak
    with hard-label mass below 1% of it). SCOREs: median |log10 ratio| of peak masses (B / A), median
    |delta t| of infall and of disruption (Gyr); a table per progenitor.

    python3 evaluate/science.py --path REF --vs OTHER [...] [--particles DIR] [--workers N]
                                [--snaps A:B] [--every N]

Writes evaluation/science.txt and .log; plots/run/: science_births.pdf (recall against purity per
birth galaxy, J against birth mass), science_exsitu.pdf (the host's ex-situ shares, J, disagreement
shares and the mass-weighted median J over galaxies >= 1e7, against time), science_progenitors.pdf
(up to 12 of the host's progenitors' mass histories, one line per catalogue, infall dashed, disruption
dotted); plots/g<id>/science_exsitu.pdf for ``plots.chosen(ref.host, peak label counts in the
reference, set())`` and the host's progenitors (its ex-situ mass, J and disagreement shares against
time, and its mass history when it is a progenitor).
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
from evaluate.load import (ABSENT, Run, UNASSIGNED, aligned, common_snaps, label_mass, log_ratio, on,
                           open_catalogue, pool, wmedian)
from evaluate.report import Report

MIN_BORN = 1e6                    # 5.1: minimum born mass in either catalogue to enter the birth-galaxy table
MIN_HOST = 1e6                    # 5.2: minimum hard-label mass in either catalogue to be a galaxy h
MIN_PROG_SHARE = 0.01             # 5.2: a progenitor must carry this share of m(X_c(h)) in some catalogue
MIN_SCORE_MASS = 1e7              # 5.2: SCOREs and the table are over galaxies at least this massive
N_TOP_PROG = 10                   # 5.3: per catalogue, the largest contributors to the host's ex-situ mass
N_BIRTH_TABLE = 15
J_LOW = 0.9                       # 5.1: J below this counts as "disagreeing"
DISRUPT_FRAC = 0.01               # 5.3: disruption is a drop below this share of the peak mass


# --- pure measurements (tested directly) -----------------------------------------------------------

def exsitu_overlap(memA, memB, bidA, bidB, m, h):
    """5.2: one galaxy ``h``'s ex-situ overlap between two catalogues (``mem.`` hard labels == h,
    ``bid.`` births, ``m`` masses, parallel arrays). X_c(h) = members of h in c with a known birth
    other than h. Returns each catalogue's ex-situ mass, their overlap, recall / purity / J (A -> B),
    and the disagreement split of m(X_A | X_B) (``mUnion``): ``membership_share`` (h in only one
    catalogue) and ``birth_share`` (h in both, ex-situ status differs) -- the two partition the symmetric
    difference.
    """
    XA = memA & (bidA >= 0) & (bidA != h)
    XB = memB & (bidB >= 0) & (bidB != h)
    mA, mB, mBoth = float(m[XA].sum()), float(m[XB].sum()), float(m[XA & XB].sum())
    sym = XA ^ XB
    membership = sym & (memA != memB)
    birth = sym & ~(memA != memB)
    denom = float(m[XA | XB].sum())
    return dict(
        mA=mA, mB=mB, mBoth=mBoth, mUnion=denom,
        recall=mBoth / mA if mA > 0 else np.nan,
        purity=mBoth / mB if mB > 0 else np.nan,
        J=mBoth / (mA + mB - mBoth) if (mA + mB - mBoth) > 0 else np.nan,
        membership_share=float(m[membership].sum()) / denom if denom > 0 else np.nan,
        birth_share=float(m[birth].sum()) / denom if denom > 0 else np.nan,
    )


def birth_overlap(bidA, bidB, m):
    """5.1 per birth galaxy, indexed by the union of galaxies with >= MIN_BORN born mass in either:
    mA, mB, their overlap m_both, recall / purity / J (A -> B), and the largest destination (other
    than itself) of A's g-born mass in B, with its share of mA (``dest`` ABSENT: unknown; None: no
    such destination)."""
    mA_all, mB_all = label_mass(bidA, m), label_mass(bidB, m)
    g = sorted({x for x in mA_all.index if x >= 0 and mA_all[x] >= MIN_BORN}
               | {x for x in mB_all.index if x >= 0 and mB_all[x] >= MIN_BORN})
    both = (bidA == bidB) & (bidA >= 0)
    mBoth_all = label_mass(bidA[both], m[both])

    known = bidA >= 0
    dest = np.where(bidB >= 0, bidB, ABSENT)
    mv = (pd.DataFrame({"g": bidA[known], "dest": dest[known], "m": m[known]})
          .groupby(["g", "dest"])["m"].sum())
    other = mv[mv.index.get_level_values("g") != mv.index.get_level_values("dest")]
    top = (other.rename("mass").reset_index().sort_values("mass", ascending=False)
           .drop_duplicates("g").set_index("g"))

    rows = []
    for x in g:
        mA, mB, mBoth = float(mA_all.get(x, 0.0)), float(mB_all.get(x, 0.0)), float(mBoth_all.get(x, 0.0))
        has_dest = x in top.index
        rows.append(dict(
            galaxy=x, mA=mA, mB=mB, mBoth=mBoth,
            recall=mBoth / mA if mA > 0 else np.nan,
            purity=mBoth / mB if mB > 0 else np.nan,
            J=mBoth / (mA + mB - mBoth) if (mA + mB - mBoth) > 0 else np.nan,
            dest=int(top.at[x, "dest"]) if has_dest else None,
            dest_share=float(top.at[x, "mass"]) / mA if has_dest and mA > 0 else np.nan,
        ))
    return pd.DataFrame(rows).set_index("galaxy")


# --- 5.1: report ------------------------------------------------------------------------------------

def report_births(ref, c, bidA, bidB, m, rep):
    """5.1 overall mass shares between ``ref`` and ``c`` on the particles both hold."""
    total = float(m.sum())
    known = (bidA >= 0) & (bidB >= 0)
    rep.score(f"{c.name}: 5.1 same known birth galaxy, mass share",
              float(m[known & (bidA == bidB)].sum() / total))
    rep.score(f"{c.name}: 5.1 known in both, different birth galaxy, mass share",
              float(m[known & (bidA != bidB)].sum() / total))
    rep.score(f"{c.name}: 5.1 unknown birth in {ref.name}, mass share", float(m[bidA < 0].sum() / total))
    rep.score(f"{c.name}: 5.1 unknown birth in {c.name}, mass share", float(m[bidB < 0].sum() / total))


def report_birth_table(ref, c, BT, rep):
    """5.1 SCOREs and the table of the largest birth galaxies, from :func:`birth_overlap`'s output."""
    rep.score(f"{c.name}: 5.1 birth galaxies measured", len(BT))
    if BT.empty:
        return
    w = (BT["mA"] + BT["mB"]).to_numpy()
    for name, col in (("J", "J"), ("recall", "recall"), ("purity", "purity")):
        rep.score(f"{c.name}: 5.1 (mA + mB)-weighted median {name} over birth galaxies",
                  wmedian(BT[col].to_numpy(), w))
    rep.score(f"{c.name}: 5.1 birth galaxies with J < {J_LOW}", int((BT["J"] < J_LOW).sum()))
    top = BT.sort_values("mA", ascending=False).head(N_BIRTH_TABLE).copy()
    top["destination"] = ["none" if pd.isna(d) else ("unknown" if int(d) == ABSENT else str(int(d)))
                          for d in top["dest"]]
    rep.table(f"5.1 {c.name} vs {ref.name}: largest birth galaxies (by mA)",
              top[["mA", "mB", "recall", "purity", "J", "destination", "dest_share"]]
              .to_string(float_format=lambda v: f"{v:.3e}"))


# --- 5.2: the workers --------------------------------------------------------------------------------

_W = {}                                 # a worker's catalogues and particle run


def _init(paths, path, source):
    _W.update(cats=[open_catalogue(p) for p in paths], particles=Run(path, particles=source))


def _snapshot(s):
    """5.2 at ``s`` in a worker: per other catalogue, the host rows and the progenitor rows (every
    galaxy h with >= 1e6 hard-label mass in either); each catalogue's hard-label mass by galaxy and
    label counts (for plots.chosen's peak, and 5.3's histories); the particle share held by every
    catalogue."""
    cats, particles = _W["cats"], _W["particles"]
    ref, others = cats[0], cats[1:]
    ids, _, m, L, share = aligned(cats, particles, s)
    t = particles.time[s]
    bid = [on(ids, *c.births(), fill=ABSENT) for c in cats]
    mass = {c.name: label_mass(lab, m) for c, lab in zip(cats, L)}
    counts = {}
    for c, lab in zip(cats, L):
        g, n = np.unique(lab[lab != UNASSIGNED], return_counts=True)
        counts[c.name] = dict(zip(g.tolist(), n.tolist()))
    hostset = set()
    for series in mass.values():
        hostset |= set(series.index[(series.index != UNASSIGNED) & (series >= MIN_HOST)].tolist())

    memA, bidA = L[0], bid[0]
    host_rows, prog_rows = [], []
    for ci, c in enumerate(others, start=1):
        labc, bidc = L[ci], bid[ci]
        agree = (memA == labc) & (bidA == bidc)
        both = (pd.Series(m[agree]).groupby([memA[agree], bidA[agree]]).sum() if agree.any()
                else pd.Series(dtype=float))
        for h in hostset:
            memAh, memBh = memA == h, labc == h
            o = exsitu_overlap(memAh, memBh, bidA, bidc, m, h)
            MA_h, MB_h = float(mass[ref.name].get(h, 0.0)), float(mass[c.name].get(h, 0.0))
            host_rows.append(dict(
                snap=s, t=t, catalogue=c.name, host=int(h), MA=MA_h, MB=MB_h,
                exA=o["mA"] / MA_h if MA_h > 0 else np.nan, exB=o["mB"] / MB_h if MB_h > 0 else np.nan,
                recall=o["recall"], purity=o["purity"], J=o["J"], mUnion=o["mUnion"],
                membership_share=o["membership_share"], birth_share=o["birth_share"],
                unknownA=float(m[memAh & (bidA < 0)].sum()) / MA_h if MA_h > 0 else np.nan,
                unknownB=float(m[memBh & (bidc < 0)].sum()) / MB_h if MB_h > 0 else np.nan,
            ))

            XA = memAh & (bidA >= 0) & (bidA != h)
            XB = memBh & (bidc >= 0) & (bidc != h)
            pA = label_mass(bidA[XA], m[XA]) if XA.any() else pd.Series(dtype=float)
            pB = label_mass(bidc[XB], m[XB]) if XB.any() else pd.Series(dtype=float)
            mXA, mXB = float(pA.sum()), float(pB.sum())
            qual = (set(pA.index[pA >= MIN_PROG_SHARE * mXA]) if mXA > 0 else set())
            qual |= (set(pB.index[pB >= MIN_PROG_SHARE * mXB]) if mXB > 0 else set())
            for p in qual:
                mA_p, mB_p = float(pA.get(p, 0.0)), float(pB.get(p, 0.0))
                mBoth_p = float(both.get((h, p), 0.0)) if not both.empty else 0.0
                denom = mA_p + mB_p - mBoth_p
                prog_rows.append(dict(
                    snap=s, t=t, catalogue=c.name, host=int(h), progenitor=int(p), mA=mA_p, mB=mB_p,
                    recall=mBoth_p / mA_p if mA_p > 0 else np.nan,
                    purity=mBoth_p / mB_p if mB_p > 0 else np.nan,
                    J=mBoth_p / denom if denom > 0 else np.nan,
                ))
    return host_rows, prog_rows, mass, counts, share


# --- 5.2: report ------------------------------------------------------------------------------------

def report_exsitu(ref, others, H, Pg, last, rep):
    """5.2 SCOREs and tables at the last snapshot."""
    Hl = H[H["snap"] == last]
    for c in others:
        host = Hl[(Hl["catalogue"] == c.name) & (Hl["host"] == ref.host)]
        if not host.empty:
            row = host.iloc[0]
            rep.score(f"{c.name}: 5.2 host ex-situ share in {ref.name}", float(row["exA"]))
            rep.score(f"{c.name}: 5.2 host ex-situ share in {c.name}", float(row["exB"]))
            rep.score(f"{c.name}: 5.2 host J", float(row["J"]))
            rep.score(f"{c.name}: 5.2 host recall", float(row["recall"]))
            rep.score(f"{c.name}: 5.2 host purity", float(row["purity"]))
            rep.score(f"{c.name}: 5.2 host membership-disagreement share", float(row["membership_share"]))
            rep.score(f"{c.name}: 5.2 host birth-disagreement share", float(row["birth_share"]))

        big = Hl[(Hl["catalogue"] == c.name) & ((Hl["MA"] >= MIN_SCORE_MASS) | (Hl["MB"] >= MIN_SCORE_MASS))]
        if not big.empty:
            w = (big["MA"] + big["MB"]).to_numpy()
            rep.score(f"{c.name}: 5.2 mass-weighted median J, galaxies >= 1e7", wmedian(big["J"].to_numpy(), w))
            big = big[big["mUnion"] > 0]                    # no ex-situ mass in either: nothing to split
            u = big["mUnion"].to_numpy()
            for kind in ("membership", "birth"):
                rep.score(f"{c.name}: 5.2 {kind}-disagreement share of the ex-situ union mass, galaxies >= 1e7",
                          float((big[f"{kind}_share"].to_numpy() * u).sum() / u.sum()) if u.sum() > 0 else np.nan)
            tab = big.sort_values("MA", ascending=False).set_index("host")[
                ["MA", "MB", "exA", "exB", "unknownA", "unknownB", "J", "membership_share", "birth_share"]]
            rep.table(f"5.2 {c.name} vs {ref.name}: galaxies >= 1e7 at the last snapshot",
                      tab.to_string(float_format=lambda v: f"{v:.3e}"))

        prog = Pg[(Pg["snap"] == last) & (Pg["catalogue"] == c.name) & (Pg["host"] == ref.host)]
        if not prog.empty:
            tab = (prog.sort_values("mA", ascending=False).set_index("progenitor")
                   [["mA", "mB", "recall", "purity", "J"]])
            rep.table(f"5.2 {c.name} vs {ref.name}: the host's progenitors "
                      "(union, >= 1% of either's ex-situ mass)", tab.to_string(float_format=lambda v: f"{v:.3e}"))


# --- 5.3 --------------------------------------------------------------------------------------------

def progenitor_histories(cats, Pg, ref, last, todo, mass_by_snap):
    """5.3: the union of the host's N_TOP_PROG largest contributors (either catalogue) at the last
    snapshot; per progenitor and catalogue, its hard-label mass at every snapshot in ``todo``, its
    peak and peak time, infall (the first snapshot the catalogue lists it as a satellite of the host)
    and disruption (the first snapshot after the peak below DISRUPT_FRAC of it)."""
    last_p = Pg[(Pg["snap"] == last) & (Pg["host"] == ref.host)]
    if last_p.empty:
        return [], pd.DataFrame()
    cols = {ref.name: last_p.drop_duplicates("progenitor").set_index("progenitor")["mA"]}
    for c in cats[1:]:
        cols[c.name] = last_p[last_p["catalogue"] == c.name].set_index("progenitor")["mB"]
    MAT = pd.DataFrame(cols).fillna(0.0)
    track = sorted(set().union(*(set(MAT[col].nlargest(N_TOP_PROG).index) for col in MAT.columns)))

    rows = []
    for p in track:
        for c in cats:
            hist = np.array([mass_by_snap[s][c.name].get(p, 0.0) for s in todo])
            i = int(np.argmax(hist))
            peak_mass, peak_t = float(hist[i]), float(c.time[todo[i]])
            infall = next((s for s in todo if ref.host in c.hosts(s).get(p, set())), None)
            after = [s for k, s in enumerate(todo[i + 1:], start=i + 1) if hist[k] < DISRUPT_FRAC * peak_mass]
            disrupt = after[0] if peak_mass > 0 and after else None
            rows.append(dict(
                progenitor=p, catalogue=c.name, history=hist, peak_mass=peak_mass, peak_t=peak_t,
                infall_t=c.time[infall] if infall is not None else np.nan,
                disrupt_t=c.time[disrupt] if disrupt is not None else np.nan,
            ))
    return track, pd.DataFrame(rows)


def report_histories(cats, ref, hist, rep):
    """5.3 SCOREs and the table of the host's main progenitors."""
    rep.score("5.3 host progenitors tracked", 0 if hist.empty else int(hist["progenitor"].nunique()))
    if hist.empty:
        return
    ref_h = hist[hist["catalogue"] == ref.name].set_index("progenitor")
    for c in cats[1:]:
        h = hist[hist["catalogue"] == c.name].set_index("progenitor")
        common = ref_h.index.intersection(h.index)
        r = log_ratio(h.loc[common, "peak_mass"].to_numpy(), ref_h.loc[common, "peak_mass"].to_numpy())
        rep.score(f"{c.name}: 5.3 median |log10 ratio| of peak masses (B / A)", float(np.nanmedian(np.abs(r))))
        for col, name in (("infall_t", "infall"), ("disrupt_t", "disruption")):
            dt = (h.loc[common, col] - ref_h.loc[common, col]).abs()
            rep.score(f"{c.name}: 5.3 median |delta t| of {name} [Gyr]", float(dt.median()))
    tab = hist.pivot(index="progenitor", columns="catalogue",
                     values=["peak_mass", "peak_t", "infall_t", "disrupt_t"])
    rep.table("5.3 the host's main progenitors: peak mass/time, infall and disruption time per catalogue",
              tab.to_string(float_format=lambda v: f"{v:.3e}"))


# --- plots ------------------------------------------------------------------------------------------

def plot_births(ref, others, BT, pdf):
    """5.1: recall against purity per birth galaxy (marker size by log birth mass), J against birth
    mass."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, c in enumerate(others):
        t = BT[c.name]
        if t.empty:
            continue
        mass = np.maximum(t["mA"] + t["mB"], 1.0)
        a1.scatter(t["recall"], t["purity"], s=4.0 + 3.0 * np.log10(mass), color=f"C{i}", alpha=0.6, label=c.name)
        a2.loglog(mass, t["J"], "o", ms=4, color=f"C{i}", alpha=0.6, label=c.name)
    a1.plot([0, 1], [0, 1], ":", color="0.5")
    a1.set_xlabel("5.1 recall")
    a1.set_ylabel("purity")
    a1.legend(fontsize=8)
    a2.set_xlabel("5.1 birth mass, mA + mB")
    a2.set_ylabel("J")
    a2.legend(fontsize=8)
    fig.suptitle(f"{ref.name}: goal 5, births")
    plots.save(fig, pdf)


def plot_exsitu(ref, others, H, pdf):
    """5.2: the host's ex-situ shares, J and disagreement shares, and the mass-weighted median J over
    galaxies >= 1e7, against time."""
    fig, ax = plt.subplots(4, 1, figsize=(9, 13), sharex=True)
    for i, c in enumerate(others):
        d = H[(H["catalogue"] == c.name) & (H["host"] == ref.host)].sort_values("t")
        ax[0].plot(d["t"], d["exA"], "-", color=f"C{i}", label=f"{c.name}: {ref.name}")
        ax[0].plot(d["t"], d["exB"], "--", color=f"C{i}", label=f"{c.name}: {c.name}")
        ax[1].plot(d["t"], d["J"], "o-", ms=3, color=f"C{i}", label=c.name)
        ax[2].plot(d["t"], d["membership_share"], "-", color=f"C{i}", label=f"{c.name}: membership")
        ax[2].plot(d["t"], d["birth_share"], "--", color=f"C{i}", label=f"{c.name}: birth")
        big = H[(H["catalogue"] == c.name) & ((H["MA"] >= MIN_SCORE_MASS) | (H["MB"] >= MIN_SCORE_MASS))]
        if not big.empty:
            med = big.groupby("t").apply(lambda d: wmedian(d["J"].to_numpy(), (d["MA"] + d["MB"]).to_numpy()),
                                         include_groups=False)
            ax[3].plot(med.index, med, "o-", ms=3, color=f"C{i}", label=c.name)
    ax[0].set_ylabel("5.2 host ex-situ share")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_ylabel("5.2 host J")
    ax[1].legend(fontsize=7)
    ax[2].set_ylabel("5.2 host disagreement share")
    ax[2].legend(fontsize=6, ncol=2)
    ax[3].set_ylabel("5.2 mass-weighted median J, >= 1e7")
    ax[3].set_xlabel("t [Gyr]")
    ax[3].legend(fontsize=7)
    fig.suptitle(f"{ref.name}: goal 5, ex-situ")
    plots.save(fig, pdf)


def plot_progenitors(cats, hist, times, pdf):
    """5.3: a grid of up to 12 of the host's progenitors' mass histories, one line per catalogue,
    infall dashed, disruption dotted."""
    progs = list(dict.fromkeys(hist["progenitor"]))[:12] if not hist.empty else []
    n = max(len(progs), 1)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, p in zip(axes.flat, progs):
        for i, c in enumerate(cats):
            row = hist[(hist["progenitor"] == p) & (hist["catalogue"] == c.name)]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.semilogy(times, np.where(r["history"] > 0, r["history"], np.nan), "-", color=f"C{i}",
                        label=c.name)                       # it ends where the galaxy is gone
            if np.isfinite(r["infall_t"]):
                ax.axvline(r["infall_t"], ls="--", color=f"C{i}", lw=0.8)
            if np.isfinite(r["disrupt_t"]):
                ax.axvline(r["disrupt_t"], ls=":", color=f"C{i}", lw=0.8)
        ax.set_title(f"progenitor {p}", fontsize=8)
        ax.set_xlim(0.0, max(times))                         # one time axis for every panel
    if progs:
        axes.flat[0].legend(fontsize=6)
    for ax in axes.flat[len(progs):]:
        ax.axis("off")
    fig.suptitle("goal 5, host progenitors' histories (-- infall, : disruption)")
    plots.save(fig, pdf)


def _draw_galaxy(item):
    """A galaxy's 5.2 ex-situ mass in A and B, its J and disagreement shares against time; its mass
    history too when it is one of the host's progenitors."""
    path, g, ref_name, others, mine, prog, times = item
    nrow = 4 if prog is not None else 3
    fig, ax = plt.subplots(nrow, 1, figsize=(7, 3.0 * nrow))
    for i, name in enumerate(others):
        d = mine[mine["catalogue"] == name].sort_values("t")
        if d.empty:
            continue
        ax[0].semilogy(d["t"], np.maximum(d["MA"] * d["exA"], 1.0), "-", color=f"C{i}",
                       label=f"{name}: {ref_name}")
        ax[0].semilogy(d["t"], np.maximum(d["MB"] * d["exB"], 1.0), "--", color=f"C{i}", label=f"{name}: {name}")
        ax[1].plot(d["t"], d["J"], "o-", ms=3, color=f"C{i}", label=name)
        ax[2].plot(d["t"], d["membership_share"], "-", color=f"C{i}", label=f"{name}: membership")
        ax[2].plot(d["t"], d["birth_share"], "--", color=f"C{i}", label=f"{name}: birth")
    ax[0].set_ylabel("5.2 ex-situ mass")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_ylabel("J")
    ax[1].legend(fontsize=7)
    ax[2].set_ylabel("disagreement share")
    ax[2].legend(fontsize=6, ncol=2)
    if prog is not None:
        for i, name in enumerate([ref_name] + list(others)):
            row = prog[prog["catalogue"] == name]
            if row.empty:
                continue
            r = row.iloc[0]
            ax[3].semilogy(times, np.where(r["history"] > 0, r["history"], np.nan), "-", color=f"C{i}",
                           label=name)
        ax[3].set_ylabel("mass history")
        ax[3].set_xlabel("t [Gyr]")
        ax[3].legend(fontsize=6)
    else:
        ax[2].set_xlabel("t [Gyr]")
    fig.suptitle(f"galaxy {g}: goal 5")
    plots.save(fig, path)


# --- the point ----------------------------------------------------------------------------------------

def evaluate(cats, particles, out, every=3, workers=1):
    ref, others = cats[0], cats[1:]
    rep = Report("Goal 5, births and ex-situ particles", " / ".join(c.name for c in cats), verdict=False)
    snaps = common_snaps(cats, particles)
    last = snaps[-1]
    rep.score("last common snapshot", int(last))

    # 5.1
    ids, _, m, L, share = aligned(cats, particles, last)
    rep.score("particles held by every catalogue at it (share of the particle data)", float(share))
    bid = [on(ids, *c.births(), fill=ABSENT) for c in cats]
    BT = {}
    for c, bidc in zip(others, bid[1:]):
        report_births(ref, c, bid[0], bidc, m, rep)
        BT[c.name] = birth_overlap(bid[0], bidc, m)
        report_birth_table(ref, c, BT[c.name], rep)
    plot_births(ref, others, BT, plots.run_plot(out, "science_births"))

    # 5.2
    todo = [int(s) for s in sorted(set(snaps[::every]) | {last})]
    parts = pool(_snapshot, todo, workers, _init, ([c.path for c in cats], particles.path, particles.source),
                desc="5.2 ex-situ")
    host_rows, prog_rows, mass_by_snap, peak, shares = [], [], {}, {}, []
    for s, (hr, pr, mass, counts, sh) in zip(todo, parts):
        host_rows += hr
        prog_rows += pr
        mass_by_snap[s] = mass
        shares.append(sh)
        for g, n in counts[ref.name].items():
            peak[g] = max(peak.get(g, 0), n)
    rep.score("5.2 snapshots measured", len(todo))
    rep.score("5.2 median particle share held by every catalogue",
              float(np.median(shares)) if shares else np.nan)
    H = pd.DataFrame(host_rows)
    Pg = pd.DataFrame(prog_rows)
    report_exsitu(ref, others, H, Pg, last, rep)
    plot_exsitu(ref, others, H, plots.run_plot(out, "science_exsitu"))

    # 5.3
    track, hist = progenitor_histories(cats, Pg, ref, last, todo, mass_by_snap)
    report_histories(cats, ref, hist, rep)
    times = [particles.time[s] for s in todo]
    plot_progenitors(cats, hist, times, plots.run_plot(out, "science_progenitors"))

    # per-galaxy plots
    chosen = sorted(set(plots.chosen(ref.host, peak, set())) | set(track))
    others_names = [c.name for c in others]
    items = [(plots.galaxy_plot(out, g, "science_exsitu"), g, ref.name, others_names,
              H[H["host"] == g], hist[hist["progenitor"] == g] if g in track else None, times)
             for g in chosen]
    plots.per_galaxy(_draw_galaxy, items, workers, desc="5 galaxy plots")

    rep.write(out / "science.txt")
    return rep


def main():
    args = cli.arguments(__doc__, every=3, compare=True)
    cats, particles, out = cli.start_compare("science", args)
    evaluate(cats, particles, out, every=args.every, workers=args.workers)


if __name__ == "__main__":                      # spawned workers import this file, never run it
    main()

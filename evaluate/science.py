"""Goal 5: are the science outputs robust? Measures and reports, for two or more catalogues, on the
particles every catalogue holds at each snapshot; the first catalogue is the reference.

5.1 the main host at the last common snapshot: stellar mass (its hard labels), ex-situ share (born in
    another galaxy, final birth table), unknown-birth share; per catalogue, and the difference of the
    ex-situ share with the reference's.
5.2 progenitors: the host's ex-situ mass by birth galaxy. Over the progenitors carrying >= 1% of the
    reference's ex-situ mass: how many, how many a catalogue lacks, and the median |log10| ratio of the
    contributions it has to the reference's; the ten largest in a table.
5.3 satellites of the host at the last common snapshot (the reference's relations): hard-label stellar
    mass; number above 1e6, 1e7, 1e8; median |log10| ratio to the reference over those >= 1e6 in both,
    and how many >= 1e6 in the reference a catalogue has below 1e6; the fifteen largest in a table.
5.4 histories: hard-label stellar mass of the host and of the eight largest satellites of 5.3 every
    --every-th common snapshot; median |log10| ratio to the reference over (galaxy, snapshot) with
    >= 1e6 in both.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate.load import ABSENT, aligned, common_snaps, on
from evaluate.report import Report

MAJOR_PROGENITOR, MIN_MASS, SAT_LEVELS, N_TRACK = 0.01, 1e6, (1e6, 1e7, 1e8), 8


def label_mass(L, m):
    """Hard-label mass of every label."""
    g, inv = np.unique(L, return_inverse=True)
    return pd.Series(np.bincount(inv.ravel(), weights=m), index=g)


def log_ratio(a, b):
    """``log10(a / b)``, NaN where either is not positive."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((a > 0) & (b > 0), np.log10(a / b), np.nan)


def evaluate(cats, particles, out, every=10):
    rep = Report("Goal 5, science outputs", " / ".join(c.name for c in cats), verdict=False)
    snaps = common_snaps(cats, particles)
    last = snaps[-1]
    ids, X, m, L, share = aligned(cats, particles, last)
    ref, names = cats[0], [c.name for c in cats]
    rep.score(f"last common snapshot", int(last))
    rep.score("particles held by every catalogue at it (share of the particle data)", float(share))

    # 5.1 and 5.2
    contrib = {}
    for c, lab in zip(cats, L):
        host = lab == c.host
        bid = on(ids, *c.births(), fill=ABSENT)
        ex = host & (bid >= 0) & (bid != c.host)
        Mh = m[host].sum()
        rep.score(f"{c.name}: 5.1 host stellar mass", float(Mh))
        rep.score(f"{c.name}: 5.1 host ex-situ share", float(m[ex].sum() / Mh))
        rep.score(f"{c.name}: 5.1 host unknown-birth share", float(m[host & (bid < 0)].sum() / Mh))
        contrib[c.name] = label_mass(bid[ex], m[ex])
    for c in cats[1:]:
        rep.score(f"{c.name}: 5.1 ex-situ share minus {ref.name}'s",
                  rep.scores[f"{c.name}: 5.1 host ex-situ share"] - rep.scores[f"{ref.name}: 5.1 host ex-situ share"])
    P = pd.DataFrame(contrib).fillna(0.0)
    big = P.index[P[ref.name] >= MAJOR_PROGENITOR * P[ref.name].sum()]
    rep.score("5.2 progenitors >= 1% of the reference's ex-situ mass", len(big))
    for c in cats[1:]:
        r = log_ratio(P.loc[big, c.name].to_numpy(), P.loc[big, ref.name].to_numpy())
        rep.score(f"{c.name}: 5.2 of them missing", int((P.loc[big, c.name] <= 0).sum()))
        rep.score(f"{c.name}: 5.2 median |log10 ratio| of contributions", float(np.nanmedian(np.abs(r))))
    rep.table("5.2 largest progenitors: ex-situ mass contributed to the host",
              P.sort_values(ref.name, ascending=False).head(10).to_string(float_format=lambda v: f"{v:.3e}"))

    # 5.3
    sats = sorted(ref.satellites(last).get(ref.host, set()))
    S = pd.DataFrame({c.name: label_mass(lab, m).reindex(sats).fillna(0.0) for c, lab in zip(cats, L)})
    for c in cats:
        for lev in SAT_LEVELS:
            rep.score(f"{c.name}: 5.3 satellites above {lev:.0e}", int((S[c.name] >= lev).sum()))
    for c in cats[1:]:
        both = (S[ref.name] >= MIN_MASS) & (S[c.name] >= MIN_MASS)
        rep.score(f"{c.name}: 5.3 median |log10 ratio| of satellite masses",
                  float(np.nanmedian(np.abs(log_ratio(S.loc[both, c.name].to_numpy(), S.loc[both, ref.name].to_numpy())))))
        rep.score(f"{c.name}: 5.3 satellites >= 1e6 in the reference, below in it",
                  int(((S[ref.name] >= MIN_MASS) & (S[c.name] < MIN_MASS)).sum()))
    rep.table("5.3 largest satellites: hard-label stellar mass",
              S.sort_values(ref.name, ascending=False).head(15).to_string(float_format=lambda v: f"{v:.3e}"))

    # 5.4
    track = [ref.host] + list(S[ref.name].sort_values(ascending=False).index[:N_TRACK])
    rows = []
    for s in sorted(set(snaps[::every]) | {last}):
        ids_s, _, m_s, L_s, _ = aligned(cats, particles, s)
        for c, lab in zip(cats, L_s):
            for g, M in label_mass(lab, m_s).reindex(track).fillna(0.0).items():
                rows.append((c.name, int(g), int(s), float(M)))
        print(f"snapshot {s}", flush=True)
    H = pd.DataFrame(rows, columns=["catalogue", "galaxy", "snap", "mass"]).pivot_table(
        index=["galaxy", "snap"], columns="catalogue", values="mass")
    for c in cats[1:]:
        both = (H[ref.name] >= MIN_MASS) & (H[c.name] >= MIN_MASS)
        rep.score(f"{c.name}: 5.4 median |log10 ratio| of histories",
                  float(np.nanmedian(np.abs(log_ratio(H.loc[both, c.name].to_numpy(), H.loc[both, ref.name].to_numpy())))))
    plot(cats, ref, H, S, P, track, out / "science.png")
    rep.write(out / "science.txt")
    return rep


def plot(cats, ref, H, S, P, track, png):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    styles = ["-", "--", ":", "-."]
    for i, g in enumerate(track):
        h = H.loc[g]
        for j, c in enumerate(cats):
            ax[0].semilogy(h.index, h[c.name], styles[j % 4], color=f"C{i % 10}", lw=1.0,
                           label=str(g) if j == 0 else None)
    ax[0].set_xlabel("snapshot")
    ax[0].set_ylabel("hard-label stellar mass")
    ax[0].set_title("5.4 host + largest satellites (line style: " + ", ".join(c.name for c in cats) + ")", fontsize=8)
    ax[0].legend(fontsize=6, ncol=2)
    for j, c in enumerate(cats[1:], start=1):
        ax[1].loglog(S[ref.name].clip(lower=1e3), S[c.name].clip(lower=1e3), "o", ms=3, color=f"C{j}", label=c.name)
        ax[2].loglog(P[ref.name].clip(lower=1e3), P[c.name].clip(lower=1e3), "o", ms=3, color=f"C{j}", label=c.name)
    for a, what in ((ax[1], "5.3 satellite mass"), (ax[2], "5.2 ex-situ contribution")):
        lo, hi = a.get_xlim()
        a.plot([lo, hi], [lo, hi], ":", color="0.5")
        a.set_xlabel(f"{what}, {ref.name}")
        a.set_ylabel("other catalogues")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)

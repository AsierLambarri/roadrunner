# evaluate/

Checks and measurements of finished catalogues: roadrunner runs and ACCME results. It reads what a run
wrote (catalogue, assignment files, merger tree, particle data) and never reruns the pipeline.

There is no ground truth. ACCME is a reference, not the truth: fairly complete, but sometimes
contaminated (satellites capturing disc stars). So most of what is here is either self-consistency
(is the run sound, stable in time, internally coherent?) or agreement between catalogues, with the
places where they disagree broken down. Only points 1 and 2 give a verdict; the rest measure, and
the decisions are yours. This file explains how to read the numbers and which decision each one
informs.

Each module's docstring is the exact specification: every check, threshold and formula. This README
is the guide; when they differ, the docstring wins.

## The points

| Script | Goal | Kind | Reads |
|---|---|---|---|
| `soundness.py` | 1. the run is technically sound | verdict | one run |
| `stability.py` | 2. assignments are stable in time | verdict | one run |
| `model.py` | 3. the probabilistic model is sound | measures | one run |
| `science.py` | 5. births and ex-situ particles agree | measures | two or more catalogues |
| `compare.py` | 6. where catalogues differ | measures | two or more catalogues |
| `sweep.py` | 7. one table of every run's scores | collects | many runs |

(Goal 4, "stars go where they belong", was dropped: there is no baseline to test it against.)

Supporting modules: `load.py` (reading runs and ACCME, aligning particles, process pools),
`report.py` (symptoms, episodes, the report file), `cli.py` (command line, output folder, log),
`plots.py` (PDF layout), `pairs.py` (goal 3's reading of the stored pairs), `model_alpha.py` (goal 3's
3.2b and 3.4), `bgmm.py` (replays the Bayesian GMM for 3.2b), `run_all.py` (every per-run point).

## Running

From the repository root, in the pipeline's environment (`model.py` imports `roadrunner`: install it, or
set `PYTHONPATH=src`):

```
python3 evaluate/run_all.py --path RUN --particles DIR --workers 24          # soundness, stability, model
python3 evaluate/model.py   --path RUN --particles DIR --workers 24 --feedback --dependence
python3 evaluate/science.py --path REF --vs OTHER [OTHER ...] --particles DIR --workers 24
python3 evaluate/compare.py --path REF --vs OTHER [OTHER ...] --particles DIR --workers 24
python3 evaluate/sweep.py   --path RUN [RUN ...] --csv table.csv --reports .../science.txt .../compare.txt
```

- `--particles` is the folder holding `particle_data/` (for halo685x095:
  `~/roadrunner_test/halo685x095_v6`).
- `--snaps A:B` restricts any point to a snapshot range; `--every N` and `--light-every N` thin the
  snapshots goal 3, 5 and 6 read.
- A comparison's `--path` and `--vs` take roadrunner runs or ACCME result folders; at least one must be
  a roadrunner run (the particles are read through it).
- Work is spread over processes per snapshot. On jotun (64 cores, mostly idle) use 24 to 50 workers.
  Times on halo685x095 with 24 to 30 workers: soundness 1 to 2 min, stability 2 to 4 min, model 5 to
  6 min with `--dependence` or both flags, science and compare about 1 min each.

Outputs:

- A run's points write to `RUN/evaluation/`: `<point>.txt` (the report), `<point>.log` (the console),
  `plots/run/*.pdf` and `plots/g<id>/*.pdf` (a galaxy gets its own plots when it is the main host,
  reaches 1000 particles, or is named in an episode).
- A comparison writes to the first roadrunner run's `evaluation/<REF>_vs_<OTHERS>/` (a run's `/` is
  written `-`), or to `--out`.
- Runs are named by their folder and the one above (`vanilla/rank`, `tempered_05/rank`); ACCME results
  `accme-<folder>`.
- Each report has the verdict (points 1 and 2), the FAILs and FLAGs, a SCORES block (one number per
  line, the same names in every run, which is what `sweep.py` collects), and tables.

## Verdicts: points 1 and 2

- **FAIL**: the run is broken there. **FLAG**: something to look at. A point passes when it has no
  FAIL.
- Checks file **symptoms**; symptoms that share a galaxy and are within 100 Myr (or adjacent
  snapshots) are one **episode**, reported once with all its symptoms, as a FAIL if any of them is one,
  under one cause. One physical event (say, a satellite absorbed by its host: a label gap, a mass step
  and a transfer) gives one finding, not three.
- **Point 1** fails on anything that should never happen: an incomplete run, inconsistent files,
  responsibilities that do not sum to one, labels that disagree with the argmax, broken Gaussians, a
  data-backed Gaussian wider than 15 R_vir^2 or 15 v_max^2 (a runaway component). Trends in `run.log`
  and the pipeline's warnings are FLAGs.
- **Point 2** fails on what a user of the catalogue would be hurt by: a galaxy of >= 100 particles that
  loses all its bound stars for two or more snapshots while they stay inside the tree halo's R_vir; a
  spike of a major galaxy, or an unexplained step of the main host, that no exchange with another
  galaxy explains. Everything else (label gaps where a neighbour took the stars, the tree losing the
  stars, steps explained by star formation, mergers or pericentres, share spikes) is a FLAG.
- A FAIL is not automatically a roadrunner bug: read its episode. On v6 phase, point 1 passes (25
  flags) and point 2 has one FAIL: galaxy 1440 drops from 7.8e6 to 1.6e6 at snapshot 8 and comes back,
  with no other galaxy involved. End-of-tree ID swaps (a branch continuing under another Sub_tree_id)
  are a merger-tree issue and show up as gaps or steps; roadrunner cannot fix them.

Point 2 also measures, without verdicts: flicker (2.3: old stars that leave a galaxy and come back
within 300 Myr), permanent transfers (2.4, split by relation: stripping, capture, swap, merger,
unbound), how much newborn mass keeps its label for 150 Myr (2.5), and the label-mass noise of each
galaxy (2.6, a lower and an upper bound on the white noise of log10 M). On v6 phase: flicker 1% of the
old mass per snapshot, unrelated flicker 2e-6, label noise between 0.007 and 0.013 dex (the medians
over galaxies of the two bounds).

## Goal 3: reading model.py and deciding with it

### The model it tests

For a particle n and candidate galaxy k, the E-step is

    log r_nk = log N_k(x_n) + log pi_k + beta log alpha_nk - log Z_n

with N_k the galaxy's Gaussian in (x, v), pi_k its weight, alpha_nk the plausibility (the physics
prior: energy, phase, rank or kinematic model) and beta the tempering (1 unless the config says
otherwise). Where a particle has two candidates k and j, the decision is the log-odds

    Delta log r = D_G + D_alpha,   D_G = Delta(log N + log pi),   D_alpha = beta Delta log alpha

(D_G is computed from the written Gaussians; it differs from the E-step's by a constant per galaxy,
which cancels in everything goal 3 reports.)

Goal 3 asks: do the Gaussians describe the galaxies, how much does alpha decide, is alpha counted once
or several times, and are the responsibilities calibrated? It looks at contested particles (two or more
candidates) in resolved groups, every 10th snapshot (3.2b and 3.4 every 3rd), and gives no verdict.

3.2b and 3.5 need the log alpha the fit read, stored per pair since commit fc95ee0 (the v6
`tempered_05` runs have it; the v6 `vanilla` runs do not). Without it, 3.2b is skipped, 3.5 is absent,
and 3.4 uses D_alpha instead (the same up to a constant per pair).

### Reference values (halo685x095, v6)

| Score | vanilla rank | vanilla phase | t05 rank | t05 phase | t05 energy | t05 kinematic |
|---|---|---|---|---|---|---|
| 3.1 depth: median rho soft | 0.830 | 0.835 | 0.830 | 0.834 | 0.833 | 0.828 |
| 3.2 median rho(D_G, D_alpha) | 0.465 | 0.467 | 0.452 | 0.461 | 0.457 | 0.188 |
| 3.2 alpha share of the decision v | 0.008 | 0.252 | 0.004 | 0.119 | 0.118 | 0.037 |
| 3.2 autocorrelation of D_G, 150 Myr | 0.716 | 0.714 | 0.716 | 0.714 | 0.713 | 0.718 |
| 3.2 autocorrelation of D_alpha, 150 Myr | 0.279 | 0.495 | 0.272 | 0.494 | 0.484 | 0.031 |
| 3.2b gamma (alpha counted 1 + gamma times) | | | ~0 | ~0 | -0.018 | ~0 |
| 3.2b label-change share of contested mass | | | 0.0004 | 0.0048 | 0.0047 | 0.0036 |
| 3.3 overconfidence O | 0.082 | 0.089 | 0.082 | 0.085 | 0.085 | 0.082 |
| 3.3 persistence at r_max >= 0.999 | 0.929 | 0.929 | 0.929 | 0.930 | 0.930 | 0.930 |
| 3.4 R2_G | | 0.763 | 0.577 | 0.763 | 0.755 | 0.528 |
| 3.4 unique share u | | 0.035 | 0.0000 | 0.012 | 0.012 | 0.006 |
| 3.5 O at beta = 1 / 0.75 / 0.25 | | | 0.0822 / 0.0822 / 0.0822 | 0.0856 / 0.0851 / 0.0855 | 0.0854 / 0.0849 / 0.0853 | 0.0822 / 0.0822 / 0.0821 |

`t05` is `tempered_05` (beta = 0.5); empty cells were not measured (no stored log alpha, or no
`--dependence`). The 3.2b values of t05 rank, phase and kinematic come from the first version of 3.2b,
which weighted pairs by their star count instead of the stars in play; they were already ~0.

### 3.1 Do the Gaussians follow the galaxies?

**What:** within each galaxy, the Spearman correlation between log N_k and how deep its stars are
(-E), how slow (-u^2), and log alpha.

**How to read it:** near a galaxy's core both log N and -E are quadratic forms in (x, v), so a
Gaussian that describes the galaxy ranks its stars like the potential does: rho around 0.8 or more.
This is a sanity check that the Gaussians are doing their job; it is not a measure of double counting.
A low rho means one Gaussian is a poor density model of that galaxy.

**On v6:** 0.83 for galaxies of every mass below 1e9, in every model, but 0.30 for the main host (the
mass-weighted median is the host's): one Gaussian is a poor model of a large galaxy with a disc.

**Decides:** nothing on its own. A low rho says a galaxy's Gaussian describes it poorly; 3.4's u for
the same galaxies says whether alpha supplies what the Gaussian misses. For the energy model, log alpha
is a monotone function of E, so its row repeats the depth row.

### 3.2 How much does alpha decide, and is it noisy?

**What:** over contested pairs, rho(D_G, D_alpha), whether the two terms push the same way; v, alpha's
share of the variance of the decision, (Var D_alpha + Cov) / Var(D_G + D_alpha); and the
autocorrelation of each term over 150 Myr, for the same stars and the same pair.

**How to read it:**

- **rho > 0:** the Gaussians and alpha agree, as two sensible descriptions of the same galaxies should.
  It cannot show that alpha leaked into the Gaussians; 3.2b does that.
- **v near 0:** alpha barely moves decisions (rank: 0.008). **v of 0.1 to 0.3:** alpha is a real part of
  the decision (phase: 0.25 at beta = 1, 0.12 at beta = 0.5; v falls roughly in proportion to beta).
- **ac_alpha well below ac_G:** alpha's push changes faster than the Gaussians' from one snapshot to the
  next. Where v is large too, alpha adds label flicker.

**On v6:** the Gaussian term is the steadier one in every model (0.71 against 0.03 to 0.49); alpha
carries a quarter of the decisions' variance in vanilla phase and an eighth at beta = 0.5.

**Decides:** whether alpha is worth its noise. Large v together with low ac_alpha argues for a smaller
beta or a smoother plausibility model; check the result in point 2's flicker (2.3) of the two runs, and
in `compare.py`.

### 3.2b Is alpha counted once? (`--feedback`)

**What:** each resolved group is replayed from its written fit in two chains: A with alpha in every
E-step (the run as written) and B with alpha applied once, at the end. If the Gaussians absorbed alpha
during EM, chain A's own Gaussian log-odds move with Delta log alpha. The leak L = D_G(A) - D_G(B)
measures that, weighted by p(1 - p) so that only the stars in play count:

- **gamma:** the slope of L on Delta log alpha. A's decision is D_G(B) + (1 + gamma) Delta log alpha:
  alpha is counted 1 + gamma times.
- **sd(L):** how far the leak moves the log-odds, in nats. rho_L: its alignment with Delta log alpha.
- **label-change share:** contested mass whose argmax differs between A and B.
- **Trust checks:** coverage (share of contested mass replayed) near 1, the reproduction residual (the
  replayed E-step against the written log r) small (median 1e-4 or less), few chains not converged.

**How to read it:** gamma near 0 with a small sd(L) means no double counting: the Gaussians fit the
stars, not alpha. A positive gamma is double counting: alpha weighs 1 + gamma times what it should.
Local values (single pairs during an interaction) of either sign and a few nats are expected; the
medians are what matter.

**On v6 (tempered_05/energy):** gamma = -0.02, sd(L) = 0.27 nats, rho_L = -0.03, and 0.5% of the
contested mass changes label (0.05% of all stellar mass); coverage 1, residual 6e-5. No double counting.

**Decides:** whether alpha needs to move out of the EM loop. If gamma is clearly positive (say above 0.1,
alpha 10% overweighted) and the label-change share is not negligible, either temper with
beta = 1 / (1 + gamma) or apply alpha once after EM (chain B's design). Otherwise leave the fit as it
is.

### 3.3 Are the responsibilities calibrated?

**What:** contested old stars at s, binned by r_max; persistence is the mass share still labelled with
their argmax at the first snapshot >= 150 Myr later. The gap is mean r - persistence per bin.
O = sum over bins of mass x max(gap, 0) / total mass (overconfidence); U the same with -gap (apparent
underconfidence; inflated, because EM starts from the previous responsibilities). The lag curve repeats
the reading at 50 Myr to 1 Gyr.

**How to read it:** persistence is a proxy. A label can change for physical reasons (stripping, a
merger), and those lower persistence too, so O mixes miscalibration with real change. The lag curve
separates them: a drop already there at 50 Myr that stays flat is fast, reversible flicker
(miscalibration); a decline that keeps growing with lag is slow loss (mostly physics).

**On v6:** O is 0.082 to 0.089 in every model. 91% of the contested mass sits at r_max >= 0.999 and
gives three quarters of O: 6% of it changes label within 50 Myr, and persistence then falls slowly, to
0.93 at 150 Myr and 0.89 at 1 Gyr. Stars with r_max between 0.8 and 0.999 keep their label only 57 to
73% of the time, already at 50 Myr, and that changes little at longer lags: those responsibilities are
overconfident (they claim 85 to 99.6%).

**Decides:** whether soft responsibilities can be used as probabilities. On v6 they cannot, below
r_max = 0.999; hard labels are fine for most of the mass. 3.5 says whether beta can fix it.

### 3.4 How much of alpha do the Gaussians already contain? (`--dependence`)

**What:** for each contested pair with >= 500 stars, a cross-fitted gradient-boosting regression of
Delta log alpha on what the Gaussians see of a star (its Mahalanobis distances to the two galaxies):

- **R2_G:** the share of alpha's push the Gaussians already express;
- **R2_phys:** the same regression on the star's distances and speeds relative to the two halos (what
  alpha is built from). It checks the regressor's capacity and should be close to 1; if it is not,
  R2_G is not trustworthy;
- **u:** alpha's push that the Gaussians cannot express, as a share of the decision's variance.

**How to read it:** alpha is a function of (x, v), so the posterior uses x twice by construction; the
question is how much alpha adds. u near 0 means alpha's push can be written as a function of the
Gaussians' own view of the star: it reshapes the decision but brings no information the Gaussians do
not have. A larger u means alpha brings information the Gaussians lack.

**On v6:** R2_phys 0.99 in every run. R2_G 0.76 (phase, energy), 0.58 (rank), 0.53 (kinematic). u is
1 to 3.5% for phase and energy, mostly in pairs of two galaxies above 1e9 (the host and its largest
satellites: u 0.02 to 0.05; the table bins a pair by its smaller galaxy); 0.6% for kinematic; 0 for
rank.

**Decides:** together with 3.2's v, whether the choice of plausibility model matters. Small v and
small u mean alpha moves few decisions and adds little of its own, so the models should agree on most
stars in `compare.py`; where they do not is where the choice matters (on v6, expect the large galaxies
next to the host, where u is largest).

### 3.5 Would tempering help? (only with stored log alpha)

**What:** 3.3 recomputed with log r' = log r + (beta - beta_run) log alpha, renormalised, for
beta = 1, 0.75, 0.5, 0.25, with the fitted Gaussians kept; and the share of stars whose argmax changes.

**How to read it:** this is a preview: in a real run at another beta, EM would move the Gaussians too.
Look for a beta that lowers O without moving many labels. Confirm with a run at that beta, whose own 3.3
should match the preview (on v6, the beta = 1 preview from the beta = 0.5 phase run gives O = 0.0856,
against 0.0893 in the real beta = 1 run).

**On v6:** O does not move with beta (0.0849 to 0.0856 for phase and energy, 0.0822 for rank and
kinematic) and at most 0.9% of the labels change. The overconfidence comes from the Gaussian term (or
from real label changes counted as misses), not from alpha.

**Decides:** beta. On v6, beta has no leverage on calibration, so choose it on other grounds: 3.2's
flicker, point 2's flicker, and agreement in `compare.py` and `science.py`. If calibrated probabilities
are needed, the knob is elsewhere: a temperature on the whole log r fitted to the 3.3 table, or the
Gaussians themselves.

### A decision path

1. **Is the run usable?** Points 1 and 2 pass, or every FAIL episode is explained (tree issues, a
   merger). If not, fix the pipeline before reading anything else.
2. **Is alpha counted once?** 3.2b gamma near 0. If clearly positive, temper with
   beta = 1 / (1 + gamma) or move alpha out of EM.
3. **What does alpha do?** 3.2's v and 3.4's u: how much it decides and how much of that the Gaussians
   could not do alone. 3.2's autocorrelations: whether it does it steadily.
4. **Can the probabilities be trusted?** 3.3 with its lag curve; 3.5 for whether beta helps.
5. **Does the choice change the science?** `compare.py` (agreement, where they differ) and
   `science.py` (births, the host's ex-situ share, its progenitors) between runs that differ only in
   the model or beta. If they agree within the spread between runs, the choice does not matter for the
   science; take the simpler one.
6. **Tabulate:** `sweep.py` gives one row per run with its configuration and every score.

## Goals 5 and 6: comparing catalogues

- **5.1 births:** per birth galaxy, recall, purity and Jaccard J of its born mass between two
  catalogues. On v6 the largest disagreements with ACCME are stars born in satellites shortly before
  they merge, which roadrunner's birth tracker gives to the remnant, and stars of a stripped satellite
  that its boundness model does not keep bound; see `audits/birth-assignment-2026-10/` (local,
  untracked) for the investigation.
- **5.2 ex-situ:** per galaxy, the overlap of its ex-situ stars, with the disagreement split into
  membership (the star is in the galaxy in only one catalogue) and birth (in both, but born elsewhere in
  one). On v6, ACCME against rank: the host's ex-situ share is 0.222 against 0.199, and the gap is about
  births, not membership.
- **5.3:** the host's main progenitors' histories: peak mass, infall, disruption.
- **6.1 to 6.4:** agreement of hard labels (all, old, newborn, contested stars), per galaxy recall,
  purity and J, the largest mass movers with their relation, where disagreements sit (distance to the
  host, inside or outside its disc, distance to the star's own galaxy in its tidal radius or R_vir, the
  galaxy's kind and mass, the star's age), and galaxy properties (Mtot, rh, sigma, positions,
  velocities) with a 16-galaxy grid.

Neither catalogue is the truth: a difference is a place to look, and the breakdowns say where.

## Development

- Tests: `PYTHONPATH=src python -m pytest -q tests/unit/test_evaluate.py` (pure measurement functions
  and the replay against the pipeline's mixture).
- SCORE names are what `sweep.py` keys on: renaming one starts a new column.
- A change meant to keep the results is checked by running the old and the new code on the same runs
  (symlinked copies of the run folders, so the outputs stay apart) and comparing every `.txt` report
  byte for byte.

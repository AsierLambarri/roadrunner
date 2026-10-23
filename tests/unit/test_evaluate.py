"""Unit tests of the evaluation rules (evaluate/): alignment, run.log, warnings, spikes, mass steps, transfers."""
import numpy as np
import pandas as pd

from evaluate.load import ABSENT, on
from evaluate.soundness import read_runlog, read_warnings, spikes
from evaluate.stability import relation, steps


def test_on_reads_sorted_ids_and_marks_absent():
    ids = np.array([2, 5, 9])
    assert on(np.array([1, 2, 9, 10]), ids, np.array([20, 50, 90])).tolist() == [ABSENT, 20, 90, ABSENT]


def test_read_runlog_keeps_last_row_of_a_resumed_snapshot(tmp_path):
    head = "RUNTIME SNAP REDSHIFT LOAD PROCESS REDUCTION BOUND UNASSIGNED AVG_CONF"
    text = "\n".join(["-halo_model: nfw", head, "-" * 40,
                      "00:01:00.000 0 5.0 1.0 2.0 -- 10 3 --",
                      "00:02:00.000 1 4.0 1.0 2.0 -- 20 4 0.9",
                      head, "-" * 40,
                      "00:03:00.000 1 4.0 1.0 2.0 -- 20 5 0.95",
                      "", "--- Run complete ---"])
    (tmp_path / "run.log").write_text(text)
    log = read_runlog(tmp_path / "run.log")
    assert log.index.tolist() == [0, 1]
    assert log.at[1, "UNASSIGNED"] == 5 and np.isnan(log.at[0, "AVG_CONF"])


def test_read_warnings_types_and_snapshots(tmp_path):
    (tmp_path / "warnings.log").write_text(
        "[2026-10-12 17:22:28] Snapshot 6: UserWarning: Numerical issue — retrying with double. (/x/gmm.py:651)\n---\n"
        "[2026-10-12 17:30:00] Snapshot 7: NegativeDistributionWarning: Eddington's f was negative and clipped to 0 "
        "in 2 inversions: at most 3.0e-03 (median 1.0e-04) of a density's mass. (/x/distribution.py:863)\n---\n")
    w = read_warnings(tmp_path / "warnings.log")
    assert w["snap"].tolist() == [6, 7]
    assert w["type"].iloc[1].startswith("NegativeDistributionWarning: Eddington's f was negative and clipped to # in #")


def test_spikes_flags_one_outlier_only():
    x = pd.Series(np.full(30, 0.99))
    x[15] = 0.5
    assert np.flatnonzero(spikes(x).to_numpy()).tolist() == [15]


def test_steps_tell_spikes_from_persistent_steps():
    M = pd.DataFrame([[1.0, 1.0, 10.0, 1.1, 1.0, 1.0],      # spike at 2, its return leg at 3 is not a step
                      [1.0, 1.0, 10.0, 10.0, 10.0, 10.0],   # persistent step at 2
                      [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]])      # no step above x2
    spike, persist = steps(M, 0.1 * np.arange(6))           # 100 Myr apart: the return window holds 3 snapshots
    assert np.argwhere(spike).tolist() == [[0, 2]]
    assert np.argwhere(persist).tolist() == [[1, 2]]
    spike, persist = steps(M, 0.2 * np.arange(6))           # 200 Myr apart: only the next one; row 0 still returns at once
    assert np.argwhere(spike).tolist() == [[0, 2]]


def test_relation_classes():
    hosts = {7: {1}, 8: {1, 7}}
    assert relation(7, 1, hosts, ends=set()) == "stripping"
    assert relation(1, 8, hosts, ends=set()) == "capture"
    assert relation(7, 8, hosts, ends=set()) == "capture"        # 7 hosts 8
    assert relation(8, 9, hosts, ends=set()) == "swap"
    assert relation(8, -1, hosts, ends=set()) == "unbound"
    assert relation(-1, 8, hosts, ends=set()) == "rebound"
    assert relation(8, 1, hosts, ends={8}) == "merger"


def test_log_gauss_matches_scipy():
    from scipy.stats import multivariate_normal
    from evaluate.model import log_gauss
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    cov, mean, x = A @ A.T + 6 * np.eye(6), rng.normal(size=6), rng.normal(size=(50, 6))
    np.testing.assert_allclose(log_gauss(x, mean, cov), multivariate_normal(mean, cov).logpdf(x), rtol=1e-10)


def test_weighted_spearman_with_equal_weights_is_spearman():
    from scipy.stats import spearmanr
    from evaluate.model import wspearman
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=200), rng.normal(size=200)
    np.testing.assert_allclose(wspearman(a, b, np.ones(200)), spearmanr(a, b)[0], rtol=1e-12)


def test_contested_pairs_recover_alpha_up_to_a_constant_per_pair():
    """D_alpha = Delta log alpha + (C_k - C_j): correlation with the stored Delta log alpha is 1."""
    from evaluate.model import contested_pairs, pair_table
    rng = np.random.default_rng(3)
    n = 60
    logN = rng.normal(size=(n, 2))
    log_alpha = 0.5 * logN + rng.normal(size=(n, 2))
    C = np.array([1.0, -2.0])                               # per-galaxy E-step constants
    w = logN + C + log_alpha
    log_resp = w - np.logaddexp(w[:, 0], w[:, 1])[:, None]
    p = {"row": np.repeat(np.arange(n), 2).astype(float), "galaxy": np.tile([5.0, 3.0], n),
         "log_resp": log_resp.ravel(), "log_alpha": log_alpha.ravel()}
    ids = 100 + np.arange(n)
    (k, j, dG, dA, dLa, pid, a, b), counts = contested_pairs(p, logN.ravel(), np.ones(2 * n, bool), ids)
    assert counts == (n, 0) and set(k) == {3} and set(j) == {5}
    np.testing.assert_array_equal(pid, ids)
    np.testing.assert_array_equal(p["galaxy"][a], k)
    np.testing.assert_array_equal(p["galaxy"][b], j)
    np.testing.assert_allclose(dA - dLa, C[1] - C[0])       # galaxy 3 minus galaxy 5
    t = pair_table(k, j, dG, dA, dLa, s=0, t=0.0)
    np.testing.assert_allclose(t["rho stored"], 1.0)
    assert t["n"].iloc[0] == n


def test_best_is_the_renormalised_argmax():
    from evaluate.model import best
    rows = np.array([1, 0, 1, 0, 1])
    g = np.array([7, 7, 8, 8, 9])
    x = np.log(np.array([0.2, 0.9, 0.5, 0.1, 0.3]))
    r, k, top = best(rows, g, x)
    assert r.tolist() == [0, 1] and k.tolist() == [7, 8]
    np.testing.assert_allclose(top, [0.9, 0.5])


def test_sweep_reads_back_report_scores(tmp_path):
    from evaluate.report import Report
    from evaluate.sweep import read_scores
    rep = Report("t", "run", verdict=False)
    rep.score("a long score name: with words", 0.25)
    rep.score("count", 7)
    rep.table("table", "  x 1")
    rep.write(tmp_path / "r.txt")
    assert read_scores(tmp_path / "r.txt") == {"a long score name: with words": 0.25, "count": 7}


def test_disc_tag_keeps_the_disc_and_drops_the_halo():
    from evaluate.compare import disc_tag
    rng = np.random.default_rng(4)
    n = 4000
    R, phi = rng.uniform(0.1, 5.0, n), rng.uniform(0, 2 * np.pi, n)
    disc = np.c_[R * np.cos(phi), R * np.sin(phi), rng.normal(0, 0.1, n),
                 -100 * np.sin(phi), 100 * np.cos(phi), np.zeros(n)]          # rotating about z
    halo = np.c_[rng.normal(0, 3.0, (500, 3)), rng.normal(0, 50.0, (500, 3))]
    X = np.vstack([disc, halo])
    tag = disc_tag(X, np.ones(len(X)), np.r_[np.ones(n, bool), np.zeros(500, bool)], np.zeros(len(X), bool),
                   np.zeros(3), np.zeros(3), rh=2.0)
    assert tag[:n].mean() > 0.8 and tag[n:].mean() < 0.3


def test_report_lists_every_item_with_minus_one(monkeypatch):
    import evaluate.report as report
    rep = report.Report("t", "run")
    for i in range(15):
        rep.flag("check", f"item {i}")
    assert report.LISTED == -1 and "item 14" in rep.text() and "more" not in rep.text()
    monkeypatch.setattr(report, "LISTED", 10)
    assert "item 9" in rep.text() and "item 10" not in rep.text() and "... and 5 more" in rep.text()


def test_has_cycle():
    from evaluate.soundness import has_cycle
    assert not has_cycle({1: {2, 3}, 2: {3}, 4: set()})
    assert has_cycle({1: {2}, 2: {3}, 3: {1}})
    assert has_cycle({1: {2}, 2: {1}, 5: {6}})


def test_halo_ratios_read_the_position_and_velocity_blocks():
    from evaluate.soundness import halo_ratios
    cov = np.diag([4.0, 1.0, 1.0, 900.0, 100.0, 100.0])
    np.testing.assert_allclose(halo_ratios(cov, np.zeros(6), r_vir=2.0, v_max=10.0), (1.0, 9.0))
    np.testing.assert_allclose(halo_ratios(np.full(6, 4.0), np.zeros(6), r_vir=2.0, v_max=1.0), (1.0, 4.0))


def test_ahead_is_a_time_window_of_at_least_one_snapshot():
    from evaluate.stability import ahead
    t = np.array([0.0, 0.05, 0.1, 0.4, 0.45, 1.0])
    assert ahead(t, 0, 0.3).tolist() == [1, 2]
    assert ahead(t, 3, 0.3).tolist() == [4]
    assert ahead(t, 4, 0.3).tolist() == [5]                 # nothing within 300 Myr: the next one
    assert ahead(t, 5, 0.3).tolist() == []


def test_gap_verdict():
    from evaluate.stability import gap_verdict
    assert gap_verdict(500, 0, None, 0.8) == ("FLAG", "absorbed by a neighbour")
    assert gap_verdict(500, 0, None, 0.2) == ("FLAG", "label gap")
    assert gap_verdict(500, 2, "out", 0.0) == ("FLAG", "the tree lost the stars")
    assert gap_verdict(500, 2, "in", 0.0) == ("FAIL", "bound gap with members in place")
    assert gap_verdict(500, 1, "in", 0.0) == ("FLAG", "bound gap with members in place")
    assert gap_verdict(50, 3, "in", 0.0) == ("FLAG", "bound gap with members in place")
    assert gap_verdict(500, 2, None, 0.0) == ("FLAG", "bound gap with members not located")


def test_label_noise_recovers_sigma_and_drops_merger_triples():
    from evaluate.stability import label_noise
    rng = np.random.default_rng(6)
    n, dt, sigma = 200, 0.1, 0.05
    t = dt * np.arange(n)
    trend = np.linspace(6.0, 6.3, n)                           # smooth growth, 0.3 dex over the run
    y = trend + rng.normal(0.0, sigma, n)                      # white label noise on top
    Mres = pd.DataFrame([10.0 ** y], index=[1], columns=np.arange(n))
    no_merge = pd.DataFrame({"into": [], "t": []})
    noise, pooled = label_noise(Mres, t, no_merge)
    assert abs(noise.at[1, "sigma_noise"] - sigma) / sigma < 0.2
    assert -0.6 < noise.at[1, "rho_1"] < -0.4
    n_pairs = int(noise.at[1, "n_pairs"])

    merged = pd.DataFrame({"into": [1], "t": [t[100]]})       # a merger mid-run drops nearby triples
    noise_m, _ = label_noise(Mres, t, merged)
    assert noise_m.at[1, "n_pairs"] < n_pairs


def test_episodes_link_by_galaxy_check_and_time():
    from evaluate.report import Report
    rep = Report("t", "run", point="2", causes=["a cause"])
    rep.symptom("gap", "FLAG", "x", s0=10, s1=12, t0=1.0, t1=1.2, galaxies={1, 2}, cause="a cause")
    rep.symptom("spike", "FAIL", "y", s0=13, s1=14, t0=1.25, t1=1.4, galaxies={2})   # galaxy 2, 50 Myr later
    rep.symptom("spike", "FLAG", "z", s0=30, s1=31, t0=2.0, t1=2.1, galaxies={2})    # 600 Myr later: alone
    rep.symptom("1.3 X", "FLAG", "u", s0=40, t0=3.0)                                  # no galaxy ...
    rep.symptom("1.3 X", "FLAG", "v", s0=41, t0=3.2)                                  # ... same check, adjacent snapshot
    rep.symptom("1.3 Y", "FLAG", "w", s0=41, t0=3.2)                                  # another check: alone
    rep.symptom("1.1 run.log", "FAIL", "no end")                                      # not in time: alone
    rep.episodes()
    assert {k: len(v) for k, v in rep.fails.items()} == {"2 episode: spike": 1, "2 episode: 1.1 run.log": 1}
    assert {k: len(v) for k, v in rep.flags.items()} == {"2 episode: spike": 1, "2 episode: 1.3 X": 1,
                                                         "2 episode: 1.3 Y": 1}
    assert rep.scores["2 episodes: a cause"] == 0 and rep.scores["2 episodes: spike"] == 2 and not rep.symptoms


def test_pool_keeps_order_in_processes_and_here():
    from evaluate.load import pool
    assert pool(abs, [-3, 1, -2], 2, dict) == [3, 1, 2]
    assert pool(abs, [-3, 1, -2], 1, dict) == [3, 1, 2]


def test_ages_from_presence():
    from types import SimpleNamespace
    from evaluate.load import newborn, seen_before
    ids = {0: np.array([5, 1]), 1: np.array([1, 5, 9]), 2: np.array([9, 1, 5, 7])}
    run = SimpleNamespace(snaps=np.array([0, 1, 2]), time={0: 0.0, 1: 0.6, 2: 1.2}, ids=ids.get)
    q = np.array([7, 9, 5, 1])
    assert newborn(run, 2, q).tolist() == [True, False, False, False]
    assert seen_before(run, 2, q, 1.0).tolist() == [False, False, True, True]     # present at snapshot 0
    assert newborn(run, 0, q).all() and not seen_before(run, 0, q, 1.0).any()


def test_resolved_groups_need_ten_particles_per_galaxy():
    from evaluate.model import components
    rows = np.r_[np.arange(5), np.arange(5), [10, 11, 12]]           # 1, 2 share 5 particles; 3 alone
    gal = np.r_[np.full(5, 1), np.full(5, 2), np.full(3, 3)]
    comp, resolved = components({"row": rows, "galaxy": gal})
    assert resolved.tolist() == [False] * 10 + [True] * 3
    assert comp[:5].tolist() == comp[5:10].tolist() and comp[0] != comp[10]
    rows = np.r_[np.arange(25), np.arange(25)]                         # 25 >= 10 x 2: resolved
    assert components({"row": rows, "galaxy": np.r_[np.full(25, 1), np.full(25, 2)]})[1].all()


def test_alpha_share_of_the_decision_adds_up():
    from evaluate.model import pair_table
    rng = np.random.default_rng(5)
    dG, dA = rng.normal(size=100), rng.normal(size=100)
    t = pair_table(np.zeros(100), np.ones(100), dG, dA, np.full(100, np.nan), s=0, t=0.0)
    vG, vA, cov = np.var(dG), np.var(dA), np.cov(dG, dA, bias=True)[0, 1]
    np.testing.assert_allclose(t["v"].iloc[0], (vA + cov) / np.var(dG + dA))


def test_autocorrelation_ignores_per_snapshot_constants_and_recovers_rho():
    """ac_alpha = Pearson(D_alpha(s), D_alpha(s')) is invariant to C_k - C_j (a constant per snapshot,
    added to every particle of a pair), and recovers a known correlation."""
    from evaluate.model import autocorr_table
    rng = np.random.default_rng(6)
    n = 2000
    k, j, pid = np.zeros(n), np.ones(n), np.arange(n)
    dG1 = rng.normal(size=n)
    dG2 = dG1 + rng.normal(scale=0.01, size=n)            # near-perfect: ac_G ~ 1
    dA1 = rng.normal(size=n)
    rho = 0.6
    dA2 = rho * dA1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    t0 = autocorr_table(k, j, pid, dG1, dA1, k, j, pid, dG2, dA2, s=0, t=0.0)
    shifted = autocorr_table(k, j, pid, dG1, dA1 + 3.0, k, j, pid, dG2, dA2 - 7.0, s=0, t=0.0)
    assert t0["n"].iloc[0] == n and t0["ac_G"].iloc[0] > 0.99
    np.testing.assert_allclose(t0["ac_alpha"].iloc[0], rho, atol=0.03)
    np.testing.assert_allclose(t0["ac_alpha"], shifted["ac_alpha"])


def test_cross_fit_r2_separates_a_real_signal_from_noise():
    from evaluate.model import cross_fit_r2
    rng = np.random.default_rng(7)
    n = 20_000                                            # large enough that noise's out-of-sample R^2
    F = rng.normal(size=(n, 2))                           # settles near 0 rather than overfitting negative
    y_signal = np.sin(F[:, 0]) + F[:, 1] ** 2 + rng.normal(scale=0.05, size=n)
    r2_signal, _ = cross_fit_r2(F, y_signal, seed=0)
    y_noise = rng.normal(size=n)
    r2_noise, _ = cross_fit_r2(F, y_noise, seed=0)
    assert r2_signal > 0.9
    assert abs(r2_noise) < 0.05


def test_dependence_table_recovers_one_pair_at_full_size():
    from evaluate.model import MIN_DEP, dependence_table
    rng = np.random.default_rng(8)
    n = 5 * MIN_DEP
    k, j = np.zeros(n), np.ones(n)
    d2k, d2j = rng.normal(size=n) ** 2, rng.normal(size=n) ** 2
    Y = d2k - d2j + rng.normal(scale=0.01, size=n)
    phys = rng.normal(size=(n, 4))
    D = Y + rng.normal(size=n)
    k[:100] = 5                                             # a second pair, under MIN_DEP: only counted
    t, few = dependence_table(k, j, Y, d2k, d2j, phys, D, s=0, t=0.0, seed=1)
    assert t["n"].iloc[0] == n - 100 and few == (1, 100)
    assert t["r2_G"].iloc[0] > 0.9


def test_corrected_r2_removes_the_small_sample_bias():
    from evaluate.model import corrected_r2, cross_fit_r2
    rng = np.random.default_rng(9)
    n = 800
    F = np.abs(rng.normal(size=(n, 2))) * [3.0, 2.0]
    f = np.log1p(F[:, 0]) - 0.6 * np.log1p(F[:, 1]) + 0.2 * np.log1p(F[:, 0] * F[:, 1])
    Y = f + f.std() * rng.normal(size=n)                    # true R^2 = 0.5
    raw, _ = cross_fit_r2(F, Y, seed=0)
    r2, se = corrected_r2(F, Y, seed=0)
    assert raw < 0.47 and abs(r2 - 0.5) < 0.07 and 0 < se < 0.06


def test_gamma_table_recovers_the_feedback_slope():
    """gamma = Cov(L, Delta log alpha) / Var(Delta log alpha), recovered from synthetic leaks
    L = gamma * Delta log alpha + noise, when Var(Delta log alpha) is not tiny next to Var(D_A)."""
    from evaluate.model import MIN_PAIR, gamma_table
    rng = np.random.default_rng(11)
    n = 2000
    k, j = np.zeros(n), np.ones(n)
    dLa = rng.normal(size=n)
    DA = rng.normal(size=n)                                 # unrelated to dLa: no conditioning issue
    gamma_true = 0.4
    L = gamma_true * dLa + rng.normal(scale=0.1, size=n)
    t = gamma_table(k, j, L, dLa, DA, s=0, t=0.0)
    assert t["n"].iloc[0] == n
    assert abs(t["gamma"].iloc[0] - gamma_true) < 0.03
    assert t["rho_L"].iloc[0] ** 2 > 0.9
    w = 1.0 / (4.0 * np.cosh(DA / 2.0) ** 2)                # p (1 - p): the stars in play
    sd = np.sqrt(np.average((L - np.average(L, weights=w)) ** 2, weights=w))
    assert abs(t["sd_L"].iloc[0] - sd) < 1e-12 and abs(t["n_play"].iloc[0] - w.sum()) < 1e-9
    assert gamma_table(np.zeros(MIN_PAIR - 1), np.ones(MIN_PAIR - 1), rng.normal(size=MIN_PAIR - 1),
                        rng.normal(size=MIN_PAIR - 1), rng.normal(size=MIN_PAIR - 1), s=0, t=0.0).empty


def test_gamma_is_nan_when_delta_log_alpha_has_tiny_variance():
    from evaluate.model import GAMMA_MIN_VAR, gamma_table
    rng = np.random.default_rng(12)
    n = 2000
    k, j = np.zeros(n), np.ones(n)
    DA = rng.normal(size=n)
    dLa = rng.normal(scale=np.sqrt(0.001 * np.var(DA)), size=n)     # well below GAMMA_MIN_VAR x Var(D_A)
    L = 0.4 * dLa + rng.normal(scale=0.1, size=n)
    t = gamma_table(k, j, L, dLa, DA, s=0, t=0.0)
    assert np.isnan(t["gamma"].iloc[0]) and GAMMA_MIN_VAR == 0.01


def test_bgmm_estep_mstep_match_the_library():
    """3.2b's replay machinery (evaluate/bgmm.py) must mirror WeightedBayesianGaussianMixture exactly:
    same E-step log r (up to the per-particle constant it never needs) and same M-step posterior,
    given the same priors and starting state, with reg_covar = 0 in both."""
    from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture, _compute_precision_cholesky
    from evaluate import bgmm

    rng = np.random.default_rng(42)
    D, K, n = 6, 3, 300
    galaxies = np.arange(K, dtype=np.int64)
    tree_pos = rng.normal(scale=50.0, size=(K, D))
    means0 = tree_pos + rng.normal(scale=2.0, size=(K, D))
    covs0 = np.stack([np.diag(rng.uniform(1.0, 4.0, D)) for _ in range(K)])

    truth = rng.integers(0, K, n)
    X = tree_pos[truth] + rng.normal(scale=5.0, size=(n, D))
    rows, gal = [], []
    for i in range(n):
        cand = rng.choice(K, size=int(rng.integers(2, K + 1)), replace=False)
        rows += [i] * cand.size
        gal += cand.tolist()
    rows, gal = np.array(rows, dtype=np.int64), np.array(gal, dtype=np.int64)
    log_alpha = rng.normal(size=rows.size)
    dense_log_alpha = np.full((n, K), -np.inf)
    dense_log_alpha[rows, gal] = log_alpha

    a0, beta0 = rng.uniform(0.5, 2.0, K), rng.uniform(0.5, 2.0, K)
    nu0_scalar = float(D + 4)
    W0inv = np.stack([np.diag(rng.uniform(1.0, 3.0, D)) for _ in range(K)])
    prior = bgmm.Prior(a0=a0, beta0=beta0, m0=tree_pos, nu0=np.full(K, nu0_scalar), W0inv=W0inv)
    N0 = rng.uniform(20.0, 80.0, K)
    post = bgmm.Posterior(m=means0, Sigma=covs0, N=N0, a=a0 + N0, beta=beta0 + N0, nu=nu0_scalar + N0)

    g, log_r = bgmm.estep(X, rows, gal, galaxies, log_alpha, post, with_alpha=True)

    ref = WeightedBayesianGaussianMixture(n_components=K, cov_type="full", reg_covar=0.0)
    ref.means_, ref.covariances_ = means0.copy(), covs0.copy()
    ref.weight_concentration_, ref.mean_precision_, ref.degrees_of_freedom_ = post.a.copy(), post.beta.copy(), post.nu.copy()
    ref.precisions_cholesky_ = _compute_precision_cholesky(ref.covariances_, "full")
    ref_log_resp, _ = ref._e_step(X, dense_log_alpha)
    np.testing.assert_allclose(log_r, ref_log_resp[rows, gal], atol=1e-8)

    ref.weight_concentration_prior_, ref.mean_precision_prior_ = a0.copy(), beta0.copy()
    ref.mean_prior_, ref.degrees_of_freedom_prior_, ref.covariance_prior_ = tree_pos.copy(), nu0_scalar, W0inv.copy()
    ref._m_step(X, np.exp(ref_log_resp), np.ones(n))

    new_post = bgmm.mstep(X, rows, gal, galaxies, np.exp(log_r), prior, reg_covar=0.0)
    np.testing.assert_allclose(new_post.N, ref.weight_concentration_ - a0, atol=1e-8)
    np.testing.assert_allclose(new_post.m, ref.means_, atol=1e-8)
    np.testing.assert_allclose(new_post.Sigma, ref.covariances_, atol=1e-8)


def test_bgmm_replay_converges_and_recovers_the_written_fit():
    """replay from theta_A = the written fit must converge (iterations < max_iter) and reproduce,
    the written log r via ``reproduce``, with with_alpha=True, nearly exactly at iteration 0."""
    from evaluate import bgmm

    rng = np.random.default_rng(7)
    D, K, n = 6, 3, 200
    galaxies = np.arange(K, dtype=np.int64)
    tree_pos = rng.normal(scale=50.0, size=(K, D))
    truth = rng.integers(0, K, n)
    X = tree_pos[truth] + rng.normal(scale=5.0, size=(n, D))

    rows, gal = [], []
    for i in range(n):
        cand = rng.choice(K, size=int(rng.integers(1, K + 1)), replace=False)
        rows += [i] * cand.size
        gal += cand.tolist()
    rows, gal = np.array(rows, dtype=np.int64), np.array(gal, dtype=np.int64)
    log_alpha = rng.normal(scale=0.3, size=rows.size)

    gauss, gauss_prev, tree_xyzv, tree_mass, tree_mass_prev = {}, {}, {}, {}, {}
    for k in range(K):
        sel = gal == k
        mean = X[rows[sel]].mean(axis=0)
        cov = np.cov(X[rows[sel]].T) + np.eye(D)
        gauss[k] = (mean, cov, 1.0 / K, float(sel.sum()))
        tree_xyzv[k] = tree_pos[k]
        tree_mass[k] = tree_mass_prev[k] = 1e10

    prior, theta_A, no_history = bgmm.state(gal, galaxies, gauss, gauss_prev, tree_xyzv, tree_mass, tree_mass_prev)
    assert no_history.all()                                  # no previous fit was supplied

    delta, delta_r = bgmm.reproduce(X, rows, gal, galaxies, log_alpha, theta_A, np.zeros(rows.size))
    assert np.isfinite(delta).all() and np.isfinite(delta_r).all()

    post, first_post, iterations, c_rows, c_gal, log_r, g = bgmm.replay(
        X, rows, gal, galaxies, log_alpha, prior, theta_A, with_alpha=True, max_iter=300)
    assert 0 < iterations < 300
    assert np.isfinite(log_r).all() and np.isfinite(g).all()
    assert post.N.sum() <= gal.size + 1e-6


def test_exsitu_overlap_recall_purity_J_and_disagreement_split():
    from evaluate.science import exsitu_overlap
    # 8 particles; h = 1. memA/memB: hard label == h. bidA/bidB: birth galaxy (-1: unknown).
    memA = np.array([True, True, True, True, False, False, True, True])
    memB = np.array([True, True, True, False, True, False, True, True])
    bidA = np.array([2, -1, 1, 2, 2, 1, 1, 3])
    bidB = np.array([2, -1, 1, 2, 2, 1, 1, 3])
    m = np.ones(8)
    out = exsitu_overlap(memA, memB, bidA, bidB, m, 1)
    # X_A(1) = {0, 3, 7} (born in 2, 2, 3; particle 2 and 6 are born in 1 = h, excluded; 1 is unknown)
    # X_B(1) = {0, 4, 7} (particle 4 is a member of B only, born in 2)
    assert out["mA"] == 3.0 and out["mB"] == 3.0 and out["mBoth"] == 2.0   # {0, 7} shared
    assert np.isclose(out["recall"], 2 / 3) and np.isclose(out["purity"], 2 / 3)
    assert np.isclose(out["J"], 2 / 4)
    # symmetric difference {3, 4}: both are membership disagreements (3 only in A, 4 only in B); as a
    # share of m(X_A | X_B) = {0, 3, 4, 7} (mass 4), that is 2 / 4
    assert np.isclose(out["membership_share"], 0.5) and out["birth_share"] == 0.0


def test_exsitu_overlap_splits_off_birth_disagreement():
    # same host membership in both catalogues, but particle 1's ex-situ status disagrees: known birth
    # in A (galaxy 2, so ex-situ), in-situ (born in h) in B.
    from evaluate.science import exsitu_overlap
    memA = memB = np.array([True, True, True])
    bidA = np.array([2, 2, -1])
    bidB = np.array([2, 1, -1])      # particle 1: in-situ in B (birth == h)
    m = np.ones(3)
    out = exsitu_overlap(memA, memB, bidA, bidB, m, 1)
    assert out["mA"] == 2.0 and out["mB"] == 1.0 and out["mBoth"] == 1.0   # only particle 0 agrees
    # symmetric difference is {1} alone (particle 0 shared, particle 2 in neither): as a share of
    # m(X_A | X_B) = {0, 1} (mass 2), the disagreement is entirely a birth mismatch
    assert out["membership_share"] == 0.0 and np.isclose(out["birth_share"], 0.5)


def test_birth_overlap_recall_purity_J_and_main_destination():
    from evaluate.science import birth_overlap
    # 10 stars born in galaxy 5 (mass 2e6 each); 8 keep that birth galaxy in B, 2 are attributed to 7
    bidA = np.full(10, 5)
    bidB = bidA.copy()
    bidB[8:] = 7
    m = np.full(10, 2e6)
    BT = birth_overlap(bidA, bidB, m)
    assert set(BT.index) == {5, 7}                            # both cross the 1e6 born-mass threshold
    assert BT.loc[5, "mA"] == 2e7 and BT.loc[5, "mB"] == 1.6e7 and BT.loc[5, "mBoth"] == 1.6e7
    assert np.isclose(BT.loc[5, "recall"], 0.8) and np.isclose(BT.loc[5, "purity"], 1.0)
    assert np.isclose(BT.loc[5, "J"], 0.8)
    assert BT.loc[5, "dest"] == 7 and np.isclose(BT.loc[5, "dest_share"], 0.2)
    assert BT.loc[7, "mA"] == 0.0 and BT.loc[7, "recall"] != BT.loc[7, "recall"]  # NaN: no mass born there in A

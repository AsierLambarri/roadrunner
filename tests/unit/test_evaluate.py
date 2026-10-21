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
    (k, j, dG, dA, dLa), counts = contested_pairs(p, logN.ravel(), np.ones(2 * n, bool))
    assert counts == (n, 0) and set(k) == {3} and set(j) == {5}
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
    from evaluate.model import resolved
    rows = np.r_[np.arange(5), np.arange(5), [10, 11, 12]]           # 1, 2 share 5 particles; 3 alone
    gal = np.r_[np.full(5, 1), np.full(5, 2), np.full(3, 3)]
    assert resolved({"row": rows, "galaxy": gal}).tolist() == [False] * 10 + [True] * 3
    rows = np.r_[np.arange(25), np.arange(25)]                         # 25 >= 10 x 2: resolved
    assert resolved({"row": rows, "galaxy": np.r_[np.full(25, 1), np.full(25, 2)]}).all()


def test_alpha_share_of_the_decision_adds_up():
    from evaluate.model import pair_table
    rng = np.random.default_rng(5)
    dG, dA = rng.normal(size=100), rng.normal(size=100)
    t = pair_table(np.zeros(100), np.ones(100), dG, dA, np.full(100, np.nan), s=0, t=0.0)
    vG, vA, cov = np.var(dG), np.var(dA), np.cov(dG, dA, bias=True)[0, 1]
    np.testing.assert_allclose(t["v"].iloc[0], (vA + cov) / np.var(dG + dA))

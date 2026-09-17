"""Test that our WeightedBayesianGaussianMixture matches sklearn's.

Generates synthetic 2D data from known Gaussians and compares:
1. Adjusted Rand index of predicted labels > 0.95
2. Component means within 2σ of true values
3. Per-cluster mean_precision_prior demonstration
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import BayesianGaussianMixture as SKLearnBGMM
from sklearn.metrics import adjusted_rand_score

from roadrunner._defaults import precision
from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture

rng = np.random.RandomState(42)

# ── Generate synthetic data: 3 Gaussians ──────────────────────────────
means_true = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, -3.0]])
covs_true = np.array([
    [[0.5, 0.0], [0.0, 0.5]],
    [[0.8, 0.3], [0.3, 0.8]],
    [[0.4, -0.2], [-0.2, 0.6]],
])
weights_true = np.array([0.40, 0.35, 0.25])
n_per = (300 * weights_true).astype(int)

X = np.vstack([
    rng.multivariate_normal(m, c, size=n)
    for m, c, n in zip(means_true, covs_true, n_per)
])
rng.shuffle(X)

# ── 1. Labels agree between sklearn and roadrunner ────────────────────
print("=== Test 1: Label agreement ===")
sk = SKLearnBGMM(n_components=3, max_iter=200, tol=1e-3,
                  random_state=42).fit(X)
with precision(math="double"):
    rr = WeightedBayesianGaussianMixture(n_components=3, max_iter=200, tol=1e-3,
                                          init_params="kmeans++",
                                          random_state=42).fit(X)

ari = adjusted_rand_score(sk.predict(X), rr.predict(X))
print(f"  Adjusted Rand Index: {ari:.4f}")
assert ari > 0.95, f"ARI too low: {ari}"
print("  PASS")

# ── 2. Means within tolerance ─────────────────────────────────────────
print("\n=== Test 2: Mean accuracy ===")
# Match component order via Hungarian on true-vs-rr cost
cost_rr = np.linalg.norm(means_true[:, None] - rr.means_[None, :], axis=-1)
_, col_rr = linear_sum_assignment(cost_rr)

for ti, ri in enumerate(col_rr):
    err = np.linalg.norm(rr.means_[ri] - means_true[ti])
    print(f"  Component {ti}: |Δμ| = {err:.4f}  "
          f"(true={means_true[ti]}, rr={rr.means_[ri].round(3)})")
    assert err < 1.0, f"Component {ti} mean error too large: {err}"
print("  PASS")

# Also check that rr and sk find similar means
cost_sk = np.linalg.norm(sk.means_[:, None] - rr.means_[None, :], axis=-1)
_, col_sk = linear_sum_assignment(cost_sk)
max_mean_diff = max(np.linalg.norm(sk.means_[i] - rr.means_[j])
                    for i, j in enumerate(col_sk))
print(f"  Max |μ_sk - μ_rr|: {max_mean_diff:.4f}")
assert max_mean_diff < 2.0, f"Mean difference too large: {max_mean_diff}"
print("  PASS")

# ── 3. Per-cluster prior demonstration ────────────────────────────────
print("\n=== Test 3: Per-cluster mean_precision_prior ===")
# Generate data where component 0 is very tight
X2 = np.vstack([
    rng.multivariate_normal([0, 0], [[0.05, 0], [0, 0.05]], size=200),   # tight
    rng.multivariate_normal([3, 0], [[0.6, 0], [0, 0.6]], size=200),
    rng.multivariate_normal([0, 3], [[0.6, 0], [0, 0.6]], size=200),
])
rng.shuffle(X2)

# Weak uniform prior — all clusters treated equally
beta_uniform = None
with precision(math="double"):
    rr_uniform = WeightedBayesianGaussianMixture(
        n_components=3, max_iter=200, tol=1e-3,
        mean_precision_prior=beta_uniform,
        init_params="kmeans++",
        random_state=42).fit(X2)

# Strong prior on component 0 (tight), weak on the rest
beta_strong = np.array([100.0, 1.0, 1.0], dtype=np.float64)
with precision(math="double"):
    rr_strong = WeightedBayesianGaussianMixture(
        n_components=3, max_iter=200, tol=1e-3,
        mean_precision_prior=beta_strong,
        init_params="kmeans++",
        random_state=42).fit(X2)

# Component 0: rr_strong should have tighter covariance (smaller det)
# than rr_uniform, because mean_precision_prior shrinks toward mean_prior
# Match components
cost_rr2 = np.linalg.norm(
    np.array([[0, 0], [3, 0], [0, 3]])[:, None] - rr_uniform.means_[None, :],
    axis=-1)
_, col_rr2 = linear_sum_assignment(cost_rr2)
comp0 = col_rr2[0]

det_uniform = np.linalg.det(rr_uniform.covariances_[comp0])
det_strong = np.linalg.det(rr_strong.covariances_[comp0])
print(f"  Det(cov) component 0 — uniform prior: {det_uniform:.6f}, "
      f"strong prior: {det_strong:.6f}")
assert det_strong < det_uniform, (
    f"Strong prior should tighten covariance, but "
    f"{det_strong} >= {det_uniform}")
print("  PASS — per-cluster prior tightens the expected component")

# ── 4. sklearn and rr produce similar means under uniform priors ──────
print("\n=== Test 4: Default-prior agreement ===")
sk2 = SKLearnBGMM(n_components=3, max_iter=200, tol=1e-3,
                   random_state=42).fit(X2)
ari2 = adjusted_rand_score(sk2.predict(X2), rr_uniform.predict(X2))
print(f"  ARI: {ari2:.4f}")
assert ari2 > 0.90, f"ARI too low with default priors: {ari2}"
print("  PASS")

print("\n✅ All tests passed.")

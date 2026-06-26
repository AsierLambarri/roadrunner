import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
try:
    import smplotlib
except:
    pass

from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture

rng = np.random.RandomState(42)

# ── Three clusters: one tight (c0), two diffuse (c1, c2) ─────────────
N = 300
X = np.vstack([
    rng.multivariate_normal([0, 0], [[0.05, 0], [0, 0.05]], size=N),     # tight
    rng.multivariate_normal([3, 0], [[0.6, 0], [0, 0.6]], size=N),
    rng.multivariate_normal([0, 3], [[0.6, 0], [0, 0.6]], size=N),
])
rng.shuffle(X)

# ── Fit with uniform prior vs strong prior on component 0 ────────────
print("Fitting BGMM with uniform prior...")
rr_uniform = WeightedBayesianGaussianMixture(
    n_components=3, max_iter=200, tol=1e-3,
    init_params="kmeans++", cast_dtype=np.float64,
    random_state=42,
).fit(X)

print("Fitting BGMM with per-cluster prior (beta=[100, 1, 1])...")
beta_strong = np.array([100.0, 1.0, 1.0], dtype=np.float64)
rr_strong = WeightedBayesianGaussianMixture(
    n_components=3, max_iter=200, tol=1e-3,
    mean_precision_prior=beta_strong,
    init_params="kmeans++", cast_dtype=np.float64,
    random_state=42,
).fit(X)

# ── Plot ─────────────────────────────────────────────────────────────
print("Plotting...")
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, figure=fig)

ax_uniform = fig.add_subplot(gs[0, 0])
ax_strong = fig.add_subplot(gs[0, 1])

# Use Hungarian matching so component ordering is consistent
from scipy.optimize import linear_sum_assignment
true_centers = np.array([[0, 0], [3, 0], [0, 3]])

for ax, model, beta, title in [
    (ax_uniform, rr_uniform, "None (uniform)", "BGMM — uniform prior"),
    (ax_strong, rr_strong, "[100, 1, 1]", "BGMM — per-cluster prior β=[100, 1, 1]"),
]:
    cost = np.linalg.norm(true_centers[:, None] - model.means_[None, :], axis=-1)
    _, col_idx = linear_sum_assignment(cost)
    colors = ["tab:orange", "tab:blue", "tab:green"]

    # Scatter with predicted labels
    labels = model.predict(X)
    ax.scatter(*X.T, s=8, c=[colors[l] for l in labels], alpha=0.3, edgecolor="none")
    ax.set_title(title, fontsize=13)

    # Ellipses
    for i in range(3):
        mean = model.means_[col_idx[i]]
        cov = model.covariances_[col_idx[i]]
        vals, vecs = np.linalg.eigh(cov[:2, :2])
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 3.5 * np.sqrt(vals) * 2
        ell = Ellipse(xy=mean[:2], width=w, height=h, angle=angle,
                       alpha=0.4, facecolor=colors[i], edgecolor="k")
        ax.add_patch(ell)
        ax.plot(*mean[:2], "kx", ms=10, mew=2)

    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-1.5, 4.5)
    ax.set_aspect("equal")

    # Report determinant
    det0 = np.linalg.det(model.covariances_[col_idx[0]])
    ax.text(0.02, 0.98, f"det(Σ₀) = {det0:.5f}", transform=ax.transAxes,
            va="top", fontsize=11)

plt.tight_layout()
plt.savefig("figures/bayes_priors.pdf")
plt.close()
print("Saved figures/bayes_priors.pdf")

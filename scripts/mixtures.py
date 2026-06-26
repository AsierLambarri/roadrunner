import sys
sys.path.insert(0, "src")

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
try:
    import smplotlib
except:
    pass

from tqdm import tqdm

from roadrunner.mixture.weighted_gmm import WeightedGaussianMixture
from roadrunner.mixture.bayesian_gmm import WeightedBayesianGaussianMixture

stds = np.random.lognormal(mean=0.1, sigma=0.375, size=50)
X, labels, centers = make_blobs(
    n_samples=int(1e5), n_features=10, centers=stds.size,
    center_box=(-50.0, 50.0), cluster_std=stds, random_state=42,
    return_centers=True,
)

shuffled_centers = centers + np.vstack(
    [np.random.normal(loc=0, scale=3 * s, size=(10,)) for s in stds]
)

# ── sklearn GMM ──────────────────────────────────────────────────────
print("Fitting sklearn GMM...")
gmm = GaussianMixture(
    n_components=stds.size,
    covariance_type="full",
    tol=1e-4,
    verbose=2,
).fit(X)
labels = gmm.predict(X)

# ── sklearn BGMM ─────────────────────────────────────────────────────
print("Fitting sklearn BGMM...")
bayes_gmm = BayesianGaussianMixture(
    n_components=stds.size,
    covariance_type="full",
    weight_concentration_prior_type="dirichlet_distribution",
    tol=1e-4,
    verbose=2,
).fit(X)
bayes_labels = bayes_gmm.predict(X)

print("mean_prior_:", np.asarray(bayes_gmm.mean_prior_).shape)
print("mean_precision_prior_:", np.asarray(bayes_gmm.mean_precision_prior_).shape, bayes_gmm.mean_precision_prior_)
print("weight_concentration_prior_:", np.asarray(bayes_gmm.weight_concentration_prior_).shape)
print("degrees_of_freedom_prior_:", np.asarray(bayes_gmm.degrees_of_freedom_prior_).shape)
print("covariance_prior_:", np.asarray(bayes_gmm.covariance_prior_).shape)

# ── roadrunner GMM ───────────────────────────────────────────────────
print("Fitting roadrunner GMM...")
my_gmm = WeightedGaussianMixture(
    n_components=stds.size,
    means_init=shuffled_centers,
    cov_type="full",
    tol=1e-4,
    verbose=2,
).fit(X)
my_labels = my_gmm.predict(X)

# ── roadrunner BGMM ──────────────────────────────────────────────────
print("Fitting roadrunner BGMM...")
my_bayes_gmm = WeightedBayesianGaussianMixture(
    n_components=stds.size,
    means_init=shuffled_centers,
    cov_type="full",
    tol=1e-4,
    verbose=2,
).fit(X)
my_bayes_labels = my_bayes_gmm.predict(X)

# ── Plot ─────────────────────────────────────────────────────────────
print("Plotting...")
cmap = plt.get_cmap("tab10")


def get_color(i):
    return cmap(i % cmap.N)


fig = plt.figure(figsize=(1.3 * 8, 1.3 * 10))
gs = GridSpec(nrows=3, ncols=2, height_ratios=[0.6, 1.0, 1.0], figure=fig)

ax_top = fig.add_subplot(gs[0, :])
ax_bottom_left = fig.add_subplot(gs[1, 0])
ax_bottom_right = fig.add_subplot(gs[1, 1], sharex=ax_bottom_left, sharey=ax_bottom_left)
ax_bbottom_left = fig.add_subplot(gs[2, 0], sharex=ax_bottom_left, sharey=ax_bottom_left)
ax_bbottom_right = fig.add_subplot(gs[2, 1], sharex=ax_bottom_left, sharey=ax_bottom_left)

ax_top.set_title("Std-Dev distribution")
ax_bottom_left.set_title(f"GMM (sklearn) log-lik={gmm.lower_bound_:.3e}")
ax_bottom_right.set_title(f"BGMM (sklearn) log-lik={bayes_gmm.lower_bound_:.3e}")
ax_bbottom_left.set_title(f"GMM (roadrunner) log-lik={my_gmm.lower_bound_:.3e}")
ax_bbottom_right.set_title(f"BGMM (roadrunner) log-lik={my_bayes_gmm.lower_bound_:.3e}")

ax_top.hist(stds, color="tab:blue")

ax_bottom_left.scatter(*X[:, [0, 1]].T, s=0.2, c=labels, edgecolor="none", cmap=cmap, zorder=10)
ax_bottom_right.scatter(*X[:, [0, 1]].T, s=0.2, c=bayes_labels, edgecolor="none", cmap=cmap, zorder=10)
ax_bbottom_left.scatter(*X[:, [0, 1]].T, s=0.2, c=my_labels, edgecolor="none", cmap=cmap, zorder=10)
ax_bbottom_right.scatter(*X[:, [0, 1]].T, s=0.2, c=my_bayes_labels, edgecolor="none", cmap=cmap, zorder=10)


def draw_ellipse(position, covariance, ax, color, alpha=0.3):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 3.5 * np.sqrt(vals) * 2
    ell = Ellipse(
        xy=position, width=width, height=height, angle=angle,
        alpha=alpha, facecolor=color, edgecolor=color,
    )
    ax.add_patch(ell)
    ell.set_zorder(20)


for i, (mean, cov) in tqdm(
    enumerate(zip(gmm.means_, gmm.covariances_)), total=gmm.means_.shape[0]
):
    if gmm.covariance_type == "diag":
        cov = np.diag(cov)
    draw_ellipse(mean[:2], cov[:2, :2], ax_bottom_left, get_color(i), alpha=0.3)

for i, (mean, cov) in enumerate(zip(bayes_gmm.means_, bayes_gmm.covariances_)):
    if bayes_gmm.covariance_type == "diag":
        cov = np.diag(cov)
    draw_ellipse(mean[:2], cov[:2, :2], ax_bottom_right, get_color(i), alpha=0.3)

for i, (mean, cov) in tqdm(
    enumerate(zip(my_gmm.means_, my_gmm.covariances_)), total=my_gmm.means_.shape[0]
):
    if my_gmm.cov_type == "diag":
        cov = np.diag(cov)
    draw_ellipse(mean[:2], cov[:2, :2], ax_bbottom_left, get_color(i), alpha=0.3)

for i, (mean, cov) in enumerate(zip(my_bayes_gmm.means_, my_bayes_gmm.covariances_)):
    if my_bayes_gmm.cov_type == "diag":
        cov = np.diag(cov)
    draw_ellipse(mean[:2], cov[:2, :2], ax_bbottom_right, get_color(i), alpha=0.3)

plt.tight_layout()
plt.savefig("figures/mixtures.pdf")
plt.close()
print("Saved figures/mixtures.pdf")

import numpy as np


def build_bgmm_priors(previous_parameters, group_subtrees, csc_b,
                      scaler, n_comp, sid_to_halo_6d,
                      cov_type, n_features):
    """Build BGMM prior_kwargs from previous snapshot's fitted parameters.

    Returns empty dict if no previous parameters (GMM, first snapshot, resume
    without checkpoint).
    """
    if previous_parameters is None:
        return {}

    dof_prior = n_features + 4
    inv_s = 1.0 / scaler.scale_

    mp = np.zeros((n_comp, n_features))
    cp = (
        np.zeros((n_comp, n_features, n_features)) if cov_type == "full"
        else np.zeros((n_comp, n_features)) if "diag" in cov_type
        else np.zeros(n_comp)
    )
    wp = np.full(n_comp, 1.0 / n_comp)
    pp = np.ones(n_comp)

    # Bound particles per component from current boundness
    bound_counts = np.array([len(c) for c in csc_b.column_indices], dtype=np.float64)

    scale_cov = (
        (lambda c: c * np.outer(inv_s, inv_s)) if cov_type == "full"
        else (lambda c: c * inv_s**2) if "diag" in cov_type
        else (lambda c: c * np.mean(inv_s**2))
    )

    for i, sid in enumerate(group_subtrees):
        sid_int = int(sid)

        # ── Mean prior from current halo catalogue position ─────
        pos = sid_to_halo_6d.get(sid_int)
        if pos is not None:
            mp[i] = (pos - scaler.mean_) * inv_s

        # ── Priors from N-1 fitted parameters ───────────────────
        p = previous_parameters.get(sid_int)
        if p is None:
            continue

        nk_n1 = max(p.get("count", 1.0), 1.0)
        n_bound_i = max(bound_counts[i], 1.0)

        # weight_concentration_prior = min(nk_n1/2, n_bound/2)
        wp[i] = max(min(nk_n1 / 2.0, n_bound_i / 2.0), 1.0 / n_comp)

        # mean_precision_prior = min(nk_n1/20, n_bound/20)
        pp[i] = max(min(nk_n1 / 20.0, n_bound_i / 20.0), 1.0)

        # covariance_prior = dof * n_bound / nk_n1 * 1/4 * cov_n1
        cp[i] = scale_cov(p["covariance"]) * dof_prior * n_bound_i / nk_n1 * 0.25

    return dict(
        mean_prior=mp,
        covariance_prior=cp,
        weight_concentration_prior=wp,
        mean_precision_prior=pp,
        degrees_of_freedom_prior=dof_prior,
    )

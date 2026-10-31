"""Explicit Gaussian profile deblending with cross-fitted structural priors.

This small numerical primitive is an integration research interface, not an
image reader, instrument pipeline, or production absorption correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class StructuralPrior:
    mean: np.ndarray
    covariance: np.ndarray
    source_run: str
    excluded_group_ids: list[str]


def deblend_profiles(design, observations, sigma, *, group_id: str,
                     prior: StructuralPrior | None = None, strength=0.1,
                     maximum_information_fraction=0.25):
    """Minimize ||W^.5(AI-y)||² + lambda||C^-.5(I-m)||², I>=0.

    A prior may be used only for an explicitly excluded group. The scalar
    strength is reduced so each nonzero data-information eigen-direction has
    at most maximum_information_fraction prior/data information. The bound
    must be chosen independently of the held-out final assessment set.
    """
    a = np.asarray(design, float)
    y, sig = np.asarray(observations, float), np.asarray(sigma, float)
    if a.ndim != 2 or y.shape != (len(a),) or sig.shape != y.shape:
        raise ValueError("Mismatched profile, observation and sigma dimensions")
    if a.shape[1] == 0 or a.shape[0] == 0 or not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError("Empty or nonfinite profile problem")
    if not np.isfinite(sig).all() or np.any(sig <= 0) or np.any(a < 0):
        raise ValueError("Profile weights must be nonnegative; sigma strictly positive")
    if not 0 <= strength <= 1 or not 0 < maximum_information_fraction <= 1:
        raise ValueError("Invalid prior-strength or information bound")
    aw, yw = a/sig[:, None], y/sig
    data_information = aw.T@aw
    eig, vec = np.linalg.eigh(data_information)
    rank_threshold = max(float(eig.max()), 0.0)*1e-12
    rank = int(np.sum(eig > rank_threshold))
    prior_precision = np.zeros_like(data_information)
    effective_strength = 0.
    augmented_a, augmented_y = aw, yw
    source = None
    if prior is not None and strength > 0:
        if not group_id or group_id not in prior.excluded_group_ids:
            raise ValueError("Structural prior was not cross-fitted with this group excluded")
        if not prior.source_run:
            raise ValueError("Prior source_run is required")
        mean, covariance = np.asarray(prior.mean, float), np.asarray(prior.covariance, float)
        if mean.shape != (a.shape[1],) or covariance.shape != data_information.shape:
            raise ValueError("Prior dimension mismatch")
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or np.any(mean < 0):
            raise ValueError("Nonfinite or negative prior mean")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("Prior covariance must be symmetric")
        c = np.linalg.cholesky(covariance)
        root_precision = np.linalg.solve(c, np.eye(len(c)))
        precision = root_precision.T@root_precision
        # An unresolved data null space cannot be made experimentally identified
        # by a structural prior. Reject that operation rather than masking it.
        if rank < a.shape[1]:
            raise ValueError("Profiles are rank deficient; prior-only separation is not reported as measured")
        invsqrt = (vec/np.sqrt(eig))@vec.T
        maximum_ratio = float(np.linalg.eigvalsh(invsqrt@precision@invsqrt).max())
        effective_strength = min(float(strength), maximum_information_fraction/maximum_ratio)
        prior_precision = effective_strength*precision
        r = np.sqrt(effective_strength)*root_precision
        augmented_a = np.vstack([aw, r])
        augmented_y = np.concatenate([yw, r@mean])
        source = prior.source_run
    result = lsq_linear(augmented_a, augmented_y, bounds=(0, np.inf), tol=1e-12)
    pseudoinverse = np.linalg.pinv(data_information+prior_precision, rcond=1e-12)
    covariance = pseudoinverse if rank == a.shape[1] else None
    boundary = np.flatnonzero(result.x <= 1e-9).tolist()
    return {"intensity_estimate": result.x.tolist(),
            "covariance_local": None if covariance is None else covariance.tolist(),
            "normal_matrix_pseudoinverse": pseudoinverse.tolist(),
            "data_information_nullspace": vec[:, eig <= rank_threshold].T.tolist(),
            "identifiable_from_data": rank == a.shape[1],
            "data_information_rank": rank, "n_components": a.shape[1],
            "prior_source_run": source, "group_id": group_id,
            "effective_prior_strength": effective_strength,
            "maximum_prior_information_fraction": maximum_information_fraction,
            "active_nonnegativity_bounds": boundary,
            "covariance_interpretation": ("unconstrained local Gaussian approximation; boundary intervals need profiling"
                                          if covariance is not None else
                                          "not identifiable: pseudoinverse is not a covariance; null-direction uncertainties are unbounded"),
            "weighted_data_residual": float(np.sum((aw@result.x-yw)**2)),
            "success": bool(result.success),
            "raw_observations_modified": False}

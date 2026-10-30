"""Analytic local rotation-prediction derivatives in DIALS conventions.

The native DIALS predictor still selects the physical entering/exiting branch,
panel and unwrapped turn. This module differentiates that selected solution;
it neither enumerates reflection identities nor approximates the Ewald sphere.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def compatibility_metrics(observed, predicted, covariance):
    """One Cholesky factor per observation, shared by every candidate residual."""
    residual = np.asarray(observed, float) - np.asarray(predicted, float)
    factor = np.linalg.cholesky(np.asarray(covariance, float))
    white = solve_triangular(factor, residual.T, lower=True, check_finite=False)
    distance2 = np.einsum('ij,ij->j', white, white)
    logdet = 2 * np.log(np.diag(factor)).sum()
    log_density = -.5 * (distance2 + logdet + 3 * np.log(2 * np.pi))
    return residual, distance2, log_density


def rotate_about_axis(vectors, axis, angles):
    """Rodrigues rotation with vector coordinates in the final dimension."""
    vectors, axis = np.asarray(vectors), np.asarray(axis)
    angles = np.asarray(angles)
    cosine, sine = np.cos(angles)[..., None], np.sin(angles)[..., None]
    return (cosine * vectors + sine * np.cross(axis, vectors)
            + (1 - cosine) * np.sum(vectors * axis, axis=-1)[..., None] * axis)


def crystal_prediction_jacobian(hkl, phi, s0, axis, fixed, setting, A, dA, detector):
    """Return d(X_mm,Y_mm,phi_rad)/dp for fixed instrument geometry.

    ``dA`` contains symmetry-preserving DIALS U/B derivatives, in the caller's
    physical parameter units. ``detector`` is the inverse detector basis for
    each selected native panel. No full reflection-by-global-parameter tensor
    is needed until the caller places this crystal's small block.

    r = S R(phi) F A h, t = (S e) cross r, s1 = s0 + r.
    Implicit differentiation of |s1|^2 = |s0|^2 gives
    dphi/dp = -(s1 dot d_r_at_fixed_phi)/(s0 dot t).
    The detector derivative then follows from the quotient rule for D s1.
    """
    hkl, phi = np.asarray(hkl, float), np.asarray(phi, float)
    s0, axis = np.asarray(s0, float), np.asarray(axis, float)
    fixed, setting, A, dA = map(lambda value: np.asarray(value, float), (fixed, setting, A, dA))
    detector = np.asarray(detector, float)
    if not len(hkl):
        return np.empty((0, 3, len(dA)))
    axis = axis / np.linalg.norm(axis)
    q = hkl @ A.T @ fixed.T
    r = rotate_about_axis(q, axis, phi) @ setting.T
    s1 = s0 + r
    tangent = np.cross(setting @ axis, r)
    denominator = tangent @ s0
    rounding_scale = 64 * np.finfo(float).eps * np.linalg.norm(s0) * np.linalg.norm(tangent, axis=1)
    if np.any(np.abs(denominator) <= rounding_scale):
        raise ValueError('Ewald tangency: rotation derivative is not numerically resolved')
    dq = np.einsum('pij,nj->npi', dA, hkl) @ fixed.T
    dr = rotate_about_axis(dq, axis, phi[:, None]) @ setting.T
    dphi = -np.einsum('npi,ni->np', dr, s1) / denominator[:, None]
    ds1 = dr + tangent[:, None, :] * dphi[:, :, None]
    projection = np.einsum('nij,nj->ni', detector, s1)
    dprojection = np.einsum('nij,npj->npi', detector, ds1)
    w = projection[:, 2]
    plane_scale = 64 * np.finfo(float).eps * np.linalg.norm(detector[:, 2], axis=1) * np.linalg.norm(s1, axis=1)
    if np.any(np.abs(w) <= plane_scale):
        raise ValueError('Detector-plane tangency: position derivative is not numerically resolved')
    xy = projection[:, :2] / w[:, None]
    dxy = (dprojection[:, :, :2] - dprojection[:, :, 2, None] * xy[:, None, :]) / w[:, None, None]
    result = np.concatenate((dxy, dphi[:, :, None]), axis=2).transpose(0, 2, 1)
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite analytic crystal-prediction derivative')
    return result

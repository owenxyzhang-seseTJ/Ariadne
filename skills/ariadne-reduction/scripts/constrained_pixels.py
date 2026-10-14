"""Frozen-background NNLS/Poisson fits; covariance is conditional, not boundary CIs."""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.special import xlogy


def fit_constrained(y, variance, profiles, valid, basis, noise, method, rcond):
    P, B, data = profiles[valid], basis[valid], y[valid]
    significant = np.any(P > .01 * np.maximum(P.max(axis=0), 1e-300), axis=1)
    selected = ~significant
    if selected.sum() < B.shape[1] + 5:
        raise ValueError('Insufficient independent background pixels for constrained fit')
    root_weight = 1 / np.sqrt(variance[valid])
    beta, _, rank_b, _ = np.linalg.lstsq(B[selected] * root_weight[selected, None],
                                       data[selected] * root_weight[selected], rcond=None)
    if rank_b != B.shape[1]:
        raise ValueError('Background-only design is rank deficient')
    background = B @ beta
    if np.any(background < 0):
        raise ValueError('Frozen background predicts negative expectation; revise background model')
    if method == 'nnls':
        result = lsq_linear(P * root_weight[:, None], (data - background) * root_weight,
                            bounds=(0., np.inf), method='trf', max_iter=300)
        intensity, fit_weight = result.x, root_weight
        converged, objective = bool(result.success), float(2 * result.cost)
    else:
        if not noise or not noise.get('source') or noise.get('poisson_counts_verified') is not True:
            raise ValueError('Poisson requires explicit calibrated genuine-count evidence')
        gain = float(noise['gain'])
        if gain <= 0 or float(noise['readout_variance']) != 0 or noise.get('offset_removed') is not True:
            raise ValueError('Poisson requires calibrated gain, removed offset and zero readout noise')
        counts, b = data / gain, background / gain
        if np.any(counts < 0) or not np.allclose(counts, np.rint(counts), atol=1e-6, rtol=0):
            raise ValueError('Corrected ADU does not represent integer photon counts')
        initial = lsq_linear(P, counts - b, bounds=(0., np.inf)).x
        def objective_and_gradient(I):
            expected = np.maximum(b + P @ I, 1e-12)
            return float(np.sum(expected - xlogy(counts, expected))), P.T @ (1 - counts / expected)
        result = minimize(objective_and_gradient, initial, jac=True, method='L-BFGS-B',
                          bounds=[(0., None)] * P.shape[1], options={'maxiter': 500, 'ftol': 1e-12})
        intensity = gain * result.x
        expected = np.maximum(b + P @ result.x, 1e-12)
        fit_weight = 1 / (gain * np.sqrt(expected))
        objective = float(2 * np.sum(xlogy(counts, counts / expected) - counts + expected))
        converged = bool(result.success)
    fitted = profiles @ intensity + basis @ beta
    design = P * fit_weight[:, None]
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales <= 1e-15):
        raise ValueError('Masked or uninformative component')
    _, singular, Vh = np.linalg.svd(design / scales, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * rcond))
    retained = Vh[:rank].T
    covariance = ((retained / singular[:rank]) @ (retained / singular[:rank]).T) / scales[:, None] / scales[None, :]
    boundary = np.flatnonzero(intensity <= 1e-7 * max(float(np.max(intensity)), 1.)).tolist()
    dof = len(data) - rank
    return {'intensities_internal': intensity, 'covariance_identifiable_subspace': covariance,
            'nullspace': Vh[rank:], 'column_scales': scales, 'rank': rank,
            'singular_values': singular, 'background_coefficients': beta, 'fitted': fitted,
            'chi2': objective, 'dof': dof, 'reduced_chi2': objective / dof if dof > 0 else None,
            'converged': converged, 'boundary_components': boundary,
            'covariance_scope': 'interior Gaussian/Fisher approximation conditional on frozen estimated background/profile/geometry; invalid as boundary confidence intervals'}

"""Retain the observable Gaussian likelihood of an overlapped pixel region.

No intensity is assigned to an unobservable component.  With fixed weights,
background/position nuisance directions eliminated, and D = U diag(s) V.T S,
z = U.T y and H = diag(s) V.T S preserve the parameter-dependent quadratic.
The retained rank is a floating-point rank, deliberately independent of a
more conservative threshold used to publish individual intensities.
"""
from __future__ import annotations

import numpy as np


def from_projected_svd(target, left, singular, right, scales, nuisance_rank, *, valid_pixels=None):
    """Construct a small, unit-covariance observation in physical amplitudes.

    ``right`` is V.T of the column-equilibrated, nuisance-projected design.
    This representation is exact for its retained numerical subspace; the
    norm of omitted normalized-design singular values is reported explicitly.
    The covariance is conditional on the supplied variance and fixed design.
    It is not a Poisson sufficient statistic or a posterior detwinning.
    """
    target = np.asarray(target, dtype=float)
    singular = np.asarray(singular, dtype=float)
    scales = np.asarray(scales, dtype=float)
    pixels = len(target) if valid_pixels is None else int(valid_pixels)
    cutoff = np.finfo(float).eps * max(pixels, len(scales)) * singular[0]
    rank = int(np.count_nonzero(singular > cutoff))
    if rank < 1:
        raise ValueError('No observable Gaussian information mode')
    observed = left[:, :rank].T @ target
    design = (singular[:rank, None] * right[:rank]) * scales[None, :]
    # Subtract vectors, not two nearly equal squared norms, for low residuals.
    orthogonal = target - left[:, :rank] @ observed
    constant = float(orthogonal @ orthogonal)
    discarded = singular[rank:]
    return {
        'schema': 'ariadne-information-modes/1',
        'y': observed.tolist(), 'design': design.tolist(),
        'covariance': np.eye(rank).tolist(), 'rank': rank,
        'variance_model': 'fixed', 'valid_pixels': pixels,
        'nuisance_rank': int(nuisance_rank),
        'rank_cutoff': float(cutoff),
        'discarded_singular_values': discarded.tolist(),
        'discarded_normalized_operator_norm': float(discarded[0]) if len(discarded) else 0.,
        'pixel_chi2_constant': constant,
        'likelihood_scope': (
            'fixed-weight Gaussian quadratic after eliminating linear nuisance; '
            'exact in retained floating-point subspace; conditional on profiles, '
            'geometry and supplied variance, including any plug-in variance assumptions'),
        'intensity_semantics': 'design acts on original ordered full-profile contributor amplitudes',
        'interpretation': 'observable linear combinations, not separately measured component intensities',
    }


def conditional_component_diagnostics(result):
    """Publish marginal weak-signal precision without selecting on its sign.

    A component's information ratio compares marginal information with its
    information if every other signal amplitude were known.  Nuisance
    background has already been eliminated in both quantities.  A vanishing
    ratio therefore describes overlap ambiguity rather than a weak count cut.
    """
    modes = result.get('information_modes')
    if modes is None:
        return None
    design = np.asarray(modes['design'], dtype=float)
    covariance = np.asarray(result['covariance_identifiable_subspace'])
    scales = np.asarray(result['column_scales'])
    nullspace = np.asarray(result['nullspace'])
    values = np.asarray(result['intensities_internal'])
    information = np.sum(design * design, axis=0)
    output = []
    for j in range(len(scales)):
        direction = np.zeros(len(scales))
        direction[j] = 1. / scales[j]
        null_fraction = float(np.linalg.norm(nullspace @ direction) / np.linalg.norm(direction))
        identifiable = null_fraction <= 10 * np.finfo(float).eps * len(scales)
        variance = float(covariance[j, j]) if identifiable else None
        positive = identifiable and variance > 0 and information[j] > 0
        output.append({
            'identifiable_in_published_subspace': bool(identifiable),
            'signed_signal_to_sigma': float(values[j] / np.sqrt(variance)) if positive else None,
            'conditional_information_ratio': float(1 / (variance * information[j])) if positive else 0.,
            'selection': 'none; weak and negative measured amplitudes are retained',
        })
    return output

"""Gaussian box masses and translation derivatives, XYZ coordinates / ZYX arrays.

One conditional axis is integrated analytically. Numerical axes retain the
existing Gauss--Legendre resolution rule; this rule is not an error certificate.
Adjacent voxels share normal-CDF boundaries. Translation derivatives are face
fluxes, eliminating a finite-difference step and six repeated volume integrals.
The scratch budget excludes caller-visible output arrays and small axis vectors.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erf, ndtr, roots_legendre


DEFAULT_WORKING_BYTES = 8 * 1024**2
_SQRT_2 = np.sqrt(2.)
_SQRT_2PI = np.sqrt(2 * np.pi)


@lru_cache(maxsize=64)
def _voxel_quadrature(order):
    nodes, weights = roots_legendre(order)
    nodes, weights = nodes / 2, weights / 2
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


def _validated(covariance, centre, bounds, max_working_bytes):
    covariance = np.asarray(covariance, dtype=float)
    centre = np.asarray(centre, dtype=float)
    if (covariance.shape != (3, 3) or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
            or np.any(np.linalg.eigvalsh(covariance) <= 0)
            or centre.shape != (3,) or not np.all(np.isfinite(centre))):
        raise ValueError('Require finite centre and symmetric positive-definite profile covariance')
    limits = np.asarray(bounds)
    if (limits.shape != (6,) or not np.all(np.isfinite(limits))
            or np.any(limits != np.floor(limits))):
        raise ValueError('Voxel bounds must contain six finite integer edges')
    limits = limits.astype(np.int64).reshape(3, 2)
    lengths = limits[:, 1] - limits[:, 0]
    if np.any(lengths <= 0):
        raise ValueError('Voxel bounds must have strictly positive lengths')
    if not isinstance(max_working_bytes, (int, np.integer)) or max_working_bytes < 4096:
        raise ValueError('max_working_bytes must be an integer of at least 4096 bytes')
    covariance = (covariance + covariance.T) * .5
    return covariance, centre, limits, tuple(lengths[::-1]), int(max_working_bytes)


def _output(out, shape):
    if out is None:
        return np.empty(shape, dtype=float)
    if not isinstance(out, np.ndarray) or out.shape != shape or out.dtype != np.float64 or not out.flags.writeable:
        raise ValueError('Output must be a writable float64 array of the required shape')
    return out


def _normal_cells(edges):
    """CDF differences along axis 0, sharing one stable tail value per edge."""
    tails = ndtr(-np.abs(edges))
    values = np.abs(np.diff(tails, axis=0))
    lower, upper = edges[:-1], edges[1:]
    central = ((lower <= 0) & (upper >= 0)) | ((lower > -.5) & (upper < .5))
    if np.any(central):
        # Around zero, subtract erf values rather than two values near 1/2.
        values[central] = .5 * (erf(upper[central] / _SQRT_2) - erf(lower[central] / _SQRT_2))
    return values


def _normal_face_difference(edges, sigma):
    lower, upper = edges[:-1], edges[1:]
    exponent_difference = .5 * (upper - lower) * (upper + lower)
    maximum_density = np.exp(-.5 * np.minimum(lower * lower, upper * upper)) / (_SQRT_2PI * sigma)
    return np.sign(exponent_difference) * maximum_density * -np.expm1(-np.abs(exponent_difference))


def _orders(precision, axes):
    orders = np.maximum(6, np.ceil(3 * np.sqrt(np.diag(precision)[axes])) + 2)
    if np.any(orders > 64):
        raise ValueError('Correlated profile needs more than 64 quadrature nodes per numerical axis; revise calibration/support')
    return orders.astype(int)


def _tile_size(count_a, quadrature_points, budget):
    # Seven live edge-sized buffers cover CDFs, selected erf operands,
    # interval masses and masks; twelve node buffers cover their inputs.
    floats = budget // 8
    if floats < 26 * quadrature_points:
        raise ValueError('Working-memory budget cannot hold one quadrature tile')
    rows = min(count_a, 32, max(1, (floats // quadrature_points - 12) // 7 - 1))
    columns = max(1, floats // (quadrature_points * (7 * (rows + 1) + 12)))
    return rows, columns


def _cdf_tiles(edges, conditional_mean, weighted_density, scale, rows):
    for start in range(0, len(edges) - 1, rows):
        stop = min(start + rows, len(edges) - 1)
        standardized = (edges[start:stop + 1, None, None] - conditional_mean[None, :, :]) * scale
        mass = _normal_cells(standardized)
        yield start, stop, np.einsum('amq,mq->am', mass, weighted_density, optimize=False)


def _diagonal(covariance, centre, limits, values, derivatives):
    masses, fluxes = [], []
    for axis, (lo, hi) in enumerate(limits):
        sigma = np.sqrt(covariance[axis, axis])
        edges = (np.arange(lo, hi + 1) - centre[axis]) / sigma
        masses.append(_normal_cells(edges))
        if derivatives is not None:
            fluxes.append(_normal_face_difference(edges, sigma))
    np.multiply(masses[2][:, None, None], masses[1][None, :, None], out=values)
    values *= masses[0][None, None, :]
    if derivatives is not None:
        for axis in range(3):
            factors = [fluxes[d] if d == axis else masses[d] for d in range(3)]
            np.multiply(factors[2][:, None, None], factors[1][None, :, None], out=derivatives[axis])
            derivatives[axis] *= factors[0][None, None, :]


def _correlated_values(covariance, precision, centre, limits, values, budget):
    axis_a = int(np.argmax(np.diag(precision)))
    axes = [axis for axis in range(3) if axis != axis_a]
    axis_b, axis_c = axes
    orders = _orders(precision, axes)
    node_b, weight_b = _voxel_quadrature(int(orders[0]))
    node_c, weight_c = _voxel_quadrature(int(orders[1]))
    quadrature_weight = (weight_b[:, None] * weight_c[None, :]).ravel()
    # Small shoeboxes should not fill a large global scratch allowance with
    # quadrature tensors. Retain a fixed rule-sized floor and scale with output.
    budget = min(budget, max(128 * 1024, 4 * values.nbytes, 26 * len(quadrature_weight) * 8))
    marginal = covariance[np.ix_(axes, axes)]
    marginal_inverse = np.linalg.inv(marginal)
    normalizer = 2 * np.pi * np.sqrt(np.linalg.det(marginal))
    regression = -precision[axis_a, axes] / precision[axis_a, axis_a]
    scale = np.sqrt(precision[axis_a, axis_a])
    edges = np.arange(limits[axis_a, 0], limits[axis_a, 1] + 1) - centre[axis_a]
    mids = [np.arange(lo, hi) + .5 - centre[axis] for axis, (lo, hi) in enumerate(limits)]
    rows, columns = _tile_size(len(edges) - 1, len(quadrature_weight), budget)
    view = values.transpose(2, 1, 0).transpose(axis_a, axis_b, axis_c)
    count_c = len(mids[axis_c])
    for start in range(0, len(mids[axis_b]) * count_c, columns):
        indices = np.arange(start, min(start + columns, len(mids[axis_b]) * count_c))
        ib, ic = indices // count_c, indices % count_c
        db = mids[axis_b][ib, None, None] + node_b[None, :, None]
        dc = mids[axis_c][ic, None, None] + node_c[None, None, :]
        exponent = (marginal_inverse[0, 0] * db * db
                    + 2 * marginal_inverse[0, 1] * db * dc + marginal_inverse[1, 1] * dc * dc)
        weighted = np.exp(-.5 * exponent).reshape(len(indices), -1)
        weighted *= quadrature_weight / normalizer
        if not np.any(weighted):
            # Exact zeros of the evaluated floating-point marginal, not an
            # intensity or tail-probability threshold.
            view[:, ib, ic] = 0.
            continue
        means = (regression[0] * db + regression[1] * dc).reshape(len(indices), -1)
        for lo, hi, tile in _cdf_tiles(edges, means, weighted, scale, rows):
            view[lo:hi, ib, ic] = tile


def _face_derivative(axis_j, covariance, precision, centre, limits, derivative, budget):
    """Integrate shared face fluxes, then apply the conservative difference."""
    remaining = [axis for axis in range(3) if axis != axis_j]
    axis_a = max(remaining, key=lambda axis: precision[axis, axis])
    axis_b = next(axis for axis in remaining if axis != axis_a)
    node, weight = _voxel_quadrature(int(_orders(precision, [axis_b])[0]))
    budget = min(budget, max(128 * 1024, 4 * derivative.nbytes, 52 * len(node) * 8))
    # A precision submatrix avoids a subtractive covariance Schur complement.
    conditional = np.linalg.inv(precision[np.ix_([axis_a, axis_b], [axis_a, axis_b])])
    variance_b = conditional[1, 1]
    regression = -precision[axis_a, axis_b] / precision[axis_a, axis_a]
    shift = covariance[:, axis_j] / covariance[axis_j, axis_j]
    sigma_j = np.sqrt(covariance[axis_j, axis_j])
    scale = np.sqrt(precision[axis_a, axis_a])
    a_edges = np.arange(limits[axis_a, 0], limits[axis_a, 1] + 1) - centre[axis_a]
    j_edges = np.arange(limits[axis_j, 0], limits[axis_j, 1] + 1) - centre[axis_j]
    b_mids = np.arange(limits[axis_b, 0], limits[axis_b, 1]) + .5 - centre[axis_b]
    rows, columns = _tile_size(len(a_edges) - 1, len(node), budget)
    if columns < 2:
        rows = 1
        columns = (budget // 8) // (26 * len(node))
        if columns < 2:
            raise ValueError('Working-memory budget cannot hold two shared face boundaries')
    face_rows = min(len(j_edges) - 1, 16, max(1, columns // 2 - 1))
    b_columns = max(1, columns // (face_rows + 1))
    view = derivative.transpose(2, 1, 0).transpose(axis_a, axis_j, axis_b)
    for j0 in range(0, len(j_edges) - 1, face_rows):
        j1 = min(j0 + face_rows, len(j_edges) - 1)
        tj = j_edges[j0:j1 + 1, None, None]
        density_j = np.exp(-.5 * (tj / sigma_j)**2) / (_SQRT_2PI * sigma_j)
        for b0 in range(0, len(b_mids), b_columns):
            b1 = min(b0 + b_columns, len(b_mids))
            db = b_mids[None, b0:b1, None] + node - shift[axis_b] * tj
            weighted = density_j * np.exp(-.5 * db**2 / variance_b) / (_SQRT_2PI * np.sqrt(variance_b))
            weighted *= weight
            if not np.any(weighted):
                view[:, j0:j1, b0:b1] = 0.
                continue
            means = shift[axis_a] * tj + regression * db
            weighted, means = weighted.reshape(-1, len(node)), means.reshape(-1, len(node))
            for a0, a1, flux in _cdf_tiles(a_edges, means, weighted, scale, rows):
                flux = flux.reshape(a1 - a0, j1 - j0 + 1, b1 - b0)
                view[a0:a1, j0:j1, b0:b1] = -np.diff(flux, axis=1)


def gaussian_voxels(covariance, centre, bounds, *, max_working_bytes=DEFAULT_WORKING_BYTES, out=None):
    """Return full-profile Gaussian voxel probabilities in ZYX order.

    Bounds do not renormalize the profile. Each numerical axis retains the
    resolution rule max(6, ceil(3/sigma_cond)+2), rejecting above 64 nodes. Scratch
    arrays are tiled independently of the output volume; out avoids a copy.
    """
    covariance, centre, limits, shape, budget = _validated(covariance, centre, bounds, max_working_bytes)
    values = _output(out, shape)
    if np.all(covariance == np.diag(np.diag(covariance))):
        _diagonal(covariance, centre, limits, values, None)
    else:
        precision = np.linalg.inv(covariance)
        _correlated_values(covariance, precision, centre, limits, values, budget)
    return values


def gaussian_voxels_with_derivatives(covariance, centre, bounds, *,
                                     max_working_bytes=DEFAULT_WORKING_BYTES,
                                     out=None, derivative_out=None):
    """Return probabilities and dP/dcentre in (XYZ, ZYX) order.

    Translation differentiates a Gaussian density as minus its spatial
    derivative. The divergence theorem gives lower-face minus upper-face
    probability flux. Shared internal faces cancel in sums and no arbitrary
    position-difference step is required. Correlated face integrals use one
    numerical axis and one analytic conditional CDF axis.
    """
    covariance, centre, limits, shape, budget = _validated(covariance, centre, bounds, max_working_bytes)
    values = _output(out, shape)
    derivatives = _output(derivative_out, (3, *shape))
    if np.shares_memory(values, derivatives):
        raise ValueError('Probability and derivative outputs must not overlap')
    if np.all(covariance == np.diag(np.diag(covariance))):
        _diagonal(covariance, centre, limits, values, derivatives)
    else:
        precision = np.linalg.inv(covariance)
        _correlated_values(covariance, precision, centre, limits, values, budget)
        for axis in range(3):
            _face_derivative(axis, covariance, precision, centre, limits, derivatives[axis], budget)
    return values, derivatives

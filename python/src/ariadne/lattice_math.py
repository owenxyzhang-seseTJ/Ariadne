"""Basis-aware Gaussian lattice sums and geometric support.

The physical lattice is ``A @ Z**3``.  Integer column operations change its
coordinates, never the lattice, reciprocal-cell volume, or noise model.  This
module has no dependency on DIALS and can be checked with independent sums.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


class PeriodicBudgetError(ValueError):
    """A selected summation representation exceeds the explicit radius budget."""


def reduced_basis(matrix):
    """Return ``B, U`` with ``B = matrix @ U`` and integer ``det(U) = +/-1``.

    Three-dimensional LLL size reduction uses integer column operations.  The
    integer transform is accumulated with Python integers; recomputing the
    floating basis in extended precision reduces cancellation after shear.
    This is a numerical lattice reduction, not a crystallographic cell change.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError('Expected a finite 3 by 3 reciprocal basis')
    if np.linalg.slogdet(matrix)[0] == 0:
        raise ValueError('Reciprocal basis is singular')
    transform = np.eye(3, dtype=object)
    extended = np.asarray(matrix, dtype=np.longdouble)

    def current():
        if max(abs(int(x)) for x in transform.flat) > 2**52:
            raise ValueError('Lattice basis transform exceeds exact floating integer range')
        return np.asarray(extended @ np.asarray(transform, dtype=np.longdouble), dtype=float)

    def gram_schmidt(basis):
        # A global scale leaves both Gram--Schmidt coefficients and Lovasz
        # comparisons unchanged, while avoiding squared-length overflow.
        basis = basis / np.max(np.abs(basis))
        orthogonal = basis.copy()
        norms = np.zeros(3)
        mu = np.zeros((3, 3))
        for i in range(3):
            for j in range(i):
                if not norms[j] > 0:
                    raise ValueError('Reciprocal basis is numerically singular')
                mu[i, j] = np.dot(basis[:, i], orthogonal[:, j]) / norms[j]
                orthogonal[:, i] -= mu[i, j] * orthogonal[:, j]
            norms[i] = np.dot(orthogonal[:, i], orthogonal[:, i])
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError('Reciprocal basis is numerically singular')
        return mu, norms

    basis = current()
    index = 1
    for _ in range(512):
        if index == 3:
            return basis, np.asarray(transform, dtype=np.int64)
        for j in range(index - 1, -1, -1):
            mu, _ = gram_schmidt(basis)
            multiplier = int(np.rint(mu[index, j]))
            if multiplier:
                transform[:, index] -= multiplier * transform[:, j]
                basis = current()
        mu, norms = gram_schmidt(basis)
        if norms[index] >= (.75 - mu[index, index - 1] ** 2) * norms[index - 1]:
            index += 1
        else:
            transform[:, [index - 1, index]] = transform[:, [index, index - 1]]
            basis = current()
            index = max(index - 1, 1)
    raise ValueError('Lattice basis reduction did not converge within its budget')


def affine_integer_rank(points):
    """Exact affine rank of 3D integer Miller indices, invariant to basis shear.

    Only three independent differences are needed.  Python integer cross and
    dot products avoid both floating SVD rank thresholds and int64 overflow.
    """
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('Expected N by 3 integer Miller indices')
    if not len(values):
        return 0
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 2**52):
        raise ValueError('Miller indices exceed exact floating integer range')
    if np.any(values != np.rint(values)):
        raise ValueError('Miller indices must be integers')
    origin = tuple(int(x) for x in values[0])
    first = normal = None
    for row in values[1:]:
        vector = tuple(int(row[j]) - origin[j] for j in range(3))
        if not any(vector):
            continue
        if first is None:
            first = vector
        elif normal is None:
            cross = (first[1]*vector[2] - first[2]*vector[1],
                     first[2]*vector[0] - first[0]*vector[2],
                     first[0]*vector[1] - first[1]*vector[0])
            if any(cross):
                normal = cross
        elif sum(normal[j] * vector[j] for j in range(3)):
            return 3
    return 2 if normal is not None else (1 if first is not None else 0)


def _offsets(radii):
    return np.asarray(list(itertools.product(*[range(-int(r), int(r) + 1) for r in radii])), dtype=float)


def _nearest(A, fractional, radii, distance=None, nearest=None):
    """Certify a finite closest-point search using the reciprocal basis metric.

    If an unsearched integer has |h_j| >= r_j + 1, its distance is at least
    (r_j + 1 - |f_j|) / ||row_j(A^-1)||.  Any failed certificate is resolved by
    an expanded search containing every possible improving integer point.
    """
    rows = np.linalg.norm(np.linalg.inv(A), axis=1)
    points = fractional @ A.T
    if distance is None:
        offsets = _offsets(radii)
        distance, indices = cKDTree(offsets @ A.T).query(points, eps=0.)
        nearest = offsets[indices]
    escape = np.min((np.asarray(radii) + 1 - np.abs(fractional)) / rows, axis=1)
    failed = distance > escape * (1 - 32 * np.finfo(float).eps)
    if np.any(failed):
        required = np.ceil(np.max(np.abs(fractional[failed])
                                  + distance[failed, None] * rows, axis=0)).astype(int)
        # Reduced dimension-three bases make this modest.  Retain an explicit
        # guard against pathological numeric inputs, before allocating nodes.
        if math.prod(int(2*r + 1) for r in required) > 1_000_000:
            raise PeriodicBudgetError('Certified closest-point search exceeds node budget')
        offsets = _offsets(required)
        distance[failed], indices = cKDTree(offsets @ A.T).query(points[failed], eps=0.)
        nearest[failed] = offsets[indices]
    return distance, nearest


def _orthogonal_sum(A, fractional, sigma, radii, chunk_size, tail_sigma, max_radius):
    """A rectangular reciprocal metric separates into three 1D theta sums."""
    lengths = np.linalg.norm(A, axis=0)
    log_ratio = np.zeros(len(fractional))
    representations = []
    actual_radii = []
    chunks = []
    terms = 0
    for axis, length in enumerate(lengths):
        radius = int(radii[axis])
        frequency_radius = int(np.ceil(max(8., tail_sigma) * length / (2*np.pi*sigma)))
        frequencies = (np.arange(1, frequency_radius + 1, dtype=float)
                       if frequency_radius < radius and frequency_radius <= 1_000_000 else None)
        coefficients = (2*np.exp(-2*(np.pi*sigma/length)**2*frequencies**2)
                        if frequencies is not None else None)
        use_fourier = coefficients is not None and coefficients.sum() < .5
        selected_radius = frequency_radius if use_fourier else radius
        if max_radius is not None and selected_radius > max_radius:
            raise PeriodicBudgetError('Selected one-dimensional lattice sum exceeds radius budget')
        if selected_radius > 1_000_000:
            raise PeriodicBudgetError('One-dimensional lattice sum exceeds node budget')
        representations.append('fourier' if use_fourier else 'real')
        actual_radii.append(selected_radius)
        terms += frequency_radius if use_fourier else 2*radius + 1
        if not use_fourier:
            nodes = np.arange(-radius, radius + 1, dtype=float)
        chunk = min(chunk_size, max(1, 32*1024**2 // (max(1, 2*selected_radius + 1)*8)))
        chunks.append(chunk)
        for start in range(0, len(fractional), chunk):
            end = min(start + chunk, len(fractional))
            f = fractional[start:end, axis, None]
            if use_fourier:
                log_ratio[start:end] += np.log1p(np.cos(2*np.pi*f*frequencies) @ coefficients)
            else:
                exponent = (f - nodes) * (length / sigma)
                exponent *= -.5 * exponent
                log_ratio[start:end] += (np.log(length / sigma) - .5*np.log(2*np.pi)
                                         + logsumexp(exponent, axis=1))
    representation = representations[0] if len(set(representations)) == 1 else 'mixed'
    return log_ratio, {'periodic_representation': representation,
                       'periodic_axis_representations': representations,
                       'periodic_selected_radius_xyz': actual_radii,
                       'periodic_terms': terms, 'effective_score_chunk': min(chunks),
                       'periodic_factorization': 'orthogonal_1d'}


def periodic_gaussian(A, fractional, sigma, radii, chunk_size, tail_sigma, *, max_radius=None):
    """Evaluate the periodic density and a certified nearest lattice point.

    The historical six positional arguments and four return values are kept.
    ``nearest_offset`` is always expressed in the caller's original basis.
    A reduced integer basis bounds work without changing the physical lattice.
    For orthogonal metrics, exact separation eliminates the tensor product.
    ``max_radius`` limits the representation actually evaluated; a broad
    Gaussian may be inexpensive in Fourier space despite a large real radius.
    """
    fractional = np.asarray(fractional, dtype=float)
    if fractional.ndim != 2 or fractional.shape[1] != 3 or not np.all(np.isfinite(fractional)):
        raise ValueError('Expected finite N by 3 fractional coordinates')
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(tail_sigma) or tail_sigma <= 0:
        raise ValueError('Gaussian width and tail extent must be finite and positive')
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError('Score chunk size must be a positive integer')
    chunk_size = int(chunk_size)
    if max_radius is not None and (not np.isfinite(max_radius) or max_radius < 1 or int(max_radius) != max_radius):
        raise ValueError('Maximum periodic radius must be a positive integer')
    radii = np.asarray(radii)
    if radii.shape != (3,) or not np.all(np.isfinite(radii)) or np.any(radii < 1) or np.any(radii != np.rint(radii)):
        raise ValueError('Periodic radii must be three positive integers')
    A, transform = reduced_basis(A)
    transformed = not np.array_equal(transform, np.eye(3, dtype=int))
    if transformed:
        fractional = np.linalg.solve(transform.astype(float), fractional.T).T
        radii = np.maximum(1, np.ceil(.5 + tail_sigma*sigma*np.linalg.norm(np.linalg.inv(A), axis=1)))
    # The caller may use unwrapped coordinates.  Centering preserves the
    # periodic density and returns the corresponding unwrapped Miller offset.
    center = np.rint(fractional)
    fractional = fractional - center
    if np.any(np.abs(center) > 2**52):
        raise ValueError('Fractional coordinates exceed exact floating integer range')
    images = math.prod(2*int(r) + 1 for r in radii)
    metric = A.T @ A
    orthogonal = np.linalg.norm(metric - np.diag(np.diag(metric)), ord=np.inf) <= (
        16*np.finfo(float).eps*np.min(np.diag(metric)))
    if orthogonal:
        log_ratio, info = _orthogonal_sum(A, fractional, sigma, radii, chunk_size, tail_sigma, max_radius)
        nearest = np.rint(fractional)
        residual = np.linalg.norm((fractional - nearest) @ A.T, axis=1)
    else:
        frequency_radius = np.ceil(max(8., tail_sigma)*np.linalg.norm(A, axis=0)/(2*np.pi*sigma))
        fourier_images = math.prod(2*int(r) + 1 for r in frequency_radius)
        frequencies = None
        if fourier_images < images and fourier_images <= 1_000_000:
            k = _offsets(frequency_radius)
            first_nonzero = np.argmax(k != 0, axis=1)
            k = k[k[np.arange(len(k)), first_nonzero] > 0]
            reciprocal = np.linalg.solve(A.T, k.T).T
            coefficients = 2*np.exp(-2*np.pi**2*sigma**2*np.sum(reciprocal**2, axis=1))
            if coefficients.sum() < .5:
                frequencies = k
        selected = radii if frequencies is None else frequency_radius
        if max_radius is not None and np.max(selected) > max_radius:
            raise PeriodicBudgetError('Selected periodic lattice sum exceeds radius budget')
        if frequencies is None and images > 1_000_000:
            raise PeriodicBudgetError('Real-space lattice sum exceeds node budget')
        terms = images if frequencies is None else len(frequencies)
        chunk = min(chunk_size, max(1, 32*1024**2 // (max(1, terms)*8)))
        log_ratio = np.empty(len(fractional))
        residual = np.empty(len(fractional))
        nearest = np.empty_like(fractional)
        points = fractional @ A.T
        if frequencies is None:
            offsets = _offsets(radii)
            lattice = offsets @ A.T
            normalizer = np.linalg.slogdet(A)[1] - 1.5*np.log(2*np.pi*sigma**2)
            for start in range(0, len(points), chunk):
                stop = min(start + chunk, len(points))
                r2 = cdist(points[start:stop], lattice, metric='sqeuclidean')
                indices = np.argmin(r2, axis=1)
                residual[start:stop] = np.sqrt(r2[np.arange(stop-start), indices])
                nearest[start:stop] = offsets[indices]
                r2 *= -.5 / sigma**2
                log_ratio[start:stop] = normalizer + logsumexp(r2, axis=1)
            residual, nearest = _nearest(A, fractional, radii, residual, nearest)
        else:
            residual, nearest = _nearest(A, fractional, np.ones(3, dtype=int))
            for start in range(0, len(points), chunk):
                stop = min(start + chunk, len(points))
                phase = 2*np.pi*(fractional[start:stop] @ frequencies.T)
                log_ratio[start:stop] = np.log1p(np.cos(phase) @ coefficients)
        info = {'periodic_representation': 'real' if frequencies is None else 'fourier',
                'periodic_terms': terms, 'effective_score_chunk': chunk,
                'periodic_selected_radius_xyz': [int(r) for r in selected],
                'periodic_factorization': 'coupled_3d'}
    return log_ratio, residual, (center + nearest) @ transform.T, {
        **info, 'periodic_images': images, 'periodic_radius_xyz': [int(r) for r in radii],
        'periodic_basis_transform': transform.tolist(), 'nearest_lattice_point_certified': True,
        'periodic_truncation': 'finite Gaussian-tail neighborhood; no certified density error bound',
    }

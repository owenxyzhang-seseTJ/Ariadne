#!/usr/bin/env python3
"""Diagnose overlap information and rank supplied candidate observations.

This is an experimental-design calculator.  The caller supplies physically
valid prediction designs in a common intensity convention and their noise
covariances.  It neither generates scan geometry nor operates an instrument.

For calibrated unit pair sums, the information matrix is a weighted signless
graph Laplacian.  General correction factors require the matrix calculation;
the graph theorem must not be applied to arbitrary overlap coefficients.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular


def _design(design, variance=None, parameter_scales=None):
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError('Design must be a finite N by P matrix')
    if matrix.shape[1] < 1:
        raise ValueError('Design must have at least one intensity parameter')
    scales = (np.ones(matrix.shape[1]) if parameter_scales is None
              else np.asarray(parameter_scales, dtype=float))
    if scales.shape != (matrix.shape[1],) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Parameter scales must be finite, positive, and common to every design')
    scaled = matrix * scales
    if variance is None:
        return scaled, scales
    noise = np.asarray(variance, dtype=float)
    if noise.ndim == 0:
        noise = np.full(matrix.shape[0], noise.item())
    if not np.all(np.isfinite(noise)):
        raise ValueError('Observation covariance must be finite')
    if noise.shape == (matrix.shape[0],):
        if np.any(noise <= 0):
            raise ValueError('Observation variances must be positive')
        return scaled / np.sqrt(noise)[:, None], scales
    if noise.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError('Expected a variance vector or full observation covariance')
    if not np.allclose(noise, noise.T, rtol=1e-12, atol=0.):
        raise ValueError('Observation covariance must be symmetric')
    try:
        lower = cholesky(noise, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Observation covariance must be positive definite') from exc
    return solve_triangular(lower, scaled, lower=True, check_finite=False), scales


def _decomposition(matrix, threshold=None, rcond=1e-10):
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError('rcond must be finite and strictly between zero and one')
    # When N >= P, economic SVD already returns all P right directions.  For
    # N < P, full V is needed but full U then has only N by N entries.
    _, singular, right = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
    if threshold is None:
        threshold = rcond * singular[0] if len(singular) else 0.
    rank = int(np.count_nonzero(singular > threshold))
    retained = singular[:rank]
    log_pdet = float(2*np.log(retained).sum()) if rank else 0.
    return {'rank': rank, 'nullspace': right[rank:], 'singular': singular,
            'threshold': float(threshold), 'log_pdet': log_pdet}


def _information_rows(matrix):
    """QR keeps R.T@R == A.T@A using at most P design rows, without a Gram solve."""
    return np.linalg.qr(matrix, mode='r') if len(matrix) > matrix.shape[1] else matrix


def observability(design, variance=None, *, rcond=1e-10, parameter_scales=None):
    """Return the measured intensity subspace without adding a ridge prior.

    ``parameter_scales`` defines theta = diag(scales) z.  Singular values and
    selection criteria refer to z; physical null directions are mapped back
    to theta.  Keep the scales identical for all compared observations.
    """
    matrix, scales = _design(design, variance, parameter_scales)
    result = _decomposition(matrix, rcond=rcond)
    nullspace = result['nullspace']
    return {'rank': result['rank'], 'nullity': matrix.shape[1] - result['rank'],
            'singular_values': result['singular'].tolist(),
            'rank_threshold': result['threshold'],
            'nullspace_normalized_parameters': nullspace.tolist(),
            'nullspace_physical_parameters': (nullspace * scales).tolist(),
            'individual_identifiable': (np.linalg.norm(nullspace, axis=0) <= rcond).tolist(),
            'log_pseudodeterminant_information': result['log_pdet'],
            'parameter_scales': scales.tolist(),
            'scope': 'conditional linear observation model; no prior or ridge added'}


def unit_pair_sum_graph(number_of_components, pairs, weights=None):
    """Exact graph rank for rows measuring I_u + I_v, with positive weights.

    For each connected bipartite component there is one alternating null
    direction.  A connected component containing an odd cycle has full rank.
    Isolated vertices are bipartite components with one unobserved intensity.
    Rows with unequal LP/scale/partiality coefficients are outside this API.
    """
    count = int(number_of_components)
    if count < 1 or count != number_of_components:
        raise ValueError('Number of components must be a positive integer')
    raw = np.asarray(pairs)
    if raw.size == 0:
        edges = np.empty((0, 2), dtype=int)
    else:
        if raw.ndim != 2 or raw.shape[1] != 2 or not np.all(np.isfinite(raw)) or np.any(raw != np.rint(raw)):
            raise ValueError('Pairs must contain two integer component indices')
        edges = raw.astype(int)
    if np.any(edges < 0) or np.any(edges >= count) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError('Pairs require two distinct valid components')
    weights = np.ones(len(edges)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(edges),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError('Pair weights must be finite and strictly positive')
    neighbors = [[] for _ in range(count)]
    information = np.zeros((count, count))
    for (u, v), weight in zip(edges, weights):
        neighbors[u].append(v)
        neighbors[v].append(u)
        information[u, u] += weight
        information[v, v] += weight
        information[u, v] += weight
        information[v, u] += weight
    colors = np.full(count, -1, dtype=int)
    components, nullspace = [], []
    for start in range(count):
        if colors[start] != -1:
            continue
        colors[start] = 0
        queue, nodes, bipartite = [start], [], True
        while queue:
            u = queue.pop()
            nodes.append(u)
            for v in neighbors[u]:
                if colors[v] == -1:
                    colors[v] = 1 - colors[u]
                    queue.append(v)
                elif colors[v] == colors[u]:
                    bipartite = False
        nodes.sort()
        if bipartite:
            vector = np.zeros(count)
            vector[nodes] = np.where(colors[nodes] == 0, 1., -1.) / np.sqrt(len(nodes))
            nullspace.append(vector)
        components.append({'nodes': nodes, 'bipartite': bipartite,
                           'contains_odd_cycle': not bipartite,
                           'rank': len(nodes) - int(bipartite),
                           'alternating_groups': [[u for u in nodes if colors[u] == color]
                                                  for color in (0, 1)] if bipartite else None})
    return {'rank': count - len(nullspace), 'nullity': len(nullspace),
            'components': components,
            'nullspace': np.asarray(nullspace).reshape(-1, count).tolist(),
            'signless_laplacian': information.tolist(),
            'coefficient_convention': 'unit pair sums I_u + I_v only',
            'scope': 'exact algebraic rank for positive independent pair weights; general coefficients require SVD'}


def select_observations(current_design, candidates, variance=None, *, budget=1,
                        rcond=1e-10, parameter_scales=None):
    """Greedily choose from supplied candidate observation packages.

    Selection is lexicographic: numerical rank gain; strength of the weakest
    newly constrained null direction; then log pseudo-determinant information
    gain.  A single absolute singular-value threshold is fixed from the whole
    supplied pool, so the meaning of numerical rank cannot change mid-greedy.
    Parameter scales are fixed too.  No fabricated diagonal regularizer is
    used to make singular information matrices appear invertible.

    Candidate dictionaries contain ``id``, ``design`` and optional ``variance``
    (vector or covariance matrix).  Candidate and current noise blocks are
    assumed independent of one another; within-block correlation is supported.
    A budget counts candidate packages, not seconds or detector exposures.
    """
    if not np.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError('rcond must be finite and strictly between zero and one')
    if int(budget) != budget or budget < 0:
        raise ValueError('Observation budget must be a nonnegative integer')
    current, scales = _design(current_design, variance, parameter_scales)
    current = _information_rows(current)
    prepared, identifiers = [], set()
    for candidate in candidates:
        identifier = candidate.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('Every candidate needs a unique nonempty string id')
        design, _ = _design(candidate['design'], candidate.get('variance'), scales)
        if design.shape[1] != current.shape[1]:
            raise ValueError('Every candidate must use the same intensity parameters')
        identifiers.add(identifier)
        prepared.append((identifier, _information_rows(design)))
    pool = np.vstack([current] + [design for _, design in prepared])
    singular = np.linalg.svd(pool, compute_uv=False)
    threshold = float(rcond*singular[0]) if len(singular) else 0.
    initial = _decomposition(current, threshold, rcond)
    selected, steps = [], []
    remaining = list(range(len(prepared)))
    for _ in range(min(int(budget), len(prepared))):
        state = _decomposition(current, threshold, rcond)
        scores = []
        for index in remaining:
            identifier, candidate = prepared[index]
            combined = _decomposition(np.vstack((current, candidate)), threshold, rcond)
            rank_gain = combined['rank'] - state['rank']
            projection = candidate @ state['nullspace'].T
            projected_singular = np.linalg.svd(projection, compute_uv=False)
            # Rank priority is determined by the combined experiment.  The
            # projected spectrum is a conditional design heuristic, not a
            # covariance that treats already-measured directions as exact.
            weakest_new = float(projected_singular[rank_gain-1]) if rank_gain > 0 else 0.
            log_gain = combined['log_pdet'] - state['log_pdet']
            key = (rank_gain, weakest_new, log_gain)
            scores.append((key, index, {'id': identifier, 'rank_gain': rank_gain,
                           'resulting_rank': combined['rank'],
                           'projected_new_direction_singular_values': projected_singular.tolist(),
                           'weakest_new_direction_singular_value': weakest_new,
                           'log_pseudodeterminant_gain': log_gain}))
        best = max(scores, key=lambda item: item[0])
        if best[0][0] == 0 and best[0][2] <= 64*np.finfo(float).eps*max(1, current.shape[1]):
            break
        _, index, evidence = best
        selected.append(prepared[index][0])
        steps.append({'selected': evidence, 'alternatives': [score[2] for score in scores]})
        current = _information_rows(np.vstack((current, prepared[index][1])))
        remaining.remove(index)
    final = _decomposition(current, threshold, rcond)
    return {'selected_ids': selected, 'budget_packages': int(budget),
            'initial_rank': initial['rank'], 'final_rank': final['rank'],
            'remaining_nullity': current.shape[1] - final['rank'],
            'rank_threshold': threshold, 'parameter_scales': scales.tolist(),
            'rank_threshold_reference': 'largest singular value of the entire supplied candidate pool',
            'remaining_nullspace_physical_parameters': (final['nullspace']*scales).tolist(),
            'steps': steps,
            'criterion': 'rank gain, weakest new null-direction singular value, log pseudo-determinant gain',
            'information_compression': 'QR square-root information; at most P rows per block; no normal-matrix inversion',
            'scope': 'greedy simulated design selection from supplied candidates; independent noise blocks; no hardware control'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSON with current and candidates design blocks')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--budget', type=int, default=1, help='Number of candidate packages to select')
    parser.add_argument('--rcond', type=float, default=1e-10)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    current = data['current']
    result = select_observations(current['design'], data['candidates'], current.get('variance'),
                                 budget=args.budget, rcond=args.rcond,
                                 parameter_scales=data.get('parameter_scales'))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()

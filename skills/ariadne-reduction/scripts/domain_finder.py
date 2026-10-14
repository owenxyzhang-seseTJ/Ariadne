#!/usr/bin/env python3
"""Independent DAFi-inspired periodic-row candidate finder (experimental, untested).

Uses reciprocal coordinates only; does not integrate, correct, identify phases,
or certify a 3-D crystal lattice. Candidate subsets require DIALS indexing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


def load_points(path):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty reciprocal-space CSV')
    points = np.asarray([[float(r[k]) for k in ('q_x', 'q_y', 'q_z')] for r in rows])
    source_rows = np.asarray([int(r['source_row']) for r in rows], dtype=np.int64)
    if len(set(source_rows.tolist())) != len(source_rows):
        raise ValueError('source_row must be unique')
    if not np.all(np.isfinite(points)):
        raise ValueError('Nonfinite q coordinates; retain their ledger and exclude explicitly before search')
    return points, source_rows


def canonical(vectors):
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12
    unit = vectors[valid] / lengths[valid, None]
    dominant = np.argmax(np.abs(unit), axis=1)
    signs = np.sign(unit[np.arange(len(unit)), dominant])
    unit *= signs[:, None]
    return unit, lengths[valid], valid


def direction_keys(unit, epsilon):
    dominant = np.argmax(np.abs(unit), axis=1)
    other = np.asarray(((1, 2), (0, 2), (0, 1)))[dominant]
    a = unit[np.arange(len(unit)), other[:, 0]]
    b = unit[np.arange(len(unit)), other[:, 1]]
    return np.column_stack((dominant, np.floor((a + 1) / epsilon), np.floor((b + 1) / epsilon))).astype(int)


def best_interval_spacing(distances, relative_error, max_order):
    """Scan interval endpoints for the period supported by most distances.

    Distances are from one centre. An interval D/(k+eps)..D/(k-eps)
    encodes a possible kth lattice step; it is not a one-dimensional phase label.
    """
    distance = np.asarray(distances, dtype=float)[:, None]
    orders = np.arange(1, max_order + 1)[None, :]
    starts = (distance / (orders + relative_error)).ravel()
    ends = (distance / (orders - relative_error)).ravel()
    positions = np.r_[starts, ends]
    change = np.r_[np.ones(len(starts), dtype=int), -np.ones(len(ends), dtype=int)]
    order = np.argsort(positions, kind='stable')
    positions, change = positions[order], change[order]
    unique, first = np.unique(positions, return_index=True)
    if len(unique) < 2:
        return None, 0
    counts = np.cumsum(np.add.reduceat(change, first))[:-1]
    best = int(np.argmax(counts))
    return float((unique[best] + unique[best + 1]) / 2), int(counts[best])


def stratified_sample(points, budget, rng):
    """Balanced radius/octant sampling; reserve representation of sparse regions."""
    if len(points) <= budget:
        return np.arange(len(points))
    radius = np.linalg.norm(points, axis=1)
    shells = np.searchsorted(np.quantile(radius, [.25, .5, .75]), radius)
    octants = ((points[:, 0] >= 0).astype(int) + 2 * (points[:, 1] >= 0) + 4 * (points[:, 2] >= 0))
    strata = shells * 8 + octants
    labels = np.unique(strata)
    selected = []
    for label in rng.permutation(labels):
        indices = np.flatnonzero(strata == label)
        quota = min(len(indices), max(1, budget // len(labels)), budget - len(selected))
        if quota:
            selected.extend(rng.choice(indices, quota, replace=False).tolist())
    if len(selected) < budget:
        remaining = np.setdiff1d(np.arange(len(points)), selected)
        selected.extend(rng.choice(remaining, budget - len(selected), replace=False).tolist())
    return np.asarray(selected, dtype=int)


def directions(points, args, rng, method=None):
    votes = defaultdict(lambda: [0, np.zeros(3)])
    n = len(points)
    method = method or args.method
    if method == 'naive':
        total_pairs = n * (n - 1) // 2
        if total_pairs <= args.max_pairs:
            pairs = np.asarray(list(itertools.combinations(range(n), 2)), dtype=int)
        else:
            # Bounded, deduplicated unordered pairs; no N x N x 3 allocation.
            pairs = rng.integers(0, n, size=(args.max_pairs, 2))
            pairs.sort(axis=1)
            pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
        unit, _, _ = canonical(points[pairs[:, 1]] - points[pairs[:, 0]])
        keys = direction_keys(unit, args.direction_epsilon)
        labels, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.column_stack([np.bincount(inverse, weights=unit[:, j]) for j in range(3)])
        for key, count, vector in zip(labels, counts, sums):
            votes[tuple(key)] = [int(count), vector]
    else:
        centres = stratified_sample(points, min(n, args.max_centres), rng)
        for centre in centres:
            unit, distance, _ = canonical(points - points[centre])
            keys = direction_keys(unit, args.direction_epsilon)
            labels, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
            order = np.argsort(inverse, kind='stable')
            offsets = np.r_[0, np.cumsum(counts)]
            for group in np.flatnonzero(counts >= args.minimum_row - 1):
                key = tuple(labels[group])
                indices = order[offsets[group]:offsets[group + 1]]
                if len(indices) < args.minimum_row - 1:
                    continue
                _, support = best_interval_spacing(distance[indices], args.relative_step_error, args.max_order)
                if support >= args.minimum_row - 1:
                    entry = votes[key]
                    entry[0] += support
                    entry[1] += np.mean(unit[indices], axis=0) * support
    ranked = sorted(votes.values(), key=lambda x: x[0], reverse=True)
    result = []
    for support, vector in ranked:
        if np.linalg.norm(vector) < 1e-12:
            continue
        vector = vector / np.linalg.norm(vector)
        # Suppress neighbouring direction-bin aliases; preserve distinct directions.
        if any(abs(float(vector @ old[0])) > np.cos(2 * args.direction_epsilon) for old in result):
            continue
        result.append((vector, int(support)))
        if len(result) >= args.max_directions:
            break
    return result


def projected_rows(points, direction, tolerance):
    """Group projections around anchors without transitive chain merging."""
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    e1 = np.cross(direction, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    projected = np.column_stack((points @ e1, points @ e2))
    tree = cKDTree(projected)
    available = np.ones(len(points), dtype=bool)
    groups = []
    for anchor in range(len(points)):
        if not available[anchor]:
            continue
        near = np.asarray(tree.query_ball_point(projected[anchor], tolerance), dtype=int)
        near = near[available[near]]
        available[near] = False
        groups.append(near)
    return groups, points @ direction


def common_gap(gaps, tolerance):
    """Find the densest interval of adjacent row gaps with a sliding window."""
    gaps = np.sort(np.asarray(gaps, dtype=float))
    if not len(gaps):
        return None
    left = best_left = best_right = 0
    for right in range(len(gaps)):
        while gaps[right] - gaps[left] > 2 * tolerance:
            left += 1
        if right - left > best_right - best_left:
            best_left, best_right = left, right
    return float(np.median(gaps[best_left:best_right + 1]))


def periodic_subset(values, spacing, tolerance, minimum):
    """Largest consistent circular remainder cluster, allowing missed nodes."""
    if spacing <= 1e-12 or len(values) < minimum:
        return np.empty(0, dtype=int)
    tolerance = min(tolerance, .1 * spacing)
    remainders = np.mod(values, spacing)
    order = np.argsort(remainders)
    expanded = np.r_[remainders[order], remainders[order] + spacing]
    left = best_left = best_right = 0
    n = len(values)
    for right in range(2 * n):
        while expanded[right] - expanded[left] > 2 * tolerance or right - left + 1 > n:
            left += 1
        if left < n and right - left > best_right - best_left:
            best_left, best_right = left, right
    chosen = order[np.arange(best_left, best_right + 1) % n]
    if len(chosen) < minimum:
        return np.empty(0, dtype=int)
    phase = float(np.median(expanded[best_left:best_right + 1])) % spacing
    nodes = np.rint((values[chosen] - phase) / spacing).astype(int)
    if len(np.unique(nodes)) < minimum:
        return np.empty(0, dtype=int)
    return chosen


def group_for_direction(points, direction, args, given_spacing=None, cache=None):
    key = tuple(np.round(direction, 10))
    if cache is not None and key in cache:
        rows, shifts = cache[key]
    else:
        rows, shifts = projected_rows(points, direction, args.row_tolerance)
        if cache is not None:
            cache[key] = (rows, shifts)
    long_rows = [r for r in rows if len(r) >= args.minimum_row]
    if given_spacing is None:
        gaps = []
        for row in long_rows:
            diffs = np.diff(np.sort(shifts[row]))
            gaps.extend(diffs[diffs > 1e-12].tolist())
        base = common_gap(gaps, args.spacing_tolerance)
        if base is None:
            return None
        possible = [base / k for k in range(1, args.max_order + 1)]
    else:
        possible = [given_spacing]
    best = None
    for spacing in possible:
        selected_rows = []
        for row in long_rows:
            subset = periodic_subset(shifts[row], spacing, args.spacing_tolerance, args.minimum_row)
            if len(subset):
                selected_rows.append(row[subset])
        selected = np.concatenate(selected_rows) if selected_rows else np.empty(0, dtype=int)
        if best is None or len(selected) > len(best['indices']):
            best = {'indices': selected, 'rows': selected_rows, 'spacing': spacing,
                    'direction': direction}
    return best


def find_domains(args):
    q, source = load_points(args.input)
    if args.minimum_row < 3 or args.min_group < args.minimum_row or args.max_groups < 1 or args.restarts < 1:
        raise ValueError('Invalid row/group limits')
    for value in (args.direction_epsilon, args.row_tolerance, args.spacing_tolerance):
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Tolerances must be finite and positive')
    if not 0 < args.relative_step_error < 1 or args.max_order < 1:
        raise ValueError('Require 0 < relative-step-error < 1 and positive max-order')
    if min(args.max_pairs, args.max_centres, args.max_search_points, args.max_directions) < 1:
        raise ValueError('Search budgets must be positive')
    centre = np.mean(q, axis=0)
    scale = np.max(np.abs(q - centre), axis=0)
    scale[scale < 1e-12] = 1
    normal = (q - centre) / scale
    # Repeated observations of the same reciprocal node do not inflate row length.
    precision = min(args.row_tolerance, args.spacing_tolerance) / 4
    quantized = np.rint(normal / precision).astype(np.int64)
    _, representatives, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    unique_points = normal[representatives]
    rng = np.random.default_rng(args.seed)
    remaining = np.ones(len(unique_points), dtype=bool)
    accepted_search_exclusions = 0
    if args.accepted_assignments:
        ledger = json.loads(args.accepted_assignments.read_text())
        if ledger.get('schema') != 'scxrd-joint-assignment/1' or Path(ledger['source_csv']).resolve() != args.input.resolve():
            raise ValueError('Assignment ledger must refer to this exact q CSV')
        accepted = set(ledger['exclusive_source_rows'])
        flags = np.asarray([int(i) in accepted for i in source], dtype=int)
        counts = np.bincount(inverse, minlength=len(unique_points))
        accepted_counts = np.bincount(inverse, weights=flags, minlength=len(unique_points))
        remaining = accepted_counts != counts
        accepted_search_exclusions = int((~remaining).sum())
    candidates = []
    search_records = []
    pool = np.flatnonzero(remaining)
    pool_points = unique_points[pool]
    cache = {}
    methods = ('smart', 'naive') if args.method == 'hybrid' else (args.method,)
    proposals = []
    for iteration in range(args.restarts):
        if len(pool) < args.min_group:
            break
        sampled = pool[stratified_sample(pool_points, min(len(pool), args.max_search_points), rng)]
        sample = unique_points[sampled]
        directions_found = []
        for method in methods:
            directions_found.extend(directions(sample, args, rng, method=method))
        for direction, support in directions_found:
            group = group_for_direction(pool_points, direction, args, cache=cache)
            if group is None or len(group['indices']) < args.min_group:
                continue
            group['direction_support'] = support
            group['restart'] = iteration + 1
            proposals.append(group)
        search_records.append({'iteration': iteration + 1, 'unique_pool': len(pool),
                               'search_sample': len(sample), 'directions_considered': len(directions_found)})
    # No provisional group is peeled off before three-dimensional verification.
    proposals.sort(key=lambda g: (len(g['indices']), g['direction_support']), reverse=True)
    used = []
    for best in proposals:
        members_set = set(pool[best['indices']].tolist())
        duplicate = next((entry for entry in used if len(members_set & entry[0]) / len(members_set | entry[0]) > .9), None)
        if duplicate is not None:
            duplicate[1]['supporting_restarts'].add(best['restart'])
            continue
        if len(used) >= args.max_groups:
            continue
        new_indices = pool[best['indices']]
        # Full-pool revisit preserves competing membership at lattice intersections.
        expanded = group_for_direction(unique_points, best['direction'], args, best['spacing'])
        members = expanded['indices'] if expanded is not None else new_indices
        mask = np.zeros(len(unique_points), dtype=bool)
        mask[members] = True
        original_members = np.flatnonzero(mask[inverse])
        candidate = {'candidate_id': len(candidates) + 1, 'status': 'requires_3d_indexing',
                     'source_rows': source[original_members].tolist(),
                     'unique_nodes': int(len(members)), 'new_unique_nodes': int(len(new_indices)),
                     'periodic_rows': int(len(best['rows'])),
                     'direction_normalized': best['direction'].tolist(),
                     'step_normalized': float(best['spacing']),
                     'direction_vote_support': int(best['direction_support']),
                     'supporting_restarts': {best['restart']},
                     'cell': None, 'A': None, 'phase_identity': None}
        candidates.append(candidate)
        used.append((members_set, candidate))
    for candidate in candidates:
        candidate['supporting_restarts'] = sorted(candidate['supporting_restarts'])
    args.output.mkdir(parents=True, exist_ok=False)
    memberships = defaultdict(list)
    for candidate in candidates:
        for row in candidate['source_rows']:
            memberships[row].append(candidate['candidate_id'])
        wanted = set(candidate['source_rows'])
        selected = [i for i, row in enumerate(source) if int(row) in wanted]
        with (args.output / f'candidate_{candidate["candidate_id"]:03d}.csv').open('x', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('source_row', 'q_x', 'q_y', 'q_z'))
            for i in selected:
                writer.writerow((int(source[i]), *q[i]))
    with (args.output / 'assignments.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('source_row', 'candidate_ids', 'status'))
        for row in source:
            ids = memberships[int(row)]
            writer.writerow((int(row), ';'.join(map(str, ids)), 'ambiguous' if len(ids) > 1 else ('candidate' if ids else 'unassigned')))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('input', 'output')}
    result = {'schema': 'scxrd-domain-candidates/1', 'algorithm': 'independent periodic-row finder inspired by DAFi',
              'implementation_status': 'experimental_untested', 'source_csv': str(args.input.resolve()),
              'q_units': 'inverse_Angstrom_without_2pi', 'normalization': {'centre': centre.tolist(), 'scale': scale.tolist()},
              'parameters': config, 'input_rows': len(q), 'search_unique_nodes': len(unique_points),
              'deduplication': 'quantized search only; every original observation retained in outputs',
              'candidates': candidates, 'search_records': search_records,
              'accepted_search_exclusions': accepted_search_exclusions,
              'provisional_peeling': False,
              'unassigned_rows': sum(not memberships[int(r)] for r in source),
              'ambiguous_rows': sum(len(memberships[int(r)]) > 1 for r in source),
              'limitations': ['row periodicity does not certify a three-dimensional lattice',
                              'finite search budgets may miss weak domains',
                              'shared axes can combine different phases; independent indexing required',
                              'not an overlap-integration or HKLF5 generator']}
    (args.output / 'domains.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.output / 'domains.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='CSV with source_row,q_x,q_y,q_z; q is 1/A without 2pi')
    parser.add_argument('output', type=Path, help='New output directory')
    parser.add_argument('--method', choices=('smart', 'naive', 'hybrid'), default='hybrid')
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--accepted-assignments', type=Path)
    parser.add_argument('--max-groups', type=int, default=8)
    parser.add_argument('--minimum-row', type=int, default=4)
    parser.add_argument('--min-group', type=int, default=35)
    parser.add_argument('--direction-epsilon', type=float, default=.02)
    parser.add_argument('--row-tolerance', type=float, default=.004)
    parser.add_argument('--spacing-tolerance', type=float, default=.002)
    parser.add_argument('--relative-step-error', type=float, default=.02)
    parser.add_argument('--max-order', type=int, default=5)
    parser.add_argument('--max-centres', type=int, default=128)
    parser.add_argument('--max-pairs', type=int, default=200000)
    parser.add_argument('--max-search-points', type=int, default=12000)
    parser.add_argument('--max-directions', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        find_domains(args)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

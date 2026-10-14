#!/usr/bin/env python3
"""Accept three-dimensional indexing evidence, then jointly assign original spots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np


def write_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def accept_checks(args):
    models, rejected = [], []
    for path in args.checks:
        check = json.loads(path.read_text())
        if check.get('schema') != 'scxrd-domain-index-check/1':
            raise ValueError('Unsupported index check: ' + str(path))
        fraction = check.get('indexed_fraction_of_input_table')
        if fraction is None:
            raise ValueError('Index check must include the original candidate denominator')
        for model in check['crystal_models']:
            reasons = []
            if not model.get('three_independent_index_directions'):
                reasons.append('insufficient three-dimensional index coverage')
            if model.get('unique_hkl', 0) < args.min_unique:
                reasons.append('insufficient unique reflections')
            if fraction < args.min_fraction:
                reasons.append('low indexed fraction of input candidate')
            residual = model.get('rlp_residual_median_p95_invA')
            if not residual or residual[1] > args.max_p95:
                reasons.append('large reciprocal-space residual')
            A = np.asarray(model['A'], dtype=float).reshape(3, 3)
            if not np.all(np.isfinite(A)) or abs(np.linalg.det(A)) < 1e-15:
                reasons.append('invalid basis matrix')
            if reasons:
                rejected.append({'source_check': str(path.resolve()), 'crystal_model_id': model['crystal_model_id'], 'reasons': reasons})
                continue
            equivalent = None
            for previous in models:
                transform = np.linalg.solve(np.asarray(previous['A']), A)
                integer = np.rint(transform)
                if np.max(np.abs(transform - integer)) < 1e-3 and abs(abs(np.linalg.det(integer)) - 1) < 1e-6:
                    equivalent = previous
                    break
            if equivalent is not None:
                equivalent.setdefault('equivalent_check_records', []).append(str(path.resolve()))
                continue
            models.append({'model_id': len(models) + 1, 'accepted': True, 'A': A.tolist(),
                           'cell': model['cell'], 'source_check': str(path.resolve()),
                           'phase_identity': None, 'twin_classification': None,
                           'acceptance': 'configurable geometry gate; chemical identity and symmetry remain unverified'})
    write_new(args.output, {'schema': 'scxrd-accepted-models/1', 'models': models, 'rejected_checks': rejected,
              'policy': {'min_unique': args.min_unique, 'min_fraction': args.min_fraction, 'max_p95_invA': args.max_p95},
              'rollback': 'rejected candidates never remove their original spots from subsequent search'})


def assign(args):
    with args.points.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty point table')
    q = np.asarray([[float(row[k]) for k in ('q_x', 'q_y', 'q_z')] for row in rows])
    source = np.asarray([int(row['source_row']) for row in rows])
    if len(np.unique(source)) != len(source) or not np.all(np.isfinite(q)):
        raise ValueError('Invalid original point table')
    document = json.loads(args.models.read_text())
    if document.get('schema') != 'scxrd-accepted-models/1':
        raise ValueError('Unsupported accepted model schema')
    models = [m for m in document['models'] if m.get('accepted') is True]
    if args.q_tolerance <= 0 or args.chunk_size < 1 or args.score_margin < 0:
        raise ValueError('Invalid assignment settings')
    covariance_keys = ('q_cov_xx', 'q_cov_yy', 'q_cov_zz', 'q_cov_xy', 'q_cov_xz', 'q_cov_yz')
    covariances = None
    if all(key in rows[0] for key in covariance_keys):
        values = np.asarray([[float(row[key]) for key in covariance_keys] for row in rows])
        covariances = np.empty((len(rows), 3, 3))
        covariances[:, 0, 0], covariances[:, 1, 1], covariances[:, 2, 2] = values[:, 0], values[:, 1], values[:, 2]
        covariances[:, 0, 1] = covariances[:, 1, 0] = values[:, 3]
        covariances[:, 0, 2] = covariances[:, 2, 0] = values[:, 4]
        covariances[:, 1, 2] = covariances[:, 2, 1] = values[:, 5]
        if not np.all(np.isfinite(covariances)) or np.any(np.linalg.eigvalsh(covariances) <= 0):
            raise ValueError('Per-spot q covariance must be finite positive definite')
    args.output.mkdir(parents=True, exist_ok=False)
    exclusive, ambiguous, unassigned = [], [], []
    coverage = {str(m['model_id']): 0 for m in models}
    with (args.output / 'assignments.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('source_row', 'status', 'model_ids', 'best_score', 'score_gap'))
        for start in range(0, len(q), args.chunk_size):
            points = q[start:start + args.chunk_size]
            scores = np.full((len(points), len(models)), np.inf)
            for j, model in enumerate(models):
                A = np.asarray(model['A']).reshape(3, 3)
                fractional = np.linalg.solve(A, points.T).T
                h = np.rint(fractional)
                delta = points - h @ A.T
                allowed = np.any(h != 0, axis=1) & (np.max(np.abs(fractional - h), axis=1) <= args.max_fractional)
                if covariances is None:
                    score = np.sum(delta * delta, axis=1) / args.q_tolerance ** 2
                    allowed &= score <= 1
                else:
                    solve = np.linalg.solve(covariances[start:start + len(points)], delta[..., None])[..., 0]
                    score = np.einsum('ij,ij->i', delta, solve)
                    allowed &= score <= args.max_mahalanobis2
                scores[allowed, j] = score[allowed]
            for i, row_scores in enumerate(scores):
                row = int(source[start + i])
                valid = np.flatnonzero(np.isfinite(row_scores))
                if not len(valid):
                    unassigned.append(row)
                    writer.writerow((row, 'unassigned', '', '', ''))
                    continue
                order = valid[np.argsort(row_scores[valid])]
                gap = float(row_scores[order[1]] - row_scores[order[0]]) if len(order) > 1 else None
                is_exclusive = len(order) == 1 or gap >= args.score_margin
                ids = [models[j]['model_id'] for j in (order[:1] if is_exclusive else order)]
                if is_exclusive:
                    exclusive.append(row)
                    coverage[str(ids[0])] += 1
                else:
                    ambiguous.append(row)
                writer.writerow((row, 'exclusive' if is_exclusive else 'ambiguous', ';'.join(map(str, ids)),
                                 float(row_scores[order[0]]), '' if gap is None else gap))
    write_new(args.output / 'assignment.json', {'schema': 'scxrd-joint-assignment/1',
              'source_csv': str(args.points.resolve()), 'accepted_models': str(args.models.resolve()),
              'exclusive_source_rows': exclusive, 'ambiguous_source_rows': ambiguous, 'unassigned_source_rows': unassigned,
              'exclusive_coverage': coverage, 'covariance_used': covariances is not None,
              'settings': {'q_tolerance_invA': args.q_tolerance, 'max_fractional': args.max_fractional,
                           'score_margin': args.score_margin, 'max_mahalanobis2': args.max_mahalanobis2},
              'interpretation': 'scores are geometric distances, not calibrated membership probabilities',
              'raw_observations': 'retained; exclusive labels exclude search seeds only, not integration pixels'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('accept-checks')
    p.add_argument('output', type=Path)
    p.add_argument('checks', nargs='+', type=Path)
    p.add_argument('--min-unique', type=int, default=35)
    p.add_argument('--min-fraction', type=float, default=.6)
    p.add_argument('--max-p95', type=float, default=.003)
    p.set_defaults(func=accept_checks)
    p = sub.add_parser('assign')
    p.add_argument('points', type=Path)
    p.add_argument('models', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--q-tolerance', type=float, default=.003)
    p.add_argument('--max-fractional', type=float, default=.15)
    p.add_argument('--max-mahalanobis2', type=float, default=14.16)
    p.add_argument('--score-margin', type=float, default=1.)
    p.add_argument('--chunk-size', type=int, default=4096)
    p.set_defaults(func=assign)
    args = parser.parse_args()
    try:
        args.func(args)
        return 0
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

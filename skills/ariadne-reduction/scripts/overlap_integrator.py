#!/usr/bin/env python3
"""Pixel-level joint profile integration with nuisance-background elimination.

Fixed calibrated profiles; weighted least squares preserves signed intensities.
SVD determines which individual/total intensity contrasts are identifiable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import qr, solve_triangular
from constrained_pixels import fit_constrained


def background_basis(shape, mode):
    columns = [np.ones(int(np.prod(shape)))]
    if mode == 'plane':
        for axis, length in enumerate(shape):
            if length <= 1:
                continue
            coordinate = np.linspace(-1, 1, length)
            reshape = [1] * len(shape)
            reshape[axis] = length
            columns.append(np.broadcast_to(coordinate.reshape(reshape), shape).ravel())
    return np.column_stack(columns)


def solve_joint(data, variance, profiles, valid, background, rcond):
    y, v = data[valid], variance[valid]
    P, B = profiles[valid], background[valid]
    if len(y) <= P.shape[1] + B.shape[1] + 2:
        raise ValueError('Too few valid pixels for joint signal/background fit')
    weight = 1 / np.sqrt(v)
    Pw, Bw, yw = P * weight[:, None], B * weight[:, None], y * weight
    # QR removes the nuisance background without constructing an Npixel-square projector.
    Q, R = qr(Bw, mode='economic')
    if np.min(np.abs(np.diag(R))) <= np.max(np.abs(np.diag(R))) * 1e-12:
        raise ValueError('Background basis is rank deficient')
    design = Pw - Q @ (Q.T @ Pw)
    target = yw - Q @ (Q.T @ yw)
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales <= 1e-15):
        raise ValueError('Signal indistinguishable from background or fully masked')
    U, singular, Vh = np.linalg.svd(design / scales, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * rcond))
    if not rank:
        raise ValueError('No identifiable signal subspace')
    retained = Vh[:rank].T
    theta = retained @ ((U[:, :rank].T @ target) / singular[:rank])
    intensities = theta / scales
    covariance = ((retained / singular[:rank]) @ (retained / singular[:rank]).T) / scales[:, None] / scales[None, :]
    beta = solve_triangular(R, Q.T @ (yw - Pw @ intensities))
    fitted = profiles @ intensities + background @ beta
    chi2 = float(np.sum((y - fitted[valid]) ** 2 / v))
    dof = len(y) - rank - B.shape[1]
    return {'intensities_internal': intensities, 'covariance_identifiable_subspace': covariance,
            'nullspace': Vh[rank:], 'column_scales': scales, 'rank': rank,
            'singular_values': singular, 'background_coefficients': beta, 'fitted': fitted,
            'chi2': chi2, 'dof': dof, 'reduced_chi2': chi2 / dof if dof > 0 else None}


def contrast(result, vector, tolerance):
    scaled = vector / result['column_scales']
    null_component = np.linalg.norm(result['nullspace'] @ scaled)
    ratio = float(null_component / max(np.linalg.norm(scaled), 1e-300))
    if ratio > tolerance:
        return {'identifiable': False, 'nullspace_fraction': ratio, 'intensity': None, 'variance': None}
    intensity = float(vector @ result['intensities_internal'])
    variance = float(vector @ result['covariance_identifiable_subspace'] @ vector)
    return {'identifiable': True, 'nullspace_fraction': ratio, 'intensity': intensity,
            'variance': max(variance, 0.)}


def fit_patch(path, metadata, args):
    with np.load(path, allow_pickle=False) as archive:
        data = np.asarray(archive['data'], dtype=float)
        variance = np.asarray(archive['variance'], dtype=float)
        mask = np.asarray(archive['valid'], dtype=bool)
        templates = np.asarray(archive['profiles'], dtype=float)
    if data.ndim != 3 or variance.shape != data.shape or mask.shape != data.shape or templates.shape[1:] != data.shape:
        raise ValueError('Expected data/variance/valid ZYX and profiles KZYX')
    if not 1 <= len(templates) <= args.max_components:
        raise ValueError('Component budget exceeded; retain group for an explicitly larger budget')
    if len(metadata['contributors']) != len(templates):
        raise ValueError('Contributor/profile mismatch')
    if not np.all(np.isfinite(templates)) or np.any(templates < 0):
        raise ValueError('Profiles must be finite, nonnegative and use full-profile normalization')
    valid = mask.ravel() & np.isfinite(data.ravel()) & np.isfinite(variance.ravel()) & (variance.ravel() > 0)
    profiles = templates.reshape(len(templates), -1).T
    masses = profiles[valid].sum(axis=0)
    if np.any(masses < args.min_profile_mass) or np.any(masses > 1 + 1e-6):
        raise ValueError('Insufficient or invalid full-profile mass; expand patch or review mask/calibration')
    B = background_basis(data.shape, args.background)
    y, v = data.ravel(), variance.ravel().copy()
    iterations = 1
    result = solve_joint(y, v, profiles, valid, B, args.svd_rcond)
    noise = metadata.get('noise_model')
    if args.estimator != 'signed-wls':
        if args.irls:
            raise ValueError('Constrained estimators do not reuse signed-WLS IRLS updates')
        result = fit_constrained(y, v, profiles, valid, B, noise, args.estimator, args.svd_rcond)
    if args.irls and not noise:
        raise ValueError('IRLS requested without calibrated pixel-noise model')
    if args.irls and noise:
        gain, readout = float(noise['gain']), float(noise['readout_variance'])
        if not noise.get('source') or gain <= 0 or readout < 0:
            raise ValueError('IRLS requires calibrated gain/readout variance and source')
        for iterations in range(2, args.max_iterations + 1):
            next_v = gain * np.maximum(result['fitted'], 0) + readout
            next_v = np.maximum(next_v, float(noise.get('variance_floor', gain)))
            fitted = solve_joint(y, next_v, profiles, valid, B, args.svd_rcond)
            change = np.linalg.norm(fitted['fitted'][valid] - result['fitted'][valid]) / max(np.linalg.norm(result['fitted'][valid]), 1.)
            result = fitted
            if change < args.iteration_tolerance:
                break
    total = contrast(result, np.ones(len(templates)), args.estimability_tolerance)
    components = [contrast(result, np.eye(len(templates))[i], args.estimability_tolerance) for i in range(len(templates))]
    status = 'joint_components' if all(c['identifiable'] for c in components) else ('total_only' if total['identifiable'] else 'unresolved')
    quality = []
    if result.get('converged') is False:
        quality.append('constrained_fit_not_converged')
    if any(components[i]['identifiable'] for i in result.get('boundary_components', [])):
        quality.append('nonnegative_boundary_requires_profile_likelihood_intervals')
    if args.estimator != 'signed-wls' and total['identifiable'] and total['intensity'] <= 1e-7:
        quality.append('nonnegative_total_boundary_requires_profile_likelihood_intervals')
    whitened = profiles[valid] / np.sqrt(v[valid, None])
    gram = whitened.T @ whitened
    norms = np.sqrt(np.maximum(np.diag(gram), 1e-300))
    correlations = gram / norms[:, None] / norms[None, :]
    condition = float(result['singular_values'][0] / result['singular_values'][-1]) if result['rank'] == len(templates) else None
    if result['reduced_chi2'] is not None and result['reduced_chi2'] > args.max_reduced_chi2:
        quality.append('profile_or_background_misfit')
    if noise and args.irls and iterations >= args.max_iterations and change >= args.iteration_tolerance:
        quality.append('irls_not_converged')
    return {'schema': 'scxrd-pixel-group/1', 'observation_id': metadata['observation_id'],
            'status': status, 'quality_flags': quality, 'contributors': metadata['contributors'],
            'basis': metadata['basis'], 'corrected': False, 'intensity_semantics': 'detector_group_counts',
            'total': total, 'components': components,
            'covariance_identifiable_subspace': result['covariance_identifiable_subspace'].tolist(),
            'covariance_scope': result.get('covariance_scope', 'conditional on supplied fixed profiles, geometry and calibrated pixel-noise model; background uncertainty included'),
            'estimator': args.estimator, 'intensity_unit': 'calibrated_adu',
            'intensity_definition': 'full-profile amplitude; no further partiality division',
            'observed_components': [{'intensity': c['intensity'] * float(m) if c['identifiable'] else None,
                                     'partiality_valid_mass': float(m)} for c, m in zip(components, masses)],
            'boundary_components': result.get('boundary_components', []),
            'overlap_diagnostics': {'bbox': 'conservative ROI/background-halo connectivity only',
                                    'profile_weighted_correlations': correlations.tolist(),
                                    'design_condition_number': condition,
                                    'class': 'C' if condition is None else ('B' if condition > args.poor_condition else 'A'),
                                    'classification_threshold': 'configured heuristic; not benchmark calibrated'},
            'signal_rank': result['rank'], 'singular_values': result['singular_values'].tolist(),
            'profile_valid_masses': masses.tolist(), 'valid_pixels': int(valid.sum()),
            'background_coefficients': result['background_coefficients'].tolist(),
            'chi2': result['chi2'], 'dof': result['dof'], 'reduced_chi2': result['reduced_chi2'],
            'iterations': iterations, 'patch': str(path.resolve()),
            'provenance': metadata.get('provenance'), 'correction_required': True}


def integrate(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema') != 'scxrd-pixel-patches/1':
        raise ValueError('Unsupported pixel manifest')
    if not 0 < args.svd_rcond < 1 or args.max_iterations < 2 or args.min_profile_mass <= 0:
        raise ValueError('Invalid fit settings')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for group in manifest['groups']:
        path = (args.manifest.parent / group['patch']).resolve()
        try:
            result = fit_patch(path, group, args)
        except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
            result = {'schema': 'scxrd-pixel-group/1', 'observation_id': group['observation_id'],
                      'status': 'unresolved', 'error': str(exc), 'contributors': group.get('contributors', []),
                      'patch': str(path), 'corrected': False}
        results.append(result)
    with (args.output / 'integrated_groups.jsonl').open('x') as handle:
        for result in results:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
    summary = {'schema': 'scxrd-overlap-integration/1', 'implementation_status': 'experimental_untested',
               'input_manifest': str(args.manifest.resolve()), 'groups': len(results),
               'joint_components': sum(r['status'] == 'joint_components' for r in results),
               'total_only': sum(r['status'] == 'total_only' for r in results),
               'unresolved': sum(r['status'] == 'unresolved' for r in results),
               'settings': {k: v for k, v in vars(args).items() if k not in ('manifest', 'output')},
               'absorption_scaling': 'not performed; detector-count outputs cannot be exported directly as corrected HKLF5'}
    (args.output / 'integration.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(args.output / 'integrated_groups.jsonl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--background', choices=('constant', 'plane'), default='plane')
    p.add_argument('--estimator', choices=('signed-wls', 'nnls', 'poisson'), default='signed-wls')
    p.add_argument('--poor-condition', type=float, default=10000.)
    p.add_argument('--svd-rcond', type=float, default=1e-6)
    p.add_argument('--estimability-tolerance', type=float, default=1e-6)
    p.add_argument('--min-profile-mass', type=float, default=.95)
    p.add_argument('--max-reduced-chi2', type=float, default=3.)
    p.add_argument('--max-components', type=int, default=32)
    p.add_argument('--irls', action='store_true')
    p.add_argument('--max-iterations', type=int, default=12)
    p.add_argument('--iteration-tolerance', type=float, default=1e-5)
    args = p.parse_args()
    try:
        integrate(args)
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

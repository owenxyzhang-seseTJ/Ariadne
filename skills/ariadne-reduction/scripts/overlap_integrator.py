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
from information_modes import from_projected_svd, conditional_component_diagnostics
from orthogonal_compression import augmented_qr


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


def solve_joint(data, variance, profiles, valid, background, rcond, *, chunk_rows=None):
    """Solve signed WLS; large patches use a bounded-workspace QR reduction.

    None selects streaming above 8192 raw pixels, 0 requests the original
    dense path, and a positive integer forces the specified streaming chunk.
    Both paths retain the same objective, rank convention and output units.
    """
    if chunk_rows is None:
        chunk_rows = 4096 if len(data) > 8192 else 0
    if isinstance(chunk_rows, (bool, np.bool_)) or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 0:
        raise ValueError('chunk_rows must be a nonnegative integer or None')
    if chunk_rows:
        return _solve_streamed(data, variance, profiles, valid, background, rcond, chunk_rows)
    y, v = data[valid], variance[valid]
    P, B = profiles[valid], background[valid]
    if len(y) <= P.shape[1] + B.shape[1] + 2:
        raise ValueError('Too few valid pixels for joint signal/background fit')
    weight = 1 / np.sqrt(v)
    Pw, Bw, yw = P * weight[:, None], B * weight[:, None], y * weight
    # QR 消去背景，避免构造像素数平方大小的投影矩阵。
    Q, R = qr(Bw, mode='economic')
    return _solve_joint_from_qr(y, v, Pw, yw, profiles, valid, background, rcond, Q, R)


def _solve_streamed(data, variance, profiles, valid, background, rcond, chunk_rows):
    factor, pixels = augmented_qr(data, variance, profiles, background, valid, chunk_rows)
    nuisance, components = background.shape[1], profiles.shape[1]
    if pixels <= nuisance + components + 2:
        raise ValueError('Too few valid pixels for joint signal/background fit')
    rb = factor[:nuisance, :nuisance]
    if np.min(np.abs(np.diag(rb))) <= np.max(np.abs(np.diag(rb))) * 1e-12:
        raise ValueError('Background basis is rank deficient')
    # Triangularity has already eliminated the first nuisance columns.
    design = factor[nuisance:, nuisance:-1]
    target = factor[nuisance:, -1]
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales <= 1e-15):
        raise ValueError('Signal indistinguishable from background or fully masked')
    left, singular, right = np.linalg.svd(design / scales, full_matrices=False)
    modes = from_projected_svd(target, left, singular, right, scales, nuisance,
                               valid_pixels=pixels)
    modes['compression'] = {'method': 'streamed augmented Householder QR',
                            'rows': len(factor), 'chunk_rows': int(chunk_rows),
                            'original_valid_pixels': pixels}
    rank = int(np.count_nonzero(singular > singular[0] * rcond))
    if not rank:
        raise ValueError('No identifiable signal subspace')
    retained = right[:rank].T
    intensities = (retained @ ((left[:, :rank].T @ target) / singular[:rank])) / scales
    root = (retained / singular[:rank]) / scales[:, None]
    covariance = root @ root.T
    beta = solve_triangular(rb, factor[:nuisance, -1] - factor[:nuisance, nuisance:-1] @ intensities)
    fitted = profiles @ intensities + background @ beta
    # A short residual vector retains the discarded-data constant without the
    # cancellation inherent in y'y - b' (A'A)^-1 b.
    residual = target - design @ intensities
    chi2 = float(residual @ residual)
    dof = pixels - rank - nuisance
    return {'intensities_internal': intensities,
            'covariance_identifiable_subspace': covariance,
            'nullspace': right[rank:], 'column_scales': scales, 'rank': rank,
            'singular_values': singular, 'background_coefficients': beta,
            'fitted': fitted, 'chi2': chi2, 'dof': dof,
            'reduced_chi2': chi2 / dof if dof > 0 else None,
            'information_modes': modes}


def _solve_joint_from_qr(y, v, Pw, yw, profiles, valid, background, rcond, Q, R):
    """复用已加权数组与背景 QR，完成投影、信号 SVD 与背景回代。"""
    if len(y) <= Pw.shape[1] + background.shape[1] + 2:
        raise ValueError('Too few valid pixels for joint signal/background fit')
    if np.min(np.abs(np.diag(R))) <= np.max(np.abs(np.diag(R))) * 1e-12:
        raise ValueError('Background basis is rank deficient')
    design = Pw - Q @ (Q.T @ Pw)
    target = yw - Q @ (Q.T @ yw)
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales <= 1e-15):
        raise ValueError('Signal indistinguishable from background or fully masked')
    U, singular, Vh = np.linalg.svd(design / scales, full_matrices=False)
    information_modes = from_projected_svd(target, U, singular, Vh, scales, background.shape[1])
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
    dof = len(y) - rank - background.shape[1]
    return {'intensities_internal': intensities, 'covariance_identifiable_subspace': covariance,
            'nullspace': Vh[rank:], 'column_scales': scales, 'rank': rank,
            'singular_values': singular, 'background_coefficients': beta, 'fitted': fitted,
            'chi2': chi2, 'dof': dof, 'reduced_chi2': chi2 / dof if dof > 0 else None,
            'information_modes': information_modes}


def solve_position_joint(data, variance, profiles, valid, background, derivatives, rcond, tolerance, max_shift):
    """消去位置导数与背景，保留强度的完整边缘协方差。"""
    count = profiles.shape[1]
    y, v = data[valid], variance[valid]
    nuisance = np.column_stack((background, derivatives))
    weights = 1 / np.sqrt(v)
    weighted = nuisance[valid] * weights[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    nonzero = np.flatnonzero(norms > 1e-15)
    Q, R, pivots = qr(weighted[:, nonzero] / norms[nonzero], mode='economic', pivoting=True)
    diagonal = np.abs(np.diag(R))
    nuisance_rank = int(np.sum(diagonal > diagonal.max() * rcond))
    selected = nonzero[pivots[:nuisance_rank]]
    basis = nuisance[:, selected] / norms[selected]
    # 单独消去所有 nuisance 方向，避免不可辨识的位置参数制造强度信息。
    result = _solve_joint_from_qr(y, v, profiles[valid] * weights[:, None], y * weights,
                                  profiles, valid, basis, rcond,
                                  Q[:, :nuisance_rank], R[:nuisance_rank, :nuisance_rank])
    coefficients = np.zeros(nuisance.shape[1])
    coefficients[selected] = result['background_coefficients'] / norms[selected]
    # 用完整设计检验每个 I*delta 是否可辨识，不发布任意零空间分解。
    complete = np.column_stack((profiles, derivatives))
    active = np.flatnonzero(np.linalg.norm(complete[valid] * weights[:, None], axis=0) > 1e-15)
    full = solve_joint(data, variance, complete[:, active], valid, background, rcond)

    def moment_contrast(index):
        location = np.flatnonzero(active == index)
        if not len(location):
            return {'identifiable': False}
        return contrast(full, np.eye(len(active))[location[0]], tolerance)

    shifts, flags = [], []
    for i in range(count):
        amplitude = moment_contrast(i)
        axes = []
        for axis in range(3):
            j = count + 3 * i + axis
            moment = moment_contrast(j)
            if not amplitude['identifiable'] or not moment['identifiable'] or abs(amplitude['intensity'] or 0) <= 1e-8:
                axes.append({'identifiable': False, 'shift_px': None, 'variance_px2': None})
                continue
            value = moment['intensity'] / amplitude['intensity']
            gradient = np.zeros(complete.shape[1])
            gradient[i], gradient[j] = -value / amplitude['intensity'], 1 / amplitude['intensity']
            error = float(gradient[active] @ full['covariance_identifiable_subspace'] @ gradient[active])
            axes.append({'identifiable': True, 'shift_px': float(value), 'variance_px2': max(error, 0.)})
            if abs(value) > max_shift:
                flags.append('position_linearization_limit_exceeded')
        shifts.append(axes)
    result['background_coefficients'] = coefficients[:background.shape[1]]
    result['position_model'] = {'method': 'first_order_independent_component_xyz_nuisance',
                                'shifts_xyz': shifts, 'max_linear_shift_px': max_shift,
                                'nuisance_rank': nuisance_rank, 'flags': sorted(set(flags)),
                                'scope': 'local linearization; intensity covariance marginalizes fitted XYZ moments and background; not instrument calibration covariance'}
    result['covariance_scope'] = 'linearized profile-position and background nuisance marginalized; conditional on supplied profile shape and calibrated noise'
    return result


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
        position_fit = getattr(args, 'fit_positions', False)
        derivatives = None
        if position_fit:
            if 'position_derivatives_xyz' not in archive:
                raise ValueError('Position fit requires patches built with --position-derivatives')
            derivatives = np.asarray(archive['position_derivatives_xyz'], dtype=float)
            if derivatives.shape != (len(templates), 3, *data.shape) or not np.all(np.isfinite(derivatives)):
                raise ValueError('Expected finite position derivatives K,XYZ,Z,Y,X')
            derivatives = derivatives.reshape(3 * len(templates), -1).T
            if args.estimator != 'signed-wls':
                raise ValueError('Position nuisance fitting currently requires signed-wls')
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
    def solve(variance):
        if position_fit:
            return solve_position_joint(y, variance, profiles, valid, B, derivatives, args.svd_rcond,
                                        args.estimability_tolerance, args.max_position_shift)
        return solve_joint(y, variance, profiles, valid, B, args.svd_rcond)

    result = solve(v)
    noise = metadata.get('noise_model')
    if args.estimator != 'signed-wls':
        if args.irls:
            raise ValueError('Constrained estimators do not reuse signed-WLS IRLS updates')
        result = fit_constrained(y, v, profiles, valid, B, noise, args.estimator, args.svd_rcond,
                                 confidence_level=getattr(args, 'confidence_level', .95))
    if args.irls and not noise:
        raise ValueError('IRLS requested without calibrated pixel-noise model')
    if args.irls and noise:
        gain, readout = float(noise['gain']), float(noise['readout_variance'])
        if not noise.get('source') or gain <= 0 or readout < 0:
            raise ValueError('IRLS requires calibrated gain/readout variance and source')
        for iterations in range(2, args.max_iterations + 1):
            next_v = gain * np.maximum(result['fitted'], 0) + readout
            next_v = np.maximum(next_v, float(noise.get('variance_floor', gain)))
            fitted = solve(next_v)
            v = next_v
            change = np.linalg.norm(fitted['fitted'][valid] - result['fitted'][valid]) / max(np.linalg.norm(result['fitted'][valid]), 1.)
            result = fitted
            if change < args.iteration_tolerance:
                break
    total = contrast(result, np.ones(len(templates)), args.estimability_tolerance)
    components = [contrast(result, np.eye(len(templates))[i], args.estimability_tolerance) for i in range(len(templates))]
    status = 'joint_components' if all(c['identifiable'] for c in components) else ('total_only' if total['identifiable'] else 'unresolved')
    quality = list(result.get('position_model', {}).get('flags', []))
    intervals = result.get('profile_intervals')
    total_interval = result.get('total_interval')
    if intervals is not None:
        for component, interval in zip(components, intervals):
            component['profile_likelihood_interval'] = interval
        total['profile_likelihood_interval'] = total_interval
        if any(c['identifiable'] and interval['status'] != 'ok' for c, interval in zip(components, intervals)):
            quality.append('profile_likelihood_interval_failed')
        if total['identifiable'] and total_interval['status'] != 'ok':
            quality.append('total_profile_likelihood_interval_failed')
    if result.get('converged') is False:
        quality.append('constrained_fit_not_converged')
    likelihood_diagnostics = result.get('likelihood_diagnostics') or {}
    covariance_usable = likelihood_diagnostics.get('covariance_estimate_usable', True)
    if not covariance_usable:
        quality.append('likelihood_curvature_unresolved_use_profile_intervals')
        # An internal fallback metric must never masquerade as an observed
        # information error bar.  Exact profile intervals remain available.
        for measurement in [total, *components]:
            measurement['variance'] = None
    if any(components[i]['identifiable'] for i in result.get('boundary_components', [])):
        quality.append('nonnegative_boundary_not_representable_by_gaussian_sigma')
    if args.estimator != 'signed-wls' and total['identifiable'] and total['intensity'] <= 1e-7:
        quality.append('nonnegative_total_boundary_not_representable_by_gaussian_sigma')
    whitened = profiles[valid] / np.sqrt(v[valid, None])
    gram = whitened.T @ whitened
    norms = np.sqrt(np.maximum(np.diag(gram), 1e-300))
    correlations = gram / norms[:, None] / norms[None, :]
    condition = float(result['singular_values'][0] / result['singular_values'][-1]) if result['rank'] == len(templates) else None
    if result['reduced_chi2'] is not None and result['reduced_chi2'] > args.max_reduced_chi2:
        quality.append('profile_or_background_misfit')
    if noise and args.irls and iterations >= args.max_iterations and change >= args.iteration_tolerance:
        quality.append('irls_not_converged')
    modes = result.get('information_modes')
    if modes is not None:
        modes['variance_model'] = 'irls_plugin' if args.irls else 'fixed'
        modes['variance_source'] = metadata.get('variance_source', 'supplied pixel variance; external calibration/estimation assumptions apply')
        modes['pixel_ownership'] = metadata.get('pixel_ownership', 'must be verified independent of other observation regions')
    return {'schema': 'scxrd-pixel-group/1', 'observation_id': metadata['observation_id'],
            'status': status, 'quality_flags': quality, 'contributors': metadata['contributors'],
            'basis': metadata['basis'], 'corrected': False, 'intensity_semantics': 'detector_group_counts',
            'total': total, 'components': components,
            'covariance_identifiable_subspace': result['covariance_identifiable_subspace'].tolist() if covariance_usable else None,
            'covariance_estimate_usable': bool(covariance_usable),
            'covariance_scope': result.get('covariance_scope', 'conditional on supplied fixed profiles, geometry and calibrated pixel-noise model; background uncertainty included'),
            'estimator': args.estimator, 'intensity_unit': 'calibrated_adu',
            'intensity_definition': 'full-profile amplitude; no further partiality division',
            'observed_components': [{'intensity': c['intensity'] * float(m) if c['identifiable'] else None,
                                     'partiality_valid_mass': float(m)} for c, m in zip(components, masses)],
            'boundary_components': result.get('boundary_components', []),
            'position_model': result.get('position_model'),
            'likelihood_diagnostics': result.get('likelihood_diagnostics'),
            'information_modes': modes,
            'weak_signal_diagnostics': conditional_component_diagnostics(result),
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
    if (not 0 < args.svd_rcond < 1 or args.max_iterations < 2 or args.min_profile_mass <= 0
            or not np.isfinite(args.max_position_shift) or not 0 < args.max_position_shift <= .5):
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
    p.add_argument('--estimator', choices=('signed-wls', 'nnls', 'poisson', 'poisson-gaussian'), default='signed-wls')
    p.add_argument('--confidence-level', type=float, default=.95)
    p.add_argument('--poor-condition', type=float, default=10000.)
    p.add_argument('--svd-rcond', type=float, default=1e-6)
    p.add_argument('--estimability-tolerance', type=float, default=1e-6)
    p.add_argument('--min-profile-mass', type=float, default=.95)
    p.add_argument('--max-reduced-chi2', type=float, default=3.)
    p.add_argument('--max-components', type=int, default=32)
    p.add_argument('--fit-positions', action='store_true', help='Marginalize first-order XYZ position nuisance in signed-WLS')
    p.add_argument('--max-position-shift', type=float, default=.3, help='Flag fitted shifts outside the local linearization range (pixels)')
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

#!/usr/bin/env python3
"""从复合观测联合估计扫描尺度与共同方向吸收，保留等效约束和误差相关性。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from composite_scaling import fit_global


def equivalent_hkl(hkl, space_group, anomalous):
    from cctbx import crystal, miller
    from cctbx.array_family import flex
    symmetry = crystal.symmetry(space_group_symbol=space_group)
    index = miller.set(symmetry, flex.miller_index([tuple(hkl)]), anomalous_flag=anomalous)
    return list(index.map_to_asu().indices()[0])


def harmonics(direction):
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or not np.isclose(np.linalg.norm(direction), 1., atol=1e-6):
        raise ValueError('Absorption directions must be unit XYZ vectors in the declared crystal frame')
    x, y, z = direction
    return np.array([x*y, x*z, y*z, x*x-y*y, 3*z*z-1])


def information_observation(record):
    """Validate a fixed-Gaussian observable-mode observation, without deconvolution.

    Conditional pixel likelihood is ||z - H a||^2_C + a known constant.  An
    unresolved component vector is allowed; z, not an arbitrary split, is the
    observation passed to the global fit.  Plug-in IRLS modes retain the legacy
    path and are never labelled an exact fixed-variance compression.
    """
    modes = record.get('information_modes')
    if modes is None:
        return None
    if not isinstance(modes, dict):
        raise ValueError('Information modes must be an object')
    if modes.get('variance_model') != 'fixed':
        return None
    if record.get('estimator') not in (None, 'signed-wls'):
        raise ValueError('Fixed Gaussian information modes require signed-wls')
    if modes.get('schema') != 'ariadne-information-modes/1':
        raise ValueError('Unsupported information-mode schema')
    k = len(record['contributors'])
    rank = modes.get('rank')
    if type(rank) is not int or not 1 <= rank <= k:
        raise ValueError('Information rank must be an integer in 1..contributors')
    arrays = []
    for key in ('y', 'design', 'covariance'):
        if np.iscomplexobj(modes.get(key)):
            raise ValueError('Information modes must be real')
        value = np.asarray(modes.get(key), dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError('Information modes must be finite')
        arrays.append(value)
    y, design, covariance = arrays
    if y.shape != (rank,) or design.shape != (rank, k) or covariance.shape != (rank, rank):
        raise ValueError('Information-mode dimensions must agree with rank and contributors')
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.):
        raise ValueError('Information covariance must be symmetric positive definite')
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Information covariance must be symmetric positive definite') from exc
    constant = modes.get('pixel_chi2_constant')
    if (isinstance(constant, bool) or not np.isscalar(constant)
            or not np.isfinite(constant) or constant < 0):
        raise ValueError('Information modes require a finite nonnegative pixel chi-square constant')
    if not modes.get('likelihood_scope'):
        raise ValueError('Information modes require a conditional likelihood scope')
    return y, covariance, design


def _legacy_observation_map(block):
    """Map mode noise to an already identifiable legacy measured contrast.

    This does not infer missing components.  The original conditional covariance
    defines its original linear estimator, T = C_I H.T C_z^-1.  Consistency is
    checked before publishing a correction for that measured quantity.
    """
    record = block['record']
    k = len(record['contributors'])
    if record['status'] not in ('joint_components', 'total_only'):
        return None
    component_cov = np.asarray(record.get('covariance_identifiable_subspace'), float)
    if component_cov.shape != (k, k) or not np.all(np.isfinite(component_cov)):
        return None
    mapping = np.linalg.solve(block['covariance'], block['design'] @ component_cov).T
    if record['status'] == 'joint_components':
        observed = np.asarray([c['intensity'] for c in record['components']], float)
        covariance = component_cov
    else:
        mapping = mapping.sum(axis=0, keepdims=True)
        observed = np.asarray([record['total']['intensity']], float)
        covariance = np.asarray([[record['total']['variance']]], float)
    # Rounding may differ between QR/SVD representations.  Acceptance requires
    # agreement on the measurement and its noise, not agreement with a fit.
    if (not np.allclose(mapping @ block['y'], observed, rtol=1e-7, atol=1e-9)
            or not np.allclose(mapping @ block['covariance'] @ mapping.T,
                               covariance, rtol=1e-7, atol=1e-9)):
        return None
    return mapping


def prepare(records, config):
    if config.get('schema') != 'ariadne-global-scaling/1':
        raise ValueError('Unsupported global scaling configuration')
    for key in ('basis', 'geometry_provenance', 'lp_provenance', 'symmetry_provenance', 'integration_provenance'):
        if not config.get(key):
            raise ValueError('Missing evidence: ' + key)
    if config.get('complete_contributor_audit') is not True:
        raise ValueError('Complete contributor audit required')
    if config.get('direction_frame') != 'crystal_cartesian_common_setting':
        raise ValueError('Require incident/outgoing directions in crystal_cartesian_common_setting')
    order = config.get('scan_order', 1)
    absorption = config.get('absorption_degree', 2)
    if type(order) is not int or not 0 <= order <= 2 or absorption not in (0, 2):
        raise ValueError('Supported scan_order=0..2 and absorption_degree=0 or 2')
    candidates, exclusions, names = [], [], set()
    for record in records:
        name = record.get('observation_id')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Require unique observation IDs')
        names.add(name)
        if record.get('quality_flags'):
            exclusions.append({'observation_id': name, 'reason': 'unresolved_or_quality_flag'})
            continue
        modes = information_observation(record) if record.get('information_modes') else None
        if record.get('status') not in ('joint_components', 'total_only', 'unresolved') or (modes is None
                and record.get('status') not in ('joint_components', 'total_only')):
            exclusions.append({'observation_id': name, 'reason': 'unresolved_or_quality_flag'})
            continue
        if record.get('basis') != config['basis'] or record.get('corrected') is not False:
            raise ValueError('Global scaling requires matching-basis uncorrected observations')
        if record.get('intensity_definition') != 'full-profile amplitude; no further partiality division':
            raise ValueError('Require full-profile amplitudes; no additional partiality division')
        contributors = record['contributors']
        meta = config.get('groups', {}).get(name)
        if not isinstance(meta, list) or len(meta) != len(contributors):
            raise ValueError('Missing ordered contributor geometry: ' + name)
        identities = []
        for component, item in zip(contributors, meta):
            if item.get('domain') != component['domain'] or item.get('hkl') != component['hkl']:
                raise ValueError('Geometry/contributor identity mismatch: ' + name)
            domain = config['domains'][str(component['domain'])]
            if type(domain.get('anomalous')) is not bool or not domain.get('space_group'):
                raise ValueError('Declare space group and anomalous policy per domain')
            if not isinstance(item.get('scan_id'), str) or not item['scan_id']:
                raise ValueError('Require raw scan identifier')
            if not np.isfinite(item['phi_rad']) or not np.isfinite(item['lp_forward']) or item['lp_forward'] <= 0:
                raise ValueError('Require finite scan radians and positive known LP forward multiplier')
            hkl = component['hkl']
            if not isinstance(hkl, list) or len(hkl) != 3 or any(type(x) is not int for x in hkl) or hkl == [0, 0, 0]:
                raise ValueError('Invalid HKL')
            identities.append(json.dumps([component['domain'], *equivalent_hkl(hkl, domain['space_group'], domain['anomalous'])]))
        if modes is not None:
            y, covariance, design = modes
            representation = 'fixed_gaussian_information_modes'
        elif record['status'] == 'joint_components':
            representation = 'components'
            y = np.array([c['intensity'] for c in record['components']], float)
            covariance = np.array(record['covariance_identifiable_subspace'], float)
            design = np.eye(len(contributors))
        else:
            representation = 'total'
            y = np.array([record['total']['intensity']], float)
            covariance = np.array([[record['total']['variance']]], float)
            design = np.ones((1, len(contributors)))
        candidates.append((record, meta, identities, y, covariance, design, representation))
    if not candidates:
        raise ValueError('No usable observations')
    scans = sorted({m['scan_id'] for _, meta, *_ in candidates for m in meta})
    ranges = {}
    for scan in scans:
        phi = [m['phi_rad'] for _, meta, *_ in candidates for m in meta if m['scan_id'] == scan]
        ranges[scan] = [float(min(phi)), float(max(phi))]
    labels = ['scan_offset:' + scan for scan in scans[1:]]
    labels += [f'scan:{scan}:power:{power}' for scan in scans for power in range(1, order+1)]
    labels += [f'absorption_l2:{j}' for j in range(5)] if absorption else []
    blocks = []
    for record, meta, ids, y, covariance, design, representation in candidates:
        features = []
        for item in meta:
            scan = item['scan_id']
            row = [float(scan == s) for s in scans[1:]]
            lo, hi = ranges[scan]
            if order and hi == lo:
                raise ValueError('Insufficient angular coverage for requested scan scale model')
            coordinate = 2*(item['phi_rad']-lo)/(hi-lo)-1 if hi > lo else 0.
            row += [coordinate**power if scan == s else 0. for s in scans for power in range(1, order+1)]
            if absorption:
                row += list(harmonics(item['incident_unit_xyz']) + harmonics(item['outgoing_unit_xyz']))
            features.append(row)
        blocks.append({'observation_id': record['observation_id'], 'y': y, 'covariance': covariance,
                       'design': design, 'equivalence_ids': ids, 'features': np.array(features).reshape(len(ids), len(labels)),
                       'known_factors': np.array([m['lp_forward'] for m in meta]), 'record': record,
                       'observation_representation': representation})
    all_features = np.vstack([b['features'] for b in blocks])
    centre = all_features.mean(axis=0)
    for block in blocks:
        block['features'] = block['features'] - centre
    return blocks, labels, centre, ranges, exclusions


def scale(args):
    records = [json.loads(line) for line in args.integrated.read_text().splitlines() if line.strip()]
    config = json.loads(args.geometry.read_text())
    blocks, labels, centre, ranges, excluded = prepare(records, config)
    fit = fit_global(blocks, max_iterations=args.max_iterations, tolerance=args.tolerance,
                     rcond=args.rcond, max_parameters=args.max_parameters, max_seconds=args.max_seconds,
                     optimizer=getattr(args, 'optimizer', 'alternating'))
    args.output.mkdir(parents=True, exist_ok=False)
    # 数值内核结果转换为可追溯主记录；不同组的相关性由联合参数协方差保留。
    serial = json.loads(json.dumps(fit, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item(), allow_nan=False))
    serial.update(schema='ariadne-global-scale-fit/1', feature_names=labels, feature_centre=centre.tolist(),
                  scan_ranges_rad=ranges, config=str(args.geometry.resolve()), integrated=str(args.integrated.resolve()),
                  excluded_observations=excluded,
                  scale_gauge='mean contributor log scale/absorption is zero; no independent domain normalization',
                  absorption_model='shared even degree-2 directional log response; empirical relative transmission, not absolute mu or crystal shape',
                  uncertainty_scope='local joint Gaussian linearization conditional on supplied geometry, LP, symmetry, feature basis and independent input groups')
    mode_blocks = [b for b in blocks if b['observation_representation'] == 'fixed_gaussian_information_modes']
    serial['observation_representations'] = {b['observation_id']: b['observation_representation'] for b in blocks}
    serial['information_mode_groups'] = len(mode_blocks)
    serial['pixel_chi2_constant'] = sum(b['record']['information_modes']['pixel_chi2_constant'] for b in mode_blocks)
    if fit['chi2'] is not None:
        serial['pixel_profiled_chi2'] = fit['chi2'] + serial['pixel_chi2_constant']
    discarded_dof = 0
    complete_counts = True
    for block in mode_blocks:
        mode = block['record']['information_modes']
        pixels, nuisance, rank = mode.get('valid_pixels'), mode.get('nuisance_rank'), mode['rank']
        if (type(pixels) is not int or type(nuisance) is not int or nuisance < 0
                or pixels < nuisance + rank):
            complete_counts = False
            break
        discarded_dof += pixels - nuisance - rank
    if complete_counts and fit['dof'] is not None:
        serial['pixel_profiled_dof'] = fit['dof'] + discarded_dof
        serial['pixel_profiled_reduced_chi2'] = (serial['pixel_profiled_chi2'] / serial['pixel_profiled_dof']
                                               if serial['pixel_profiled_dof'] > 0 else None)
    else:
        serial['pixel_profiled_dof'] = serial['pixel_profiled_reduced_chi2'] = None
    serial['pixel_profiled_scope'] = ('all pixel-mode groups; background/position nuisance eliminated'
                                     if len(mode_blocks) == len(blocks) else
                                     'mixed observations: add pixel constants and eliminated residual dimensions only for fixed-mode groups; legacy blocks retain their measurement-domain rows')
    serial['information_mode_scope'] = 'conditional fixed-Gaussian pixel modes; individual unresolved components are not exported'
    write_json(args.output / 'scaling.json', serial)
    if fit['status'] != 'converged':
        print(args.output / 'scaling.json')
        return
    corrections = make_corrections(blocks, config, fit, args)
    write_json(args.output / 'corrections.json', corrections)
    from correct_overlap_groups import correct
    from types import SimpleNamespace
    correct(SimpleNamespace(integrated=args.integrated, corrections=args.output / 'corrections.json', output=args.output / 'corrected'))
    print(args.output / 'scaling.json')


def make_corrections(blocks, config, fit, args):
    beta = np.asarray(fit['beta'])
    covariance = np.asarray(fit['joint_covariance'])
    n_beta = len(beta)
    beta_cov = covariance[-n_beta:, -n_beta:] if n_beta else np.zeros((0, 0))
    groups = {}
    for block, cross in zip(blocks, fit['block_parameter_data_covariance']):
        features = block['features']
        factors = 1 / (block['known_factors'] * np.exp(features @ beta))
        derivative = -factors[:, None] * features
        factor_covariance = derivative @ beta_cov @ derivative.T
        beta_data = np.asarray(cross)[-n_beta:] if n_beta else np.zeros((0, len(block['y'])))
        if block.get('observation_representation') == 'fixed_gaussian_information_modes':
            legacy_map = _legacy_observation_map(block)
            if legacy_map is None:
                continue
            beta_data = beta_data @ legacy_map.T
        data_factor = beta_data.T @ derivative.T
        item = {'factors': factors.tolist(), 'factor_covariance': factor_covariance.tolist(),
                'estimated_from_same_intensities': True,
                'partiality_policy': 'already_full_profile_amplitude'}
        if block['record']['status'] == 'joint_components':
            item['intensity_factor_cross_covariance'] = data_factor.tolist()
        else:
            # 不可测分量不使用等效反射的预测拆分替代；仅共同随机校正可直接作用于总量。
            if (not np.allclose(factors, factors[0], rtol=1e-8, atol=0)
                    or not np.allclose(derivative, derivative[0], rtol=1e-8, atol=1e-12)):
                continue
            item['total_factor_cross_covariance'] = float(data_factor[0, 0])
        groups[block['observation_id']] = item
    return {'schema': 'scxrd-group-corrections/1', 'basis': config['basis'],
            'domain_count': len(config['domains']), 'groups': groups,
            'complete_contributor_audit': True, 'contains_lp_scaling_absorption': True,
            'scale_convention': 'common_across_domains_keep_twin_fraction',
            'phase_relation': config.get('phase_relation', 'unknown'),
            'phase_evidence_provenance': config.get('phase_evidence_provenance'),
            'integration_provenance': str(args.integrated.resolve()),
            'correction_provenance': {'global_fit': str((args.output / 'scaling.json').resolve()),
                                      'config': str(args.geometry.resolve()),
                                      'model': 'shared log scale/directional response; local joint Gaussian covariance'},
            'global_parameter_covariance': str((args.output / 'scaling.json').resolve()),
            'fit_only_observations': [b['observation_id'] for b in blocks if b['observation_id'] not in groups],
            'fit_only_reason': 'unresolved information modes, unverified measured-contrast map, or total-only with unequal corrections; retained in global fit without deconvolution or HKLF export'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('integrated', type=Path)
    p.add_argument('geometry', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--max-iterations', type=int, default=100)
    p.add_argument('--tolerance', type=float, default=1e-8)
    p.add_argument('--rcond', type=float, default=1e-9)
    p.add_argument('--max-parameters', type=int, default=500)
    p.add_argument('--max-seconds', type=float, default=300.)
    p.add_argument('--optimizer', choices=('alternating', 'profiled'), default='alternating',
                   help='profiled eliminates signed intensities at every scale trial and uses the exact profile Jacobian')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Output directory already exists')
    scale(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""从复合观测联合估计扫描尺度与共同方向吸收，保留等效约束和误差相关性。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
        if record.get('status') not in ('joint_components', 'total_only') or record.get('quality_flags'):
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
        if record['status'] == 'joint_components':
            y = np.array([c['intensity'] for c in record['components']], float)
            covariance = np.array(record['covariance_identifiable_subspace'], float)
            design = np.eye(len(contributors))
        else:
            y = np.array([record['total']['intensity']], float)
            covariance = np.array([[record['total']['variance']]], float)
            design = np.ones((1, len(contributors)))
        candidates.append((record, meta, identities, y, covariance, design))
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
    for record, meta, ids, y, covariance, design in candidates:
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
                       'known_factors': np.array([m['lp_forward'] for m in meta]), 'record': record})
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
                     rcond=args.rcond, max_parameters=args.max_parameters, max_seconds=args.max_seconds)
    args.output.mkdir(parents=True, exist_ok=False)
    # 数值内核结果转换为可追溯主记录；不同组的相关性由联合参数协方差保留。
    serial = json.loads(json.dumps(fit, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item(), allow_nan=False))
    serial.update(schema='ariadne-global-scale-fit/1', feature_names=labels, feature_centre=centre.tolist(),
                  scan_ranges_rad=ranges, config=str(args.geometry.resolve()), integrated=str(args.integrated.resolve()),
                  excluded_observations=excluded,
                  scale_gauge='mean contributor log scale/absorption is zero; no independent domain normalization',
                  absorption_model='shared even degree-2 directional log response; empirical relative transmission, not absolute mu or crystal shape',
                  uncertainty_scope='local joint Gaussian linearization conditional on supplied geometry, LP, symmetry, feature basis and independent input groups')
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
            'fit_only_reason': 'total-only with unequal component corrections; retained in global fit, not deconvolved or exported'}


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
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Output directory already exists')
    scale(args)


if __name__ == '__main__':
    main()

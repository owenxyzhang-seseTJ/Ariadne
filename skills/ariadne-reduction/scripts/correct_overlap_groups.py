#!/usr/bin/env python3
"""Apply documented LP/scale/absorption factors to joint pixel observations.

Propagates full intensity covariance and optional correction-factor covariance.
Does not estimate correction factors or equalize twin-domain volume fractions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np


def checked_covariance(value, size, label):
    covariance = np.asarray(value, dtype=float)
    if (covariance.shape != (size, size) or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14)):
        raise ValueError(label + ' covariance must be finite, symmetric and match its components')
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if np.min(np.linalg.eigvalsh(covariance)) < -1e-10 * scale:
        raise ValueError(label + ' covariance must be positive semidefinite')
    return covariance


def correct(args):
    records = [json.loads(line) for line in args.integrated.read_text().splitlines() if line.strip()]
    corrections = json.loads(args.corrections.read_text())
    if corrections.get('schema') != 'scxrd-group-corrections/1':
        raise ValueError('Unsupported correction schema')
    for key in ('basis', 'correction_provenance', 'integration_provenance', 'domain_count'):
        if not corrections.get(key):
            raise ValueError('Missing correction evidence: ' + key)
    if corrections.get('complete_contributor_audit') is not True or corrections.get('contains_lp_scaling_absorption') is not True:
        raise ValueError('Require complete contributor audit and actual LP/scaling/absorption corrections')
    if corrections.get('scale_convention') != 'common_across_domains_keep_twin_fraction':
        raise ValueError('Do not normalize individual twin domains to equal intensities')
    args.output.mkdir(parents=True, exist_ok=False)
    observations, rejected = [], []
    for record in records:
        group_id = record['observation_id']
        reason = None
        correction = corrections.get('groups', {}).get(group_id)
        if record.get('status') == 'unresolved' or record.get('quality_flags'):
            reason = record.get('error') or 'unresolved integration or profile/background quality flag'
        elif correction is None:
            reason = 'missing correction factors'
        elif record.get('basis') != corrections['basis']:
            reason = 'basis mismatch'
        if reason:
            rejected.append({'observation_id': group_id, 'reason': reason})
            continue
        if record.get('intensity_definition') == 'full-profile amplitude; no further partiality division':
            if correction.get('partiality_policy') != 'already_full_profile_amplitude':
                rejected.append({'observation_id': group_id, 'reason': 'state partiality_policy=already_full_profile_amplitude to prevent double partiality correction'})
                continue
        contributors = record['contributors']
        n = len(contributors)
        factors = np.asarray(correction['factors'], dtype=float)
        if factors.shape != (n,) or not np.all(np.isfinite(factors)) or np.any(factors <= 0):
            raise ValueError('Invalid correction factors for ' + group_id)
        Cf = checked_covariance(correction.get('factor_covariance', np.zeros((n, n))), n, 'Correction-factor')
        if 'factor_covariance' not in correction and correction.get('factor_uncertainty') != 'treated_as_fixed':
            raise ValueError('Explicitly provide factor covariance or state treated_as_fixed')
        if record['status'] == 'joint_components':
            intensity = np.asarray([c['intensity'] for c in record['components']], dtype=float)
            if intensity.shape != (n,) or not np.all(np.isfinite(intensity)):
                raise ValueError('Component intensities must be finite and match their contributors')
            covariance = checked_covariance(record['covariance_identifiable_subspace'], n, 'Intensity')
            value = float(factors @ intensity)
            # Independent intensity/factor errors; cross-covariance must be supplied if jointly estimated.
            variance = float(factors @ covariance @ factors + intensity @ Cf @ intensity)
            corrected_covariance = factors[:, None] * covariance * factors[None, :] + intensity[:, None] * Cf * intensity[None, :]
            corrected_components = factors * intensity
            if correction.get('estimated_from_same_intensities') is True:
                cross = np.asarray(correction.get('intensity_factor_cross_covariance'), dtype=float)
                if cross.shape != (n, n) or not np.all(np.isfinite(cross)):
                    raise ValueError('Jointly estimated corrections require intensity-factor cross covariance')
                variance += float(2 * factors @ cross @ intensity)
                corrected_covariance += factors[:, None] * cross * intensity[None, :] + intensity[:, None] * cross.T * factors[None, :]
            checked_covariance(corrected_covariance, n, 'Propagated component')
        else:
            corrected_covariance, corrected_components = None, None
            if not np.allclose(factors, factors[0], rtol=1e-8, atol=0):
                rejected.append({'observation_id': group_id, 'reason': 'only raw total identifiable; unequal component correction factors require a global observation model'})
                continue
            if not np.allclose(Cf, Cf[0, 0], rtol=1e-8, atol=1e-12):
                rejected.append({'observation_id': group_id, 'reason': 'total-only observation requires a common random correction factor'})
                continue
            total = record['total']
            value = factors[0] * total['intensity']
            variance = factors[0] ** 2 * total['variance'] + total['intensity'] ** 2 * Cf[0, 0]
            if correction.get('estimated_from_same_intensities') is True:
                cross = correction.get('total_factor_cross_covariance')
                if not isinstance(cross, (int, float)) or not np.isfinite(cross):
                    raise ValueError('Provide total-factor cross covariance')
                variance += 2 * factors[0] * total['intensity'] * cross
        if not np.isfinite(variance) or variance <= 0:
            rejected.append({'observation_id': group_id, 'reason': 'nonpositive propagated variance'})
            continue
        observations.append({'record': 'observation', 'observation_id': group_id, 'intensity': float(value),
                             'sigma': float(np.sqrt(variance)), 'contributors': contributors,
                             'integration_status': record['status'], 'correction_factors': factors.tolist(),
                             'source_covariance': record['covariance_identifiable_subspace'],
                             'corrected_component_intensities': corrected_components.tolist() if corrected_components is not None else None,
                             'corrected_component_covariance': corrected_covariance.tolist() if corrected_covariance is not None else None,
                             'intensity_definition': record.get('intensity_definition'),
                             'covariance_scope': ('local joint scale/intensity Gaussian covariance, including same-data cross terms; conditional on geometry, LP and model basis'
                                                  if corrections.get('global_parameter_covariance') else record.get('covariance_scope'))})
    reject_path = args.output / 'rejected_groups.json'
    reject_path.write_text(json.dumps(rejected, indent=2) + '\n')
    metadata = {'record': 'metadata', 'schema': 'scxrd-observations/1', 'corrected': True,
                'basis': corrections['basis'], 'backend': 'SCXRD joint QR/SVD pixel integrator 0.3.0',
                'integration_provenance': corrections['integration_provenance'],
                'correction_provenance': corrections['correction_provenance'],
                'domain_count': corrections['domain_count'], 'intensity_semantics': 'measured_group_total',
                'phase_relation': corrections.get('phase_relation', 'unknown'),
                'phase_evidence_provenance': corrections.get('phase_evidence_provenance'),
                'all_contributors_accounted': True, 'excluded_observations_provenance': str(reject_path.resolve()),
                'overlap_model': 'joint fixed empirical profiles; nuisance background eliminated; identifiability-aware covariance',
                'source_integrated_groups': str(args.integrated.resolve()),
                'export_loss': 'HKLF5 does not encode full component/cross-observation covariance; preserve this master JSONL and original integrated groups',
                'global_parameter_covariance': corrections.get('global_parameter_covariance'),
                'cross_group_covariance': 'retain joint parameter covariance and same-data sensitivities in global scaling master record; HKLF5 does not encode it' if corrections.get('global_parameter_covariance') else 'retain correction model separately; HKLF5 does not encode it'}
    with (args.output / 'corrected_observations.jsonl').open('x') as handle:
        handle.write(json.dumps(metadata) + '\n')
        for observation in observations:
            handle.write(json.dumps(observation, allow_nan=False) + '\n')
    print(args.output / 'corrected_observations.jsonl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('integrated', type=Path)
    p.add_argument('corrections', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    try:
        correct(args)
        return 0
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

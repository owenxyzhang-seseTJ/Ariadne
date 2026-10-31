"""Stream validated measured groups into fixed-width HKLF 5 records.

One common positive multiplier acts on both I and sigma; no contribution is
split or sign-clipped. Two passes avoid retaining observation/covariance payloads
and output lines. A set of observation IDs remains necessary for uniqueness.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


def _header(handle):
    header = json.loads(next(handle))
    if header.get('record') != 'metadata' or header.get('schema') != 'scxrd-observations/1':
        raise ValueError('Unsupported observation schema')
    if header.get('corrected') is not True or header.get('all_contributors_accounted') is not True:
        raise ValueError('Require corrected genuine groups with all contributors')
    if header.get('intensity_semantics') != 'measured_group_total':
        raise ValueError('Only measured_group_total intensities are accepted')
    if header.get('phase_relation') != 'same_phase_supported' or not header.get('phase_evidence_provenance'):
        raise ValueError('HKLF5 twin export requires same-phase evidence; unrelated phases need separate structure models')
    for key in ('basis', 'backend', 'integration_provenance', 'correction_provenance',
                'excluded_observations_provenance', 'overlap_model'):
        if not header.get(key):
            raise ValueError('Missing provenance: '+key)
    domains = header.get('domain_count')
    if type(domains) is not int or not 2 <= domains <= 9999:
        raise ValueError('Require at least two declared domains')
    return header


def _observations(handle):
    for number, line in enumerate(handle, 2):
        if line.strip():
            yield number, json.loads(line)


def _inventory(handle, domains):
    names, seen_domains = set(), set()
    count = lines = overlap = 0
    upper, minimum_sigma = sys.float_info.max, math.inf
    for line_no, item in _observations(handle):
        if item.get('record') != 'observation':
            raise ValueError(f'Not an observation at line {line_no}')
        name = item.get('observation_id')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Empty or duplicated observation_id')
        names.add(name)
        intensity, sigma = float(item['intensity']), float(item['sigma'])
        if not math.isfinite(intensity) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('Intensity must be finite and sigma positive')
        magnitude = max(abs(intensity), sigma)
        upper = min(upper, 99999.0/magnitude)
        if intensity < 0:
            upper = min(upper, 9999.0/abs(intensity))
        minimum_sigma = min(minimum_sigma, sigma)
        contributors = item.get('contributors')
        if not isinstance(contributors, list) or not contributors:
            raise ValueError('Observation must have contributors')
        keys = set()
        for index, component in enumerate(contributors):
            domain, hkl = component['domain'], component['hkl']
            if type(domain) is not int or not 1 <= domain <= domains:
                raise ValueError('Invalid domain number')
            if not isinstance(hkl, list) or len(hkl) != 3 or any(type(v) is not int for v in hkl) or hkl == [0, 0, 0]:
                raise ValueError('Invalid reflection indices')
            if any(len(f'{value:4d}') != 4 for value in hkl):
                raise ValueError('Indices do not fit SHELX I4 fields')
            batch = domain if index == len(contributors)-1 else -domain
            if len(f'{batch:4d}') != 4:
                raise ValueError('Signed domain number does not fit SHELX I4 field')
            key = (domain, *hkl)
            if key in keys:
                raise ValueError('Duplicate contributor in observation')
            keys.add(key)
            seen_domains.add(domain)
        count += 1
        lines += len(contributors)
        overlap += len(contributors) > 1
    if not count or seen_domains != set(range(1, domains+1)):
        raise ValueError('All declared domains must retain at least one observation contribution')
    multiplier = min(1., upper)
    rescued = minimum_sigma*multiplier < .01
    if rescued:
        # Preserve historical scales unless small sigma would suffer severe
        # format quantization. Prefer min(sigma_scaled) >= 1 when the same
        # global scale fits every signed intensity and sigma field.
        exponent = math.ceil(-math.log10(minimum_sigma))
        preferred = 10.**exponent if exponent <= 308 else sys.float_info.max
        multiplier = min(upper, max(multiplier, preferred))
    if multiplier <= 0 or float(f'{minimum_sigma*multiplier:.2f}') <= 0:
        raise ValueError('F8.2 dynamic range cannot retain positive sigma for all observations; retain master JSONL')
    return count, lines, overlap, multiplier, rescued


def _identity(handle):
    stat = os.fstat(handle.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def export(source, target):
    source, target = Path(source), Path(target)
    sidecar = target.with_suffix('.export.json')
    if target.exists() or sidecar.exists():
        raise ValueError('Output already exists')
    with source.open(encoding='utf-8') as handle:
        identity = _identity(handle)
        header = _header(handle)
        count, lines, overlap, multiplier, rescued = _inventory(handle, header['domain_count'])
        report = {'schema':'scxrd-hklf5-export/1', 'source':str(source.resolve()),
                  'metadata':header, 'observations':count, 'component_lines':lines,
                  'intensity_and_sigma_multiplier':multiplier, 'overlap_groups':overlap,
                  'purpose':'format conversion only; provenance assertions supplied by integration backend',
                  'export_loss':['component covariance not encoded in HKLF5', 'cross-observation covariance not encoded',
                                 'retain source master JSONL, integrated groups, corrections and hypothesis evidence']}
        if _identity(handle) != identity:
            raise ValueError('Observation source changed during validation')
        handle.seek(0)
        next(handle)
        with TemporaryDirectory(prefix='ariadne-export-', dir=target.parent) as temporary:
            staged = Path(temporary)/'reflections.hkl'
            staged_report = Path(temporary)/'report.json'
            worst_relative_sigma_error = 0.
            with staged.open('w', encoding='ascii') as output, staged_report.open('w', encoding='utf-8') as audit:
                audit.write(json.dumps(report, ensure_ascii=False, allow_nan=False)[:-1]+', "observation_to_lines": [')
                line_start = 1
                for observation_index, (_, item) in enumerate(_observations(handle)):
                    intensity, sigma = float(item['intensity'])*multiplier, float(item['sigma'])*multiplier
                    i_field, s_field = f'{intensity:8.2f}', f'{sigma:8.2f}'
                    if len(i_field) != 8 or len(s_field) != 8 or float(s_field) <= 0:
                        raise ValueError('Value overflows SHELX fixed-width field or sigma underflows')
                    worst_relative_sigma_error = max(worst_relative_sigma_error, abs(float(s_field)-sigma)/sigma)
                    contributors = item['contributors']
                    for index, component in enumerate(contributors):
                        h, k, l = component['hkl']
                        batch = component['domain'] if index == len(contributors)-1 else -component['domain']
                        output.write(f'{h:4d}{k:4d}{l:4d}{i_field}{s_field}{batch:4d}\n')
                    if observation_index:
                        audit.write(',')
                    json.dump({'observation_id':item['observation_id'], 'first_line':line_start,
                               'line_count':len(contributors)}, audit, ensure_ascii=False)
                    line_start += len(contributors)
                output.write(f'{0:4d}{0:4d}{0:4d}{0.:8.2f}{0.:8.2f}{0:4d}\n')
                audit.write('], "format_quantization": ')
                json.dump({'weak_sigma_scale_rescue':rescued,
                           'quantum_in_source_units':.01/multiplier,
                           'max_absolute_rounding_error_in_source_units':.005/multiplier,
                           'rounding_bound_scope':'nominal half quantum for decimal format rounding; excludes binary multiplication and parsing error',
                           'maximum_relative_sigma_rounding_error':worst_relative_sigma_error,
                           'statistical_effect':'one common scale preserves I/sigma before decimal rounding; no measurement selection'},
                          audit, allow_nan=False)
                audit.write('}\n')
            if _identity(handle) != identity:
                raise ValueError('Observation source changed during export')
            # Exclusive hard-link installation refuses overwriting an output
            # that appeared during the two passes; staging errors leave no HKL.
            os.link(staged, target)
            try:
                os.link(staged_report, sidecar)
            except OSError:
                target.unlink()
                raise
    print(target)

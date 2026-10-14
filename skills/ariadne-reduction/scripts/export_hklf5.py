#!/usr/bin/env python3
"""Format genuine corrected observation groups as HKLF 5; does not integrate pixels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def export(source, target):
    if target.exists() or target.with_suffix('.export.json').exists():
        raise ValueError('Output already exists')
    with source.open(encoding='utf-8') as handle:
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
                raise ValueError('Missing provenance: ' + key)
        domains = header.get('domain_count')
        if type(domains) is not int or not 2 <= domains <= 9999:
            raise ValueError('Require at least two declared domains')
        observations = []
        names = set()
        seen_domains = set()
        for line_no, line in enumerate(handle, 2):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('record') != 'observation':
                raise ValueError(f'Not an observation at line {line_no}')
            name = item.get('observation_id')
            if not isinstance(name, str) or not name or name in names:
                raise ValueError('Empty or duplicated observation_id')
            names.add(name)
            intensity, sigma = float(item['intensity']), float(item['sigma'])
            if not math.isfinite(intensity) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Intensity must be finite and sigma positive')
            item['intensity'], item['sigma'] = intensity, sigma
            contributors = item.get('contributors')
            if not isinstance(contributors, list) or not contributors:
                raise ValueError('Observation must have contributors')
            keys = set()
            for component in contributors:
                domain, hkl = component['domain'], component['hkl']
                if type(domain) is not int or not 1 <= domain <= domains:
                    raise ValueError('Invalid domain number')
                if not isinstance(hkl, list) or len(hkl) != 3 or any(type(v) is not int for v in hkl) or hkl == [0, 0, 0]:
                    raise ValueError('Invalid reflection indices')
                if any(len(f'{v:4d}') != 4 for v in hkl):
                    raise ValueError('Indices do not fit SHELX I4 fields')
                key = (domain, *hkl)
                if key in keys:
                    raise ValueError('Duplicate contributor in observation')
                keys.add(key)
                seen_domains.add(domain)
            observations.append(item)
    if not observations or seen_domains != set(range(1, domains + 1)):
        raise ValueError('All declared domains must retain at least one observation contribution')
    # Leave headroom for rounding; transform all intensities and sigmas identically.
    limits = [99999.0 / max(abs(float(o['intensity'])), float(o['sigma']), 1e-300) for o in observations]
    multiplier = min(1.0, min(limits))
    # More precise conservative bound for negative F8.2 fields (minus sign uses one column).
    for item in observations:
        if item['intensity'] < 0:
            multiplier = min(multiplier, 9999.0 / abs(item['intensity']))
    lines = []
    for item in observations:
        value, sigma = float(item['intensity']) * multiplier, float(item['sigma']) * multiplier
        if round(sigma, 2) <= 0:
            raise ValueError('F8.2 sigma underflow; obtain an appropriate backend export scale')
        components = item['contributors']
        for i, component in enumerate(components):
            h, k, l = component['hkl']
            batch = component['domain'] if i == len(components) - 1 else -component['domain']
            columns = [f'{h:4d}', f'{k:4d}', f'{l:4d}', f'{value:8.2f}', f'{sigma:8.2f}', f'{batch:4d}']
            if any(len(col) != width for col, width in zip(columns, (4, 4, 4, 8, 8, 4))):
                raise ValueError('Value overflows SHELX fixed-width field')
            lines.append(''.join(columns))
    lines.append(f'{0:4d}{0:4d}{0:4d}{0.:8.2f}{0.:8.2f}{0:4d}')
    with target.open('x', encoding='ascii') as handle:
        handle.write('\n'.join(lines) + '\n')
    report = {'schema': 'scxrd-hklf5-export/1', 'source': str(source.resolve()),
              'metadata': header, 'observations': len(observations),
              'component_lines': len(lines) - 1, 'intensity_and_sigma_multiplier': multiplier,
              'overlap_groups': sum(len(o['contributors']) > 1 for o in observations),
              'purpose': 'format conversion only; provenance assertions supplied by integration backend'}
    report['export_loss'] = ['component covariance not encoded in HKLF5', 'cross-observation covariance not encoded',
                             'retain source master JSONL, integrated groups, corrections and hypothesis evidence']
    report['observation_to_lines'] = []
    line_start = 1
    for observation in observations:
        report['observation_to_lines'].append({'observation_id': observation['observation_id'], 'first_line': line_start,
                                              'line_count': len(observation['contributors'])})
        line_start += len(observation['contributors'])
    with target.with_suffix('.export.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        export(args.source, args.output)
        return 0
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

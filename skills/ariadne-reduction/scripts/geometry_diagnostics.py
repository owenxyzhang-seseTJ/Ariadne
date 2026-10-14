#!/usr/bin/env python3
"""Summarise common vs domain-dependent centroid residuals, without assigning cause."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from dials.array_family import flex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reflections', type=Path)
    parser.add_argument('domain_mapping', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scan-bin-frames', type=int, default=20)
    parser.add_argument('--min-observations', type=int, default=10)
    args = parser.parse_args()
    if args.scan_bin_frames < 1 or args.min_observations < 3:
        raise ValueError('Invalid binning settings')
    table = flex.reflection_table.from_file(str(args.reflections))
    mapping = json.loads(args.domain_mapping.read_text())
    observed = np.asarray(table['xyzobs.px.value'], dtype=float)
    predicted = np.asarray(table['xyzcal.px'], dtype=float)
    ids = np.asarray(table['id'], dtype=int)
    panels = np.asarray(table['panel'], dtype=int)
    groups = defaultdict(list)
    omitted = []
    for row in range(len(table)):
        if ids[row] < 0 or not np.all(np.isfinite([observed[row], predicted[row]])):
            omitted.append(row)
            continue
        descriptor = mapping['experiments'].get(str(ids[row]))
        if descriptor is None:
            raise ValueError('Need explicit experiment -> domain and raw scan mapping')
        # Frame zero and raw_scan identity must share a physical scan, not sweep id.
        scan = str(descriptor['raw_scan'])
        domain = int(descriptor['domain'])
        frame = observed[row, 2] + float(descriptor.get('frame_offset', 0))
        key = (scan, int(panels[row]), int(np.floor(frame / args.scan_bin_frames)), domain)
        groups[key].append(observed[row] - predicted[row])
    bins = []
    shared = defaultdict(list)
    for key, values in groups.items():
        if len(values) < args.min_observations:
            bins.append({'raw_scan': key[0], 'panel': key[1], 'scan_bin': key[2], 'domain': key[3],
                         'n': len(values), 'status': 'insufficient_observations'})
            continue
        values = np.asarray(values)
        centre = np.median(values, axis=0)
        record = {'raw_scan': key[0], 'panel': key[1], 'scan_bin': key[2], 'domain': key[3],
                  'n': len(values), 'median_residual_xyz_px': centre.tolist(),
                  'mad_xyz_px': (1.4826 * np.median(abs(values - centre), axis=0)).tolist(),
                  'status': 'descriptive_only'}
        bins.append(record)
        shared[key[:3]].append(record)
    contrasts = []
    for key, records in shared.items():
        if len(records) < 2:
            continue
        centres = np.asarray([r['median_residual_xyz_px'] for r in records])
        common = np.median(centres, axis=0)
        contrasts.append({'raw_scan': key[0], 'panel': key[1], 'scan_bin': key[2],
                          'common_residual_xyz_px': common.tolist(),
                          'domains': [{'domain': r['domain'], 'contrast_xyz_px': (v - common).tolist()}
                                      for r, v in zip(records, centres)],
                          'cause': 'unresolved; common motion can be instrument or sample motion'})
    result = {'schema': 'ariadne-geometry-diagnostics/1', 'status': 'experimental_untested',
              'bins': bins, 'cross_domain_contrasts': contrasts, 'omitted_rows': omitted,
              'decision': 'diagnosis only; compare constrained static/scan-varying physical refinements on common holdout observations',
              'limitations': 'not a calibrated significance test; no causal separation without geometry Jacobians and independent calibration'}
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

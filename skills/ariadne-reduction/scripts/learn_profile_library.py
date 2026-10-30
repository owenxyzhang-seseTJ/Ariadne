#!/usr/bin/env python3
"""Learn pooled and bounded-complexity local empirical profiles from real shoeboxes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from profile_learning import signal_moments, learning_array_bytes, stream_templates


def learn(args):
    from dials.array_family import flex
    from dials.algorithms.shoebox import MaskCode
    references = flex.reflection_table.from_file(str(args.references))
    config = json.loads(args.calibration.read_text())
    if config.get('schema') != 'scxrd-profile-calibration/1':
        raise ValueError('Unsupported calibration schema')
    for key in ('isolation_provenance', 'isolated_source_rows', 'background_provenance', 'noise_model', 'basis',
                'experiment_to_domain', 'complete_prediction_provenance'):
        if not config.get(key):
            raise ValueError('Missing calibration evidence: ' + key)
    if config.get('calibration_support_complete') is not True:
        raise ValueError('Reference profile support must have been inspected as complete')
    source = np.asarray(references['scxrd_source_row'], dtype=int) if 'scxrd_source_row' in references else np.arange(len(references))
    selected = set(config['isolated_source_rows'])
    boxes = references['shoebox']
    positions = np.asarray(references['xyzcal.px'] if 'xyzcal.px' in references else references['xyzobs.px.value'], dtype=float)
    if not np.all(np.isfinite(positions)) or min(args.local_grid) < 1 or np.prod(args.local_grid) > 128 or args.shrinkage_references <= 0 or args.min_local_references < 1:
        raise ValueError('Invalid reference positions or local profile settings')
    keys = defaultdict(list)
    rejected = []
    for i in range(len(references)):
        if int(source[i]) not in selected:
            continue
        box = boxes[i]
        data = np.asarray(box.data.as_numpy_array(), dtype=float)
        background = np.asarray(box.background.as_numpy_array(), dtype=float)
        flags = np.asarray(box.mask.as_numpy_array(), dtype=int)
        if data.ndim != 3 or data.size == 0 or background.shape != data.shape or flags.shape != data.shape:
            rejected.append({'source_row': int(source[i]), 'reason': 'missing_pixel_or_background_data'})
            continue
        if data.shape[0] < args.min_reference_frames:
            rejected.append({'source_row': int(source[i]), 'reason': 'insufficient_rocking_profile_sampling'})
            continue
        intensity = float(references['intensity.sum.value'][i])
        variance = float(references['intensity.sum.variance'][i])
        if not np.isfinite(variance) or variance <= 0 or intensity / np.sqrt(variance) < args.reference_min_snr:
            rejected.append({'source_row': int(source[i]), 'reason': 'low_snr_for_template_learning_only'})
            continue
        valid = (flags & int(MaskCode.Valid)) != 0
        foreground = (flags & int(MaskCode.Foreground)) != 0
        signal = np.where(valid & foreground, np.maximum(data - background, 0), 0)
        if not np.all(np.isfinite(signal)) or signal.sum() <= 0:
            rejected.append({'source_row': int(source[i]), 'reason': 'no_positive_foreground_signal'})
            continue
        centre, covariance = signal_moments(signal)
        keys[(int(references['id'][i]), int(references['panel'][i]))].append({
            'reference_index': i, 'shape': signal.shape, 'centre': centre - .5, 'covariance_zyx': covariance,
            'position_xyz': positions[i], 'source_row': int(source[i])})
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    for (exp_id, panel), profiles in keys.items():
        if len(profiles) < args.min_references:
            rejected.append({'experiment_id': exp_id, 'panel': panel, 'reason': 'too_few_isolated_references'})
            continue
        covariance = np.mean([p['covariance_zyx'] for p in profiles], axis=0)
        # Preserve the full supplied empirical support after centroid alignment.
        support = np.max([np.maximum(p['centre'], np.asarray(p['shape']) - 1 - p['centre']) for p in profiles], axis=0)
        radii = np.ceil(np.maximum(support + 1, args.support_sigma * np.sqrt(np.diag(covariance)))).astype(int)
        shape = tuple((2 * radii + 1).tolist())
        if np.prod(shape) > args.max_template_voxels or learning_array_bytes(profiles, shape) + 4096 > args.max_learning_bytes:
            rejected.append({'experiment_id': exp_id, 'panel': panel, 'reason': 'template_memory_budget'})
            continue
        def load_signal(profile):
            box = boxes[profile['reference_index']]
            data = np.asarray(box.data.as_numpy_array(), dtype=float)
            background = np.asarray(box.background.as_numpy_array(), dtype=float)
            mask = np.asarray(box.mask.as_numpy_array(), dtype=int)
            accepted = ((mask & int(MaskCode.Valid)) != 0) & ((mask & int(MaskCode.Foreground)) != 0)
            return np.where(accepted, np.maximum(data - background, 0), 0)

        try:
            name, local_model = stream_templates(
                profiles, load_signal, shape, args.output, f'profile_exp{exp_id}_panel{panel}',
                grid_xyz=args.local_grid, min_local_references=args.min_local_references,
                shrinkage_references=args.shrinkage_references, max_learning_bytes=args.max_learning_bytes)
        except ValueError as exc:
            rejected.append({'experiment_id': exp_id, 'panel': panel,
                             'reason': 'failed_profile_alignment', 'detail': str(exc)})
            continue
        entry = {'experiment_id': exp_id, 'panel': panel, 'template': name, 'template_shape_zyx': list(shape),
                        'covariance_px_xyz': covariance[::-1, ::-1].tolist(), 'noise_model': config['noise_model'],
                        'provenance': {'reference_file': str(args.references.resolve()),
                                       'source_rows': [p['source_row'] for p in profiles],
                                       'isolation': config['isolation_provenance'], 'background': config['background_provenance'],
                                       'regularization': 'uniform within-voxel variance 1/12; local templates shrink to pooled reference'}}
        if local_model is not None:
            entry['local_model'] = local_model
        entry['provenance']['accumulation'] = 'cellwise normalized sums; separable affine alignment with bounded-tile fallback; temporary local sums until pooled mean is known'
        entries.append(entry)
    library = {'schema': 'scxrd-profile-library/1', 'implementation_status': 'experimental_untested',
               'entries': entries, 'basis': config['basis'], 'experiment_to_domain': config['experiment_to_domain'],
               'experiment_to_phase': config.get('experiment_to_phase', {}),
               'complete_prediction_provenance': config['complete_prediction_provenance'],
               'rejected_reference_rows': rejected,
               'limitations': 'conditional empirical detector-space templates; no instrument/crystal Jacobian uncertainty; local grid complexity not cross-validated'}
    (args.output / 'profiles.json').write_text(json.dumps(library, indent=2, allow_nan=False) + '\n')
    print(args.output / 'profiles.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('references', type=Path)
    p.add_argument('calibration', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--reference-min-snr', type=float, default=10.)
    p.add_argument('--min-reference-frames', type=int, default=2)
    p.add_argument('--min-references', type=int, default=10)
    p.add_argument('--support-sigma', type=float, default=5.)
    p.add_argument('--max-template-voxels', type=int, default=1000000)
    p.add_argument('--max-learning-bytes', type=int, default=536870912)
    p.add_argument('--local-grid', type=int, nargs=3, default=(1, 1, 1), metavar=('NX', 'NY', 'NZ'))
    p.add_argument('--min-local-references', type=int, default=5)
    p.add_argument('--shrinkage-references', type=float, default=20.)
    args = p.parse_args()
    try:
        learn(args)
        return 0
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

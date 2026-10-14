#!/usr/bin/env python3
"""Learn pooled and bounded-complexity local empirical profiles from real shoeboxes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import map_coordinates
from dials.array_family import flex
from dials.algorithms.shoebox import MaskCode


def learn(args):
    references = flex.reflection_table.from_file(str(args.references))
    config = json.loads(args.calibration.read_text())
    if config.get('schema') != 'scxrd-profile-calibration/1':
        raise ValueError('Unsupported calibration schema')
    for key in ('isolation_provenance', 'background_provenance', 'noise_model', 'basis',
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
            continue
        normalized = signal / signal.sum()
        grid = np.asarray(np.meshgrid(*[np.arange(n) + .5 for n in signal.shape], indexing='ij'))
        centre = np.asarray([(normalized * coordinate).sum() for coordinate in grid])
        delta = grid.reshape(3, -1).T - centre
        covariance = (delta.T * normalized.ravel()) @ delta
        # Uniform within-voxel approximation; recorded as modelling regularization.
        covariance += np.eye(3) / 12
        keys[(int(references['id'][i]), int(references['panel'][i]))].append({
            'signal': normalized, 'centre': centre - .5, 'covariance_zyx': covariance,
            'position_xyz': positions[i], 'source_row': int(source[i])})
    args.output.mkdir(parents=True, exist_ok=False)
    entries = []
    for (exp_id, panel), profiles in keys.items():
        if len(profiles) < args.min_references:
            rejected.append({'experiment_id': exp_id, 'panel': panel, 'reason': 'too_few_isolated_references'})
            continue
        covariance = np.mean([p['covariance_zyx'] for p in profiles], axis=0)
        # Preserve the full supplied empirical support after centroid alignment.
        support = np.max([np.maximum(p['centre'], np.asarray(p['signal'].shape) - 1 - p['centre']) for p in profiles], axis=0)
        radii = np.ceil(np.maximum(support + 1, args.support_sigma * np.sqrt(np.diag(covariance)))).astype(int)
        shape = tuple((2 * radii + 1).tolist())
        if np.prod(shape) > args.max_template_voxels or np.prod(shape) * (len(profiles) + 3) * 8 > args.max_learning_bytes:
            rejected.append({'experiment_id': exp_id, 'panel': panel, 'reason': 'template_memory_budget'})
            continue
        target = np.asarray(np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape], indexing='ij')).reshape(3, -1)
        total = np.zeros(shape)
        for profile in profiles:
            coordinates = target + profile['centre'][:, None]
            aligned = map_coordinates(profile['signal'], coordinates, order=1, mode='constant', cval=0).reshape(shape)
            if aligned.sum() > 0:
                profile['aligned'] = aligned / aligned.sum()
                total += profile['aligned']
        if not np.isfinite(total.sum()) or total.sum() <= 0:
            rejected.append({'experiment_id': exp_id, 'panel': panel, 'reason': 'failed_profile_alignment'})
            continue
        total /= total.sum()
        name = f'profile_exp{exp_id}_panel{panel}.npy'
        np.save(args.output / name, total, allow_pickle=False)
        entry = {'experiment_id': exp_id, 'panel': panel, 'template': name, 'template_shape_zyx': list(shape),
                        'covariance_px_xyz': covariance[::-1, ::-1].tolist(), 'noise_model': config['noise_model'],
                        'provenance': {'reference_file': str(args.references.resolve()),
                                       'source_rows': [p['source_row'] for p in profiles],
                                       'isolation': config['isolation_provenance'], 'background': config['background_provenance'],
                                       'regularization': 'uniform within-voxel variance 1/12; local templates shrink to pooled reference'}}
        if np.prod(args.local_grid) > 1:
            # Fixed grid bounds and complexity; no intensity-dependent template fitting.
            coordinates = np.asarray([p['position_xyz'] for p in profiles])
            low, high = coordinates.min(axis=0), coordinates.max(axis=0)
            span = np.maximum(high - low, 1.)
            dimensions = np.asarray(args.local_grid)
            bins = np.minimum(np.floor((coordinates - low) / span * dimensions).astype(int), dimensions - 1)
            nodes = []
            for cell in np.ndindex(*args.local_grid):
                local = [p for p, index in zip(profiles, bins) if np.array_equal(index, cell) and 'aligned' in p]
                if len(local) < args.min_local_references:
                    continue
                alpha = len(local) / (len(local) + args.shrinkage_references)
                local_template = np.mean([p['aligned'] for p in local], axis=0)
                shrunk = alpha * local_template + (1 - alpha) * total
                shrunk /= shrunk.sum()
                local_covariance = alpha * np.mean([p['covariance_zyx'] for p in local], axis=0) + (1 - alpha) * covariance
                local_name = f'profile_exp{exp_id}_panel{panel}_node_{cell[0]}_{cell[1]}_{cell[2]}.npy'
                np.save(args.output / local_name, shrunk, allow_pickle=False)
                nodes.append({'grid_index': list(cell), 'template': local_name,
                              'covariance_px_xyz': local_covariance[::-1, ::-1].tolist(),
                              'shrinkage_weight': alpha, 'n_references': len(local),
                              'source_rows': [p['source_row'] for p in local]})
            entry['local_model'] = {'grid_xyz': list(args.local_grid), 'low_xyz': low.tolist(), 'span_xyz': span.tolist(),
                                    'nodes': nodes, 'prior_reference_count': args.shrinkage_references,
                                    'outside_support': 'pooled reference; no local extrapolation',
                                    'selection': 'trilinear nonnegative weights; missing nodes use pooled reference'}
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

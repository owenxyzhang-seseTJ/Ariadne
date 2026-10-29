#!/usr/bin/env python3
"""DIALS raw-frame adapter: connected detector/scan overlaps -> pixel patches.

Requires complete multi-domain predictions and an evidence-backed profile library.
No q-neighbour grouping, intensity duplication, or per-domain pixel double-counting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import map_coordinates
from local_profiles import select_profile
from profile_geometry import gaussian_voxels


def connected_boxes(boxes):
    parent = list(range(len(boxes)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    active = []
    for i in sorted(range(len(boxes)), key=lambda j: boxes[j][4]):
        b = boxes[i]
        active = [j for j in active if boxes[j][5] > b[4]]
        for j in active:
            c = boxes[j]
            if b[0] < c[1] and c[0] < b[1] and b[2] < c[3] and c[2] < b[3]:
                parent[root(i)] = root(j)
        active.append(i)
    groups = defaultdict(list)
    for i in range(len(boxes)):
        groups[root(i)].append(i)
    return list(groups.values())


def template_values(entry, load_template, centre, bounds):
    if not entry.get('template'):
        covariance = np.asarray(entry['covariance_px_xyz'], dtype=float).reshape(3, 3)
        return gaussian_voxels(covariance, centre, bounds)
    x0, x1, y0, y1, z0, z1 = bounds
    z, y, x = np.mgrid[z0:z1, y0:y1, x0:x1]
    offsets = np.stack((x + .5 - centre[0], y + .5 - centre[1], z + .5 - centre[2]), axis=-1)
    values = np.zeros(z.shape)
    for item in entry.get('template_mixture', [{'template': entry['template'], 'weight': 1.}]):
        template = load_template(item['template'])
        origin = (np.asarray(template.shape) - 1) / 2
        xyz = offsets[..., ::-1] + origin
        values += item['weight'] * map_coordinates(template, xyz.reshape(-1, 3).T, order=1, mode='constant', cval=0).reshape(z.shape)
    return values


def prepare(args):
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    expts = ExperimentListFactory.from_json_file(str(args.expt), check_format=True)
    predictions = flex.reflection_table.from_file(str(args.predictions))
    library = json.loads(args.library.read_text())
    if library.get('schema') != 'scxrd-profile-library/1' or library.get('complete_prediction_provenance') is None:
        raise ValueError('Require calibrated library and complete multi-domain prediction provenance')
    if args.frame_cache < 1 or args.template_cache < 1 or args.background_margin < 0 or args.support_sigma <= 0:
        raise ValueError('Invalid patch/cache settings')
    ids = np.asarray(predictions['id'], dtype=int)
    panels = np.asarray(predictions['panel'], dtype=int)
    positions = np.asarray(predictions['xyzcal.px'], dtype=float)
    indices = np.asarray(predictions['miller_index'], dtype=int)
    args.output.mkdir(parents=True, exist_ok=False)
    entries = {(int(e['experiment_id']), int(e['panel'])): e for e in library['entries']}
    template_cache = OrderedDict()

    def load_template(name):
        path = (args.library.parent / name).resolve()
        if not path.is_relative_to(args.library.parent.resolve()):
            raise ValueError('Template path escapes the profile library')
        if name not in template_cache:
            template = np.load(path, allow_pickle=False)
            if template.ndim != 3 or not np.all(np.isfinite(template)) or np.any(template < 0) or template.sum() <= 0:
                raise ValueError('Invalid empirical profile template')
            template_cache[name] = template / template.sum()
            while len(template_cache) > args.template_cache:
                template_cache.popitem(last=False)
        template_cache.move_to_end(name)
        return template_cache[name]
    partitions = defaultdict(list)
    geometries = {}
    experiment_cache = {}
    records = []
    issues = []
    for i, (exp_id, panel_id, centre) in enumerate(zip(ids, panels, positions)):
        if exp_id < 0 or exp_id >= len(expts) or not np.any(indices[i]) or not np.all(np.isfinite(centre)):
            issues.append({'prediction_row': i, 'reason': 'invalid_prediction'})
            continue
        entry = entries.get((int(exp_id), int(panel_id)))
        if entry is None or not entry.get('provenance') or not entry.get('noise_model', {}).get('source'):
            raise ValueError(f'Missing calibrated profile/noise for experiment {exp_id}, panel {panel_id}')
        entry = select_profile(entry, centre)
        if int(exp_id) not in experiment_cache:
            expt = expts[int(exp_id)]
            if not expt.imageset or not expt.scan:
                raise ValueError('Raw rotation imageset and scan are required')
            experiment_cache[int(exp_id)] = (expt, tuple(expt.imageset.paths()), tuple(expt.scan.get_array_range()))
        expt, paths, array_range = experiment_cache[int(exp_id)]
        key = (paths, array_range, int(panel_id))
        panel_model = expt.detector[int(panel_id)]
        geometry = np.asarray([*expt.beam.get_s0(), *panel_model.get_origin(), *panel_model.get_fast_axis(),
                               *panel_model.get_slow_axis(), *panel_model.get_pixel_size(),
                               *expt.scan.get_oscillation(),
                               *(expt.goniometer.get_fixed_rotation() if expt.goniometer else ()),
                               *(expt.goniometer.get_setting_rotation() if expt.goniometer else ())], dtype=float)
        if key in geometries and (geometry.shape != geometries[key].shape or not np.allclose(geometry, geometries[key], rtol=1e-9, atol=1e-8)):
            raise ValueError('Contributors sharing raw pixels have inconsistent instrument geometry; jointly calibrate before integration')
        geometries[key] = geometry
        if entry.get('template'):
            shape = entry.get('template_shape_zyx') or load_template(entry['template']).shape
            radius = (np.asarray(shape)[::-1] + 1) / 2
        else:
            covariance = np.asarray(entry['covariance_px_xyz']).reshape(3, 3)
            radius = args.support_sigma * np.sqrt(np.diag(covariance))
        x0, y0, z0 = np.floor(centre - radius).astype(int)
        x1, y1, z1 = np.ceil(centre + radius).astype(int)
        box = (int(x0), int(x1), int(y0), int(y1), int(z0), int(z1))
        record = {'row': i, 'experiment_id': int(exp_id), 'panel': int(panel_id), 'centre': centre,
                  'box': box, 'entry': entry, 'partition': key}
        records.append(record)
        partitions[key].append(len(records) - 1)
    groups = []
    raw_cache = OrderedDict()

    def frame(imageset, cache_key, frame_index):
        if cache_key not in raw_cache:
            raw_cache[cache_key] = (imageset.get_corrected_data(frame_index), imageset.get_mask(frame_index))
            while len(raw_cache) > args.frame_cache:
                raw_cache.popitem(last=False)
        raw_cache.move_to_end(cache_key)
        return raw_cache[cache_key]

    for key, selected in partitions.items():
        boxes = [records[i]['box'] for i in selected]
        margin = args.background_margin
        expanded_boxes = [(b[0]-margin, b[1]+margin, b[2]-margin, b[3]+margin, b[4]-margin, b[5]+margin) for b in boxes]
        for component in connected_boxes(expanded_boxes):
            chosen = [records[selected[i]] for i in component]
            expt = expts[chosen[0]['experiment_id']]
            detector = expt.detector[chosen[0]['panel']]
            width, height = detector.get_image_size()
            start, stop = expt.scan.get_array_range()
            x0 = min(r['box'][0] for r in chosen) - args.background_margin
            x1 = max(r['box'][1] for r in chosen) + args.background_margin
            y0 = min(r['box'][2] for r in chosen) - args.background_margin
            y1 = max(r['box'][3] for r in chosen) + args.background_margin
            z0 = min(r['box'][4] for r in chosen) - args.background_margin
            z1 = max(r['box'][5] for r in chosen) + args.background_margin
            bounds = (max(0, x0), min(width, x1), max(0, y0), min(height, y1), max(start, z0), min(stop, z1))
            x0, x1, y0, y1, z0, z1 = bounds
            shape = (z1 - z0, y1 - y0, x1 - x0)
            if min(shape) <= 0 or np.prod(shape) > args.max_voxels or len(chosen) > args.max_components:
                issues.append({'prediction_rows': [r['row'] for r in chosen], 'reason': 'patch_or_component_budget_or_detector_extent'})
                continue
            noises = [r['entry']['noise_model'] for r in chosen]
            noise = noises[0]
            if float(noise['gain']) <= 0 or float(noise['readout_variance']) < 0:
                raise ValueError('Invalid calibrated gain/readout variance')
            if any(n['gain'] != noise['gain'] or n['readout_variance'] != noise['readout_variance'] for n in noises):
                raise ValueError('Overlapping contributors on one detector must share the calibrated pixel-noise model')
            data = np.empty(shape)
            valid = np.empty(shape, dtype=bool)
            for zi, absolute_frame in enumerate(range(z0, z1)):
                raw, masks = frame(expt.imageset, (key, absolute_frame), absolute_frame - start)
                data[zi] = raw[chosen[0]['panel']].as_numpy_array()[y0:y1, x0:x1]
                valid[zi] = masks[chosen[0]['panel']].as_numpy_array()[y0:y1, x0:x1]
            variance = float(noise['gain']) * np.maximum(data, 0) + float(noise['readout_variance'])
            variance = np.maximum(variance, float(noise.get('variance_floor', noise['gain'])))
            profiles = np.asarray([template_values(r['entry'], load_template, r['centre'], bounds) for r in chosen])
            group_id = f'pixel-group-{len(groups) + 1:06d}'
            patch_name = group_id + '.npz'
            arrays = {'data': data, 'variance': variance, 'valid': valid, 'profiles': profiles}
            if args.position_derivatives:
                step = .001
                derivatives = []
                for r in chosen:
                    axes = []
                    for axis in range(3):
                        delta = np.eye(3)[axis] * step
                        plus = template_values(r['entry'], load_template, r['centre'] + delta, bounds)
                        minus = template_values(r['entry'], load_template, r['centre'] - delta, bounds)
                        axes.append((plus - minus) / (2 * step))
                    derivatives.append(axes)
                arrays['position_derivatives_xyz'] = np.asarray(derivatives)
            np.savez_compressed(args.output / patch_name, **arrays)
            contributors = []
            for r in chosen:
                mapping = library.get('experiment_to_domain', {}).get(str(r['experiment_id']))
                if mapping is None:
                    raise ValueError('Provide explicit experiment_to_domain mapping; sweep id is not domain id')
                contributors.append({'domain': int(mapping), 'hkl': indices[r['row']].tolist(),
                                     'experiment_id': r['experiment_id'], 'prediction_row': r['row'],
                                     'phase_id': library.get('experiment_to_phase', {}).get(str(r['experiment_id']), 'unknown')})
            groups.append({'observation_id': group_id, 'patch': patch_name, 'contributors': contributors,
                           'basis': library['basis'], 'bounds_xyz': list(bounds), 'noise_model': noise,
                           'position_derivatives': {'available': args.position_derivatives, 'units': 'per_pixel_xyz',
                                                    'finite_difference_step_px': .001 if args.position_derivatives else None},
                           'provenance': {'raw_first_path': key[0][0], 'raw_last_path': key[0][-1],
                                          'expt': str(args.expt.resolve()), 'predictions': str(args.predictions.resolve()),
                                          'library': str(args.library.resolve()), 'panel': key[2],
                                          'profile_sources': [r['entry']['provenance'] for r in chosen]}})
    manifest = {'schema': 'scxrd-pixel-patches/1', 'implementation_status': 'experimental_untested',
                'groups': groups, 'retained_unprocessed_predictions': issues,
                'support_sigma': args.support_sigma, 'background_margin': args.background_margin,
                'complete_prediction_provenance': library['complete_prediction_provenance'],
                'limitations': 'fixed regularized local profile interpolation; covariance conditional on calibration; no geometry refinement or absorption scaling inside this adapter'}
    (args.output / 'patches.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(args.output / 'patches.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('expt', type=Path)
    p.add_argument('predictions', type=Path)
    p.add_argument('library', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--support-sigma', type=float, default=5.)
    p.add_argument('--background-margin', type=int, default=3)
    p.add_argument('--max-voxels', type=int, default=2000000)
    p.add_argument('--max-components', type=int, default=32)
    p.add_argument('--frame-cache', type=int, default=8)
    p.add_argument('--template-cache', type=int, default=32)
    p.add_argument('--position-derivatives', action='store_true', help='Store XYZ profile derivatives for signed-WLS position nuisance fitting')
    args = p.parse_args()
    try:
        prepare(args)
        return 0
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

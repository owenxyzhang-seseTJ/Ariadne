#!/usr/bin/env python3
"""Nonexclusive q + native detector/scan compatibility, with real covariance units.

Input predictions must cover all observable reflections from the supplied
multi-experiment model. Compatibility is not a calibrated membership posterior,
pixel intensity fraction, or twin volume fraction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory


def raw_key(experiment):
    if not experiment.imageset or not experiment.scan:
        raise ValueError('This adapter requires rotation imagesets and scans')
    return (tuple(experiment.imageset.paths()), tuple(experiment.scan.get_array_range()))


def signature(experiment, panel):
    detector = experiment.detector[panel]
    return np.asarray([*experiment.beam.get_s0(), *detector.get_origin(), *detector.get_fast_axis(),
                       *detector.get_slow_axis(), *detector.get_pixel_size(), *experiment.scan.get_oscillation(),
                       *experiment.goniometer.get_fixed_rotation(), *experiment.goniometer.get_setting_rotation(),
                       *experiment.goniometer.get_rotation_axis()], dtype=float)


def assign(args):
    observed_experiments = ExperimentListFactory.from_json_file(str(args.observed_expt), check_format=False)
    models = ExperimentListFactory.from_json_file(str(args.models_expt), check_format=False)
    if any(not m.crystal or not m.goniometer or m.crystal.num_scan_points or m.beam.num_scan_points or m.goniometer.num_scan_points for m in models):
        raise ValueError('Soft-matching MVP requires static crystals/shared instrument models; segment dynamic data explicitly')
    spots = flex.reflection_table.from_file(str(args.spots))
    predictions = flex.reflection_table.from_file(str(args.predictions))
    if not len(spots) or args.max_mahalanobis2 <= 0:
        raise ValueError('Need observed spots and a positive compatibility threshold')
    if 'imageset_id' not in spots and len(observed_experiments) == 1 and spots['id'].all_eq(-1):
        spots['imageset_id'] = flex.int(len(spots), 0)
    config = json.loads(args.calibration.read_text())
    for name in ('model_covariance_source', 'centroid_covariance_source', 'instrument_geometry_provenance',
                 'q_tolerance_source', 'complete_prediction_provenance', 'experiment_to_lattice'):
        if not config.get(name):
            raise ValueError('Missing evidence: ' + name)
    if config.get('prediction_scope') != 'all_observable_reflections':
        raise ValueError('Supply full model predictions, not a list of detected strong spots')
    model_cov = np.asarray(config['model_covariance_mm_rad'], dtype=float)
    if model_cov.shape != (3, 3) or not np.allclose(model_cov, model_cov.T) or np.any(np.linalg.eigvalsh(model_cov) < 0):
        raise ValueError('Model covariance must be PSD in mm/mm/rad units')
    q_tolerance = float(config['q_tolerance_invA'])
    if not np.isfinite(q_tolerance) or q_tolerance <= 0 or args.max_candidates_per_spot < 1:
        raise ValueError('Invalid geometric budget')
    if 'xyzobs.mm.value' not in spots or 'xyzobs.mm.variance' not in spots:
        if 'xyzobs.px.variance' not in spots:
            raise ValueError('Require measured centroid variances; do not use DIALS default unit weights')
        spots.centroid_px_to_mm(observed_experiments)
    spots.map_centroids_to_reciprocal_space(observed_experiments)
    y = np.asarray(spots['xyzobs.mm.value'], dtype=float)
    variances = np.asarray(spots['xyzobs.mm.variance'], dtype=float)
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError('Centroids/variances must be finite and variances positive')
    q = np.asarray(spots['rlp'], dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError('Nonfinite observed reciprocal coordinates')
    source = np.asarray(spots['scxrd_source_row'], dtype=int) if 'scxrd_source_row' in spots else np.arange(len(spots))
    observed_ids = np.asarray(spots['imageset_id'] if 'imageset_id' in spots else spots['id'], dtype=int)
    if len(observed_experiments) == 1 and np.all(observed_ids == -1):
        observed_ids = np.zeros(len(spots), dtype=int)
    panels = np.asarray(spots['panel'], dtype=int)
    pred_ids = np.asarray(predictions['id'], dtype=int)
    pred_panels = np.asarray(predictions['panel'], dtype=int)
    predicted = np.asarray(predictions['xyzcal.mm'], dtype=float)
    hkl = np.asarray(predictions['miller_index'], dtype=int)
    model_raw_keys = [raw_key(m) for m in models]
    observed_raw_keys = [raw_key(m) for m in observed_experiments]
    model_signatures = {(eid, panel): signature(m, panel) for eid, m in enumerate(models) for panel in range(len(m.detector))}
    observed_signatures = {(eid, panel): signature(m, panel) for eid, m in enumerate(observed_experiments) for panel in range(len(m.detector))}
    matrices = [np.asarray(m.crystal.get_A()).reshape(3, 3) for m in models]
    buckets = defaultdict(list)
    geometry = {}
    for row, (eid, panel) in enumerate(zip(pred_ids, pred_panels)):
        if not 0 <= eid < len(models) or not np.any(hkl[row]) or not np.all(np.isfinite(predicted[row])):
            continue
        key = (model_raw_keys[eid], int(panel))
        if str(eid) not in config['experiment_to_lattice']:
            raise ValueError('Experiment id is not a lattice id; provide explicit mapping')
        sig = model_signatures[(int(eid), int(panel))]
        if key in geometry and not np.allclose(sig, geometry[key], rtol=1e-9, atol=1e-8):
            raise ValueError('Models do not share calibrated instrument geometry')
        geometry[key] = sig
        buckets[key].append(row)
    # An upper bound on marginal uncertainty supports a conservative KD-tree screen.
    screen_scale = np.sqrt(variances.max(axis=0) + np.diag(model_cov))
    trees = {key: (np.asarray(rows), cKDTree(predicted[rows] / screen_scale)) for key, rows in buckets.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    independent = defaultdict(int)
    support = defaultdict(float)
    for i in range(len(spots)):
        eid, panel = int(observed_ids[i]), int(panels[i])
        if not 0 <= eid < len(observed_experiments):
            raise ValueError('Invalid observed imageset identifier')
        key = (observed_raw_keys[eid], panel)
        matches = []
        status = 'unexplained'
        if key in trees:
            if not np.allclose(observed_signatures[(eid, panel)], geometry[key], rtol=1e-9, atol=1e-8):
                raise ValueError('Observed and predicted coordinates use different instrument geometry')
            rows, tree = trees[key]
            # Correlations can stretch the standardized 3-D ellipsoid by sqrt(3).
            nearby = tree.query_ball_point(y[i] / screen_scale, np.sqrt(3 * args.max_mahalanobis2))
            if len(nearby) > args.max_candidates_per_spot:
                status = 'candidate_budget_exceeded'  # Retain the spot; never silently truncate possible contributors.
            else:
                C = np.diag(variances[i]) + model_cov
                for local in nearby:
                    row = int(rows[local])
                    predicted_eid = int(pred_ids[row])
                    A = matrices[predicted_eid]
                    q_error = q[i] - A @ hkl[row]
                    if np.linalg.norm(q_error) > q_tolerance:
                        continue
                    residual = y[i] - predicted[row]  # Unwrapped scan radians, not modulo 2pi across different exposures.
                    D2 = float(residual @ np.linalg.solve(C, residual))
                    if D2 <= args.max_mahalanobis2:
                        log_density = -.5 * (D2 + np.linalg.slogdet(C)[1] + 3 * np.log(2 * np.pi))
                        matches.append({'prediction_row': row, 'experiment_id': predicted_eid,
                                        'lattice_id': config['experiment_to_lattice'][str(predicted_eid)],
                                        'hkl': hkl[row].tolist(), 'mahalanobis2': D2,
                                        'compatibility': float(np.exp(-.5 * D2)), 'log_position_density': float(log_density),
                                        'q_residual_invA': float(np.linalg.norm(q_error)),
                                        'residual_mm_mm_rad': residual.tolist()})
                if matches:
                    status = 'single_explanation' if len(matches) == 1 else 'multiple_explanations'
                    if len(matches) == 1:
                        independent[str(matches[0]['lattice_id'])] += 1
                    for match in matches:
                        support[str(match['lattice_id'])] += match['compatibility']
        results.append({'spot_row': i, 'source_row': int(source[i]), 'status': status,
                        'observed_experiment_id': eid, 'frame': float(spots['xyzobs.px.value'][i][2]),
                        'matches': matches})
    manifest = {'schema': 'ariadne-soft-compatibility/1', 'status': 'experimental_untested',
                'source_spots': str(args.spots.resolve()), 'observed_expt': str(args.observed_expt.resolve()),
                'models_expt': str(args.models_expt.resolve()), 'predictions': str(args.predictions.resolve()),
                'calibration': config, 'spots': results, 'independent_support': dict(independent),
                'soft_support': dict(support), 'normalization': 'nonexclusive; no sum-to-one constraint',
                'meaning': 'geometric compatibility, not posterior membership/intensity proportion/twin fraction',
                'angle_convention': 'unwrapped native scan radians; no cross-exposure periodic aliases',
                'unexplained_and_multiple_spots': 'retained for hypothesis generation and joint pixel integration'}
    (args.output / 'compatibility.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observed_expt', type=Path)
    parser.add_argument('spots', type=Path)
    parser.add_argument('models_expt', type=Path)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-mahalanobis2', type=float, default=25.)
    parser.add_argument('--max-candidates-per-spot', type=int, default=128)
    assign(parser.parse_args())


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Joint crystal refinement with fixed instrument geometry.

Default: native DIALS unique-anchor weighting (legacy interface).
--method em: full-covariance exclusive single-source centroid EM; independent
single-source/overlap evidence is mandatory via --centroid-selection. Predeclared
--compare-models share raw scan-block holdout rows and normalized density.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
from dials.array_family import flex
from dials.algorithms.refinement import RefinerFactory
from dials.command_line.refine import phil_scope
from dxtbx.model.experiment_list import ExperimentListFactory


def refine(args):
    evidence = json.loads(args.compatibility.read_text())
    if evidence.get('schema') != 'ariadne-soft-compatibility/1':
        raise ValueError('Require nonexclusive geometric compatibility evidence')
    if Path(evidence['models_expt']).resolve() != args.models_expt.resolve() or Path(evidence['source_spots']).resolve() != args.spots.resolve():
        raise ValueError('Compatibility provenance does not match requested models/spots')
    models = ExperimentListFactory.from_json_file(str(args.models_expt), check_format=False)
    originals = flex.reflection_table.from_file(str(args.spots))
    observed_models = ExperimentListFactory.from_json_file(evidence['observed_expt'], check_format=False)
    if 'imageset_id' not in originals and len(observed_models) == 1 and originals['id'].all_eq(-1):
        originals['imageset_id'] = flex.int(len(originals), 0)
    if 'xyzobs.mm.value' not in originals or 'xyzobs.mm.variance' not in originals:
        if 'xyzobs.px.variance' not in originals:
            raise ValueError('Missing measured centroid variance')
        originals.centroid_px_to_mm(observed_models)
    blocks = sorted({(s['observed_experiment_id'], int(np.floor(s['frame'] / args.block_frames))) for s in evidence['spots']})
    if len(blocks) < 2:
        raise ValueError('Insufficient independent scan blocks for refinement holdout')
    rng = np.random.default_rng(args.seed)
    heldout = {blocks[i] for i in rng.permutation(len(blocks))[:max(1, int(len(blocks) * args.holdout_fraction))]}
    rows, matches, weights, validation_rows = [], [], [], []
    omitted_multiple = []
    for spot in evidence['spots']:
        block = (spot['observed_experiment_id'], int(np.floor(spot['frame'] / args.block_frames)))
        if block in heldout:
            validation_rows.append(spot['source_row'])
            continue
        if spot['status'] != 'single_explanation':
            omitted_multiple.append(spot['source_row'])
            continue
        match = spot['matches'][0]
        if match['compatibility'] < args.min_weight:
            continue
        rows.append(spot['spot_row'])
        matches.append(match)
        weights.append(match['compatibility'])
    coverage = Counter(m['experiment_id'] for m in matches)
    if any(coverage[eid] < args.min_anchors for eid in range(len(models))):
        raise ValueError('Each experiment needs sufficient unique training anchors; keep unresolved models without inventing support')
    table = originals.select(flex.size_t(rows))
    table['id'] = flex.int([m['experiment_id'] for m in matches])
    table['miller_index'] = flex.miller_index([tuple(m['hkl']) for m in matches])
    if 'imageset_id' in table:
        del table['imageset_id']
    mm_variance = np.asarray(table['xyzobs.mm.variance'], dtype=float)
    model_covariance = np.asarray(evidence['calibration']['model_covariance_mm_rad'])
    mm_variance = (mm_variance + np.diag(model_covariance)) / np.asarray(weights)[:, None]
    table['xyzobs.mm.variance'] = flex.vec3_double([tuple(v) for v in mm_variance])
    for key in list(table.experiment_identifiers().keys()):
        del table.experiment_identifiers()[key]
    for eid, model in enumerate(models):
        if model.identifier:
            table.experiment_identifiers()[eid] = model.identifier
    params = copy.deepcopy(phil_scope.extract())
    params.refinement.parameterisation.beam.fix = ['all']
    params.refinement.parameterisation.detector.fix = 'all'
    params.refinement.parameterisation.goniometer.fix = ['all']
    params.refinement.parameterisation.scan_varying = False
    params.refinement.parameterisation.auto_reduction.action = 'fail'
    params.refinement.refinery.max_iterations = args.max_iterations
    refiner = RefinerFactory.from_parameters_data_experiments(params, table, models)
    refiner.run()
    args.output.mkdir(parents=True, exist_ok=False)
    refiner.get_experiments().as_file(str(args.output / 'refined.expt'))
    table.as_file(str(args.output / 'training.refl'))
    report = {'schema': 'ariadne-joint-geometry/1', 'status': 'proposal_requires_holdout_review',
              'method': 'native DIALS joint refinement; unique anchors weighted by compatibility; shared instrument fixed',
              'training_source_rows': [evidence['spots'][r]['source_row'] for r in rows],
              'validation_source_rows': validation_rows, 'heldout_blocks': [list(b) for b in sorted(heldout)],
              'ambiguous_training_rows_not_duplicated': omitted_multiple, 'seed': args.seed,
              'rmsds_mm_mm_rad': list(refiner.rmsds()),
              'parent_models': str(args.models_expt.resolve()), 'compatibility': str(args.compatibility.resolve()),
              'covariance_approximation': 'diagonal centroid/model variance in DIALS objective; soft weighting is algorithmic, not independent repeated observations',
              'next': 'predict all reflections from refined.expt, rerun soft matching on all original spots, compare these fixed heldout rows; do not accept training RMSD alone',
              'tests': 'not_run_as_requested'}
    (args.output / 'joint_refinement.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('models_expt', type=Path)
    p.add_argument('spots', type=Path)
    p.add_argument('compatibility', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--block-frames', type=int, default=20)
    p.add_argument('--holdout-fraction', type=float, default=.2)
    p.add_argument('--min-anchors', type=int, default=25)
    p.add_argument('--min-weight', type=float, default=.05)
    p.add_argument('--max-iterations', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--method', choices=('anchors', 'em'), default='anchors')
    p.add_argument('--centroid-selection', type=Path,
                   help='JSON: single_source_source_rows, composite_source_rows, provenance; optional upstream training row provenance')
    p.add_argument('--compare-models', type=Path, nargs='*', default=[],
                   help='Predeclared alternatives, independently fitted on the same training rows')
    p.add_argument('--em-iterations', type=int, default=8)
    p.add_argument('--em-tolerance', type=float, default=1e-6)
    p.add_argument('--outlier-fraction', type=float, default=.05)
    p.add_argument('--max-predictions', type=int, default=20000)
    p.add_argument('--max-pairs', type=int, default=2000000)
    p.add_argument('--max-hypotheses', type=int, default=4)
    p.add_argument('--max-prediction-calls', type=int, default=500)
    p.add_argument('--max-seconds', type=float, default=300.)
    p.add_argument('--orientation-bound-mrad', type=float, default=5.)
    p.add_argument('--cell-relative-bound', type=float, default=.02)
    args = p.parse_args()
    if (args.em_iterations < 1 or args.em_tolerance <= 0
            or not 0 < args.outlier_fraction < 1 or args.max_predictions < 1
            or args.max_pairs < 1 or args.max_hypotheses < 1
            or args.max_prediction_calls < 2 or args.max_seconds <= 0
            or args.orientation_bound_mrad <= 0 or not 0 < args.cell_relative_bound < .1):
        raise ValueError('Invalid EM budget/noise settings')
    if args.block_frames < 1 or not 0 < args.holdout_fraction < .5 or args.max_iterations < 1 or args.min_anchors < 5 or not 0 < args.min_weight <= 1:
        raise ValueError('Invalid refinement/holdout budget')
    if args.method == 'em':
        from centroid_em import run
        run(args)
    else:
        refine(args)


if __name__ == '__main__':
    main()

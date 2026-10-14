#!/usr/bin/env python3
"""Compare native geometry compatibility on identical declared scan-block holdouts.

Scores are diagnostic gains, not Bayes evidence or a valid composite-spot BIC.
Alternative models remain available; this script never accepts a hypothesis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def compare(args):
    partition = json.loads(args.partition.read_text())
    heldout = set(partition['validation_source_rows'])
    if not heldout or not partition.get('heldout_blocks'):
        raise ValueError('Require actual scan-block heldout rows and partition provenance')
    evaluations = []
    sources = None
    calibration = None
    for path in args.compatibilities:
        document = json.loads(path.read_text())
        if document.get('schema') != 'ariadne-soft-compatibility/1':
            raise ValueError('Unsupported compatibility evidence')
        identity = (document['source_spots'], document['observed_expt'])
        assumptions = document['calibration']
        # Complete-prediction provenance legitimately changes with a hypothesis.
        assumptions = {k: v for k, v in assumptions.items() if k not in
                       ('experiment_to_lattice', 'complete_prediction_provenance')}
        if sources is not None and (identity != sources or assumptions != calibration):
            raise ValueError('Models must use the same observations, calibration and noise assumptions')
        sources, calibration = identity, assumptions
        rows = {s['source_row']: s for s in document['spots']}
        if not heldout.issubset(rows):
            raise ValueError('A hypothesis omits heldout observations')
        selected = [rows[i] for i in sorted(heldout)]
        scores = [max((m['compatibility'] for m in s['matches']), default=0.) for s in selected]
        independent = {}
        for spot in selected:
            if len(spot['matches']) == 1:
                model = str(spot['matches'][0]['lattice_id'])
                independent[model] = independent.get(model, 0) + 1
        evaluations.append({'compatibility_artifact': str(path.resolve()), 'models': document['models_expt'],
                            'heldout_rows': len(heldout), 'heldout_mean_best_compatibility': float(np.mean(scores)),
                            'heldout_explained_fraction': float(np.mean([len(s['matches']) > 0 for s in selected])),
                            'heldout_unique_support': independent,
                            'heldout_multiple_explanation_fraction': float(np.mean([len(s['matches']) > 1 for s in selected])),
                            'budget_exceeded_rows': [s['source_row'] for s in selected if s['status'] == 'candidate_budget_exceeded']})
    baseline = evaluations[0]['heldout_mean_best_compatibility']
    for evaluation in evaluations:
        evaluation['gain_against_first_baseline'] = evaluation['heldout_mean_best_compatibility'] - baseline
    output = {'schema': 'ariadne-hypothesis-comparison/1', 'evaluations': evaluations,
              'partition': str(args.partition.resolve()), 'decision': 'requires scientific evidence selection; no automatic acceptance',
              'score_semantics': 'nonexclusive geometric diagnostic; not normalized model likelihood or BIC',
              'additional_evidence_required': ['candidate equivalence/supercell checks', 'generation stability',
                  'independent support', 'full pixel-fit gain', 'phase/twin-law evidence'],
              'holdout_scope': 'refinement holdout; candidate generation may already have seen these spots'}
    with args.output.open('x') as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('partition', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('compatibilities', type=Path, nargs='+')
    compare(parser.parse_args())


if __name__ == '__main__':
    main()

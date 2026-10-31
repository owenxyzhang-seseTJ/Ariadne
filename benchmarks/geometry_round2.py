#!/usr/bin/env python3
"""First-principles geometry changes against the frozen first-round source.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Native DIALS measures
the same synthetic refinement, not experimental indexing or detector accuracy.
Peak memory is tracemalloc-traced allocation during the call, excluding inputs.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc
import types
import zipfile

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'skills/ariadne-reduction/scripts'), str(ROOT / 'tests')]
import centroid_em as updated
from rotation_geometry import compatibility_metrics


def baseline(root):
    name = 'skills/ariadne-reduction/scripts/centroid_em.py'
    if root is not None:
        source = (root / name).read_text()
    else:
        with zipfile.ZipFile(ROOT / 'benchmarks/baseline-first-principles.zip') as archive:
            source = archive.read(name).decode()
    module = types.ModuleType('ariadne_geometry_first_round')
    sys.modules[module.__name__] = module
    exec(compile(source, str(root or 'baseline-first-principles.zip') + '/' + name, 'exec'), module.__dict__)
    return module


def timed(function, repeats):
    seconds = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        seconds.append(time.perf_counter() - started)
    gc.collect()
    tracemalloc.start()
    measured = function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del measured
    return output, {'seconds': seconds, 'median_seconds': float(np.median(seconds)),
                    'peak_traced_bytes_excluding_inputs': int(peak)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/geometry_round2_measurements.json')
    parser.add_argument('--skip-native', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old = baseline(args.baseline_root)
    rng = np.random.default_rng(20261029)
    y, means = rng.normal(size=(400, 3)), rng.normal(size=(5000, 3))
    matrix = rng.normal(size=(400, 3, 3))
    covariance = matrix @ matrix.transpose(0, 2, 1) + .2 * np.eye(3)
    white, normalizer = updated.covariance_factors(covariance)
    allowed = rng.uniform(size=(400, 5000)) > .2
    inputs = (y, means, white, normalizer, allowed, np.full(400, 5000), np.full(400, 1e-5), .05)
    old_value, old_time = timed(lambda: old.expectation(*inputs), args.repeats)
    new_value, new_time = timed(lambda: updated.expectation(*inputs), args.repeats)
    rows = [{'name': 'complete_gaussian_mixture_2000000_pairs',
             'observations': 400, 'predictions': 5000,
             'before': old_time, 'after': new_time,
             'elapsed_ratio': old_time['median_seconds'] / new_time['median_seconds'],
             'peak_allocation_ratio': old_time['peak_traced_bytes_excluding_inputs'] / new_time['peak_traced_bytes_excluding_inputs'],
             'maximum_absolute_differences': {name: float(np.max(np.abs(before - after)))
                 for name, before, after in zip(('responsibility', 'outlier', 'log_density'), old_value, new_value)}}]
    print(rows[-1]['name'], flush=True)
    targets = means[:128]
    C = covariance[0]

    def old_metrics():
        output = []
        for observation in y:
            values = []
            for target in targets:
                residual = observation - target
                distance = residual @ np.linalg.solve(C, residual)
                values.append(-.5 * (distance + np.linalg.slogdet(C)[1] + 3 * np.log(2 * np.pi)))
            output.append(values)
        return np.asarray(output)

    before, before_time = timed(old_metrics, args.repeats)
    after, after_time = timed(lambda: np.array([compatibility_metrics(row, targets, C)[2] for row in y]), args.repeats)
    rows.append({'name': 'nonexclusive_compatibility_400_by_128_candidates', 'before': before_time,
                 'after': after_time, 'elapsed_ratio': before_time['median_seconds'] / after_time['median_seconds'],
                 'max_log_density_difference': float(np.max(np.abs(before - after)))})
    print(rows[-1]['name'], flush=True)
    native = None
    if not args.skip_native:
        import test_centroid_em as fixtures

        def refinement(module):
            adapter = module.DialsCrystalAdapter(fixtures.DialsAdapterTests.experiments(), module.Budget(500, 120))
            truth = adapter.reference.copy()
            selected = np.concatenate([np.flatnonzero(adapter.ids == eid)[np.linspace(0, (adapter.ids == eid).sum() - 1, 35, dtype=int)] for eid in (0, 1)])
            observations = truth[selected]
            adapter.x0 = adapter.x0.copy()
            adapter.x0[[0, 9]] += .4
            adapter.x0[[3, 12]] *= 1.0003
            C = np.array([[.02, .004, 1e-5], [.004, .018, -1e-5], [1e-5, -1e-5, 1e-6]])
            started = time.perf_counter()
            result = module.fit_em(adapter, observations, np.tile(C, (len(observations), 1, 1)),
                np.ones((len(observations), len(truth)), bool), np.full(len(observations), len(truth)),
                np.full(len(observations), 1e-9), min_support=20, em_iterations=5, max_nfev=12)
            elapsed = time.perf_counter() - started
            rms = np.sqrt(np.mean((result['means'][selected] - observations)**2, axis=0))
            return result, {'seconds': elapsed, 'prediction_calls_including_support': adapter.budget.calls,
                            'status': result['status'], 'rank': result['jacobian_rank'],
                            'predictions': len(truth), 'parameters': len(adapter.x0),
                            'rms_mm_mm_rad': rms.tolist(),
                            'log_likelihood_history': result['training_log_likelihood_history']}

        before_runs, after_runs = [], []
        for _ in range(args.repeats):
            before_fit, row = refinement(old)
            before_runs.append(row)
            after_fit, row = refinement(updated)
            after_runs.append(row)
        before_median, after_median = (float(np.median([x['seconds'] for x in runs])) for runs in (before_runs, after_runs))
        native = {'scenario': 'two static crystals; complete native prediction set; 70 synthetic single-source observations; correlated fixed covariance',
                  'before_runs': before_runs, 'after_runs': after_runs,
                  'before_median_seconds': before_median, 'after_median_seconds': after_median,
                  'equivalent_completion_speedup': before_median / after_median if before_fit['status'] == after_fit['status'] == 'converged' else None,
                  'max_prediction_difference': float(np.max(np.abs(before_fit['means'] - after_fit['means']))),
                  'final_log_likelihood_difference': float(after_fit['log_density'].sum() - before_fit['log_density'].sum())}
        print('native_refinement', flush=True)
    report = {'schema': 'ariadne-geometry-round2-benchmark/1', 'baseline': 'first-principles delivery of 2026-10-29',
              'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform()},
              'scope': 'synthetic known-geometry kernels; no experimental weak-spot recovery claim',
              'memory_scope': 'tracemalloc peak allocation during one call; preallocated inputs excluded; not process RSS',
              'repeats': args.repeats, 'measurements': rows, 'native_refinement': native}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

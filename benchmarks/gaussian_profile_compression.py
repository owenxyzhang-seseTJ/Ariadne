#!/usr/bin/env python3
"""Reproduce Gaussian NNLS + all profile intervals with/without augmented QR.

Run with Python, NumPy and SciPy. Defaults write the raw measurements to
reports/current_validation/gaussian_profile_compression.json. The baseline
uses the same solver with only _compress_gaussian replaced by the identity;
no archived source, test imports, native extension or temporary files are used.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

# Set before importing NumPy/SciPy so each standalone invocation uses one thread.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/ariadne-reduction/scripts'))
import constrained_pixels


def problem(rows):
    rng = np.random.default_rng(6817)
    coordinate = np.linspace(-3, 3, rows)
    profiles = np.exp(-.5 * ((coordinate[:, None] - [-.8, .2, 1.1]) / .65)**2)
    profiles /= profiles.sum(axis=0)
    signal = profiles @ np.array([2500., 0., 1800.]) + rng.normal(0., .8, rows)
    profiles = np.vstack((profiles, np.zeros((16, 3))))
    variance = np.r_[rng.uniform(.7, 1.8, rows), np.ones(16)]
    data = np.r_[signal + 3., np.full(16, 3.)]
    valid = np.ones(len(data), bool)
    valid[13:19] = False
    return data, variance, profiles, valid, np.ones((len(data), 1))


def summary(result, args):
    data, variance, profiles, valid, _ = args
    weighted = profiles[valid] / np.sqrt(variance[valid, None])
    units = 1 / np.linalg.norm(weighted, axis=0)
    gradient = (weighted * units).T @ ((result['fitted'][valid] - data[valid]) / np.sqrt(variance[valid]))
    intervals = result['profile_intervals'] + [result['total_interval']]
    return {
        'converged': bool(result['converged']), 'rank': int(result['rank']),
        'boundary_components': result['boundary_components'],
        'intensities': result['intensities_internal'].tolist(),
        'objective': float(result['chi2']),
        'pixel_projected_gradient_inf_norm': constrained_pixels._projected_error(
            result['intensities_internal'] / units, gradient),
        'intervals': [{key: value[key] for key in ('status', 'lower', 'upper')} for value in intervals],
        'optimization_observations': result['likelihood_diagnostics']['likelihood_factorization']['optimization_observations'],
    }


def compare(before, after):
    assert before['converged'] and after['converged']
    assert before['rank'] == after['rank']
    assert before['boundary_components'] == after['boundary_components']
    for result in (before, after):
        assert result['pixel_projected_gradient_inf_norm'] <= 1e-6
    differences = []
    for first, second in zip(before['intervals'], after['intervals']):
        assert first['status'] == second['status'] == 'ok'
        for key in ('lower', 'upper'):
            differences.append(abs(first[key] - second[key]))
            np.testing.assert_allclose(first[key], second[key], rtol=2e-9, atol=2e-6)
    np.testing.assert_allclose(before['intensities'], after['intensities'], rtol=2e-9, atol=2e-6)
    np.testing.assert_allclose(before['objective'], after['objective'], rtol=2e-12, atol=2e-9)
    return {
        'max_endpoint_absolute_difference': max(differences),
        'max_intensity_absolute_difference': float(np.max(np.abs(np.array(before['intensities']) - after['intensities']))),
        'objective_absolute_difference': abs(before['objective'] - after['objective']),
    }


def measure(rows, repeats):
    args = problem(rows)
    times = {'uncompressed': [], 'compressed': []}
    runs = []
    for repeat in range(repeats):
        results = {}
        # Reverse the order on alternate repeats to limit ordering effects.
        order = ('uncompressed', 'compressed') if repeat % 2 == 0 else ('compressed', 'uncompressed')
        for name in order:
            if name == 'uncompressed':
                context = patch.object(constrained_pixels, '_compress_gaussian', side_effect=lambda A, b: (A, b))
            else:
                context = nullcontext()
            with context:
                started = time.perf_counter()
                result = constrained_pixels.fit_constrained(*args, None, 'nnls', 1e-8)
                elapsed = time.perf_counter() - started
            times[name].append(elapsed)
            results[name] = summary(result, args)
        runs.append({'repeat': repeat + 1, 'order': list(order),
                     'seconds': {name: times[name][-1] for name in times},
                     'results': results,
                     'comparison': compare(results['uncompressed'], results['compressed'])})
    medians = {name: statistics.median(values) for name, values in times.items()}
    return {'requested_signal_rows': rows, 'valid_signal_rows': rows - 6,
            'component_count': 3, 'seconds': times, 'median_seconds': medians,
            'speedup': medians['uncompressed'] / medians['compressed'], 'runs': runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--rows', type=int, nargs='+', default=[64, 700, 1600, 12000, 60000])
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports/current_validation/gaussian_profile_compression.json')
    args = parser.parse_args()
    if args.repeats < 1 or min(args.rows) < 20:
        parser.error('Require repeats >= 1 and rows >= 20')
    report = {
        'schema': 'ariadne-gaussian-profile-compression-benchmark/1',
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'numpy': np.__version__, 'scipy': scipy.__version__,
                        'platform': platform.platform(), 'machine': platform.machine(),
                        'thread_environment': {key: os.environ[key] for key in (
                            'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}},
        'baseline': 'Current fit_constrained with only _compress_gaussian replaced by identity; original uncompressed pixel solver',
        'timing': 'perf_counter around complete fit_constrained including all component/total profile intervals; no excluded warmup; alternating order',
        'input': 'Synthetic seed 6817; three normalized overlapping Gaussian profiles; 16 background-only pixels; six masked signal pixels; heterogeneous fixed variance',
        'limitations': 'Synthetic Gaussian NNLS only; single process/thread; not real observations or Poisson/PG; timings vary by machine and load',
        'cases': [],
    }
    for rows in args.rows:
        case = measure(rows, args.repeats)
        report['cases'].append(case)
        print(f"{rows} rows: {1000*case['median_seconds']['uncompressed']:.3f} -> "
              f"{1000*case['median_seconds']['compressed']:.3f} ms; {case['speedup']:.2f}x", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()

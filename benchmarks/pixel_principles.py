#!/usr/bin/env python3
"""Compare exact PG kernels with the frozen Ariadne baseline; emit real timings."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import zipfile

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills' / 'ariadne-reduction' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / 'tests'))
from poisson_gaussian import PoissonGaussian
from test_pixel_principles import decimal_high_count_density, fit_pixels


def baseline_class(ref, source=None):
    frozen = ROOT / 'benchmarks/baseline-797ac22.zip'
    if source:
        code = Path(source).read_text()
    elif frozen.exists() and ref == '797ac22ffa9b81da2dd977e013bea8f114b49eb2':
        with zipfile.ZipFile(frozen) as archive:
            code = archive.read('skills/ariadne-reduction/scripts/poisson_gaussian.py').decode()
    else:
        code = subprocess.check_output(
            ['git', 'show', ref + ':skills/ariadne-reduction/scripts/poisson_gaussian.py'],
            cwd=ROOT, text=True)
    module = types.ModuleType('ariadne_baseline_poisson_gaussian')
    exec(compile(code, 'baseline_poisson_gaussian.py', 'exec'), module.__dict__)
    return module.PoissonGaussian


def measure(factory, y, means, gain, readout, repeats):
    times = []
    for _ in range(repeats):
        evaluator = factory(y, gain, readout)
        start = time.perf_counter()
        logs, scores, diagnostics = evaluator.evaluate(means)
        times.append(time.perf_counter() - start)
    return {'seconds': times, 'median_seconds': statistics.median(times),
            'terms_evaluated': diagnostics['terms_evaluated']}, logs, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='797ac22ffa9b81da2dd977e013bea8f114b49eb2')
    parser.add_argument('--baseline-source', help='Optional extracted baseline poisson_gaussian.py')
    parser.add_argument('--pixels', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    old = baseline_class(args.baseline_ref, args.baseline_source)
    rng = np.random.default_rng(20261029)
    means = np.exp(rng.uniform(-6, 9, args.pixels))
    gain, readout = 1., 4.
    y = gain * rng.poisson(means) + rng.normal(0, np.sqrt(readout), args.pixels)
    before, old_logs, old_scores = measure(old, y, means, gain, readout, args.repeats)
    after, logs, scores = measure(PoissonGaussian, y, means, gain, readout, args.repeats)
    high_count = []
    for mean in (1e8, 1e9, 1e12):
        observed = mean + .3
        reference = decimal_high_count_density(observed, mean)
        b = old([observed], 1., .2).evaluate([mean])[0][0]
        a = PoissonGaussian([observed], 1., .2).evaluate([mean])[0][0]
        high_count.append({'mean_photons': mean, 'reference_log_density_80_digit': reference,
            'before_absolute_log_error': abs(float(b)-reference),
            'after_absolute_log_error': abs(float(a)-reference)})
    P = np.array([[.6], [.3], [.1]])
    fit = fit_pixels(np.array([1.2, -.3, 2.4]), P, background=.5, gain=1.4, readout=1.1)
    amplitude = float(fit['intensities_internal'][0])
    proxy = 1/np.sum(P[:, 0]**2 / (1.4*(.5+P[:, 0]*amplitude)+1.1))
    axis = np.linspace(-4, 4, 41)
    profiles = np.column_stack((np.exp(-(axis+.4)**2/.8), np.exp(-(axis-.4)**2/.8)))
    profiles /= profiles.sum(axis=0)
    generator = np.random.default_rng(21)
    truth = np.array([1000., 1.])
    signal = generator.poisson(.2 + profiles @ truth) + generator.normal(0, 1., len(axis))
    overlap = fit_pixels(signal, profiles, background=.2)
    result = {
        'baseline_commit': args.baseline_ref,
        'scope': 'Synthetic exact PG numerical kernels and conditional fixed-background profile fits; not raw diffraction reduction',
        'seed': 20261029,
        'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                        'platform': platform.platform()},
        'batched_convolution': {'pixels': args.pixels, 'gain_adu_per_photon': gain,
            'readout_variance_adu2': readout, 'before': before, 'after': after,
            'speedup': before['median_seconds']/after['median_seconds'],
            'max_old_new_log_difference': float(np.max(np.abs(logs-old_logs))),
            'max_old_new_score_difference': float(np.max(np.abs(scores-old_scores)))},
        'high_count_stability': high_count,
        'low_count_covariance': {'fitted_intensity_adu': amplitude, 'moment_proxy_variance_adu2': float(proxy),
            'observed_information_variance_adu2': float(fit['covariance_identifiable_subspace'][0, 0]),
            'comparison': 'Observed information is the finite-difference-verified curvature of the exact PG likelihood; this does not prove frequentist coverage'},
        'strong_weak_overlap': {'truth_adu': truth.tolist(),
            'frozen_background_adu': float(overlap['background_coefficients'][0]),
            'fitted_intensities_adu': overlap['intensities_internal'].tolist(),
            'weak_profile_interval': overlap['profile_intervals'][1],
            'interpretation': 'The weak 1 ADU component is not resolved in this realization; a boundary estimate and broad conditional interval are retained'}
    }
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(output)
        print(args.output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()

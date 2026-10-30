#!/usr/bin/env python3
"""Reproduce information recovery and signed weak-intensity Monte Carlo.

Synthetic, fixed-geometry independent observations with known pixel variances.
This is an identifiable inverse-problem test, not raw-frame reduction validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/ariadne-reduction/scripts'))
from overlap_integrator import solve_joint, contrast
from composite_scaling import fit_global


def run(seed=20261029, trials=500):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, 120)
    p = np.exp(-.5 * ((x + 1.6) / .6)**2)
    q = np.exp(-.5 * ((x - 1.4) / .7)**2)
    p /= p.sum()
    q /= q.sum()
    designs = [np.column_stack((p, p, q)), np.column_stack((p, q, q))]
    truth = np.array([1200., 12., .8])
    background = .6
    repeats = 40
    means = [background + design @ truth for design in designs]
    variances = [mean / repeats for mean in means]
    B, valid = np.ones((len(x), 1)), np.ones(len(x), bool)
    labels = ['domain-A:hkl', 'domain-B:hkl', 'domain-C:hkl']

    def compress(data):
        blocks, totals, local = [], [], []
        for P, y, v in zip(designs, data, variances):
            result = solve_joint(y, v, P, valid, B, 1e-8)
            modes = result['information_modes']
            blocks.append(dict(y=modes['y'], design=modes['design'], covariance=modes['covariance'],
                               equivalence_ids=labels, features=np.empty((3, 0)), known_factors=np.ones(3)))
            total = contrast(result, np.ones(3), 1e-7)
            totals.append(dict(y=[total['intensity']], design=np.ones((1, 3)), covariance=[[total['variance']]],
                               equivalence_ids=labels, features=np.empty((3, 0)), known_factors=np.ones(3)))
            local.append(result)
        return blocks, totals, local

    blocks, totals, local = compress(means)
    new, old = fit_global(blocks), fit_global(totals)
    covariance = np.asarray(new['joint_covariance'])
    sigma = np.sqrt(np.diag(covariance))
    # An independent raw-pixel simultaneous solve includes separate background
    # amplitudes for the two observations; its I covariance must equal modes.
    raw_design = np.zeros((2 * len(x), 5))
    raw_target = []
    for i, (P, y, v) in enumerate(zip(designs, means, variances)):
        rows = slice(i * len(x), (i + 1) * len(x))
        raw_design[rows, :3] = P / np.sqrt(v[:, None])
        raw_design[rows, 3 + i] = 1 / np.sqrt(v)
        raw_target.extend(y / np.sqrt(v))
    reference = np.linalg.lstsq(raw_design, raw_target, rcond=None)[0]
    raw_covariance = np.linalg.inv(raw_design.T @ raw_design)[:3, :3]
    estimates = []
    start = perf_counter()
    for _ in range(trials):
        observed = [rng.poisson(repeats * mean) / repeats for mean in means]
        observations, _, _ = compress(observed)
        fit = fit_global(observations)
        if fit['status'] != 'converged':
            raise RuntimeError('Synthetic fixed-geometry fit unexpectedly unresolved')
        estimates.append(fit['I'])
    estimates = np.asarray(estimates)
    errors = estimates - truth
    return {
        'schema': 'ariadne-twin-information-experiment/1', 'seed': seed,
        'scope': 'synthetic independent Poisson pixels; true fixed variance supplied; calibrated profiles and static known geometry',
        'domain_intensities': truth.tolist(), 'strong_to_weak_ratio': float(truth[0] / truth[-1]),
        'repeats_per_geometry': repeats, 'geometries': 2,
        'local_signal_ranks': [result['rank'] for result in local],
        'legacy_total_only_rank': old['rank'], 'legacy_total_only_status': old['status'],
        'information_modes_rank': new['rank'], 'information_modes_status': new['status'],
        'noiseless_recovered_intensities': np.asarray(new['I']).tolist(),
        'maximum_noiseless_amplitude_error': float(np.max(np.abs(new['I'] - truth))),
        'maximum_raw_pixel_reference_difference': float(np.max(np.abs(new['I'] - reference[:3]))),
        'maximum_raw_pixel_covariance_difference': float(np.max(np.abs(covariance - raw_covariance))),
        'pixel_observations': 2 * len(x), 'compressed_observations': sum(len(b['y']) for b in blocks),
        'conditional_standard_errors': sigma.tolist(),
        'monte_carlo_trials': trials, 'monte_carlo_seconds': perf_counter() - start,
        'mean_estimates': estimates.mean(axis=0).tolist(),
        'mean_bias': errors.mean(axis=0).tolist(),
        'bias_monte_carlo_standard_error': (estimates.std(axis=0, ddof=1) / np.sqrt(trials)).tolist(),
        'rmse': np.sqrt(np.mean(errors * errors, axis=0)).tolist(),
        'negative_estimate_fractions': np.mean(estimates < 0, axis=0).tolist(),
        'nominal_95pct_gaussian_interval_coverage': np.mean(np.abs(errors) <= 1.95996398454 * sigma, axis=0).tolist(),
        'positive_clipping_bias': (np.maximum(estimates, 0).mean(axis=0) - truth).tolist(),
        'qualification': 'coverage is this synthetic experiment only; no universal low-count Gaussian coverage or weak-peak detection claim',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trials', type=int, default=500)
    args = parser.parse_args()
    value = run(trials=args.trials)
    serialized = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    else:
        print(serialized, end='')


if __name__ == '__main__':
    main()

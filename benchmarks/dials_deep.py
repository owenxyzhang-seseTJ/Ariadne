#!/usr/bin/env python3
"""Deep, held-out DIALS comparison for spot masks and profile intensities.

The spot suite reports threshold-plus-component accuracy, not a model of DIALS'
entire indexing pipeline. The profile suite uses the same fixed background and
DIALS-compatible IRLS policy. No test-set parameter selection is performed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

from ariadne.spots import propose
from dials.algorithms.image.threshold import (
    DispersionExtendedThreshold,
    DispersionThreshold,
)
from dials.array_family import flex
from dials.algorithms.integration.fit import ProfileFitter
from overlap_integrator import solve_irls_known_background


IMAGE_SIZE = 512


def synthetic_image(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))
    base = float(rng.choice([1., 4., 10.]))
    readout = float(rng.choice([0., 3., 8.])) if rng.random() < .6 else 0.
    gy, gx = rng.normal(0, .004, 2)
    expectation = base + gy*(yy-IMAGE_SIZE/2) + gx*(xx-IMAGE_SIZE/2)
    expectation = np.maximum(expectation, .2)
    truth = []
    for y0 in np.linspace(34, IMAGE_SIZE-34, 7):
        for x0 in np.linspace(34, IMAGE_SIZE-34, 7):
            flux = 10*np.exp(rng.uniform(np.log(2), np.log(160)))
            sigma = rng.uniform(.55, 2.5)
            cy, cx = y0+rng.normal(0, 4), x0+rng.normal(0, 4)
            gaussian = np.exp(-.5*((yy-cy)**2+(xx-cx)**2)/sigma**2)
            expectation += flux*gaussian/gaussian.sum()
            truth.append(dict(y=cy, x=cx, flux=flux, sigma=sigma,
                              background=base, readout_variance=readout))
    image = rng.poisson(expectation).astype(float)
    if readout:
        image += rng.normal(0, np.sqrt(readout), image.shape)
    mask = rng.random(image.shape) > .008
    return image, mask, truth


def components(mask, minimum=4):
    groups, count = label(mask)
    flat = groups.ravel()
    sizes = np.bincount(flat, minlength=count+1)
    yy, xx = np.indices(mask.shape)
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=count+1)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=count+1)
    keep = np.flatnonzero((sizes >= minimum) & (sizes <= 1000))
    keep = keep[keep != 0]
    denominator = sizes[keep]
    return np.column_stack((sum_y[keep]/denominator, sum_x[keep]/denominator))


def score_components(found, truth):
    truth_points = np.asarray([[item['y'], item['x']] for item in truth])
    if not len(found):
        pairs = []
    else:
        distances = np.linalg.norm(truth_points[:, None]-found[None, :], axis=2)
        rows, columns = linear_sum_assignment(distances)
        pairs = [(int(row), int(column), float(distances[row, column]))
                 for row, column in zip(rows, columns) if distances[row, column] <= 3]
    matched = {row for row, _, _ in pairs}
    return {
        'tp': len(pairs),
        'fp': len(found)-len(pairs),
        'fn': len(truth)-len(pairs),
        'weak_tp': sum(truth[row]['flux'] < 50 for row in matched),
        'weak_total': sum(item['flux'] < 50 for item in truth),
        'strong_tp': sum(truth[row]['flux'] >= 200 for row in matched),
        'strong_total': sum(item['flux'] >= 200 for item in truth),
        'localization_error': [error for _, _, error in pairs],
    }


def aggregate(rows):
    total = {key: int(sum(row[key] for row in rows))
             for key in ('tp', 'fp', 'fn', 'weak_tp', 'weak_total', 'strong_tp', 'strong_total')}
    precision = total['tp']/max(1, total['tp']+total['fp'])
    recall = total['tp']/max(1, total['tp']+total['fn'])
    total.update(
        precision=precision, recall=recall,
        f1=2*precision*recall/max(1e-12, precision+recall),
        weak_recall=total['weak_tp']/max(1, total['weak_total']),
        strong_recall=total['strong_tp']/max(1, total['strong_total']),
        localization_median_pixels=statistics.median(
            [value for row in rows for value in row['localization_error']]) if any(
                row['localization_error'] for row in rows) else None,
    )
    return total


def ariadne_mask(image, mask, item, mode):
    readout = item['readout_variance']
    if mode == 'legacy_deviance':
        statistic, threshold, support, minimum = 'deviance', 6., 1., 3
    else:
        statistic, threshold, support, minimum = 'standardized', 4., .5, 4
    return propose(
        image, mask, gain=1., offset=0., readout_variance=readout,
        noise_model='poisson_gaussian' if readout else 'poisson',
        background_width=31, exclusion_width=7, scales=[1, 3, 5],
        score_threshold=threshold, support_sigma=support,
        min_valid_fraction=.5, background_iterations=0,
        statistic=statistic, component_min_size=minimum)


def dials_mask(image, mask, sigma_background=6., sigma_strong=3., extended=False):
    algorithm_type = DispersionExtendedThreshold if extended else DispersionThreshold
    algorithm = algorithm_type(
        image.shape, (3, 3), sigma_background, sigma_strong, 0., 2)
    output = flex.bool(flex.grid(*image.shape), False)
    algorithm(flex.double(image), flex.bool(mask), output)
    return output.as_numpy_array()


def spot_suite(seeds, dials_config):
    cases = [synthetic_image(seed) for seed in seeds]
    algorithms = {
        'ariadne_legacy_deviance': lambda image, mask, truth:
            components(ariadne_mask(image, mask, truth[0], 'legacy_deviance'), 3),
        'ariadne_standardized': lambda image, mask, truth:
            components(ariadne_mask(image, mask, truth[0], 'standardized'), 4),
        'dials_dispersion': lambda image, mask, truth:
            components(dials_mask(
                image, mask, dials_config['sigma_background'],
                dials_config['sigma_strong'], False), dials_config['min_size']),
        'dials_dispersion_extended': lambda image, mask, truth:
            components(dials_mask(
                image, mask, dials_config['sigma_background'],
                dials_config['sigma_strong'], True), dials_config['min_size']),
    }
    result = {}
    for name, function in algorithms.items():
        rows = []
        for image, mask, truth in cases:
            rows.append(score_components(function(image, mask, truth), truth))
        result[name] = aggregate(rows)
    return result


def select_dials_configuration(seeds):
    cases = [synthetic_image(seed) for seed in seeds]
    candidates = []
    for sigma_background in (4., 6., 8.):
        for sigma_strong in (3., 4.):
            for minimum in (3, 4, 5):
                for extended in (False, True):
                    rows = []
                    for image, mask, truth in cases:
                        found = components(dials_mask(
                            image, mask, sigma_background, sigma_strong, extended),
                            minimum)
                        rows.append(score_components(found, truth))
                    summary = aggregate(rows)
                    candidates.append(dict(
                        sigma_background=sigma_background,
                        sigma_strong=sigma_strong,
                        min_size=minimum,
                        extended=extended,
                        training=summary))
    return max(candidates, key=lambda item: item['training']['f1'])


def gaussian_profiles(count, rng):
    shape = (13, 17, 17)
    coordinates = np.indices(shape).reshape(3, -1).T
    centres = []
    for index in range(count):
        centres.append([4.5+.37*index, 7.5+1.4*(index % 3),
                        8.5+1.6*((index//3) % 3)])
    profiles = np.column_stack([
        np.exp(-.5*np.sum(((coordinates-centre)/1.35)**2, axis=1))
        for centre in centres])
    profiles /= profiles.sum(axis=0)
    return profiles


def profile_suite(seed_start, trials):
    rows = []
    timings = {'dials': [], 'ariadne': []}
    for case_index, count in enumerate((1, 2, 4, 8)):
        for index in range(trials):
            rng = np.random.default_rng(seed_start+case_index*1000+index)
            profiles = gaussian_profiles(count, rng)
            truth = np.geomspace(1000., max(4., 20./max(1, count-1)), count)
            background = np.full(len(profiles), 2.)
            expectation = background+profiles@truth
            if index % 2:
                observed = rng.normal(expectation, np.sqrt(np.maximum(expectation, .5)))
            else:
                observed = rng.poisson(expectation).astype(float)
            valid = rng.random(len(profiles)) > .02
            dials_fit = ProfileFitter(
                flex.double(observed), flex.double(background), flex.bool(valid),
                flex.double(np.ascontiguousarray(profiles.T)), 1e-6, 100)
            dials_intensity = np.asarray(dials_fit.intensity())
            started = time.perf_counter()
            ariadne = solve_irls_known_background(
                observed, background, profiles, valid, 1e-10,
                tolerance=1e-12, max_iterations=100)
            ariadne_seconds = time.perf_counter()-started
            ariadne_intensity = ariadne['intensities_internal']
            np.testing.assert_allclose(ariadne_intensity, dials_intensity,
                                       rtol=1e-8, atol=1e-6)
            rows.append(dict(
                components=count,
                ariadne_rmse=float(np.sqrt(np.mean(
                    (ariadne_intensity-truth)**2/np.maximum(truth, 1)**2))),
                dials_rmse=float(np.sqrt(np.mean(
                    (dials_intensity-truth)**2/np.maximum(truth, 1)**2))),
                maximum_intensity_difference=float(np.max(np.abs(
                    ariadne_intensity-dials_intensity))),
                ariadne_iterations=ariadne['iterations'],
                dials_iterations=int(dials_fit.niter())+1))
            timings['ariadne'].append(ariadne_seconds)
            started = time.perf_counter()
            ProfileFitter(
                flex.double(observed), flex.double(background), flex.bool(valid),
                flex.double(np.ascontiguousarray(profiles.T)), 1e-6, 100)
            timings['dials'].append(time.perf_counter()-started)
    return {
        'trials_per_component_count': trials,
        'timing_samples': trials*4,
        'time_median_seconds': {
            name: statistics.median(values) for name, values in timings.items()},
        'ariadne_over_dials': (
            statistics.median(timings['ariadne'])/statistics.median(timings['dials'])),
        'maximum_scaled_rmse_difference': max(
            abs(row['ariadne_rmse']-row['dials_rmse']) for row in rows),
        'maximum_intensity_difference': max(
            row['maximum_intensity_difference'] for row in rows),
        'results': rows,
    }


def covariance_coverage(seed_start, trials):
    z = 1.959963984540054
    result = {}
    for count in (1, 2, 4):
        covered_components = {'ariadne': [], 'dials': []}
        covered_totals = {'ariadne': [], 'dials': []}
        for index in range(trials):
            rng = np.random.default_rng(seed_start+count*1000+index)
            profiles = gaussian_profiles(count, rng)
            truth = np.geomspace(1000., 100., count)
            background = np.full(len(profiles), 2.)
            expectation = background+profiles@truth
            observed = rng.normal(expectation, np.sqrt(expectation))
            valid = np.ones(len(observed), bool)
            dials_fit = ProfileFitter(
                flex.double(observed), flex.double(background), flex.bool(valid),
                flex.double(np.ascontiguousarray(profiles.T)), 1e-6, 100)
            ariadne = solve_irls_known_background(
                observed, background, profiles, valid, 1e-10)
            dials_intensity = np.asarray(dials_fit.intensity())
            dials_variance = np.asarray(dials_fit.variance())
            ariadne_variance = np.diag(ariadne['covariance_identifiable_subspace'])
            covered_components['ariadne'].extend(
                np.abs(ariadne['intensities_internal']-truth)
                <= z*np.sqrt(np.maximum(ariadne_variance, 0)))
            covered_components['dials'].extend(
                np.abs(dials_intensity-truth)
                <= z*np.sqrt(np.maximum(dials_variance, 0)))
            total_variance = np.ones(count)@ariadne['covariance_identifiable_subspace']@np.ones(count)
            covered_totals['ariadne'].append(
                abs(ariadne['intensities_internal'].sum()-truth.sum())
                <= z*np.sqrt(max(total_variance, 0)))
            covered_totals['dials'].append(
                abs(dials_intensity.sum()-truth.sum())
                <= z*np.sqrt(max(dials_variance.sum(), 0)))
        result[str(count)] = {
            'trials': trials,
            'component_coverage_95': {
                name: float(np.mean(values))
                for name, values in covered_components.items()},
            'total_coverage_95': {
                name: float(np.mean(values))
                for name, values in covered_totals.items()},
        }
    return {
        'setup': (
            'known heteroscedastic Gaussian variance equal to expected counts; '
            'truth is independent of both fits'),
        'results': result,
    }


def timing_suite(repeats, dials_config, size=1024):
    image, mask, truth = synthetic_image(991)
    if size != IMAGE_SIZE:
        image = np.resize(image, (size, size))
        mask = np.resize(mask, (size, size))
    functions = {
        'ariadne_standardized': lambda: ariadne_mask(
            image, mask, truth[0], 'standardized'),
        'dials_dispersion': lambda: dials_mask(image, mask, False),
        'dials_dispersion_selected': lambda: dials_mask(
            image, mask, dials_config['sigma_background'],
            dials_config['sigma_strong'], dials_config['extended']),
    }
    result = {}
    for name, function in functions.items():
        function()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            function()
            samples.append(time.perf_counter()-started)
        result[name] = {
            'seconds': samples,
            'median_seconds': statistics.median(samples),
        }
    return {
        'shape': [size, size],
        'scope': 'threshold only; Ariadne includes component pruning, DIALS does not',
        'measurements': result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile-trials', type=int, default=25)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    dials_selection = select_dials_configuration(range(100, 120))
    result = {
        'schema': 'ariadne-dials-deep-comparison/1',
        'date': '2026-10-31',
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'dials': importlib.metadata.version('dials'),
        'splits': {
            'selection_train': [100, 120],
            'validation': [200, 220],
            'final_test': [300, 320],
        },
        'spot_results': {
            name: spot_suite(range(*limits), dials_selection)
            for name, limits in (
                ('selection_train', (100, 120)),
                ('validation', (200, 220)),
                ('final_test', (300, 320)),
            )
        },
        'dials_spot_selection': dials_selection,
        'profiles': profile_suite(5000, args.profile_trials),
        'profile_covariance': covariance_coverage(9000, 200),
        'timing': timing_suite(args.repeats, dials_selection),
        'semantics': {
            'spots': (
                'threshold mask followed by identical 4-connected component '
                'extraction and size filtering; not the full DIALS indexing pipeline'),
            'profiles': (
                'same fixed background, profiles, mask and DIALS-compatible IRLS '
                'weights; Ariadne additionally reports full conditional covariance'),
        },
        'limits': [
            'Synthetic images rather than experimental rotation frames',
            'DIALS spot result excludes later reciprocal-space filtering and indexing',
            'Spot operating points were selected on the training split and frozen for validation/test',
            'No general claim of DIALS algorithm equivalence',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for split, measures in result['spot_results'].items():
        print(split, {name: (round(value['f1'], 4), round(value['precision'], 4),
                              round(value['recall'], 4))
                      for name, value in measures.items()})
    print('profiles', result['profiles']['ariadne_over_dials'])
    print('coverage', result['profile_covariance']['results'])


if __name__ == '__main__':
    main()

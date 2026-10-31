#!/usr/bin/env python3
"""Measure optional Apple MLX/Metal proposals against the exact C++ decision.

MLX executes in float32 on Metal. This is an experimental comparison because
thresholded pixel masks are expected to differ; it is not a production backend.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
from scipy.ndimage import label

from ariadne.spots import propose


def box_sum(value, width):
    radius = width // 2
    height, columns = value.shape
    cumulative = mx.cumsum(mx.cumsum(
        mx.pad(value, [(radius, radius), (radius, radius)]), axis=0), axis=1)
    cumulative = mx.pad(cumulative, [(1, 0), (1, 0)])
    stop_y, stop_x = 2*radius+1, 2*radius+1
    return (cumulative[stop_y:stop_y+height, stop_x:stop_x+columns]
            - cumulative[:height, stop_x:stop_x+columns]
            - cumulative[stop_y:stop_y+height, :columns]
            + cumulative[:height, :columns])


def gpu_propose(image, mask, readout, poisson, background_width,
                exclusion_width, scales, threshold, support_sigma,
                min_valid_fraction, background_iterations):
    valid_np = np.asarray(mask, bool) & np.isfinite(image)
    data_np = np.where(valid_np, image, 0.).astype(np.float32)
    data = mx.array(data_np)
    valid = mx.array(valid_np)
    valid_float = valid.astype(mx.float32)
    retained = valid_float
    annulus_area = background_width**2 - exclusion_width**2

    def background(selection):
        counts = (box_sum(selection, background_width)
                  - box_sum(selection, exclusion_width))
        sums = (box_sum(data*selection, background_width)
                - box_sum(data*selection, exclusion_width))
        usable = counts >= min_valid_fraction*annulus_area
        mean = mx.maximum(mx.where(counts > 0, sums/counts, 0), 0)
        return mean, counts, usable

    for _ in range(background_iterations):
        mean, counts, usable = background(retained)
        noise = mean + readout
        clipped = usable & (data > mean + 4*mx.sqrt(noise + noise/mx.maximum(counts, 1)))
        retained = mx.where(clipped, 0, retained)
    mean, counts, usable = background(retained)
    support = data > mean + support_sigma*mx.sqrt(mean + readout)
    background_variance = (mean + readout)/mx.maximum(counts, 1)
    best = mx.zeros_like(data)
    for width in scales:
        valid_count = box_sum(valid_float, width)
        observed = box_sum(data, width)
        expected = box_sum(mean, width)
        enough = valid_count >= width**2*min_valid_fraction
        if poisson:
            expected = mx.maximum(expected, .5)
            excess = mx.maximum(observed, expected)
            score = mx.sqrt(mx.maximum(
                2*(excess*mx.log(excess/expected) - excess + expected), 0))
        else:
            variance = (expected + readout*valid_count
                        + width**2*box_sum(background_variance, width))
            score = mx.maximum(observed-expected, 0)/mx.sqrt(mx.maximum(variance, .5))
        best = mx.maximum(best, mx.where(enough & usable, score, 0))
    result = valid & usable & support & (best >= threshold)
    mx.eval(result, best)
    return np.asarray(result), np.asarray(best)


def fixture(size):
    rng = np.random.default_rng(317+size)
    yy, xx = np.indices((size, size))
    expectation = np.full((size, size), 4.)
    centres = []
    for i, y in enumerate(np.linspace(28, size-28, 8)):
        for j, x in enumerate(np.linspace(28, size-28, 8)):
            flux = [30., 100., 400., 1200.][(i+j) % 4]
            sigma = [.8, 1.3, 2.][(i*3+j) % 3]
            gaussian = np.exp(-.5*((yy-y)**2 + (xx-x)**2)/sigma**2)
            expectation += flux*gaussian/gaussian.sum()
            centres.append([y, x])
    image = rng.poisson(expectation).astype(float)
    mask = rng.random(image.shape) > .01
    return image, mask, np.asarray(centres)


def quality(mask, centres):
    groups, count = label(mask)
    sizes = np.bincount(groups.ravel())
    detections = [np.argwhere(groups == i).mean(axis=0)
                  for i in range(1, count+1) if sizes[i] >= 3]
    matched = set()
    for point in detections:
        nearest = int(np.argmin(np.linalg.norm(centres-point, axis=1)))
        if np.linalg.norm(centres[nearest]-point) <= 3:
            matched.add(nearest)
    return {
        'true_spots': len(centres),
        'detected_components': len(detections),
        'true_positive': len(matched),
        'recall': len(matched)/len(centres),
        'precision': len(matched)/max(1, len(detections)),
    }


def timed(function, repeats):
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter()-started)
    return {
        'seconds': samples,
        'median_seconds': statistics.median(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sizes', type=int, nargs='+', default=[512, 1024, 2048])
    args = parser.parse_args()
    if importlib.util.find_spec('mlx') is None:
        raise RuntimeError('Install the optional Apple MLX runtime to run this comparison')
    global mx
    import mlx.core as mx
    settings = dict(gain=1., offset=0., readout_variance=0., noise_model='poisson',
                    background_width=31, exclusion_width=7, scales=[1, 3, 5],
                    score_threshold=6., support_sigma=1., min_valid_fraction=.5,
                    background_iterations=2, statistic='deviance', component_min_size=1)
    measurements = []
    for size in args.sizes:
        image, mask, centres = fixture(size)
        cpu_function = lambda: propose(image, mask, **settings)
        gpu_function = lambda: gpu_propose(
            image, mask, 0., True, 31, 7, [1, 3, 5], 6., 1., .5, 2)
        cpu_mask = cpu_function()
        gpu_mask, _ = gpu_function()
        difference = np.count_nonzero(cpu_mask != gpu_mask)
        cpu_time = timed(cpu_function, args.repeats)
        gpu_time = timed(lambda: gpu_function()[0], args.repeats)
        measurements.append({
            'shape': [size, size],
            'cpu_exact_cpp': cpu_time,
            'apple_gpu_mlx_float32': gpu_time,
            'cpu_over_gpu': cpu_time['median_seconds']/gpu_time['median_seconds'],
            'decision_pixels_equal': bool(difference == 0),
            'differing_pixels': int(difference),
            'cpu_quality': quality(cpu_mask, centres),
            'gpu_quality': quality(gpu_mask, centres),
        })
    result = {
        'schema': 'ariadne-gpu-spot-comparison/1',
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': sys.version,
        'backend': 'Apple MLX/Metal float32',
        'reference': 'exact Ariadne C++17 float64 proposal kernel',
        'status': 'experimental_nonproduction',
        'reason_not_default': (
            'float32 Metal changes a small number of thresholded weak-spot pixels; '
            'production defaults require the exact reference mask'),
        'timing_scope': 'preloaded kernels; includes Metal transfers and synchronisation for GPU',
        'measurements': measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for row in measurements:
        print(row['shape'], row['cpu_over_gpu'], row['differing_pixels'])


if __name__ == '__main__':
    main()

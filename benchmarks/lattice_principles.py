#!/usr/bin/env python3
"""Reproducible lattice-kernel measurements against the exact task baseline.

Run with one BLAS thread.  A checkout of the baseline may be supplied when git
history is unavailable; otherwise the named commit is read with ``git show``.
Only the baseline's eager scitbx import is omitted.  No numerical expression
is rewritten, and no native DIALS execution is claimed by these measurements.
"""
import argparse
import ast
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from types import ModuleType
import zipfile

import numpy as np
import scipy
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python/src'))
from ariadne.ranking import periodic_gaussian

BASELINE = '797ac22ffa9b81da2dd977e013bea8f114b49eb2'


def old_module(path=None):
    frozen = ROOT / 'benchmarks/baseline-797ac22.zip'
    if path:
        source = (Path(path) / 'python/src/ariadne/ranking.py').read_text()
    elif frozen.exists():
        with zipfile.ZipFile(frozen) as archive:
            source = archive.read('python/src/ariadne/ranking.py').decode()
    else:
        source = subprocess.check_output(['git', 'show', f'{BASELINE}:python/src/ariadne/ranking.py'],
                                         cwd=ROOT, text=True)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == 'scitbx.array_family')]
    module = ModuleType('baseline_ranking')
    exec(compile(tree, '<baseline ranking; native import omitted>', 'exec'), module.__dict__)
    return module


def measure(function, arguments, repeats):
    result = function(*arguments)
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        result = function(*arguments)
        timings.append(perf_counter() - start)
    return result, {'seconds': timings, 'median_seconds': float(np.median(timings))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    old = old_module(args.baseline_root)
    rng = np.random.default_rng(20261029)
    fractional = rng.uniform(-.5, .5, (5000, 3))
    measurements = []
    scenarios = [
        ('orthogonal_narrow', np.diag([.03, .04, .05]), .001),
        ('orthogonal_broad', np.diag([.003, .004, .005]), .003),
        ('coupled_broad', np.array([[.004, .0013, -.0007], [0, .005, .0008], [0, 0, .006]]), .004),
        ('sheared_equivalent_lattice', np.diag([.03, .04, .05]) @ np.array([[1, 12, 0], [0, 1, 0], [0, 0, 1]]), .001),
    ]
    for name, matrix, sigma in scenarios:
        radii = np.maximum(1, np.ceil(.5 + 8*sigma*np.linalg.norm(np.linalg.inv(matrix), axis=1))).astype(int)
        parameters = (matrix, fractional, sigma, radii, 512, 8)
        before, bt = measure(old.periodic_gaussian, parameters, args.repeats)
        after, at = measure(periodic_gaussian, parameters, args.repeats)
        # Independent enlarged 3D density sums, including a physical-coordinate
        # oracle for the sheared case where the baseline truncation is wrong.
        reference_A = np.diag([.03, .04, .05]) if name == 'sheared_equivalent_lattice' else matrix
        reference_fraction = np.linalg.solve(reference_A, (fractional[:129] @ matrix.T).T).T
        reference_fraction -= np.rint(reference_fraction)
        radius = 13 if 'broad' in name else 3
        nodes = np.asarray(list(itertools.product(range(-radius, radius + 1), repeat=3)))
        r2 = np.sum(((reference_fraction[:, None] - nodes) @ reference_A.T)**2, axis=2)
        reference = (np.linalg.slogdet(reference_A)[1] - 1.5*np.log(2*np.pi*sigma**2)
                     + logsumexp(-r2/(2*sigma**2), axis=1))
        measurements.append({'name': name, 'rows': len(fractional), 'before': bt, 'after': at,
                             'speedup': bt['median_seconds']/at['median_seconds'],
                             'max_log_density_difference': float(np.max(np.abs(after[0]-before[0]))),
                             'max_nearest_distance_difference': float(np.max(np.abs(after[1]-before[1]))),
                             'independent_direct_reference_rows': len(reference),
                             'baseline_max_log_reference_error': float(np.max(np.abs(before[0][:129]-reference))),
                             'updated_max_log_reference_error': float(np.max(np.abs(after[0][:129]-reference))),
                             'baseline_max_nearest_reference_error': float(np.max(np.abs(before[1][:129]-np.sqrt(r2.min(axis=1))))),
                             'updated_max_nearest_reference_error': float(np.max(np.abs(after[1][:129]-np.sqrt(r2.min(axis=1))))),
                             'algorithm': after[3]})
    output = {'baseline_commit': BASELINE, 'seed': 20261029,
              'measurement_scope': 'synthetic lattice kernels; excludes raw diffraction images and native DIALS',
              'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                              'platform': platform.platform(),
                              'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
                              'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')},
              'measurements': measurements}
    text = json.dumps(output, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()

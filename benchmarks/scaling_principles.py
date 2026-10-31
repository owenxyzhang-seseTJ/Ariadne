#!/usr/bin/env python3
"""Reproduce scaling kernel accuracy and timings against the pinned source.

Run with OPENBLAS_NUM_THREADS=1.  A standalone package may pass --baseline-file;
a checkout can retrieve the pinned baseline with read-only git show.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types
import zipfile

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'skills/ariadne-reduction/scripts'), str(ROOT/'tests')]
from composite_scaling import fit_global
from test_composite_scaling import composite_truth
from test_scaling_principles import linear_block

BASE = '797ac22ffa9b81da2dd977e013bea8f114b49eb2'
SOURCE = 'skills/ariadne-reduction/scripts/composite_scaling.py'


def baseline(path):
    frozen = ROOT / 'benchmarks/baseline-797ac22.zip'
    if path:
        source = path.read_text()
    elif frozen.exists():
        with zipfile.ZipFile(frozen) as archive:
            source = archive.read(SOURCE).decode()
    else:
        source = subprocess.check_output(['git', 'show', f'{BASE}:{SOURCE}'], cwd=ROOT, text=True)
    module = types.ModuleType('ariadne_scaling_baseline')
    exec(compile(source, str(path or f'{BASE}:{SOURCE}'), 'exec'), module.__dict__)
    return module.fit_global


def measure(function, blocks, repeats):
    times = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function(blocks, max_seconds=120.)
        times.append(time.perf_counter()-started)
    return result, dict(seconds=times, median_seconds=float(np.median(times)),
                        status=result['status'], reason=result['reason'],
                        chi2=result['chi2'], gradient_norm=result['gradient_norm'],
                        trace_states=len(result['trace']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-file', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/scaling_principles_measurements.json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    old = baseline(args.baseline_file)
    rows = []
    for name, noisy in [('correlated_composites_exact', False), ('correlated_composites_noisy', True)]:
        blocks, _, _ = composite_truth(noisy)
        blocks = [deepcopy(b) for _ in range(12) for b in blocks]
        before, before_t = measure(old, blocks, args.repeats)
        after, after_t = measure(fit_global, blocks, args.repeats)
        errors = {key: float(np.max(np.abs(after[key]-before[key])))
                  for key in ('I', 'beta', 'joint_covariance', 'parameter_response')}
        ratio = before_t['median_seconds']/after_t['median_seconds']
        rows.append(dict(name=name, blocks=len(blocks), measurements=sum(len(b['y']) for b in blocks),
                         before=before_t, after=after_t, elapsed_ratio=ratio,
                         equivalent_completion_speedup=ratio if before['status'] == after['status'] == 'converged' else None,
                         max_absolute_differences=errors))
        print(name, before_t['status'], after_t['status'], flush=True)
    truth, known = np.array([4e12, -3e-12]), np.array([1e-12, 1e12])
    blocks = [linear_block(np.eye(2), truth, known)]
    before, after = old(blocks), fit_global(blocks)
    units = dict(name='intensity_unit_invariance_24_orders', true_intensity=truth.tolist(),
                 before=dict(rank=before['rank'], status=before['status'], I=before['I'].tolist()),
                 after=dict(rank=after['rank'], status=after['status'], I=after['I'].tolist(),
                            covariance=after['joint_covariance'].tolist()))
    report = dict(baseline_commit=BASE, repeats=args.repeats,
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.platform()),
                  scope='Synthetic small dense composite-scaling kernels; excludes raw-frame DIALS processing and I/O',
                  timing_semantics='A ratio is labelled a speedup only when both fits report converged under the same tolerances.',
                  measurements=rows, unit_invariance=units)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()

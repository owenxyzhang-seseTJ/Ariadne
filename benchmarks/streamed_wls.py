#!/usr/bin/env python3
"""Measure signed-WLS workspace and timing against the first delivery."""
from pathlib import Path
import argparse
import json
import platform
import sys
import time
import tracemalloc
import zipfile
import types

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'skills/ariadne-reduction/scripts'))
from overlap_integrator import solve_joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--rows', type=int, default=120000)
    parser.add_argument('--components', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/streamed_wls_measurements.json')
    args = parser.parse_args()
    path = 'skills/ariadne-reduction/scripts/overlap_integrator.py'
    if args.baseline_root:
        code = (args.baseline_root/path).read_text()
    else:
        with zipfile.ZipFile(ROOT/'benchmarks/baseline-first-principles.zip') as archive:
            code = archive.read(path).decode()
    old = types.ModuleType('baseline_wls')
    exec(compile(code, 'baseline/'+path, 'exec'), old.__dict__)
    rng = np.random.default_rng(20261029)
    p = rng.uniform(.1, 1., (args.rows, args.components))
    p /= p.sum(axis=0)
    x = np.linspace(-1, 1, args.rows)
    b = np.column_stack((np.ones(args.rows), x, x*x-.3))
    y = b @ [1.2, -.2, .05]+p @ rng.normal(0, 1000, args.components)+rng.normal(0, .2, args.rows)
    v = rng.uniform(.5, 2., args.rows)
    valid = rng.random(args.rows) > .03
    inputs = (y, v, p, valid, b, 1e-10)
    results, arrays = {}, {}
    for name, function in [('before', old.solve_joint), ('after', solve_joint)]:
        elapsed = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = function(*inputs)
            elapsed.append(time.perf_counter()-started)
        arrays[name] = result
        tracemalloc.start()
        measured = function(*inputs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del measured
        results[name] = {'seconds':elapsed, 'median_seconds':float(np.median(elapsed)),
                         'traced_peak_additional_bytes':peak, 'traced_retained_bytes':current}
    results['speedup'] = results['before']['median_seconds']/results['after']['median_seconds']
    results['peak_workspace_ratio'] = results['before']['traced_peak_additional_bytes']/results['after']['traced_peak_additional_bytes']
    results['maximum_absolute_differences'] = {key:float(np.max(np.abs(np.asarray(arrays['before'][key])-np.asarray(arrays['after'][key]))))
        for key in ['intensities_internal','covariance_identifiable_subspace','chi2','fitted']}
    report = {'baseline':'first-principles delivery 20261029', 'rows':args.rows,
              'components':args.components, 'repeats':args.repeats,
              'scope':'synthetic fixed-profile signed WLS; inputs allocated before tracemalloc; additional NumPy/Python allocations, not total process RSS',
              'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
              'measurements':results}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()

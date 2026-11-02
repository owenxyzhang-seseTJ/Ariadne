#!/usr/bin/env python3
"""同进程交替比较缩放装配内核及完整求解，保留状态和数值误差。"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'skills/ariadne-reduction/scripts'), str(ROOT/'tests')]
import composite_scaling as scaling
from test_composite_scaling import composite_truth
from numerical_kernels import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    report = {'policy': 'Same process; alternating order; one warmup; unchanged solvers and tolerances.', 'cases': []}
    rng = np.random.default_rng(311)
    for rows, contributions, count in ((2, 3, 3), (32, 256, 64), (256, 1024, 128)):
        design = rng.normal(size=(rows, contributions))
        gain = np.exp(rng.normal(size=contributions))
        indices = rng.integers(count, size=contributions)
        results, timing = measure(lambda: scaling._intensity_design(design, gain, indices, count),
                                  scaling, '_native_scaling_design', args.repeats)
        np.testing.assert_array_equal(*results)
        report['cases'].append(dict(kind='assembly', rows=rows, contributions=contributions,
                                    intensity_count=count, bitwise_equal=True, timing=timing))
    for noisy in (False, True):
        blocks = [deepcopy(b) for _ in range(12) for b in composite_truth(noisy)[0]]
        results, timing = measure(lambda: scaling.fit_global(blocks, max_seconds=120.),
                                  scaling, '_native_scaling_design', args.repeats)
        assert all(r['status'] == 'converged' for r in results)
        errors = {}
        for key in ('I', 'beta', 'joint_covariance', 'parameter_response'):
            np.testing.assert_array_equal(results[0][key], results[1][key])
            errors[key] = float(np.max(np.abs(results[0][key]-results[1][key])))
        report['cases'].append(dict(kind='complete_fit', noisy=noisy, blocks=len(blocks),
                                    statuses=[r['status'] for r in results], max_errors=errors, timing=timing))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

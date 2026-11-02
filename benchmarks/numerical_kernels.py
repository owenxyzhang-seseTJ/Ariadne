#!/usr/bin/env python3
"""交替测量原生卷积、完整拟合区间和完整候选 EM，保存实际数值误差。"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'skills/ariadne-reduction/scripts'))
import poisson_gaussian as pg
import centroid_em as em
from likelihood_round2 import case, interval_values
from constrained_pixels import fit_constrained


def measure(call, module, attribute, repeats):
    native = getattr(module, attribute)
    if native is None:
        raise RuntimeError(f'{attribute} is unavailable')
    samples = [[], []]
    results = [None, None]
    for repeat in range(repeats+1):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            with patch.object(module, attribute, native if index else None):
                start = time.perf_counter()
                results[index] = call()
                elapsed = time.perf_counter()-start
            if repeat:
                samples[index].append(elapsed*1000)
    old, new = map(np.median, samples)
    return results, dict(reference_ms=samples[0], native_ms=samples[1],
                         reference_median_ms=float(old), native_median_ms=float(new),
                         speedup=float(old/new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('repeats must be >= 3')
    report = dict(policy='Same process; reverse order each repetition; one warmup per method; no fast-math or scientific tolerance changes.', cases=[])
    rng = np.random.default_rng(20261102)
    for n, variance in ((64, 1.), (4096, 4.), (4096, 100.)):
        mu = np.exp(rng.uniform(np.log(.001), np.log(1e5), n))
        y = rng.poisson(mu)+rng.normal(0, np.sqrt(variance), n)
        results, timing = measure(lambda: pg.PoissonGaussian(y, 1., variance).evaluate(mu), pg, '_native_pg_sums', args.repeats)
        for j in (0, 1):
            np.testing.assert_allclose(results[0][j], results[1][j], rtol=2e-9, atol=2e-9)
        report['cases'].append(dict(kind='pg_evaluate', pixels=n, readout_variance=variance,
            timing=timing, max_log_density_error=float(np.max(np.abs(results[0][0]-results[1][0]))),
            max_score_error=float(np.max(np.abs(results[0][1]-results[1][1]))),
            terms=[r[2]['terms_evaluated'] for r in results]))
    for name in ('pg_background_halo_32768', 'pg_1000_to_1_overlap'):
        data = case(name)
        results, timing = measure(lambda: fit_constrained(*data), pg, '_native_pg_sums', args.repeats)
        before, after = results
        np.testing.assert_allclose(before['intensities_internal'], after['intensities_internal'], rtol=2e-7, atol=3e-4)
        intervals = list(map(interval_values, results))
        np.testing.assert_allclose(*intervals, rtol=2e-7, atol=3e-4)
        statuses = [[x['status'] for x in [*r['profile_intervals'], r['total_interval']]] for r in results]
        assert statuses[0] == statuses[1] and all(s == 'ok' for s in statuses[0])
        assert before['converged'] and after['converged']
        report['cases'].append(dict(kind='complete_fit_and_intervals', name=name, timing=timing,
            statuses=statuses, converged=[r['converged'] for r in results],
            max_intensity_error=float(np.max(np.abs(before['intensities_internal']-after['intensities_internal']))),
            max_interval_error=float(np.max(np.abs(intervals[0]-intervals[1])))))
    for n, m, fraction in ((128, 64, 1.), (1024, 512, 1.), (1024, 512, .05)):
        y, means = rng.normal(size=(n, 3)), rng.normal(size=(m, 3))
        matrices = rng.normal(size=(n, 3, 3))
        white, norm = em.covariance_factors(matrices@matrices.transpose(0, 2, 1)+np.eye(3)*.01)
        allowed = rng.random((n, m)) < fraction
        parameters = y, means, white, norm, allowed, np.full(n, m), np.full(n, 1e-6), .05
        results, timing = measure(lambda: em.expectation(*parameters), em, '_native_expectation', args.repeats)
        for before, after in zip(*results):
            np.testing.assert_allclose(before, after, rtol=2e-12, atol=2e-13)
        report['cases'].append(dict(kind='centroid_expectation', observations=n, predictions=m,
            allowed_fraction=fraction, timing=timing,
            max_absolute_errors=[float(np.max(np.abs(a-b))) for a, b in zip(*results)]))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([dict(kind=r['kind'], name=r.get('name'), pixels=r.get('pixels'),
        observations=r.get('observations'), allowed_fraction=r.get('allowed_fraction'),
        reference_ms=r['timing']['reference_median_ms'], native_ms=r['timing']['native_median_ms'],
        speedup=r['timing']['speedup']) for r in report['cases']], indent=2), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""同进程交替、预热后测量分组和窗口，保存精确一致性与原始计时。"""
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python/src'))
sys.path.insert(0, str(ROOT/'skills/ariadne-reduction/scripts'))
import overlap_prepare as overlap
import domain_finder as domain


def measure(call, module, attribute, repeats, batch=1):
    native = getattr(module, attribute)
    if native is None:
        raise RuntimeError(f'{attribute} is unavailable: compile/register grouping kernels first')
    samples = [[], []]
    results = [None, None]
    for repeat in range(repeats+2):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            with patch.object(module, attribute, native if index else None):
                start = time.perf_counter_ns()
                for _ in range(batch):
                    results[index] = call()
                elapsed = (time.perf_counter_ns()-start)/1e6/batch
            if repeat >= 2:
                samples[index].append(elapsed)
    old, new = map(np.median, samples)
    return results, dict(reference_ms=samples[0], native_ms=samples[1],
                         reference_median_ms=float(old), native_median_ms=float(new),
                         speedup=float(old/new), calls_per_sample=batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/current_validation/grouping_kernels.json')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('repeats must be >= 3')
    report = dict(policy='Same process; alternate execution order; two warmup samples per path; '
                        'timing includes dispatch, input ownership copy and output construction. '
                        'Exact group/member order and window endpoint equality required.',
                  dispatch_policy=dict(box_minimum=32, window_minimum_original_points=16,
                      pilot_small_boxes_speedup=.916, pilot_small_common_gap_speedup=.980,
                      decision='Retain Python below these sizes after initial alternating measurements found no useful benefit.'),
                  cases=[])
    rng = np.random.default_rng(20261102)
    for name, n, extent, size in [('small_sparse', 16, 100, 2), ('sparse_scan', 4096, 200, 5),
                                   ('dense_scan', 2048, 40, 8), ('all_intersect', 512, 1, 5),
                                   ('chain', 4096, 100, 2)]:
        starts = rng.integers(0, extent, (n, 3))
        if name == 'chain':
            starts[:] = 0
            starts[:, 0] = np.arange(n)
            starts[:, 2] = np.arange(n)//64
        boxes = np.stack((starts, starts+size), axis=2).reshape(n, 6).tolist()
        results, timing = measure(lambda: overlap.connected_boxes(boxes), overlap,
                                  '_native_connected_boxes', args.repeats, 20 if n < 100 else 1)
        before, after = results
        record = dict(kind='connected_boxes', name=name, boxes=n, timing=timing, dispatch_used=n >= 32,
                      exact_group_and_member_order=before == after,
                      groups=[len(x) for x in results],
                      all_input_candidates_retained=sorted(i for group in after for i in group) == list(range(n)))
        report['cases'].append(record)
    for n in (8, 64, 4096, 65536):
        values = np.sort(rng.uniform(0., 1., n))
        expanded = np.r_[values, values+1.]
        for period, data in ((0, values), (n, expanded)):
            results, timing = measure(lambda: domain._densest_window(data, .02, period),
                                      domain, '_native_densest_window', args.repeats, 100 if n < 100 else 3)
            report['cases'].append(dict(kind='densest_window', points=n, period_size=period, timing=timing, dispatch_used=n >= 16,
                endpoints=[list(x) for x in results], exact_endpoints=results[0] == results[1],
                max_endpoint_error=max(abs(a-b) for a, b in zip(*results))))
        gaps = rng.uniform(.9, 1.1, n)
        results, timing = measure(lambda: domain.common_gap(gaps, .01), domain,
                                  '_native_densest_window', args.repeats, 20 if n < 100 else 1)
        report['cases'].append(dict(kind='common_gap', points=n, timing=timing, dispatch_used=n >= 16,
            exact_result=results[0] == results[1], max_absolute_error=abs(results[0]-results[1])))
        # 弱峰、噪声行及缺失节点仍进入完全相同的候选簇。
        row = np.arange(n, dtype=float)+rng.normal(0., .003, n)
        row[::11] += .5
        results, timing = measure(lambda: domain.periodic_subset(row, 1., .01, 3), domain,
                                  '_native_densest_window', args.repeats, 20 if n < 100 else 1)
        before, after = results
        equal = bool(np.array_equal(before, after))
        report['cases'].append(dict(kind='periodic_subset', points=n, timing=timing, dispatch_used=n >= 16,
            exact_candidate_order=equal, candidates=[len(x) for x in results],
            max_index_error=int(np.max(np.abs(before-after), initial=0)) if before.shape == after.shape else None))
    report['all_exact'] = all(r.get('exact_group_and_member_order', True)
        and r.get('all_input_candidates_retained', True) and r.get('exact_endpoints', True)
        and r.get('exact_result', True) and r.get('exact_candidate_order', True) for r in report['cases'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for row in report['cases']:
        print(f"{row['kind']:20} {row.get('name', row.get('points'))!s:16} "
              f"period={row.get('period_size', 0):6} "
              f"reference={row['timing']['reference_median_ms']:.6f} ms "
              f"native={row['timing']['native_median_ms']:.6f} ms speedup={row['timing']['speedup']:.3f}x")
    print(f"all_exact={report['all_exact']} report={args.output}")
    if not report['all_exact']:
        raise SystemExit('Grouping/window result mismatch; inspect saved report')


if __name__ == '__main__':
    main()

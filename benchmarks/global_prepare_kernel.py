#!/usr/bin/env python3
"""比较本轮等效索引缓存与扫描范围归约，核验完整侧表。"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'skills/ariadne-reduction/scripts'), str(ROOT/'tests')]
import global_scale
from test_global_scale import inputs


def equal(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main():
    source = subprocess.check_output(['git', 'show', 'HEAD:skills/ariadne-reduction/scripts/global_scale.py'], cwd=ROOT, text=True)
    old = types.ModuleType('global_prepare_reference')
    exec(compile(source, 'HEAD:global_scale.py', 'exec'), old.__dict__)
    records, config, *_ = inputs()
    rows = []
    for scans in (1, 16):
        expanded, settings = [], deepcopy(config)
        settings['groups'] = {}
        for scan in range(scans):
            for record in records:
                new = deepcopy(record)
                name = f'{scan}:{record["observation_id"]}'
                new['observation_id'] = name
                meta = deepcopy(config['groups'][record['observation_id']])
                for item in meta:
                    item['scan_id'] = f'scan{scan:02}'
                settings['groups'][name] = meta
                expanded.append(new)
        results = [None, None]
        samples = [[], []]
        for repeat in range(8):
            for j in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = time.perf_counter()
                results[j] = (old.prepare if j == 0 else global_scale.prepare)(expanded, settings)
                if repeat:
                    samples[j].append(1000*(time.perf_counter()-start))
        equal(*results)
        medians = [float(np.median(s)) for s in samples]
        rows.append(dict(scans=scans, records=len(expanded), exact_outputs=True,
                         reference_ms=samples[0], optimized_ms=samples[1],
                         reference_median_ms=medians[0], optimized_median_ms=medians[1], speedup=medians[0]/medians[1]))
    report = dict(policy='Same process; alternating order; one warmup; identical full preparation outputs.', cases=rows)
    (ROOT/'reports/current_validation/global_prepare_kernel.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

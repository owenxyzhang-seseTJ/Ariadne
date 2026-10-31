#!/usr/bin/env python3
"""核验加速路径与逐像素 NumPy 参考的一致性，保存实际误差。"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from ariadne import spots
from matched_spots import read_bruker_absolute_underflows, radial
from matched_spots_fixtures import GENERATORS


def compare(image, mask, settings, name):
    native = spots._native_matched
    if native is None:
        raise RuntimeError('Native extension is required for parity verification')
    reference_info = {}
    try:
        spots._native_matched = None
        reference = spots.propose_matched(image, mask, info=reference_info, **settings)
    finally:
        spots._native_matched = native
    rows = []
    before = os.environ.get('ARIADNE_MATCHED_THREADS')
    try:
        for threads in (1, 4):
            os.environ['ARIADNE_MATCHED_THREADS'] = str(threads)
            info = {}
            result = spots.propose_matched(image, mask, info=info, **settings)
            differences = int(np.count_nonzero(result != reference))
            if differences:
                raise AssertionError(f'{name}: {threads} threads changed {differences} pixels')
            errors = {}
            for key in ('gain', 'offset', 'readout_variance', 'kappa', 'kappa_inf', 'thresholds'):
                a, b = np.asarray(info[key]), np.asarray(reference_info[key])
                np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)
                errors[key] = float(np.max(np.abs(a-b)))
            np.testing.assert_allclose(list(info['correlation'].values()),
                                       list(reference_info['correlation'].values()), rtol=1e-7, atol=1e-9)
            rows.append(dict(threads=threads, pixel_differences=differences, pixels=int(result.sum()),
                             max_absolute_errors=errors))
    finally:
        if before is None:
            os.environ.pop('ARIADNE_MATCHED_THREADS', None)
        else:
            os.environ['ARIADNE_MATCHED_THREADS'] = before
    print(name, rows, flush=True)
    return dict(name=name, shape=list(image.shape), results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(cases=[], policy='Exact boolean masks; float64 noise-model tolerance rtol=1e-7 atol=1e-9.')
    for suite, generator in GENERATORS.items():
        for seed in (200, 301, 319):
            case = generator(seed)
            settings = dict(gain=case['gain'], offset=case['offset'],
                            readout_variance=case['readout_adu'], bins=radial(case))
            report['cases'].append(compare(case['image'], case['mask'], settings, f'{suite}-{seed}'))
    if args.experiments:
        from dxtbx.model.experiment_list import ExperimentListFactory
        el = ExperimentListFactory.from_json_file(str(args.experiments), check_format=False)
        iset = el.imagesets()[0]
        panel, beam = iset.get_detector()[0], iset.get_beam()
        tt = panel.get_two_theta_array(beam.get_unit_s0()).as_numpy_array().reshape(panel.get_image_size()[::-1])
        bins = None
        trusted = panel.get_trusted_range()
        for frame in (20, 21, 22, 23):
            image = read_bruker_absolute_underflows(iset.get_path(frame))
            mask = (image >= trusted[0]) & (image < trusted[1])
            if bins is None:
                bins = spots.radial_bins(tt, mask)
            report['cases'].append(compare(image, mask, dict(gain=panel.get_gain(), bins=bins), f'real-{frame}'))
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

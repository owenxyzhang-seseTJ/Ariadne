#!/usr/bin/env python3
"""真实帧交替计时，区分首次掩膜、缓存复用与 NumPy/flex 转换。"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from dxtbx.model.experiment_list import ExperimentListFactory
from dials.algorithms.image.threshold import DispersionExtendedThreshold
from dials.array_family import flex
from ariadne import spots
from matched_spots import read_bruker_absolute_underflows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument('--compare-threads', action='store_true',
                        help='Compare one/four matched workers in the same alternating run')
    args = parser.parse_args()
    if args.repeat < 3:
        parser.error('repeat must be >= 3')
    el = ExperimentListFactory.from_json_file(args.experiments, check_format=False)
    iset = el.imagesets()[0]
    panel, beam = iset.get_detector()[0], iset.get_beam()
    gain, trusted = panel.get_gain(), panel.get_trusted_range()
    frames = [read_bruker_absolute_underflows(iset.get_path(f)) for f in (20, 21, 22, 23)]
    masks = [(r >= trusted[0]) & (r < trusted[1]) for r in frames]
    tt = panel.get_two_theta_array(beam.get_unit_s0()).as_numpy_array().reshape(panel.get_image_size()[::-1])
    bins = spots.radial_bins(tt, masks[0])
    result = dict(shape=list(frames[0].shape), frames=[20, 21, 22, 23], repeat=args.repeat,
                  platform=platform.platform(), numpy=np.__version__, native=spots._native_matched is not None,
                  requested_threads=os.environ.get('ARIADNE_MATCHED_THREADS', 'default'),
                  policy='Alternate/reverse order per repetition; discard first two samples of each method; exclude I/O and radial bins.',
                  timing_ms={})
    info = {}
    start = time.perf_counter()
    out = spots.propose_matched(frames[0], masks[0], gain=gain, bins=bins, info=info)
    result.update(first_mask_ms=1000*(time.perf_counter()-start), first_stage_ms=info['stage_ms'],
                  first_pixels=int(out.sum()))
    alg = DispersionExtendedThreshold(frames[0].shape, (3, 3), 6., 3., 0., 2)
    output = flex.bool(flex.grid(*frames[0].shape), False)
    images = [flex.double(r/gain) for r in frames]
    flex_masks = [flex.bool(m) for m in masks]

    def matched(i, same):
        details = {}
        spots.propose_matched(frames[i], masks[0] if same else masks[i], gain=gain, bins=bins, info=details)
        return details['stage_ms']

    def dials(i, convert):
        alg(flex.double(frames[i]/gain) if convert else images[i],
            flex.bool(masks[i]) if convert else flex_masks[i], output)

    def threaded(i, threads):
        os.environ['ARIADNE_MATCHED_THREADS'] = str(threads)
        return matched(i, False)

    calls = (dict(matched_threads_1=lambda i: threaded(i, 1),
                  matched_threads_4=lambda i: threaded(i, 4)) if args.compare_threads else
             dict(matched_mask_vary=lambda i: matched(i, False),
                  matched_same_mask=lambda i: matched(i, True)))
    calls.update(dials_cpp_only=lambda i: dials(i, False),
                 dials_flex_conversion=lambda i: dials(i, True))
    result['compare_threads'] = args.compare_threads
    raw = {name: [] for name in calls}
    stages = {name: [] for name in calls}
    before = os.environ.get('ARIADNE_MATCHED_THREADS')
    try:
        for repeat in range(args.repeat):
            order = list(calls) if repeat % 2 == 0 else list(reversed(calls))
            for name in order:
                start = time.perf_counter()
                details = calls[name](repeat % 4)
                raw[name].append(1000*(time.perf_counter()-start))
                if details is not None:
                    stages[name].append(details)
    finally:
        if before is None:
            os.environ.pop('ARIADNE_MATCHED_THREADS', None)
        else:
            os.environ['ARIADNE_MATCHED_THREADS'] = before
    for name in calls:
        result['timing_ms'][name] = dict(raw=raw[name], median=float(np.median(raw[name][2:])))
        if stages[name]:
            result['timing_ms'][name]['stage_ms'] = stages[name]
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value['median'] for key, value in result['timing_ms'].items()}, indent=2))


if __name__ == '__main__':
    main()

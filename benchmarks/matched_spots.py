#!/usr/bin/env python3
"""自校准 matched-filter 的可复现 DIALS 对照，不评估完整结构解析流程。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import label, binary_dilation, find_objects
from ariadne import spots
from dials.algorithms.image.threshold import DispersionThreshold, DispersionExtendedThreshold
from dials.array_family import flex
from matched_spots_fixtures import GENERATORS, components, score, aggregate


def dials_mask(image, mask, gain, offset, extended=True, variance_gain=None):
    photons = (image-offset)/gain
    alg = (DispersionExtendedThreshold if extended else DispersionThreshold)(
        photons.shape, (3, 3), 6., 3., 0., 2)
    out = flex.bool(flex.grid(*photons.shape), False)
    im, mk = flex.double(photons), flex.bool(mask)
    if variance_gain is None:
        alg(im, mk, out)
    else:
        gain_map = np.broadcast_to(variance_gain, photons.shape).copy()
        alg(im, mk, flex.double(gain_map), out)
    return out.as_numpy_array()


def radial(case):
    if 'centre' not in case:
        return None
    yy, xx = np.indices(case['image'].shape)
    cy, cx = case['centre']
    return spots.radial_bins(np.hypot(yy-cy, xx-cx), case['mask'], 1.)


def synthetic(seeds, settings):
    result = {}
    for suite, generator in GENERATORS.items():
        names = ('matched', 'dials_extended', 'dials_dispersion',
                 'dials_extended_oracle_noise', 'dials_dispersion_oracle_noise')
        rows = {name: {minimum: [] for minimum in (1, 6)} for name in names}
        timing = {name: [] for name in rows}
        pixels = 0
        for seed in seeds:
            case = generator(seed)
            bins = radial(case)
            for name in rows:
                start = time.perf_counter()
                if name == 'matched':
                    mask = spots.propose_matched(case['image'], case['mask'], gain=case['gain'],
                        offset=case['offset'], readout_variance=case['readout_adu'], bins=bins,
                        **settings)
                else:
                    variance_gain = (1.+case['readout_adu']/case['gain']**2/case['background']
                                     if name.endswith('oracle_noise') else None)
                    mask = dials_mask(case['image'], case['mask'], case['gain'], case['offset'],
                                      extended=name.startswith('dials_extended'), variance_gain=variance_gain)
                timing[name].append(1000*(time.perf_counter()-start))
                for minimum, values in rows[name].items():
                    values.append(score(components(mask, minimum=minimum), case['truth'],
                                        ignore=case.get('ignore', ())))
            pixels += case['mask'].size
        result[suite] = {name: dict(by_minimum_size={str(minimum): aggregate(values, pixels)
            for minimum, values in groups.items()}, ms_per_frame=float(np.median(timing[name])))
            for name, groups in rows.items()}
        print(suite, json.dumps(result[suite]), flush=True)
    return result


def read_bruker_absolute_underflows(path):
    """实验用 FORMAT-100 解释：underflow 值作为绝对 ADU；未改安装的 dxtbx。"""
    data = Path(path).read_bytes()
    def header(nbytes):
        out = {}
        for i in range(0, nbytes, 80):
            out.setdefault(data[i:i+7].decode('latin1').strip(), data[i+8:i+80].decode('latin1'))
        return out
    hdr = header(2560)
    offset = int(hdr['HDRBLKS'])*512
    hdr = header(offset)
    if hdr['FORMAT'].strip() != '100':
        raise ValueError('Only Bruker FORMAT 100 is supported')
    nr, nc = int(hdr['NROWS'].split()[0]), int(hdr['NCOLS'].split()[0])
    nb = [int(v) for v in hdr['NPIXELB'].split()]
    nu, n2, n4 = [int(v) for v in hdr['NOVERFL'].split()]
    if nb[1] != 1 or nb[0] not in (1, 2):
        raise ValueError('Unsupported pixel storage width')
    image = np.frombuffer(data, {1: 'u1', 2: '<u2'}[nb[0]], nr*nc, offset).astype(np.int64)
    offset += nr*nc*nb[0]
    zero = image == 0
    def table(count, dtype, width):
        nonlocal offset
        values = np.frombuffer(data, dtype, count, offset).astype(np.int64)
        offset += (count*width+15) & ~15
        return values
    under = table(nu, 'i1', 1) if nu > 0 else None
    if n2 > 0:
        image[image == 255] = table(n2, '<u2', 2)
    if n4 > 0:
        image[image == 65535] = table(n4, '<u4', 4)
    if nu != -1:
        image += int(hdr['NEXP'].split()[2])
        if under is not None:
            image[zero] = under
    return image.reshape(nr, nc).astype(float)


def real(work, frames_per_scan, reader, settings):
    import dxtbx
    from dxtbx.model.experiment_list import ExperimentListFactory
    experiments = ExperimentListFactory.from_json_file(str(work/'integrated.expt'), check_format=False)
    refl = flex.reflection_table.from_file(str(work/'integrated.refl'))
    isets = experiments.imagesets()
    paths = [iset.get_path(0) for iset in isets]
    scans = sorted(set(paths))
    scan_of = np.array([scans.index(paths[i]) for i in refl['imageset_id'].as_numpy_array()])
    xyz = refl['xyzcal.px'].as_numpy_array()
    bbox = np.array(list(refl['bbox']), int)
    intensity = refl['intensity.prf.value'].as_numpy_array()
    variance = refl['intensity.prf.variance'].as_numpy_array()
    isig = np.where(variance > 0, intensity/np.sqrt(np.maximum(variance, 1e-30)), 0.)
    edges = np.array([-np.inf, 1, 2, 3, 5, 10, 20, 50, np.inf])
    names = ('matched', 'dials_extended', 'dials_gain_only', 'dials_calibrated')
    stats = {name: dict(found=np.zeros(8), total=np.zeros(8), components=0, supported=0,
                       milliseconds=[], model=[]) for name in names}
    for si, first in enumerate(scans):
        iset = isets[paths.index(first)]
        panel, beam = iset.get_detector()[0], iset.get_beam()
        gain = panel.get_gain()
        trusted = panel.get_trusted_range()
        tt = panel.get_two_theta_array(beam.get_unit_s0()).as_numpy_array().reshape(panel.get_image_size()[::-1])
        bins = None
        for f in np.linspace(3, len(iset)-4, frames_per_scan).round().astype(int):
            path = iset.get_path(int(f))
            image = (read_bruker_absolute_underflows(path) if reader == 'absolute_underflows' else
                     dxtbx.load(path).get_raw_data().as_numpy_array().astype(float))
            valid = (image >= trusted[0]) & (image < trusted[1])
            if bins is None:
                bins = spots.radial_bins(tt, valid)
            on = (scan_of == si) & (np.floor(xyz[:, 2]) == f)
            cover = (scan_of == si) & (bbox[:, 4] <= f) & (bbox[:, 5] > f)
            model = {}
            # 先计算帧内校准，DIALS 使用相同有效增益，避免借助参考反射调参。
            start = time.perf_counter()
            matched = spots.propose_matched(image, valid, gain=gain, bins=bins, info=model, **settings)
            matched_ms = 1000*(time.perf_counter()-start)
            for name in names:
                if name == 'matched':
                    out, elapsed = matched.copy(), matched_ms
                else:
                    start = time.perf_counter()
                    if name == 'dials_calibrated':
                        out = dials_mask(image, valid, model.get('gain', gain), model.get('offset', 0.))
                    else:
                        out = dials_mask(image, valid, gain, 0., variance_gain=(
                            model.get('gain', gain)/gain if name == 'dials_gain_only' else None))
                    elapsed = 1000*(time.perf_counter()-start)
                groups, count = label(out, np.ones((3, 3)))
                sizes = np.bincount(groups.ravel(), minlength=count+1)
                out &= sizes[groups] >= 6
                groups, count = label(out, np.ones((3, 3)))
                st = stats[name]
                st['milliseconds'].append(elapsed)
                near = binary_dilation(out, structure=np.ones((7, 7)))
                px = np.clip(np.round(xyz[on, 0]-.5).astype(int), 0, image.shape[1]-1)
                py = np.clip(np.round(xyz[on, 1]-.5).astype(int), 0, image.shape[0]-1)
                k = np.digitize(isig[on], edges)-1
                np.add.at(st['total'], k, 1)
                np.add.at(st['found'], k, near[py, px])
                cx, cy = xyz[cover, 0]-.5, xyz[cover, 1]-.5
                st['components'] += count
                for sl in find_objects(groups):
                    st['supported'] += int(np.any((cx >= sl[1].start-3) & (cx <= sl[1].stop+2) &
                                                    (cy >= sl[0].start-3) & (cy <= sl[0].stop+2)))
                if name == 'matched':
                    st['model'].append({key: value for key, value in model.items()
                                        if key not in ('correlation', 'stage_ms')})
        print('scan', si, 'done', flush=True)
    result = {}
    for name, st in stats.items():
        result[name] = dict(recall_by_isig=(st['found']/np.maximum(st['total'], 1)).tolist(),
            n=st['total'].astype(int).tolist(), recall_isig_ge3=float(st['found'][3:].sum()/max(st['total'][3:].sum(), 1)),
            components=st['components'], reference_supported_fraction=st['supported']/max(st['components'], 1),
            ms_per_frame=float(np.median(st['milliseconds'])), model=st['model'])
    return dict(reader=reader, frames_per_scan=frames_per_scan, results=result,
        limitation='Integrated reflections are an incomplete, DIALS-derived truth proxy; component support is not precision.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--frames-per-scan', type=int, default=8)
    parser.add_argument('--reader', choices=('dxtbx', 'absolute_underflows'), default='dxtbx')
    args = parser.parse_args()
    settings = dict(local_cells=4, false_alarms=.02)
    seeds = range(200, 220) if args.split == 'validation' else range(300, 320)
    report = dict(platform=platform.platform(), numpy=np.__version__, dials=importlib.metadata.version('dials'),
        native=spots._native_matched is not None, settings=settings,
        timing_scope='Public threshold call including conversion; excludes I/O, radial-bin construction and component extraction.')
    if args.work:
        report['real'] = real(args.work, args.frames_per_scan, args.reader, settings)
    else:
        report.update(split=args.split, seeds=list(seeds), synthetic=synthetic(seeds, settings))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

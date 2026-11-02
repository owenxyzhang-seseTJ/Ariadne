#!/usr/bin/env python3
"""优化前后弱峰、近邻双峰与真实双域参考的一致性检查。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from ariadne import spots
from matched_spots import dials_mask, read_bruker_absolute_underflows
from matched_spots_fixtures import components, score


def run_pair(image, mask, settings, previous):
    current = spots._native_matched
    before_info, after_info = {}, {}
    try:
        spots._native_matched = previous
        before = spots.propose_matched(image, mask, info=before_info, **settings)
    finally:
        spots._native_matched = current
    after = spots.propose_matched(image, mask, info=after_info, **settings)
    different = int(np.count_nonzero(before != after))
    if different:
        raise AssertionError(f'Changed {different} detection pixels')
    errors = {}
    for key in ('gain', 'offset', 'readout_variance', 'kappa', 'kappa_inf', 'thresholds'):
        a, b = np.asarray(before_info[key]), np.asarray(after_info[key])
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)
        errors[key] = float(np.max(np.abs(a-b)))
    np.testing.assert_allclose(list(before_info['correlation'].values()),
                               list(after_info['correlation'].values()), rtol=1e-7, atol=1e-9)
    return before, after, dict(pixel_differences=different, max_absolute_errors=errors)


def doublets(seed, separation, ratio):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((512, 513))
    background = 2.+.003*xx+.002*yy
    mean = background.copy()
    truth = []
    for j, cy in enumerate((64., 160., 256., 352., 448.)):
        for k, cx in enumerate((64., 160., 256., 352., 448.)):
            flux = (40., 80., 160., 320., 640.)[k]
            theta = (j*.43+k*.29)
            dy, dx = separation*np.sin(theta)/2, separation*np.cos(theta)/2
            for domain, sign in enumerate((-1, 1)):
                y, x = cy+sign*dy, cx+sign*dx
                value = flux if domain == 0 else flux*ratio
                g = np.exp(-.5*((yy-y)**2+(xx-x)**2)/1.1**2)
                g /= g.sum()
                mean += value*g
                truth.append(dict(y=y, x=x, flux=value, domain=domain,
                                  snr=float(value*np.sqrt(np.sum(g*g/(background+1.))))))
    image = 64.+25.*(rng.poisson(mean)+rng.normal(0, 1., mean.shape))
    return image, np.ones(mean.shape, bool), truth, background


def synthetic(previous):
    rows = []
    for seed in (801, 802, 803):
        for separation in (2., 4., 6., 8.):
            for ratio in (1., 4.):
                image, mask, truth, background = doublets(seed, separation, ratio)
                settings = dict(gain=25., offset=64., readout_variance=625., component_min_size=6)
                before, after, parity = run_pair(image, mask, settings, previous)
                # 一个合并组件不能匹配两颗峰；门限随间距收紧。
                gate = min(1.5, separation/3)
                metrics = {}
                dials = dials_mask(image, mask, 25., 64., variance_gain=1.+1./background)
                for name, out in (('before', before), ('after', after), ('dials_oracle_noise', dials)):
                    found = components(out, minimum=6)
                    result = score(found, truth, gate=gate)
                    matched = result['matched']
                    snr = result['snr']
                    metrics[name] = dict(tp=result['tp'], fp=result['fp'], fn=result['fn'],
                        pairs_resolved=int(np.count_nonzero(matched.reshape(-1, 2).all(axis=1))),
                        pairs_total=len(truth)//2,
                        recall_by_snr=[dict(lo=lo, hi=hi, n=int(sel.sum()),
                            found=int(matched[sel].sum())) for lo, hi in ((0., 3.), (3., 5.), (5., 10.), (10., 1e9))
                            for sel in [(snr >= lo) & (snr < hi)]])
                assert metrics['before'] == metrics['after']
                rows.append(dict(seed=seed, separation_px=separation, flux_ratio=ratio,
                                 matching_gate_px=gate, parity=parity, metrics=metrics))
    return rows


def real(work, previous):
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory
    experiments = ExperimentListFactory.from_json_file(str(work/'integrated.expt'), check_format=False)
    refl = flex.reflection_table.from_file(str(work/'integrated.refl'))
    isets = experiments.imagesets()
    paths = [iset.get_path(0) for iset in isets]
    scans = sorted(set(paths))
    scan_of = np.array([scans.index(paths[i]) for i in refl['imageset_id'].as_numpy_array()])
    ids = refl['id'].as_numpy_array()
    crystals = experiments.crystals()
    domain_of = np.array([next(j for j, c in enumerate(crystals) if c == ex.crystal) for ex in experiments])
    domains = domain_of[ids]
    xyz = refl['xyzcal.px'].as_numpy_array()
    intensity, variance = (refl[f'intensity.prf.{key}'].as_numpy_array() for key in ('value', 'variance'))
    isig = np.where(variance > 0, intensity/np.sqrt(np.maximum(variance, 1e-30)), 0.)
    edges = np.array([-np.inf, 1., 2., 3., 5., 10., 20., 50., np.inf])
    stats = {str(domain): {name: dict(n=np.zeros(8, int), found=np.zeros(8, int))
                          for name in ('before', 'after', 'dials')} for domain in np.unique(domains)}
    rows = []
    for si, first in enumerate(scans):
        iset = isets[paths.index(first)]
        panel, beam = iset.get_detector()[0], iset.get_beam()
        tt = panel.get_two_theta_array(beam.get_unit_s0()).as_numpy_array().reshape(panel.get_image_size()[::-1])
        bins = None
        for frame in np.linspace(3, len(iset)-4, 8).round().astype(int):
            image = read_bruker_absolute_underflows(iset.get_path(int(frame)))
            lower, upper = panel.get_trusted_range()
            mask = (image >= lower) & (image < upper)
            if bins is None:
                bins = spots.radial_bins(tt, mask)
            before, after, parity = run_pair(image, mask, dict(gain=panel.get_gain(), bins=bins), previous)
            dials = dials_mask(image, mask, panel.get_gain(), 0.)
            on = (scan_of == si) & (np.floor(xyz[:, 2]) == frame) & (ids >= 0)
            py = np.clip(np.round(xyz[on, 1]-.5).astype(int), 0, image.shape[0]-1)
            px = np.clip(np.round(xyz[on, 0]-.5).astype(int), 0, image.shape[1]-1)
            group = np.digitize(isig[on], edges)-1
            for name, out in (('before', before), ('after', after), ('dials', dials)):
                # 与既有真实帧报告一致，先剔除不足六像素的八连通组件。
                from scipy.ndimage import label
                labels, count = label(out, np.ones((3, 3)))
                sizes = np.bincount(labels.ravel(), minlength=count+1)
                near = binary_dilation(out & (sizes[labels] >= 6), structure=np.ones((7, 7)))
                for domain in stats:
                    sel = domains[on] == int(domain)
                    np.add.at(stats[domain][name]['n'], group[sel], 1)
                    np.add.at(stats[domain][name]['found'], group[sel], near[py[sel], px[sel]])
            rows.append(dict(scan=si, frame=int(frame), parity=parity))
        print(f'real scan {si}: done', flush=True)
    for methods in stats.values():
        for entry in methods.values():
            entry['recall'] = (entry['found']/np.maximum(entry['n'], 1)).tolist()
            entry['n'] = entry['n'].tolist()
            entry['found'] = entry['found'].tolist()
        assert methods['before'] == methods['after']
    return dict(frames=rows, isig_edges=['-inf', 1, 2, 3, 5, 10, 20, 50, 'inf'], crystal_models=stats,
                model_scope='Each scan has separate crystal models; these indices are not independently identified physical twin domains.',
                limitation='Integrated reference is incomplete and DIALS-derived. Proximity recall does not prove twin separation or correct overlapping intensities.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-native', required=True, type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_native', args.baseline_native)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    os.environ['ARIADNE_MATCHED_THREADS'] = '4'
    report = dict(policy='No tuning; fixed seeds and settings. Exact before/after masks; noise diagnostics rtol=1e-7 atol=1e-9.',
                  synthetic=synthetic(baseline.matched_spot_proposals))
    if args.work:
        report['real'] = real(args.work, baseline.matched_spot_proposals)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()

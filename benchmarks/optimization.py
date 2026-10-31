#!/usr/bin/env python3
"""固定基线与当前源码的数值、时间及分配测量。使用 ariadne-env python 运行。"""
from __future__ import annotations
import argparse
import gc
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import scipy
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/ariadne-reduction/scripts'
sys.path[:0] = [str(ROOT / 'python/src'), str(SCRIPTS), str(ROOT / 'tests')]


def module(name, code, filename):
    value = ModuleType(name)
    value.__file__ = str(filename)
    sys.modules[name] = value
    exec(compile(code, str(filename), 'exec'), value.__dict__)
    return value


def load_old(name, path):
    with zipfile.ZipFile(ROOT / 'benchmarks/baseline-2b9bd04.zip') as archive:
        return module('baseline_' + name, archive.read(path), path)


def measured(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(seconds=times, median_seconds=statistics.median(times), peak_traced_bytes=peak)


def compare(name, old, new, repeats, **details):
    before, after = measured(old, repeats), measured(new, repeats)
    record = dict(name=name, before=before, after=after,
                  speedup=before['median_seconds'] / after['median_seconds'], **details)
    print(name, 'speedup', round(record['speedup'], 3), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/optimization_measurements.json')
    parser.add_argument('--only', nargs='+', help='只重测指定核心，其余记录保留')
    args = parser.parse_args()
    records = []
    if args.only and args.output.exists():
        records = json.loads(args.output.read_text())['measurements']
        records = [record for record in records if record['name'] not in args.only]
    def record(name, old, new, **details):
        if args.only and name not in args.only:
            return
        records.append(compare(name, old, new, args.repeats, **details))
    from ariadne import ranking, spots
    import overlap_prepare, overlap_integrator, composite_scaling
    from dials.algorithms.indexing import model_evaluation as installed_model
    from dxtbx.model import Crystal
    from dials.array_family import flex
    old_ranking = load_old('ranking', 'python/src/ariadne/ranking.py')
    old_spots = load_old('spots', 'python/src/ariadne/spots.py')
    old_prepare = load_old('prepare', 'skills/ariadne-reduction/scripts/overlap_prepare.py')
    old_integrator = load_old('integrator', 'skills/ariadne-reduction/scripts/overlap_integrator.py')
    old_scaling = load_old('scaling', 'skills/ariadne-reduction/scripts/composite_scaling.py')
    old_model = load_old('model', 'dials/src/dials/algorithms/indexing/model_evaluation.py')
    path = ROOT.parent / 'vendor/dials-3.30.1/src/dials/algorithms/indexing/model_evaluation.py'
    if not path.exists():
        path = Path(installed_model.__file__)
    new_model = module('optimized_dials_model', path.read_text(), path)
    rng = np.random.default_rng(20261029)
    with threadpool_limits(limits=1):
        for label, A, sigma in [
            ('periodic_narrow', np.diag([.03, .04, .05]), .001),
            ('periodic_wide', np.array([[.004, .0013, -.0007], [0, .005, .0008], [0, 0, .006]]), .003),
        ]:
            fractional = rng.uniform(-.5, .5, (5000, 3))
            radii = np.maximum(1, np.ceil(.5 + 6 * sigma * np.linalg.norm(np.linalg.inv(A), axis=1)).astype(int))
            offsets = np.asarray(list(itertools.product(*[range(-r, r+1) for r in radii])))
            def direct():
                logs, distances, nearest = [], [], []
                chunk = min(512, max(1, 64*1024**2 // (len(offsets)*48)))
                for start in range(0, len(fractional), chunk):
                    delta = (fractional[start:start+chunk, None, :] - offsets) @ A.T
                    r2 = np.sum(delta**2, axis=2)
                    logs.append(logsumexp(-.5*r2/sigma**2, axis=1))
                    distances.append(np.sqrt(r2.min(axis=1)))
                    nearest.append(offsets[np.argmin(r2, axis=1)])
                return (np.log(abs(np.linalg.det(A))) - 1.5*np.log(2*np.pi*sigma**2) + np.concatenate(logs),
                        np.concatenate(distances), np.concatenate(nearest))
            def adaptive():
                return ranking.periodic_gaussian(A, fractional, sigma, radii, 512, 6)
            before, after = direct(), adaptive()
            np.testing.assert_allclose(after[0], before[0], atol=1e-9, rtol=1e-12)
            np.testing.assert_allclose(after[1], before[1], atol=1e-16)
            np.testing.assert_array_equal(after[2], before[2])
            record(label, direct, adaptive, rows=5000,
                                   radii=radii.tolist(), representation=after[3],
                                   max_log_error=float(np.max(abs(after[0]-before[0]))))
        import profile_geometry
        from test_conditional_gaussian import _rotated_covariance, _reference_voxel
        old_geometry = load_old('geometry', 'skills/ariadne-reduction/scripts/profile_geometry.py')
        for label, widths in [('broad', [1.1, 1.4, 2.]), ('rotated', [.35, .8, 1.3]), ('narrow', [.1, .45, 1.2])]:
            covariance = _rotated_covariance(widths)
            center = np.array([6.37, 6.81, 4.22])
            bounds = (0, 15, 0, 13, 0, 9)
            old = lambda:old_geometry.gaussian_voxels(covariance, center, bounds)
            new = lambda:profile_geometry.gaussian_voxels(covariance, center, bounds)
            ref72 = _reference_voxel(covariance, center, (6, 6, 4), 72)
            ref96 = _reference_voxel(covariance, center, (6, 6, 4), 96)
            assert abs(ref72-ref96) < 2e-12
            old_error = float(abs(old()[4,6,6]-ref96))
            new_error = float(abs(new()[4,6,6]-ref96))
            assert new_error < 1e-11
            record('conditional_gaussian_'+label,old,new,principal_widths=widths,
                                   covariance=covariance.tolist(),center=center.tolist(),bounds=list(bounds),
                                   reference_voxel=[6,6,4],reference96=ref96,reference_convergence=abs(ref72-ref96),
                                   old_reference_error=old_error,new_reference_error=new_error)
        n, candidates = 30000, 20
        reflections = flex.reflection_table()
        reflections['id'] = flex.int(np.where(np.arange(n)%5, 0, -1))
        reflections['imageset_id'] = flex.int(np.repeat([0, 1], n//2))
        reflections['xyzobs.px.value'] = flex.vec3_double(np.column_stack((np.zeros(n), np.zeros(n), np.arange(n)%1200)))
        params = SimpleNamespace(seed=42, holdout_block_frames=20, holdout_fraction=.2, min_training=25, min_validation=20)
        def partition_old():
            for _ in range(candidates): old_ranking.refinement_partition(reflections, params)
        def partition_new():
            selection = ranking.holdout_selection(reflections, params)
            for _ in range(candidates): ranking.refinement_partition(reflections, params, selection)
        record('shared_holdout', partition_old, partition_new, rows=n, candidates=candidates)
        results=[]
        for i in range(40):
            crystal = Crystal((30+i*.2,0,0),(0,35,0),(0,0,40),'P 1')
            results.append(new_model.Result(.91+i*.0002, crystal, (.02,.03,.01),1000+i*5,.8+i*.003,(0,0,0)))
        def rank_old():
            rank = old_model.ModelRankFilter()
            for result in results: rank.append(result)
            return rank
        def rank_new():
            rank = new_model.ModelRankFilter(); rank.extend(results)
            return rank
        assert str(rank_old()) == str(rank_new())
        assert rank_old().best_model() is rank_new().best_model()
        record('native_filter_batch', rank_old, rank_new, candidates=len(results))
        image = rng.poisson(4., (1024,1024)).astype(float)
        image[490:500,510:520] += 40
        mask = rng.random(image.shape) > .03
        settings = dict(gain=1., offset=0., readout_variance=3., noise_model='poisson_gaussian',
                        background_width=31, exclusion_width=7, scales=[1,3,5], score_threshold=6.,
                        support_sigma=1., min_valid_fraction=.5, background_iterations=2)
        new_settings = {**settings, 'statistic': 'deviance', 'component_min_size': 1}
        np.testing.assert_array_equal(
            old_spots.propose(image,mask,**settings),
            spots.propose(image,mask,**new_settings))
        record('multiscale_spots', lambda:old_spots.propose(image,mask,**settings),
                               lambda:spots.propose(image,mask,**new_settings),
               pixels=image.size,mask_exact=True)
        entry = {'covariance_px_xyz':np.diag([1.2,1.4,.9]).ravel().tolist()}
        center, bounds = [48.2,48.3,24.1], (0,96,0,96,0,48)
        old = lambda:old_prepare.template_values(entry,None,center,bounds)
        new = lambda:overlap_prepare.template_values(entry,None,center,bounds)
        np.testing.assert_array_equal(old(),new())
        record('gaussian_template_dispatch',old,new,voxels=96*96*48,values_exact=True)
        n,k=6000,8
        profiles = rng.normal(size=(n,k))
        derivatives = rng.normal(size=(n,3*k))
        background=np.column_stack((np.ones(n),rng.normal(size=(n,3))))
        variance=rng.uniform(2,4,n);valid=np.ones(n,bool)
        data=profiles@np.linspace(20,50,k)+background@np.arange(4)+derivatives@rng.normal(size=3*k)
        inputs=(data,variance,profiles,valid,background,derivatives,1e-9,1e-7,.5)
        old=lambda:old_integrator.solve_position_joint(*inputs)
        new=lambda:overlap_integrator.solve_position_joint(*inputs)
        before,after=old(),new()
        errors={}
        for key in ['intensities_internal','covariance_identifiable_subspace','fitted']:
            np.testing.assert_allclose(after[key],before[key],atol=1e-10,rtol=1e-10)
            errors[key]=float(np.max(abs(after[key]-before[key])))
        record('position_qr_reuse',old,new,rows=n,components=k,max_errors=errors)
        from test_composite_scaling import composite_truth
        from copy import deepcopy
        for label, noisy in [('scaling_callback_reuse', True), ('scaling_callback_converged', False)]:
            if args.only and label not in args.only:
                continue
            blocks,_,_=composite_truth(noisy=noisy)
            blocks=[deepcopy(block) for _ in range(12) for block in blocks]
            old=lambda:old_scaling.fit_global(blocks,max_seconds=120.)
            new=lambda:composite_scaling.fit_global(blocks,max_seconds=120.)
            before,after=old(),new()
            errors={}
            for key in ['I','beta','joint_covariance','parameter_response']:
                np.testing.assert_array_equal(after[key],before[key])
                errors[key]=float(np.max(abs(after[key]-before[key])))
            assert after['status']==before['status']
            record(label,old,new,blocks=len(blocks),max_errors=errors,status=after['status'],
                   reason=after['reason'],gradient_norm=after['gradient_norm'],trace_states=len(after['trace']))
    result=dict(baseline_commit='2b9bd04',seed=20261029,repeats=args.repeats,
                environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
                                 platform=platform.platform(),blas_threads=1),measurements=records,
                measurement_scope='Synthetic numerical kernels and native ranking; excludes raw-frame reduction and file I/O')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output,flush=True)


if __name__ == '__main__':
    main()

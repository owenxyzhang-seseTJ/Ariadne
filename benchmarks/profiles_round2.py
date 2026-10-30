#!/usr/bin/env python3
"""Measure box integration, analytic fluxes and bounded template learning.

Use OPENBLAS_NUM_THREADS=1. Native DIALS enables the actual shoebox-learning
comparison; pure NumPy/SciPy environments still run all Gaussian comparisons.
Peak memory is Python/NumPy traced allocation, not whole-process resident RSS.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import gc
import io
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time
import tracemalloc
import types
from types import SimpleNamespace
import zipfile

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'skills/ariadne-reduction/scripts'
sys.path.insert(0, str(SCRIPTS))
import profile_geometry
import learn_profile_library


def previous_source(relative, baseline_root):
    if baseline_root:
        return (baseline_root / relative).read_text()
    with zipfile.ZipFile(ROOT/'benchmarks/baseline-first-principles.zip') as archive:
        return archive.read(relative).decode()


def previous_module(filename, baseline_root):
    module = types.ModuleType('baseline_' + filename.removesuffix('.py'))
    source = previous_source('skills/ariadne-reduction/scripts/' + filename, baseline_root)
    exec(compile(source, 'first_principles_round1/'+filename, 'exec'), module.__dict__)
    return module


def measure(function, repeats):
    warm = function()
    del warm
    times, cpu_times = [], []
    for _ in range(repeats):
        started = time.perf_counter()
        cpu_started = time.process_time()
        result = function()
        cpu_times.append(time.process_time()-cpu_started)
        times.append(time.perf_counter()-started)
        del result
    gc.collect()
    tracemalloc.start()
    result = function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {'seconds':times, 'median_seconds':float(np.median(times)),
                    'cpu_seconds':cpu_times,'median_cpu_seconds':float(np.median(cpu_times)),
                    'peak_traced_bytes':peak}


def gaussian_comparisons(old, repeats):
    rotation = np.linalg.qr(np.array([[.7,-.4,.8],[.6,.8,-.1],[-.3,.5,.7]]))[0]
    cases = [
        ('correlated_volume', (33,35,21), rotation@np.diag(np.square([.35,.8,1.3]))@rotation.T, False),
        ('narrow_value_and_xyz_derivatives', (13,15,9), rotation@np.diag(np.square([.11,.45,1.2]))@rotation.T, True),
        ('large_correlated_volume', (129,127,65), np.array([[1.4,.3,.1],[.3,1.9,.25],[.1,.25,.8]]), False),
        ('large_value_and_xyz_derivatives', (65,67,33), np.array([[1.4,.3,.1],[.3,1.9,.25],[.1,.25,.8]]), True),
    ]
    results = []
    for name,shape,covariance,derivatives in cases:
        centre = np.asarray(shape)/2+[.17,.23,-.12]
        bounds = tuple(edge for count in shape for edge in (0,count))
        def before():
            values = old.gaussian_voxels(covariance, centre, bounds)
            if not derivatives:
                return values
            gradients = [(old.gaussian_voxels(covariance, centre+np.eye(3)[axis]*.001, bounds)
                          -old.gaussian_voxels(covariance, centre-np.eye(3)[axis]*.001, bounds))/.002
                         for axis in range(3)]
            return values, np.asarray(gradients)
        def after():
            function = profile_geometry.gaussian_voxels_with_derivatives if derivatives else profile_geometry.gaussian_voxels
            return function(covariance, centre, bounds)
        first, a = measure(before,repeats)
        second, b = measure(after,repeats)
        value_error = np.max(abs(first[0]-second[0])) if derivatives else np.max(abs(first-second))
        row = {'name':name,'shape_xyz':shape,'voxel_count':int(np.prod(shape)),
               'before':a,'after':b,'speedup':a['median_seconds']/b['median_seconds'],
               'cpu_speedup':a['median_cpu_seconds']/b['median_cpu_seconds'],
               'traced_memory_ratio':a['peak_traced_bytes']/b['peak_traced_bytes'],
               'max_absolute_value_difference':float(value_error)}
        if derivatives:
            row['max_difference_from_legacy_step_0_001_gradient'] = float(np.max(abs(first[1]-second[1])))
            row['derivative_reference'] = 'Independent original-density score integral in tests/test_profiles_round2.py; old finite differences have O(step^2) bias.'
        results.append(row)
        print(name, row['speedup'], row['traced_memory_ratio'], flush=True)
    return results


def native_learning_comparison(baseline_root, repeats, references):
    try:
        from dials.array_family import flex
        from dials.algorithms.shoebox import MaskCode
    except ImportError:
        return {'status':'not_run','reason':'requires native DIALS shoebox runtime'}
    old = previous_module('learn_profile_library.py', baseline_root)
    with TemporaryDirectory(prefix='ariadne_profile_benchmark_') as temporary:
        directory = Path(temporary)
        shape = (9,13,17)
        bounds = (0,17,0,13,0,9)
        boxes = flex.shoebox(flex.size_t(references,0),flex.int6([bounds]*references),allocate=True)
        z,y,x = np.ogrid[:9,:13,:17]
        positions=[]
        for index in range(references):
            centre = np.array([4.,6.,8.])+[.13*np.sin(index),.17*np.cos(index),.1*np.sin(.7*index)]
            signal = 3000*np.exp(-.5*((z-centre[0])**2/1.5**2+(y-centre[1])**2/2.2**2+(x-centre[2])**2/3.**2))
            box=boxes[index]
            box.data=flex.float(signal+12)
            box.background=flex.float(np.full(shape,12.))
            box.mask=flex.uint8(np.full(shape,int(MaskCode.Valid)|int(MaskCode.Foreground),dtype=np.uint8))
            positions.append((float(index%17),float(index//17),float(index%9)))
        table=flex.reflection_table()
        table['id']=flex.int(references,0);table['panel']=flex.size_t(references,0)
        table['xyzcal.px']=flex.vec3_double(positions);table['shoebox']=boxes
        table['intensity.sum.value']=flex.double(references,1e5)
        table['intensity.sum.variance']=flex.double(references,1e5)
        reference_path=directory/'reference.refl';table.as_file(str(reference_path))
        calibration=directory/'calibration.json'
        calibration.write_text(json.dumps({'schema':'scxrd-profile-calibration/1',
            'isolation_provenance':'synthetic known isolated profiles','isolated_source_rows':list(range(references)),
            'background_provenance':'synthetic constant 12 ADU','noise_model':{'gain':1,'readout_variance':1},
            'basis':{'scope':'synthetic fixture'},'experiment_to_domain':{'0':'0'},
            'complete_prediction_provenance':'synthetic support fixture','calibration_support_complete':True}))
        counter = [0]
        def run(function):
            counter[0]+=1
            output=directory/f'run_{counter[0]}'
            args=SimpleNamespace(references=reference_path,calibration=calibration,output=output,
                local_grid=(2,2,1),shrinkage_references=20.,min_local_references=5,
                min_reference_frames=2,reference_min_snr=10.,min_references=10,
                support_sigma=5.,max_template_voxels=1000000,max_learning_bytes=536870912)
            with redirect_stdout(io.StringIO()):
                function(args)
            return output
        before,a=measure(lambda:run(old.learn),repeats)
        after,b=measure(lambda:run(learn_profile_library.learn),repeats)
        first=json.loads((before/'profiles.json').read_text())
        second=json.loads((after/'profiles.json').read_text())
        if len(first['entries'])!=1 or len(second['entries'])!=1:
            raise AssertionError('Native template learning did not produce its expected entry')
        errors=[]
        for path in before.glob('*.npy'):
            errors.append(float(np.max(abs(np.load(path)-np.load(after/path.name)))))
        entry=second['entries'][0]
        return {'status':'compared_native_dials_shoeboxes','references':references,
                'template_shape_zyx':entry['template_shape_zyx'],'local_nodes':len(entry['local_model']['nodes']),
                'before':a,'after':b,'speedup':a['median_seconds']/b['median_seconds'],
                'cpu_speedup':a['median_cpu_seconds']/b['median_cpu_seconds'],
                'traced_memory_ratio':a['peak_traced_bytes']/b['peak_traced_bytes'],
                'max_absolute_template_difference':max(errors),
                'max_absolute_covariance_difference':float(np.max(abs(np.array(first['entries'][0]['covariance_px_xyz'])-entry['covariance_px_xyz']))),
                'same_reference_provenance':first['entries'][0]['provenance']['source_rows']==entry['provenance']['source_rows'],
                'temporary_node_sums_remaining':len(list(after.glob('.profile-sums-*')))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--references',type=int,default=200)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/profiles_round2_measurements.json')
    args=parser.parse_args()
    if args.repeats<1 or args.references<20:
        parser.error('repeats >= 1 and references >= 20 are required')
    old=previous_module('profile_geometry.py',args.baseline_root)
    report={'baseline':'first-principles delivery 2026-10-29; baseline-first-principles.zip',
            'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
                           'thread_settings':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}},
            'repeats':args.repeats,'scope':'Synthetic Gaussian boxes and synthetic native DIALS shoebox calibration; no experimental spot accuracy claim',
            'memory_scope':'tracemalloc peak includes Python/NumPy allocations during one call and outputs, excludes preloaded libraries and externally held inputs; not whole-process RSS',
            'working_byte_budget':profile_geometry.DEFAULT_WORKING_BYTES,
            'working_budget_scope':'Gaussian quadrature scratch only; excludes inputs, outputs and small axis vectors; conservative buffer accounting, not a process-RSS hard limit',
            'quadrature_accuracy':'Per numerical axis max(6, ceil(3*sqrt(precision[j,j]))+2), reject above 64; retained resolution heuristic, not a rigorous integration error certificate',
            'gaussian':gaussian_comparisons(old,args.repeats),
            'template_learning':native_learning_comparison(args.baseline_root,args.repeats,args.references)}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__':
    main()

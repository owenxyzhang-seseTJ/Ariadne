#!/usr/bin/env python3
"""Reproduce complete constrained fits/intervals against the first-round source.

Run with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. Peak allocations use
tracemalloc in a separate untimed run, after input arrays have been constructed;
they are incremental Python/NumPy-traced allocations, not process peak RSS.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc
import types
import zipfile

import numpy as np
import scipy
from scipy.stats import chi2

ROOT=Path(__file__).resolve().parents[1]
SOURCE='skills/ariadne-reduction/scripts/constrained_pixels.py'
sys.path.insert(0,str(ROOT/'skills/ariadne-reduction/scripts'))
from constrained_pixels import fit_constrained


def baseline(root):
    if root:
        source=(root/SOURCE).read_text()
    else:
        with zipfile.ZipFile(ROOT/'benchmarks/baseline-first-principles.zip') as archive:
            source=archive.read(SOURCE).decode()
    module=types.ModuleType('first_round_constrained_baseline')
    exec(compile(source,'first-principles:'+SOURCE,'exec'),module.__dict__)
    return module.fit_constrained


def calibration(readout):
    return {'source':'synthetic reproducible benchmark','gain':1.,'readout_variance':readout,
            'offset_removed':True,'poisson_counts_verified':True}


def case(name):
    rng=np.random.default_rng(20261029)
    if name=='poisson_disjoint_4096':
        k,n=4,4096
        P=np.zeros((n+8,k))
        for j in range(k):
            P[j*(n//k):(j+1)*(n//k),j]=k/n
        y=rng.poisson(P@np.array([2000.,500.,100.,2.])).astype(float)
        method,noise='poisson',calibration(0.)
    elif name=='pg_background_halo_32768':
        axis=np.linspace(-3,3,31)
        profiles=np.column_stack((np.exp(-(axis+.4)**2/4),np.exp(-(axis-.4)**2/4)))
        profiles/=profiles.sum(axis=0)
        P=np.vstack((profiles,np.zeros((32768,2))))
        y=np.r_[rng.poisson(profiles@np.array([100.,2.]))+rng.normal(size=len(axis)),np.zeros(32768)]
        method,noise='poisson-gaussian',calibration(1.)
    elif name=='pg_1000_to_1_overlap':
        axis=np.linspace(-4,4,41)
        profiles=np.column_stack((np.exp(-(axis+.4)**2/.8),np.exp(-(axis-.4)**2/.8)))
        profiles/=profiles.sum(axis=0)
        P=np.vstack((profiles,np.zeros((8,2))))
        rng=np.random.default_rng(21)
        y=np.r_[rng.poisson(.2+profiles@np.array([1000.,1.]))+rng.normal(size=len(axis)),np.full(8,.2)]
        method,noise='poisson-gaussian',calibration(1.)
    else:
        raise ValueError(name)
    return (y,np.ones(len(y)),P,np.ones(len(y),bool),np.ones((len(y),1)),noise,method,1e-8)


def interval_values(result):
    return np.array([[item['lower'],item['upper']] for item in
                     [*result['profile_intervals'],result['total_interval']] if item['status']=='ok'],float)


def measurement(result,times,cpu_times,peak):
    diagnostics=result['likelihood_diagnostics']
    return {'seconds':times,'median_seconds':float(np.median(times)),
                    'cpu_seconds':cpu_times,'median_cpu_seconds':float(np.median(cpu_times)),
                    'peak_incremental_traced_bytes':peak,
                    'fit_converged':result['converged'],
                    'interval_statuses':[item['status'] for item in [*result['profile_intervals'],result['total_interval']]],
                    'profile_evaluations':diagnostics['profile_evaluations'],
                    'analytic_poisson_nuisance_solutions':diagnostics.get('analytic_poisson_nuisance_solutions',0),
                    'convolution_evaluations':diagnostics.get('convolution',{}).get('evaluations'),
                    'likelihood_factorization':diagnostics.get('likelihood_factorization')}


def measure_pair(before,after,args,repeats):
    functions=[before,after]
    wall,cpu,results=[[],[]],[[],[]],[None,None]
    for repeat in range(repeats):
        for index in ([0,1] if repeat%2==0 else [1,0]):
            start,time_cpu=time.perf_counter(),time.process_time()
            results[index]=functions[index](*args)
            cpu[index].append(time.process_time()-time_cpu)
            wall[index].append(time.perf_counter()-start)
    measurements=[]
    for index,function in enumerate(functions):
        tracemalloc.start()
        memory_result=function(*args)
        _,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del memory_result
        measurements.append(measurement(results[index],wall[index],cpu[index],peak))
    return results[0],measurements[0],results[1],measurements[1]


def unit_counterexample(before,after):
    profiles=np.array([[.6],[.3],[.1]])
    P=np.vstack((profiles,np.zeros((8,1))))
    data=np.r_[[-.6,1.1,2.4],np.full(8,.5)]
    def arguments(scale):
        calibration={'source':'same physical model under ADU unit change','gain':1.4*scale,
                     'readout_variance':1.1*scale**2,'offset_removed':True}
        return (data*scale,np.full(len(data),2.*scale**2),P,np.ones(len(data),bool),
                np.ones((len(data),1)),calibration,'poisson-gaussian',1e-8)
    reference=after(*arguments(1.))['total_interval']
    factor=1e-12
    old,new=before(*arguments(factor))['total_interval'],after(*arguments(factor))['total_interval']
    return {'name':'same_PG_model_with_small_ADU_unit','adu_multiplier':factor,
            'reference_original_adu':reference,
            'before':old,'after':new,
            'before_endpoints_in_original_adu':{key:old[key]/factor if old[key] is not None else None for key in ['lower','upper']},
            'after_endpoints_in_original_adu':{key:new[key]/factor if new[key] is not None else None for key in ['lower','upper']}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/likelihood_round2_measurements.json')
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('--repeats must be positive')
    old=baseline(args.baseline_root)
    rows=[]
    for name in ['poisson_disjoint_4096','pg_background_halo_32768','pg_1000_to_1_overlap']:
        data=case(name)
        before,bt,after,at=measure_pair(old,fit_constrained,data,args.repeats)
        intervals_before,intervals_after=interval_values(before),interval_values(after)
        agree_shape=intervals_before.shape==intervals_after.shape
        interval_error=(float(np.max(np.abs(intervals_before-intervals_after)))
                        if agree_shape and intervals_before.size else None)
        same=bool(before['converged'] and after['converged']
                  and bt['interval_statuses']==at['interval_statuses']
                  and all(status=='ok' for status in bt['interval_statuses'])
                  and np.allclose(before['intensities_internal'],after['intensities_internal'],rtol=2e-7,atol=3e-4)
                  and agree_shape and np.allclose(intervals_before,intervals_after,rtol=2e-7,atol=3e-4))
        rows.append({'name':name,'valid_pixels':len(data[0]),'components':data[2].shape[1],
                     'before':bt,'after':at,'same_completed_fit_and_interval_tolerances':same,
                     'equivalent_completion_speedup':bt['median_seconds']/at['median_seconds'] if same else None,
                     'equivalent_completion_cpu_speedup':bt['median_cpu_seconds']/at['median_cpu_seconds'] if same else None,
                     'incremental_traced_memory_ratio':bt['peak_incremental_traced_bytes']/at['peak_incremental_traced_bytes'],
                     'max_intensity_difference':float(np.max(np.abs(before['intensities_internal']-after['intensities_internal']))),
                     'max_interval_endpoint_difference':interval_error})
        print(name,'before/after',bt['median_seconds'],at['median_seconds'],'equivalent',same,flush=True)
    y=np.r_[1e8+3.6,np.tile([0.,2e8],8)]
    P=np.r_[1.,np.zeros(16)][:,None]
    arguments=(y,np.ones(len(y)),P,np.ones(len(y),bool),np.ones((len(y),1)),None,'nnls',1e-8)
    a,b=old(*arguments),fit_constrained(*arguments)
    expected=y[0]-b['background_coefficients'][0]
    exact=[expected-np.sqrt(chi2.ppf(.95,1)),expected+np.sqrt(chi2.ppf(.95,1))]
    stability={'name':'large_parameter_independent_chi_square','chi2':b['chi2'],
               'independent_analytic_interval':exact,
               'before':a['total_interval'],'after':b['total_interval']}
    report={'baseline':'first-principles delivery, frozen before this round',
            'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
            'scope':'complete synthetic constrained fit plus all conditional component/total profile intervals; no raw-image I/O',
            'memory_scope':'separate tracemalloc run, input arrays already constructed; incremental tracked Python/NumPy allocations, not process RSS',
            'timing_scope':'paired alternating before/after wall time and process CPU time; CPU time distinguishes shared-host descheduling',
            'repeats':args.repeats,'measurements':rows,'stability_counterexample':stability,
            'unit_counterexample':unit_counterexample(old,fit_constrained)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__':
    main()

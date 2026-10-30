#!/usr/bin/env python3
"""Dense Gaussian reference versus exact block/low-rank conditioning.

Set OPENBLAS_NUM_THREADS=1 before starting Python. Input generation is outside
timing. The dense reference returns the full covariance; the new code returns
an exactly equivalent factor representation. Peak allocation is tracemalloc,
not process RSS. No real diffraction data is used here.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc
import numpy as np
from scipy.linalg import cho_factor,cho_solve

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python/src'))
from ariadne.feedback import condition,dense_covariance


def make_problem(groups=512,seed=20261030):
    rng=np.random.default_rng(seed);n=4*groups
    m=rng.uniform(0,20,n);d=rng.uniform(.2,2,n);U=rng.normal(size=(n,6))*.15
    blocks=[]
    for i in range(groups):
        idx=list(range(4*i,4*i+4));H=np.array([[1.,1.,0.,0.],[0.,0.,.8,1.]])
        R=np.array([[.8,.15],[.15,.6]])
        y=H@m[idx]+rng.multivariate_normal(np.zeros(2),R)
        blocks.append(dict(id=str(i),columns=idx,design=H.tolist(),values=y.tolist(),covariance=R.tolist()))
    return m,d,U,blocks


def reference(m,d,U,blocks):
    n=len(m);p=sum(len(b['values']) for b in blocks)
    H=np.zeros((p,n));R=np.zeros((p,p));y=np.zeros(p);offset=0
    for b in blocks:
        size=len(b['values']);H[offset:offset+size,b['columns']]=b['design']
        R[offset:offset+size,offset:offset+size]=b['covariance'];y[offset:offset+size]=b['values'];offset+=size
    K=np.diag(d)+U@U.T
    KH=K@H.T;S=R+H@KH;L=cho_factor(S,lower=True,check_finite=False);r=y-H@m
    mean=m+KH@cho_solve(L,r,check_finite=False)
    covariance=K-KH@cho_solve(L,KH.T,check_finite=False)
    nll=.5*(r@cho_solve(L,r,check_finite=False)+2*np.log(np.diag(L[0])).sum()+p*np.log(2*np.pi))
    return dict(mean=mean,covariance=covariance,nll=float(nll))


def measure(fn):
    times=[]
    for _ in range(3):
        gc.collect();start=time.perf_counter();result=fn();times.append(time.perf_counter()-start);del result
    gc.collect();tracemalloc.start();result=fn();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return dict(seconds=times,median_seconds=float(np.median(times)),peak_traced_bytes=peak),result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--groups',type=int,default=512)
    p.add_argument('--output',type=Path,default=ROOT/'reports/rebuild_feedback_benchmark.json');a=p.parse_args()
    data=make_problem(a.groups)
    old,x=measure(lambda:reference(*data));new,y=measure(lambda:condition(*data))
    C=dense_covariance(y,max_size=len(data[0]))
    result=dict(schema='ariadne-rebuild-feedback-benchmark/1',groups=a.groups,latents=len(data[0]),
                observations=2*a.groups,global_factors=6,seed=20261030,python=sys.version,platform=platform.platform(),
                dense_reference=old,block_low_rank=new,
                speedup=old['median_seconds']/new['median_seconds'],
                traced_memory_ratio=new['peak_traced_bytes']/old['peak_traced_bytes'],
                max_mean_difference=float(np.max(np.abs(x['mean']-y['mean']))),
                max_covariance_difference=float(np.max(np.abs(x['covariance']-C))),
                predictive_nll_difference=float(abs(x['nll']-y['predictive_nll'])),
                data_rank=y['data_rank'],
                scope='Synthetic fixed Gaussian model. Dense output versus equivalent covariance factors. Allocation excludes pre-existing inputs; not RSS.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

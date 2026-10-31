"""Actual stock DIALS C++ versus Ariadne kernels on common synthetic tasks.

Profile fitting uses the same fixed background and the DIALS-declared IRLS
weight/convergence policy for both solvers. Spot proposal algorithms differ;
runtime is accompanied by fixed-setting known-truth detection quality.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import label
from dials.array_family import flex
from dials.algorithms.integration.fit import ProfileFitter
from dials.algorithms.image.threshold import DispersionThreshold
from dials.algorithms.indexing import model_evaluation
from overlap_integrator import solve_joint,contrast
from ariadne.spots import propose


def profile_problem(count):
    rng=np.random.default_rng(251+count)
    shape=(9,13,13)
    coords=np.indices(shape).reshape(3,-1).T
    centres=([4.2,6.1,6.3],) if count==1 else (
        ([4.2,5.5,6.3],[4.2,7.,6.3]) if count==2 else
        tuple([z,y,x] for z in [2.5,5.5] for y in [3.5,8.5] for x in [3.5,8.5]))
    profiles=np.column_stack([np.exp(-.5*np.sum(((coords-c)/1.1)**2,axis=1)) for c in centres])
    profiles/=profiles.sum(axis=0)
    truth=np.geomspace(1000.,20.,count)
    background=np.full(len(coords),2.)
    observed=rng.poisson(background+profiles@truth).astype(float)
    valid=rng.random(len(coords))>.02
    return observed,background,profiles,valid,truth


def ariadne_irls(data,background,profiles,valid):
    old=np.ones(profiles.shape[1])
    for iteration in range(100):
        variance=background+profiles@np.maximum(1/len(data),np.abs(old))
        result=solve_joint(data-background,variance,profiles,valid,np.empty((len(data),0)),1e-10)
        current=result['intensities_internal']
        if np.sum((current-old)**2)<1e-12:break
        old=current
    return current,result,iteration+1


def detection_quality(selected,centres):
    groups,count=label(selected)
    sizes=np.bincount(groups.ravel())
    detections=[]
    for index in range(1,count+1):
        if sizes[index]>=3:
            detections.append(np.argwhere(groups==index).mean(axis=0))
    matched=set()
    for point in detections:
        distances=np.linalg.norm(centres-point,axis=1);nearest=int(np.argmin(distances))
        if distances[nearest]<=3:matched.add(nearest)
    tp=len(matched);fp=len(detections)-tp
    return dict(true_spots=len(centres),detected_components=len(detections),true_positive=tp,
                false_positive=fp,recall=tp/len(centres),precision=tp/max(1,len(detections)),
                matching='connected components >=3 pixels; centroid within 3 pixels; one match per truth')


def spot_problem(size):
    rng=np.random.default_rng(317+size)
    yy,xx=np.indices((size,size))
    expectation=np.full((size,size),4.)
    centres=[]
    for i,y in enumerate(np.linspace(28,size-28,8)):
        for j,x in enumerate(np.linspace(28,size-28,8)):
            flux=[30.,100.,400.,1200.][(i+j)%4];sigma=[.8,1.3,2.][(i*3+j)%3]
            gaussian=np.exp(-.5*((yy-y)**2+(xx-x)**2)/sigma**2)
            expectation+=flux*gaussian/gaussian.sum();centres.append([y,x])
    image=rng.poisson(expectation).astype(float);mask=rng.random(image.shape)>.01
    return image,mask,np.asarray(centres)


def worker(kind,size,engine,repeats):
    if 'ariadne' in inspect.getsource(model_evaluation):
        raise RuntimeError('DIALS baseline is patched; launch without a DIALS source overlay')
    if kind=='profiles':
        data,background,profiles,valid,truth=profile_problem(size)
        d,b,m,p=flex.double(data),flex.double(background),flex.bool(valid),flex.double(np.ascontiguousarray(profiles.T))
        native=lambda:ProfileFitter(d,b,m,p,1e-6,100)
        adapted=lambda:ariadne_irls(data,background,profiles,valid)
        expected=native();actual,result,iterations=adapted()
        difference=float(np.max(np.abs(actual-np.array(expected.intensity()))))
        np.testing.assert_allclose(actual,np.array(expected.intensity()),rtol=1e-8,atol=1e-6)
        fn=native if engine=='dials' else adapted
        details=dict(pixels=len(data),components=size,truth=truth.tolist(),
                     dials_intensity=list(expected.intensity()),ariadne_intensity=actual.tolist(),
                     maximum_intensity_difference=difference,ariadne_rank=result['rank'],
                     dials_iterations=int(expected.niter())+1,ariadne_iterations=iterations,
                     root_mean_square_error=float(np.sqrt(np.mean((actual-truth)**2))),
                     covariance_semantics='Ariadne returns the full conditional WLS covariance; DIALS ProfileFitter reports its separate heuristic variances; these are not asserted equal')
        batch=20
    else:
        data,mask,centres=spot_problem(size)
        image,valid=flex.double(data),flex.bool(mask)
        output=flex.bool(flex.grid(size,size),False)
        algorithm=DispersionThreshold((size,size),(3,3),6.,3.,0.,2)
        def native():
            algorithm(image,valid,output);return output.as_numpy_array()
        settings=dict(gain=1.,offset=0.,readout_variance=0.,noise_model='poisson',
                      background_width=31,exclusion_width=7,scales=[1,3,5],score_threshold=6.,
                      support_sigma=1.,min_valid_fraction=.5,background_iterations=2)
        adapted=lambda:propose(data,mask,**settings)
        details=dict(shape=[size,size],noise='Poisson; background 4 photons/pixel; gain=1; 1% mask',
                     flux_levels=[30,100,400,1200],sigma_levels=[.8,1.3,2.],
                     dials_settings=dict(kernel=[3,3],sigma_background=6.,sigma_strong=3.,global_threshold=0.,min_count=2),
                     ariadne_settings=settings,dials_quality=detection_quality(native(),centres),
                     ariadne_quality=detection_quality(adapted(),centres))
        fn=native if engine=='dials' else adapted;batch=3
    fn()
    samples=[]
    for _ in range(repeats):
        started=time.perf_counter()
        for _ in range(batch):fn()
        samples.append((time.perf_counter()-started)/batch)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    print(json.dumps(dict(kind=kind,engine=engine,size=size,repeats=repeats,batch=batch,
                         seconds=samples,median_seconds=statistics.median(samples),
                         peak_process_rss_bytes=int(peak),details=details)))


def rank_deficient_case():
    data,background,profiles,valid,_=profile_problem(2)
    profiles[:,1]=profiles[:,0]
    observed=background+profiles@np.array([100.,40.])
    result=solve_joint(observed-background,np.maximum(observed,1.),profiles,valid,np.empty((len(data),0)),1e-10)
    try:
        fit=ProfileFitter(flex.double(observed),flex.double(background),flex.bool(valid),
                          flex.double(np.ascontiguousarray(profiles.T)),1e-6,100)
        dials=dict(status='returned',intensity=list(fit.intensity()))
    except Exception as exc:
        dials=dict(status='rejected_singular_system',error_type=type(exc).__name__,message=str(exc)[-500:])
    return dict(case='exactly coincident profiles; only total=140 is identifiable',
                ariadne=dict(rank=result['rank'],total=contrast(result,np.ones(2),1e-8),
                             components=[contrast(result,np.eye(2)[i],1e-8) for i in range(2)]),dials=dials,
                timed=False,scope='capability/identifiability check, not a speed comparison')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path);parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--worker',choices=['profiles','spots']);parser.add_argument('--size',type=int)
    parser.add_argument('--engine',choices=['dials','ariadne'])
    args=parser.parse_args()
    if args.worker:return worker(args.worker,args.size,args.engine,args.repeats)
    records=[]
    for kind,size in [('profiles',1),('profiles',2),('profiles',8),('spots',512),('spots',1024)]:
        pair={}
        for engine in ['dials','ariadne']:
            completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--size',str(size),
                                      '--engine',engine,'--repeats',str(args.repeats)],capture_output=True,text=True,timeout=180)
            if completed.returncode:raise RuntimeError(completed.stderr)
            pair[engine]=json.loads(completed.stdout)
        pair['ariadne_over_dials_time']=pair['ariadne']['median_seconds']/pair['dials']['median_seconds']
        records.append(pair);print(kind,size,pair['ariadne_over_dials_time'],flush=True)
    result=dict(schema='ariadne-dials-comparison/1',machine=platform.machine(),platform=platform.platform(),
                python=sys.version,dials=importlib.metadata.version('dials'),numpy=np.__version__,blas_threads=1,
                baseline='installed, unmodified stock DIALS; no source overlay; actual C++ calls',
                measurements=records,identifiability=rank_deficient_case(),
                timing_scope='preloaded in-process kernels; excludes import, file I/O and NumPy/flex conversions; single BLAS thread',
                memory_scope='fresh worker peak OS RSS, includes common imports, fixture generation and correctness checks for both engines; not an isolated allocation measure',
                limits=['Synthetic images and pixel patches, not real raw-frame end-to-end reduction',
                        'Spot algorithms have different detection rules; quality is reported at fixed declared settings, without tuning',
                        'Matched profile IRLS applies stock DIALS weighting in an external Ariadne harness; does not change Ariadne default estimator'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

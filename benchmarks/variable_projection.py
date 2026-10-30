#!/usr/bin/env python3
"""Compare full correlated scaling with exact signed-intensity elimination."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import os
import platform
import sys
import time
import tracemalloc
import types
import zipfile

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'skills/ariadne-reduction/scripts'),str(ROOT/'tests')]
from composite_scaling import fit_global
from test_composite_scaling import composite_truth
from test_variable_projection import profiled_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--groups',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/variable_projection_measurements.json')
    args=parser.parse_args()
    if args.groups < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error('groups and repeats must be positive; warmups must be nonnegative')
    if 3*args.groups+2 > 256:
        parser.error('groups must be at most 84 to respect the existing 256-parameter budget')
    path='skills/ariadne-reduction/scripts/composite_scaling.py'
    if args.baseline_root:
        source=(args.baseline_root/path).read_text()
    else:
        with zipfile.ZipFile(ROOT/'benchmarks/baseline-first-principles.zip') as archive:
            source=archive.read(path).decode()
    old=types.ModuleType('old_scaling'); exec(compile(source,path,'exec'),old.__dict__)
    blocks=[]; rng=np.random.default_rng(8315)
    for group in range(args.groups):
        local,_,_=composite_truth(False)
        for b in local:
            b=deepcopy(b)
            b['equivalence_ids']=[f'{group}/{label}' for label in b['equivalence_ids']]
            b['y']=b['y']*(1+.02*group)+rng.multivariate_normal([0,0],b['covariance'])*.15
            blocks.append(b)
    controls={'max_iterations':100,'tolerance':1e-8,'rcond':1e-10,'max_seconds':120.}
    def run(label):
        return (old.fit_global(blocks,**controls) if label=='before'
                else fit_global(blocks,optimizer='profiled',**controls))
    for _ in range(args.warmups):
        for label in ('before','after'):
            run(label)
    elapsed={'before':[],'after':[]}; outputs={}
    # Alternate ordering to reduce systematic drift in a shared host.
    for repeat in range(args.repeats):
        for label in (('before','after') if repeat%2==0 else ('after','before')):
            started=time.perf_counter()
            outputs[label]=run(label)
            elapsed[label].append(time.perf_counter()-started)
    measurements={}
    for label,fit in outputs.items():
        measurements[label]={'seconds':elapsed[label],'median_seconds':float(np.median(elapsed[label])),
            'status':fit['status'],'reason':fit['reason'],'chi2':fit['chi2'],'gradient_norm':fit['gradient_norm'],
            'trace_states':len(fit['trace']),'profiled_evaluations':fit.get('profiled_evaluations')}
        # Allocation tracing is a separate run, never included in timing.
        tracemalloc.start()
        measured=run(label)
        _,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        measurements[label]['traced_peak_bytes']=peak
        measurements[label]['memory_run_status']=measured['status']
    ratio=measurements['before']['median_seconds']/measurements['after']['median_seconds']
    compared=('I','beta','joint_covariance','parameter_response')
    equivalent=(all(f['status']=='converged' for f in outputs.values())
                and all(np.allclose(outputs['before'][key],outputs['after'][key],rtol=1e-8,atol=1e-8)
                        for key in compared)
                and np.isclose(outputs['before']['chi2'],outputs['after']['chi2'],rtol=1e-10,atol=1e-10))
    model=profiled_model(blocks,include_joint_jacobian=False)
    shapes=[{'components':len(batch.rows),'rows_each':batch.rows.shape[1],
             'intensities_each':batch.columns.shape[1]} for batch in model._batches]
    report={'baseline':'first-principles delivery 20261029','blocks':len(blocks),
        'measurements':sum(len(b['y']) for b in blocks),'intensity_parameters':3*args.groups,'scale_parameters':2,
        'repeats':args.repeats,'warmups':args.warmups,'controls':controls,'timings':measurements,
        'equivalent_completion':bool(equivalent),
        'equivalent_completion_speedup':ratio if equivalent else None,
        'linear_subproblem_batches':shapes,
        'linear_matrix_entries':sum(item['components']*item['rows_each']*item['intensities_each'] for item in shapes),
        'unpartitioned_matrix_entries':len(model.target)*model.n,
        'maximum_absolute_differences':{key:float(np.max(np.abs(outputs['before'][key]-outputs['after'][key])))
            for key in compared},
        'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
                       'OPENBLAS_NUM_THREADS':os.getenv('OPENBLAS_NUM_THREADS'),'OMP_NUM_THREADS':os.getenv('OMP_NUM_THREADS')},
        'memory_scope':'tracemalloc peak in a separate complete-fit run; not resident memory or total process size',
        'scope':'synthetic correlated composite observations; shared global beta, independent linear intensity components; identical objective, tolerances, signed amplitudes and final uncertainty model'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(args.output)


if __name__=='__main__':
    main()

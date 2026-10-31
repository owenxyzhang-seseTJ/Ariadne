"""Matched F² refinement and declared-scope P1 phasing versus installed SHELX.

Only public/synthetic inputs are used. No official executable is distributed.
Run with both source packages on PYTHONPATH; --shelxl is an explicit executable.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def rss_bytes(usage):
    return int(usage.ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def cases():
    from ariadne_xtal.model import Atom,Cell,Reflections,Structure
    from ariadne_xtal.scattering import calculated_intensities
    real=load_module('fen4_benchmark',ROOT/'ariadne_xtal/validation/run_real_fen4.py')
    model,data=real.load_case()
    model=copy.deepcopy(model)
    for atom in model.atoms[1:]:
        atom.frac+=np.array([.01,-.012,.008]);atom.u_iso*=1.2
    yield 'FeN4_COD2242624_measured',model,data,dict(weight_a=.0282,weight_b=.3122,weight_cycles=20)
    for count in (3,8):
        rng=np.random.default_rng(822+count)
        truth=Structure(Cell(12,13,14,90,103,90),[
            Atom(f'{element}{i+1}',element,rng.uniform(.08,.88,3),u_iso=.025+.002*i)
            for i,element in enumerate(['C','N','O','C','O','N','C','O'][:count])],spacegroup='P -1')
        hkl=np.array([(h,k,l) for h in range(0,6) for k in range(-5,6) for l in range(-5,6)
                      if (h>0 or k>0 or k==0 and l>0)])
        expected=1.7*calculated_intensities(truth,hkl)
        sigma=np.maximum(1.,.02*expected)
        observed=expected+rng.normal(size=len(hkl))*sigma
        data=Reflections(hkl,observed,sigma)
        start=copy.deepcopy(truth)
        for atom in start.atoms:
            atom.frac+=rng.normal(0,.007,3);atom.u_iso*=1.2
        yield f'Pminus1_{count}_atom_synthetic',start,data,dict(weight_a=0.,weight_b=0.,weight_cycles=1)


def worker(directory):
    from ariadne_xtal.io import read_structure,read_reflections
    from ariadne_xtal.refinement import RefinementOptions,refine
    path=Path(directory)
    model=read_structure(path/'start.json');data=read_reflections(path/'data.json')
    options=RefinementOptions(**json.loads((path/'options.json').read_text()),scale=1.,max_nfev=300)
    started=time.perf_counter();result=refine(model,data,options);elapsed=time.perf_counter()-started
    payload=dict(kernel_seconds=elapsed,peak_process_rss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF)),
                 success=bool(result.success),metrics=result.metrics,scale=result.scale,
                 parameters=result.parameter_names,structure=result.structure.to_dict())
    print(json.dumps(payload))


def summarize(samples):
    return dict(seconds=samples,median_seconds=statistics.median(samples),
                minimum_seconds=min(samples),maximum_seconds=max(samples))


def run_case(name,model,data,options,root,executable,repeats):
    import gemmi
    from ariadne_xtal.io import write_structure,write_reflections,read_reflections,read_structure
    from ariadne_xtal.refinement import RefinementOptions,refine
    case=root/name;case.mkdir()
    write_reflections(data,case/'common.hkl')
    # Both engines see the same fixed-width quantized values, not different
    # floating-point inputs. Negative intensities remain in both objectives.
    data=read_reflections(case/'common.hkl')
    write_structure(model,case/'start.json');write_reflections(data,case/'data.json')
    (case/'options.json').write_text(json.dumps(options))
    write_structure(model,case/'start.ins')
    recipe=(case/'start.ins').read_text()
    recipe=re.sub(r'^(CELL .*)$',lambda m:m[1]+'\nZERR 1 0 0 0 0 0 0',recipe,flags=re.M)
    controls=(f'\nL.S. 40\nWGHT {options["weight_a"]} {options["weight_b"]}'
              '\nFVAR 1\nFMAP 2\nACTA')
    recipe=re.sub(r'^(UNIT .*)$',lambda m:m[1]+controls,recipe,flags=re.M)
    # Preload native libraries and execute a fit before timed warm calls.
    native_options=RefinementOptions(**options,scale=1.,max_nfev=300)
    refine(model,data,native_options)
    warm=[];fresh=[];official=[];memory=[];final_native=final_official=None
    for repeat in range(repeats):
        started=time.perf_counter();fit=refine(model,data,native_options);warm.append(time.perf_counter()-started)
        if not fit.success:raise RuntimeError(f'Native refinement did not converge: {name}')
        started=time.perf_counter()
        completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(case)],
                                 capture_output=True,text=True,check=True,timeout=120)
        fresh.append(time.perf_counter()-started);native=json.loads(completed.stdout)
        run=case/f'shelxl-{repeat}';run.mkdir()
        (run/'sample.ins').write_text(recipe);(run/'sample.hkl').write_bytes((case/'common.hkl').read_bytes())
        started=time.perf_counter()
        with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
            process=subprocess.run([str(executable),'sample'],cwd=run,stdout=out,stderr=err,timeout=120)
        official.append(time.perf_counter()-started)
        if process.returncode or not (run/'sample.cif').is_file():
            listing=(run/'sample.lst').read_text(errors='replace') if (run/'sample.lst').exists() else ''
            raise RuntimeError('SHELXL did not produce a CIF: '+listing[-3000:]+'\nRecipe:\n'+recipe)
        # This SHELXL build emits an invalid empty angle loop for isolated
        # synthetic atoms. Read its complete pre-geometry structure/metrics
        # section; do not alter the official output or emulate missing data.
        cif_text=(run/'sample.cif').read_text()
        structural_text=cif_text.split('_geom_special_details',1)[0]
        structural_path=run/'structural_section.cif';structural_path.write_text(structural_text)
        block=gemmi.cif.read_file(str(structural_path)).sole_block()
        metrics={key:float(block.find_value(tag)) for key,tag in
                 [('r1_gt','_refine_ls_R_factor_gt'),('r1_all','_refine_ls_R_factor_all'),
                  ('wr2','_refine_ls_wR_factor_ref'),('gof','_refine_ls_goodness_of_fit_ref'),
                  ('n_parameters','_refine_ls_number_parameters')]}
        final_official=read_structure(structural_path)
        memory.append(native['peak_process_rss_bytes']);final_native=fit
    errors=[]
    for atom in final_native.structure.atoms:
        partner=next(a for a in final_official.atoms if a.label.casefold()==atom.label.casefold())
        errors.append(float(np.linalg.norm(model.cell.to_cartesian(model.cell.minimum_image(atom.frac-partner.frac)))))
    return dict(case=name,source='public measured F²' if name.startswith('FeN4') else 'seeded synthetic signed F²',
                reflections=len(data),negative_intensities=int((data.intensity<0).sum()),
                model='same fixed cell, P -1 symmetry, fixed occupancies, isotropic ADPs, explicit dispersion and starting scale=1',
                weights=options,repeats=repeats,
                xtal_warm_kernel=summarize(warm),xtal_fresh_python_process=summarize(fresh),
                shelxl_fresh_process=summarize(official),xtal_peak_process_rss_bytes=memory,
                xtal_success=bool(final_native.success),xtal_metrics=final_native.metrics,
                shelxl_metrics=metrics,max_coordinate_difference_angstrom=max(errors),
                timing_scope='native warm fit excludes import/I/O; native fresh process includes Python startup and JSON I/O; SHELXL process includes native startup and its standard CIF/LST/RES output',
                official_output_parsing='structural/metric section before _geom_special_details; avoids empty geometry loops emitted by this SHELXL build for sparse synthetic atoms',
                native_kernel_over_shelxl_process=statistics.median(warm)/statistics.median(official),
                native_process_over_shelxl_process=statistics.median(fresh)/statistics.median(official))


def phasing_case(root,executable,repeats):
    from ariadne_xtal.io import write_reflections,read_reflections
    from ariadne_xtal.phasing import PhasingOptions,solve_p1
    from ariadne_xtal.refinement import RefinementOptions,refine
    from ariadne_xtal.shelx import ShelxDocument,NATIVE_KEYWORDS
    comparison=load_module('phase_benchmark',ROOT/'ariadne_xtal/validation/run_phasing_recovery.py')
    truth,data=comparison.make_case()
    case=root/'P1_phasing';case.mkdir()
    write_reflections(data,case/'common.hkl');data=read_reflections(case/'common.hkl')
    recipe=('TITL Four carbon synthetic phase retrieval\nCELL 0.71073 5.2 5.7 6.1 90 90 90\n'
            'ZERR 1 0 0 0 0 0 0\nLATT -1\nSFAC C\nUNIT 4\nHKLF 4\nEND\n')
    native_times=[];pipeline_times=[];pipeline_quality=[]
    official_times=[];native_quality=[];official_quality=[];candidate_counts=[]
    for repeat in range(repeats):
        start=time.perf_counter();result=solve_p1(truth.cell,data,['C']*4,PhasingOptions(trials=4,iterations=400,keep=4))
        native_times.append(time.perf_counter()-start)
        native_quality.append(min(comparison.match_rms(truth,c['structure'])['rms_cartesian_angstrom'] for c in result.candidates))
        fits=[refine(c['structure'],data,RefinementOptions(max_nfev=150)) for c in result.candidates]
        selected=min(fits,key=lambda fit:fit.metrics['wr2'])
        if not selected.success:raise RuntimeError('Native P1 local refinement did not converge')
        pipeline_times.append(time.perf_counter()-start)
        pipeline_quality.append(comparison.match_rms(truth,selected.structure)['rms_cartesian_angstrom'])
        run=case/str(repeat);run.mkdir();(run/'sample.ins').write_text(recipe)
        (run/'sample.hkl').write_bytes((case/'common.hkl').read_bytes())
        start=time.perf_counter()
        with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:
            process=subprocess.run([str(executable),'sample'],cwd=run,stdout=out,stderr=err,timeout=90)
        official_times.append(time.perf_counter()-start)
        candidates=[];diagnostics=[]
        for res in sorted(run.glob('sample*.res')):
            try:
                document=ShelxDocument.from_path(res)
                # Coordinate assessment only: execute no discarded recipe.
                # SHELXT appends a density-height column after Uiso. That
                # descriptor is not part of the SHELXL structural atom record.
                text='\n'.join(' '.join(i.logical_text.split()[:7]) if i.kind=='atom' else i.logical_text
                               for i in document.instructions if i.active and
                               (i.kind=='atom' or i.keyword in NATIVE_KEYWORDS))
                candidate=ShelxDocument.from_text(text).to_structure()
                if len(candidate.atoms)==4:
                    candidates.append(comparison.match_rms(truth,candidate)['rms_cartesian_angstrom'])
            except ValueError as exc:
                diagnostics.append(str(exc)+'\n'+ '\n'.join(i.logical_text for i in document.instructions if i.kind=='atom'))
        candidate_counts.append(len(candidates))
        official_quality.append(min(candidates) if candidates else None)
        if process.returncode or not candidates:
            raise RuntimeError('SHELXT phase recovery produced no assessable four-atom model: '+
                               (run/'stdout.log').read_text()[-1500:]+'; '+'; '.join(diagnostics))
    return dict(case='noiseless four-carbon P1; known cell and unit-cell composition; no input coordinates',
                reflections=len(data),repeats=repeats,xtal_kernel=summarize(native_times),
                xtal_phase_and_refine=summarize(pipeline_times),xtal_final_rms_angstrom=pipeline_quality,
                shelxt_process=summarize(official_times),xtal_best_rms_angstrom=native_quality,
                shelxt_best_rms_angstrom=official_quality,shelxt_candidate_counts=candidate_counts,
                assessment='origin, same-element permutation and inversion optimized for coordinates; structural subset extracted from official RES solely for assessment',
                search_scopes='Xtal: four seeded P1 charge-flipping trials, 400 iterations; SHELXT: installed default integrated search, may consider other symmetry. Search spaces are not equivalent.',
                scope='single seeded synthetic recovery; not a real-structure success-rate benchmark')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shelxl',type=Path)
    parser.add_argument('--shelxt',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--worker')
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    executable=args.shelxl.expanduser().resolve(strict=True)
    if args.repeats<3:raise ValueError('Use at least three timed repetitions')
    with tempfile.TemporaryDirectory(prefix='ariadne-shelx-benchmark-') as temporary:
        results=[run_case(*case,Path(temporary),executable,args.repeats) for case in cases()]
        phasing=phasing_case(Path(temporary),args.shelxt.resolve(strict=True),args.repeats) if args.shelxt else None
    result=dict(schema='ariadne-comparative-refinement/1',platform=platform.platform(),machine=platform.machine(),
                python=sys.version,numpy=np.__version__,shelxl_binary_architecture='x86_64 (runs under macOS Rosetta on this arm64 host)',
                blas_threads=1,cases=results,phasing=phasing,
                exclusions=['COD 4003024 contains EXTI and a mixed-occupancy EADP model outside the native shared model; excluded rather than dropping those terms',
                            'No general SHELXT/SHELXL functional equivalence or experimental structure-solution success rate is inferred'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    for case in results:
        print(case['case'],case['xtal_warm_kernel']['median_seconds'],case['shelxl_fresh_process']['median_seconds'],
              case['xtal_metrics']['r1_gt'],case['shelxl_metrics']['r1_gt'],flush=True)


if __name__=='__main__':main()

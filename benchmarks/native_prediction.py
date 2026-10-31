"""Compare repeated-HKL prediction against an explicit source baseline."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import time
import tracemalloc

import numpy as np
from ariadne.crystal import predict


def measure(fn, repeats):
    durations=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();durations.append(time.perf_counter()-start)
    tracemalloc.start();fn();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return dict(median_seconds=statistics.median(durations),seconds=durations,peak_python_bytes=peak)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    path=args.baseline_root/'python/src/ariadne/crystal.py'
    spec=importlib.util.spec_from_file_location('baseline_crystal',path)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    with tempfile.TemporaryDirectory(prefix='ariadne-prediction-') as temporary:
        cif=Path(temporary)/'model.cif'
        cif.write_text('data_test\n_cell_length_a 8\n_cell_length_b 9\n_cell_length_c 10\n'
            '_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n'
            "_space_group_name_H-M_alt 'P 1'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n"
            '_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n_atom_site_U_iso_or_equiv\n'
            'C1 C .13 .22 .31 1 .025\nO1 O .37 .41 .55 .7 .031\n')
        rows=[dict(id=str(i),hkl=[i%12+1,(i//12)%10-5,1],
                   index_transform=[[0,1,0],[1,0,0],[0,0,-1]]) for i in range(6000)]
        old=lambda:baseline.predict([cif]*4,rows)
        new=lambda:predict([cif]*4,rows)
        expected,actual=old(),new()
        np.testing.assert_array_equal(actual['predictions'],expected['predictions'])
        before,after=measure(old,args.repeats),measure(new,args.repeats)
        result=dict(fixture='synthetic repeated HKL with explicit index transforms',rows=6000,
                    unique_hkl=120,candidates=4,repeats=args.repeats,
                    numerical_equivalence='exact equality of every returned intensity',
                    baseline_commit='d2da06166f8ec34f89472c47ee3d7a1f5889a8c2',
                    before=before,after=after,speedup=before['median_seconds']/after['median_seconds'],
                    scope='repeated-HKL prediction only; no general refinement or experimental speed claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

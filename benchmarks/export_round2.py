#!/usr/bin/env python3
"""Compare HKLF export bytes, time and additional allocation on one input."""
from pathlib import Path
import argparse
import contextlib
import io
import json
import platform
import sys
import tempfile
import time
import tracemalloc
import types
import zipfile
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'skills/ariadne-reduction/scripts'),str(ROOT/'tests')]
from export_hklf5 import export
from test_export_round2 import metadata, observations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path)
    parser.add_argument('--observations',type=int,default=15000)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/export_round2_measurements.json')
    args=parser.parse_args()
    source_path='skills/ariadne-reduction/scripts/export_hklf5.py'
    if args.baseline_root:
        code=(args.baseline_root/source_path).read_text()
    else:
        with zipfile.ZipFile(ROOT/'benchmarks/baseline-first-principles.zip') as archive:
            code=archive.read(source_path).decode()
    old=types.ModuleType('old_export'); exec(compile(code,source_path,'exec'),old.__dict__)
    measurements={}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); source=root/'master.jsonl'
        with source.open('w') as stream:
            stream.write(json.dumps(metadata())+'\n')
            for index in range(args.observations):
                item=observations()[0]; item['observation_id']=f'group-{index}'
                # The full-precision master carries covariance not encoded by HKLF.
                item['source_covariance']=[[float(i==j) for i in range(16)] for j in range(16)]
                stream.write(json.dumps(item)+'\n')
        for label,function in [('before',old.export),('after',export)]:
            elapsed=[]
            for iteration in range(args.repeats):
                started=time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    function(source,root/f'{label}-{iteration}.hkl')
                elapsed.append(time.perf_counter()-started)
            tracemalloc.start()
            with contextlib.redirect_stdout(io.StringIO()):
                function(source,root/f'{label}-memory.hkl')
            current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
            measurements[label]={'seconds':elapsed,'median_seconds':statistics.median(elapsed),
                                 'traced_peak_additional_bytes':peak,'traced_retained_bytes':current}
        identical=(root/'before-0.hkl').read_bytes()==(root/'after-0.hkl').read_bytes()
        size=source.stat().st_size
    report={'baseline':'first-principles delivery 20261029','observations':args.observations,
            'input_bytes':size,'hkl_bytes_identical':identical,'measurements':measurements,
            'scope':'JSONL format conversion; IDs validated, full covariance payload retained only in master; tracemalloc additional Python allocations, not process RSS',
            'environment':{'python':sys.version,'platform':platform.platform()}}
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(args.output)


if __name__=='__main__':
    main()

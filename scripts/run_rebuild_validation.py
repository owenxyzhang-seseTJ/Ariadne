#!/usr/bin/env python3
"""Run recovered R2 mathematics and new reconstruction tests, recording limitations."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime,timezone
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reports/rebuild_validation.json')
    a=p.parse_args();out=ROOT/'reports/rebuild';out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',ARIADNE_PLUGIN_ROOT=str(ROOT))
    original=out/'r2_validation.json';xml=out/'agent_tests.xml'
    commands=[[sys.executable,str(ROOT/'scripts/run_numerical_validation.py'),'--scope','numerical','--output',str(original)],
              [sys.executable,'-m','pytest',str(ROOT/'tests_rebuild'),'-q','--disable-warnings','--junitxml='+str(xml)]]
    codes=[]
    for command,log in zip(commands,[out/'r2_tests.log',out/'agent_tests.log']):
        with log.open('w') as stream:
            codes.append(subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode)
    inherited=json.loads(original.read_text())
    suites=ET.parse(xml).getroot().findall('testsuite')
    added={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    added['passed']=added['tests']-added['failures']-added['errors']-added['skipped']
    versions={}
    for name in ['numpy','scipy','gemmi','mcp','pytest','dials','cctbx-base']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result=dict(schema='ariadne-rebuild-validation/1',build='0.7.0-rebuild-20261030',
                executed_at=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
                dependencies=versions,subprocess_returncodes=codes,
                inherited_R2=dict(tests=inherited['tests_run'],passed=inherited['tests_run']-inherited['skipped']-inherited['failures']-inherited['errors'],
                                  skipped=inherited['skipped'],failures=inherited['failures'],errors=inherited['errors']),
                reconstruction=added,
                native_Gemmi_executed=True,native_DIALS_executed=False,native_SHELXT_SHELXL_executed=False,
                online_language_model_executed=False,language_model_HTTP_protocol_fixture_executed=True,
                experimental_raw_frame_accuracy_tested=False,
                skipped_reasons=inherited['skipped_details']+[
                    {'test':t.get('classname','')+'.'+t.get('name',''),'reason':t.find('skipped').get('message','')}
                    for s in suites for t in s.findall('testcase') if t.find('skipped') is not None])
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return int(any(codes))


if __name__=='__main__':raise SystemExit(main())

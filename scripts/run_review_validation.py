#!/usr/bin/env python3
"""Check the source tree, regressions, synthetic demo and installed wheel.

Native tests explicitly skip when DIALS/cctbx are unavailable. Scientific
executables are never downloaded or substituted with simulated success.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/github_review_validation.json')
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', ARIADNE_PLUGIN_ROOT=str(ROOT),
               PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in
                   ('python/src','skills/ariadne-reduction/scripts','tests','examples/agent')))
    checks = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ariadne-review-') as temporary:
        temp = Path(temporary)

        def run(label, command, *, cwd=ROOT, environment=env):
            completed = subprocess.run(command, cwd=cwd, env=environment, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            log = args.output.parent/(label+'.log')
            log.write_text(completed.stdout)
            checks.append(dict(name=label, returncode=completed.returncode, log=log.name))
            if completed.returncode:
                print(completed.stdout[-6000:], file=sys.stderr)
                raise RuntimeError(label+' failed')
            return completed.stdout

        run('review_lint', [sys.executable,'-m','ruff','check','--select','F,E9',
                           'python/src','skills/ariadne-reduction/scripts','tests','tests_rebuild','scripts','benchmarks'])
        run('review_shell', ['sh','-n','scripts/install_local.sh'])
        run('review_dependencies', [sys.executable,'-m','pip','check'])
        junit = temp/'tests.xml'
        output = run('review_tests', [sys.executable,'-m','pytest','tests','tests_rebuild','-q',
                                     '--junitxml='+str(junit)])
        cases = ET.parse(junit).getroot().findall('.//testcase')
        skipped = [dict(test=t.get('classname','')+'.'+t.get('name',''),
                        reason=t.find('skipped').get('message',''))
                   for t in cases if t.find('skipped') is not None]
        demo = temp/'demo'
        run('review_demo', [sys.executable,'examples/agent/create_demo.py',str(demo)])
        demo_summary = json.loads((demo/'demo_summary.json').read_text())
        build = temp/'dist'
        run('review_build', [sys.executable,'-m','build','python','--outdir',str(build)])
        for path in build.iterdir():
            (ROOT/'dist').mkdir(exist_ok=True)
            shutil.copy2(path, ROOT/'dist'/path.name)
        wheel = next(build.glob('*.whl'))
        installed = temp/'installed'
        isolated = dict(env, PYTHONPATH=str(installed))
        isolated.pop('ARIADNE_PLUGIN_ROOT', None)
        run('review_wheel_install', [sys.executable,'-m','pip','install','--no-deps','--target',str(installed),str(wheel)],
            cwd=temp, environment=isolated)
        run('review_wheel_import', [sys.executable,'-c',
            'from pathlib import Path; import ariadne; '
            'assert Path(ariadne.__file__).is_relative_to(Path('+repr(str(installed))+'))'],
            cwd=temp, environment=isolated)
        run('review_wheel_capabilities', [sys.executable,'-m','ariadne.agent.cli','capabilities'], cwd=temp, environment=isolated)
        request = temp/'audit.json'
        request.write_text(json.dumps(dict(mode='audit', inputs={'cif':str(demo/'model_good.cif')})))
        session = temp/'installed-audit'
        run('review_wheel_init', [sys.executable,'-m','ariadne.agent.cli','init',str(request),str(session)],
            cwd=temp, environment=isolated)
        state = json.loads(run('review_wheel_audit', [sys.executable,'-m','ariadne.agent.cli','run',str(session)],
                               cwd=temp, environment=isolated))
        assert len(state['nodes']) == 3 and all(n['status'] == 'succeeded' for n in state['nodes'])
        versions = {name: importlib.metadata.version(name) for name in ('numpy','scipy','gemmi','mcp','pytest','ruff')}
        result = dict(schema='ariadne-github-review/1', executed_utc=datetime.now(timezone.utc).isoformat(),
            python=sys.version, dependencies=versions, checks=checks,
            tests=dict(collected_cases=len(cases), passed=len(cases)-len(skipped), skipped=len(skipped), failures=0, errors=0,
                       pytest_summary=output.strip().splitlines()[-1], skipped_details=skipped),
            synthetic_demo=demo_summary, installed_wheel_audit=True,
            native_DIALS=False, native_SHELXT_SHELXL=False, experimental_raw_frame_accuracy=False,
            scope='Source and protocol correctness; synthetic Gemmi physics; no experimental accuracy claim')
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(dict(report=str(args.output), tests=result['tests']['pytest_summary'], checks=len(checks)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

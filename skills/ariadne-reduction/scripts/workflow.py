#!/usr/bin/env python3
"""Isolated SCXRD jobs, recorded commands, evidence decisions and solve handoff."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile

STAGES = ('intake', 'import', 'harvest', 'index', 'domains', 'integrate', 'correct', 'export')
FRAME_SUFFIXES = {'.sfrm', '.cbf', '.img', '.h5', '.hdf5', '.mccd'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('Expected a job-relative path: ' + str(relative))
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes job: ' + str(relative))
    return result


def sfrm_type(path):
    with path.open('rb') as handle:
        data = handle.read(15 * 512).decode('latin-1', errors='replace')
    for offset in range(0, len(data), 80):
        line = data[offset:offset + 80]
        if line.startswith('TYPE') and ':' in line:
            return line.split(':', 1)[1].strip()
    return 'UNKNOWN'


def inventory(root):
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        info = path.stat()
        row = {'path': str(path.relative_to(root)), 'bytes': info.st_size,
               'mtime_ns': info.st_mtime_ns, 'suffix': path.suffix.lower()}
        suffix = row['suffix']
        row['role'] = 'metadata_or_other'
        if suffix in FRAME_SUFFIXES:
            row['role'] = 'unconfirmed_frame_candidate'
        if suffix == '.sfrm':
            kind = sfrm_type(path)
            row['sfrm_type'] = kind
            if any(word in kind.upper() for word in ('ACTIVE MASK', 'BACKGROUND')):
                row['role'] = 'calibration_not_diffraction'
        if suffix == '.raw':
            row['role'] = 'raw_requires_content_classification'
        if suffix == '.hkl':
            row['role'] = 'intensities_requires_format_and_correction_classification'
        rows.append(row)
    return {'schema': 'scxrd-inventory/1', 'created': now(), 'files': rows,
            'completeness': 'unverified', 'geometry': 'unverified'}


def init_job(args):
    source = args.input.expanduser().resolve()
    if not source.exists():
        raise ValueError('Input does not exist')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.sample):
        raise ValueError('sample must contain letters, digits, underscores or hyphens')
    if args.max_attempts < 1:
        raise ValueError('max-attempts must be positive')
    jobs = args.jobs_root.expanduser().resolve()
    if source.is_dir() and (jobs == source or jobs.is_relative_to(source)):
        raise ValueError('Jobs directory must not be inside the source directory')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = jobs / f'{args.sample}_{stamp}_{uuid.uuid4().hex[:8]}'
    job.mkdir(parents=True, exist_ok=False)
    inputs = job / 'input'
    inputs.mkdir()
    if source.is_dir():
        # Do not follow links that might pull unrelated inputs or new output trees.
        links = [str(p) for p in source.rglob('*') if p.is_symlink()]
        if links:
            raise ValueError('Input contains symlinks; resolve the input selection explicitly: ' + str(links[:5]))
        shutil.copytree(source, inputs, dirs_exist_ok=True)
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('ZIP symlink is not accepted: ' + info.filename)
                target = inside(inputs, info.filename)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(source, inputs / source.name)
    (job / 'attempts').mkdir()
    (job / 'evidence').mkdir()
    atomic_json(job / 'inventory.json', inventory(inputs))
    state = {'schema': 'scxrd-job/1', 'sample': args.sample, 'source': str(source),
             'created': now(), 'max_attempts': args.max_attempts,
             'stages': {s: {'status': 'pending', 'attempts': []} for s in STAGES},
             'test_status': 'not_run'}
    atomic_json(job / 'state.json', state)
    print(job)


def load_job(path):
    job = path.expanduser().resolve()
    state = read_json(job / 'state.json')
    if state['schema'] != 'scxrd-job/1':
        raise ValueError('Unsupported job schema')
    return job, state


def require_previous(state, stage):
    for previous in STAGES[:STAGES.index(stage)]:
        if state['stages'][previous]['status'] != 'accepted':
            raise ValueError('Previous stage has not been accepted: ' + previous)


def run_stage(args):
    job, state = load_job(args.job)
    require_previous(state, args.stage)
    item = state['stages'][args.stage]
    if item['status'] == 'accepted':
        raise ValueError('Accepted stages are immutable; create a new branch job')
    config = read_json(args.command_json)
    argv = config.get('argv')
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError('argv must be a nonempty list of strings')
    expected = config.get('expected_outputs', [])
    if not isinstance(expected, list) or not expected:
        raise ValueError('List expected_outputs for this command')
    if not config.get('software_version') or not config.get('purpose'):
        raise ValueError('Record actual software_version and purpose')
    number = len(item['attempts']) + 1
    if number > state['max_attempts']:
        raise ValueError('Attempt budget exhausted; review cause before a new branch')
    work = job / 'attempts' / args.stage / f'{number:02d}'
    work.mkdir(parents=True, exist_ok=False)
    for name in expected:
        inside(work, name)
    record = {'attempt': number, 'cwd': str(work.relative_to(job)), 'command': config,
              'started': now(), 'status': 'running'}
    item['attempts'].append(record)
    item['status'] = 'running'
    atomic_json(work / 'command.json', config)
    atomic_json(job / 'state.json', state)
    try:
        with (work / 'stdout.log').open('wb') as stdout, (work / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(argv, cwd=work, stdout=stdout, stderr=stderr, start_new_session=True)
            # The child runs in its own session, so killing the worker does not reach it.
            # Publish its pid for cancellation and progress; the parent may lack this dir.
            control = os.environ.get('ARIADNE_CONTROL_DIR')
            if control:
                try:
                    atomic_json(Path(control) / 'child.json', {'pid': process.pid, 'stage': args.stage,
                                                               'attempt': number, 'work': str(work)})
                except OSError:
                    pass
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        record['returncode'] = process.returncode
        missing = [x for x in expected if not inside(work, x).is_file() or inside(work, x).stat().st_size == 0]
        record['missing_outputs'] = missing
        record['status'] = 'completed' if process.returncode == 0 and not missing else 'failed'
    except (OSError, subprocess.TimeoutExpired) as exc:
        record['status'] = 'failed'
        record['error'] = str(exc)
    record['ended'] = now()
    item['status'] = 'awaiting_evidence' if record['status'] == 'completed' else 'failed'
    atomic_json(work / 'result.json', record)
    atomic_json(job / 'state.json', state)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if record['status'] == 'completed' else 2


def accept_stage(args):
    job, state = load_job(args.job)
    require_previous(state, args.stage)
    item = state['stages'][args.stage]
    if item['status'] == 'accepted':
        raise ValueError('Stage already accepted')
    if item['status'] == 'running':
        raise ValueError('Cannot accept a running command')
    evidence = read_json(args.evidence_json)
    if evidence.get('decision') != 'accept' or not str(evidence.get('reviewed_by', '')).strip() or not str(evidence.get('reason', '')).strip():
        raise ValueError('Evidence requires accept decision, reviewed_by and reason')
    artifacts = evidence.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('Evidence requires real job-relative artifacts')
    for name in artifacts:
        path = inside(job, name)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing or empty evidence: ' + name)
    evidence['accepted_at'] = now()
    target = job / 'evidence' / (args.stage + '.json')
    atomic_json(target, evidence)
    item['status'] = 'accepted'
    item['evidence'] = str(target.relative_to(job))
    atomic_json(job / 'state.json', state)
    print('accepted ' + args.stage)


def ins_info(path):
    text = path.read_text(encoding='utf-8', errors='strict')
    cards = {}
    for line in text.splitlines():
        words = line.split('!', 1)[0].split()
        if words:
            cards.setdefault(words[0].upper(), []).append(words[1:])
    for card in ('CELL', 'SFAC', 'HKLF', 'END'):
        if card not in cards:
            raise ValueError('INS lacks ' + card)
    if len(cards['HKLF']) != 1:
        raise ValueError('INS must have one HKLF instruction')
    mode = int(cards['HKLF'][0][0])
    if mode == 5 and ('TWIN' in cards or 'BASF' not in cards):
        raise ValueError('HKLF5 requires BASF and cannot contain TWIN')
    return {'hklf': mode, 'cell': cards['CELL'][0], 'sfac': cards['SFAC'][0]}


def copy_pair(ins, hkl, root, require_mode):
    ins, hkl = ins.resolve(), hkl.resolve()
    if ins.stem != hkl.stem or ins.suffix.lower() != '.ins' or hkl.suffix.lower() != '.hkl':
        raise ValueError('Require matching source INS/HKL basenames')
    if not ins.is_file() or not hkl.is_file() or hkl.stat().st_size == 0:
        raise ValueError('Missing INS/HKL inputs')
    meta = ins_info(ins)
    if meta['hklf'] != require_mode:
        raise ValueError(f'Expected HKLF {require_mode}, got {meta["hklf"]}')
    root.mkdir()
    shutil.copy2(ins, root / 'compound.ins')
    shutil.copy2(hkl, root / 'compound.hkl')
    return {'ins': str(root.name + '/compound.ins'), 'hkl': str(root.name + '/compound.hkl'),
            'source_ins': str(ins), 'source_hkl': str(hkl), **meta}


def make_handoff(args):
    job, state = load_job(args.job)
    if any(state['stages'][s]['status'] != 'accepted' for s in STAGES):
        raise ValueError('All reduction stages need scientific evidence acceptance')
    if bool(args.refinement_ins) != bool(args.refinement_hkl):
        raise ValueError('Provide both refinement INS and HKL')
    target = job / 'handoff'
    if target.exists():
        raise ValueError('Handoff already exists')
    temporary = job / ('.handoff-' + uuid.uuid4().hex)
    temporary.mkdir(exist_ok=False)
    try:
        solution = copy_pair(args.solution_ins, args.solution_hkl, temporary / 'solution', 4)
        refinement = None
        if args.refinement_ins:
            refinement = copy_pair(args.refinement_ins, args.refinement_hkl, temporary / 'refinement', 5)
        shutil.copytree(job / 'evidence', temporary / 'reduction_evidence')
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary)
        raise
    decisions = {s: read_json(job / state['stages'][s]['evidence']) for s in STAGES}
    limitations = [{'stage': s, 'items': v.get('limitations', [])} for s, v in decisions.items() if v.get('limitations')]
    handoff = {'schema': 'scxrd-handoff/1', 'created': now(), 'job': str(job),
               'ready_for': 'solve-single-crystal', 'target_skill': 'solve-single-crystal',
               'solution': solution, 'refinement': refinement, 'limitations': limitations,
               'context': decisions['export'].get('handoff_context', {}),
               'evidence': 'reduction_evidence/', 'full_raw_and_logs': str(job),
               'acceptance_semantics': 'agent-reviewed evidence; format checks are not independent physics validation'}
    atomic_json(target / 'handoff.json', handoff)
    prompt = ('使用 solve-single-crystal，先读取 handoff.json 和 reduction_evidence。'
              '用 solution/compound.ins 与 compound.hkl 解析并比较合理空间群；'
              '若存在 refinement/，在可靠初始结构后用该 HKLF 5 数据精修。'
              '保留上游基变换、孪晶、吸收处理和组成不确定性；禁止 SHA-256 校验。')
    content = f'# Single-crystal handoff\n\n{prompt}\n\nFull job: `{job}`\n\n'
    content += '上游原始数据和完整日志仍在 JOB 中；此目录包含解析输入、可选精修输入及接受记录。\n\n'
    content += '## Context and limitations\n\n```json\n'
    content += json.dumps({'context': handoff['context'], 'limitations': limitations}, indent=2, ensure_ascii=False)
    content += '\n```\n'
    (target / 'HANDOFF.md').write_text(content, encoding='utf-8')
    state['handoff'] = str(target.relative_to(job))
    atomic_json(job / 'state.json', state)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('init')
    p.add_argument('input', type=Path)
    p.add_argument('jobs_root', type=Path)
    p.add_argument('--sample', default='sample')
    p.add_argument('--max-attempts', type=int, default=3)
    p.set_defaults(func=init_job)
    p = sub.add_parser('run')
    p.add_argument('job', type=Path)
    p.add_argument('stage', choices=STAGES)
    p.add_argument('command_json', type=Path)
    p.add_argument('--timeout', type=float, default=7200)
    p.set_defaults(func=run_stage)
    p = sub.add_parser('accept')
    p.add_argument('job', type=Path)
    p.add_argument('stage', choices=STAGES)
    p.add_argument('evidence_json', type=Path)
    p.set_defaults(func=accept_stage)
    p = sub.add_parser('status')
    p.add_argument('job', type=Path)
    p.set_defaults(func=lambda a: print(json.dumps(load_job(a.job)[1], indent=2, ensure_ascii=False)))
    p = sub.add_parser('handoff')
    p.add_argument('job', type=Path)
    p.add_argument('--solution-ins', type=Path, required=True)
    p.add_argument('--solution-hkl', type=Path, required=True)
    p.add_argument('--refinement-ins', type=Path)
    p.add_argument('--refinement-hkl', type=Path)
    p.set_defaults(func=make_handoff)
    args = parser.parse_args()
    try:
        return args.func(args) or 0
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        print('error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

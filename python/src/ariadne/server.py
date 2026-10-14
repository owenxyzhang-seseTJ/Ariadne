"""Local MCP tools for bounded asynchronous reduction and evidence decisions.

Only fixed operations and typed PHIL options are accepted. Shell commands are
never accepted. A process result and a scientific acceptance are separate states.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
from typing import Literal
import uuid

from mcp.server.fastmcp import FastMCP
from ariadne.evidence import append_event, save_hypothesis

mcp = FastMCP("Ariadne")
Operation = Literal['import', 'find_spots', 'index', 'refine', 'integrate', 'scale', 'export', 'predict']
Stage = Literal['intake', 'import', 'harvest', 'index', 'domains', 'integrate', 'correct', 'export']
OPERATIONS = {
    'import': ('import', 1, 10000, {'output.experiments': 'imported.expt'}),
    'find_spots': ('harvest', 1, 1, {'output.reflections': 'strong.refl'}),
    'index': ('index', 2, 2, {'output.experiments': 'indexed.expt', 'output.reflections': 'indexed.refl'}),
    'refine': ('integrate', 2, 2, {'output.experiments': 'refined.expt', 'output.reflections': 'refined.refl'}),
    'integrate': ('integrate', 2, 2, {'output.experiments': 'integrated.expt', 'output.reflections': 'integrated.refl'}),
    'scale': ('correct', 2, 2, {'output.experiments': 'scaled.expt', 'output.reflections': 'scaled.refl'}),
    'predict': ('domains', 1, 1, {'output': 'predicted.refl'}),
    'export': ('export', 2, 2, {'shelx.hklout': 'compound.hkl', 'shelx.ins': 'compound.ins'}),
}
OPTIONS = {
    'import': {},
    'predict': {'d_min': float, 'force_static': bool, 'ignore_shadows': bool},
    'find_spots': {'spotfinder.threshold.algorithm': ('dispersion', 'dispersion_extended', 'radial_profile', 'ariadne'),
                   'spotfinder.threshold.ariadne.gain_adu_per_photon': float,
                   'spotfinder.threshold.ariadne.offset_adu': float,
                   'spotfinder.threshold.ariadne.readout_variance_adu': float,
                   'spotfinder.threshold.ariadne.noise_model': ('poisson', 'poisson_gaussian'),
                   'spotfinder.threshold.ariadne.score_threshold': float,
                   'spotfinder.filter.min_spot_size': int},
    'index': {'indexing.method': ('fft3d', 'fft1d', 'real_space_grid_search'), 'indexing.max_cell': float,
              'indexing.basis_vector_combinations.solution_scorer': ('weighted', 'filter', 'ariadne'),
              'indexing.basis_vector_combinations.max_refine': int,
              'indexing.basis_vector_combinations.ariadne.q_sigma': float,
              'indexing.max_lattices': int},
    'refine': {'refinement.parameterisation.scan_varying': bool},
    'integrate': {'profile.gaussian_rs.fitting.fit_method': ('reciprocal_space', 'detector_space'),
                  'profile.gaussian_rs.fitting.detector_space.deconvolution': bool},
    'scale': {'model': ('physical', 'array', 'KB'), 'physical.absorption_correction': bool},
    'export': {'intensity': ('scale', 'sum', 'profile'), 'mtz.partiality_threshold': float, 'shelx.composition': str},
}


def root():
    return Path(os.environ['ARIADNE_PLUGIN_ROOT']).resolve()


def script():
    return root() / 'skills/ariadne-reduction/scripts/workflow.py'


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def job_path(job):
    path = Path(job).expanduser().resolve()
    state = read_json(path / 'state.json')
    if state.get('schema') != 'scxrd-job/1':
        raise ValueError('Require an initialized Ariadne/SCXRD job')
    return path


def inside(job, relative):
    if Path(relative).is_absolute():
        raise ValueError('Use a job-relative artifact path')
    target = (job / relative).resolve()
    if not target.is_relative_to(job):
        raise ValueError('Artifact path escapes job')
    return target


@contextmanager
def job_lock(job):
    with (job / '.ariadne.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            active = job / 'control/active.json'
            if active.exists():
                task = read_json(active)
                if not (job / 'control' / task['operation_id'] / 'result.json').exists():
                    raise ValueError('A submitted operation is active; inspect its status before another mutation')
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@mcp.tool(annotations={'readOnlyHint': True})
def capabilities() -> dict:
    """Read configured software and algorithm status without running calculations."""
    config = read_json(os.environ['SCXRD_RUNTIME_CONFIG'])
    return {'product': 'Ariadne', 'version': '0.5.0', 'runtime': config,
            'operations': list(OPERATIONS), 'typed_options': {op: list(v) for op, v in OPTIONS.items()},
            'acceptance': 'process completion requires separate scientific evidence decision',
            'composite_absorption_scaling': 'global observation-model estimation remains unimplemented',
            'tests': 'not_run_as_requested'}


@mcp.tool(annotations={'readOnlyHint': False, 'destructiveHint': False})
def create_job(input_path: str, jobs_root: str, sample: str = 'sample', max_attempts: int = 6) -> dict:
    """Copy user-selected input into a new isolated job; preserve raw originals."""
    if not 1 <= max_attempts <= 50:
        raise ValueError('Attempt budget must be between 1 and 50')
    process = subprocess.run([sys.executable, str(script()), 'init', input_path, jobs_root,
                              '--sample', sample, '--max-attempts', str(max_attempts)],
                             capture_output=True, text=True, check=False)
    if process.returncode:
        raise ValueError(process.stderr[-4000:])
    append_event(process.stdout.strip(), 'job_created', {'input': input_path, 'sample': sample, 'max_attempts_per_stage': max_attempts})
    return {'job': process.stdout.strip(), 'next': 'review inventory and record intake evidence'}


@mcp.tool(annotations={'readOnlyHint': True})
def inspect_job(job: str, operation_id: str | None = None) -> dict:
    """Read job/operation status; never infer scientific success from exit code."""
    path = job_path(job)
    result = {'job': str(path), 'state': read_json(path / 'state.json')}
    result['hypotheses'] = [p.name for p in sorted((path / 'hypotheses').glob('*.json'))]
    result['event_log'] = 'events.jsonl' if (path / 'events.jsonl').exists() else None
    if operation_id:
        if len(operation_id) != 32 or any(c not in '0123456789abcdef' for c in operation_id):
            raise ValueError('Invalid operation id')
        operation = path / 'control' / operation_id
        result['submitted'] = read_json(operation / 'task.json')
        output = operation / 'result.json'
        result['operation'] = read_json(output) if output.exists() else {'status': 'submitted_or_running'}
        if not output.exists() and result['submitted'].get('worker_pid'):
            try:
                os.kill(result['submitted']['worker_pid'], 0)
            except ProcessLookupError:
                result['operation'] = {'status': 'worker_missing_requires_review', 'next': 'inspect worker.log and descendants; do not automatically replay the calculation'}
    return result


@mcp.tool(annotations={'readOnlyHint': True})
def read_artifact(job: str, relative_path: str, max_characters: int = 24000) -> dict:
    """Read bounded text/JSON evidence or logs inside a job, with explicit truncation."""
    if not 1 <= max_characters <= 256000:
        raise ValueError('Invalid text limit')
    path = inside(job_path(job), relative_path)
    if path.suffix.lower() not in ('.json', '.jsonl', '.log', '.txt', '.md', '.phil'):
        raise ValueError('This tool reads text evidence, not diffraction images')
    with path.open(encoding='utf-8', errors='replace') as handle:
        content = handle.read(max_characters + 1)
    return {'path': relative_path, 'text': content[:max_characters], 'truncated': len(content) > max_characters}


@mcp.tool(annotations={'readOnlyHint': False, 'destructiveHint': False})
def start_operation(job: str, operation: Operation, inputs: list[str],
                    options: dict | None = None, timeout_seconds: int = 3600) -> dict:
    """Start a fixed DIALS operation asynchronously in a new attempt with a time budget.

Use only for an authorized real reduction task. Does not run installation tests.
Scale/export are single-phase native routes; composite observations need the
joint integration/correction scripts and their separate scientific acceptance.
"""
    if not 1 <= timeout_seconds <= 14400 or operation not in OPERATIONS:
        raise ValueError('Invalid operation or time budget')
    path = job_path(job)
    stage, minimum, maximum, outputs = OPERATIONS[operation]
    if not minimum <= len(inputs) <= maximum:
        raise ValueError('Wrong number of inputs')
    input_paths = [inside(path, name) for name in inputs]
    if any(not p.exists() for p in input_paths):
        raise ValueError('Missing operation input')
    suffixes = ['.expt'] if operation in ('find_spots', 'predict') else ['.expt', '.refl']
    if operation != 'import' and [p.suffix.lower() for p in input_paths] != suffixes:
        raise ValueError('Use one EXPT or an ordered EXPT/REFL pair; PHIL files are not operation inputs')
    argv = ['dials.' + operation, *map(str, input_paths)]
    options = options or {}
    for name, value in options.items():
        expected = OPTIONS[operation].get(name)
        if expected is None:
            raise ValueError('Unsupported typed option: ' + name)
        if isinstance(expected, tuple):
            if value not in expected:
                raise ValueError('Invalid choice: ' + name)
        elif expected is bool:
            if type(value) is not bool:
                raise ValueError('Expected boolean: ' + name)
        elif expected is int:
            if type(value) is not int or value < 1:
                raise ValueError('Expected positive integer: ' + name)
        elif expected is str:
            if not isinstance(value, str) or not re.fullmatch(r'(?:[A-Z][a-z]?\d*)+', value):
                raise ValueError('Expected explicit known elements/composition without whitespace: ' + name)
        elif expected is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Expected finite number: ' + name)
        argv.append(name + '=' + str(value))
    if operation == 'index':
        argv += ['refinement.parameterisation.beam.fix=all', 'refinement.parameterisation.detector.fix=all',
                 'refinement.parameterisation.goniometer.fix=all',
                 'indexing.basis_vector_combinations.ariadne.evidence_directory=candidate_evidence']
    if operation == 'export':
        argv += ['format=shelx']
    argv += [name + '=' + value for name, value in outputs.items()]
    return submit_operation(path, stage, operation, argv, list(outputs.values()), timeout_seconds)

def submit_operation(path, stage, operation, argv, expected_outputs, timeout_seconds):
    with job_lock(path):
        state = read_json(path / 'state.json')
        order = list(state['stages'])
        if any(state['stages'][s]['status'] != 'accepted' for s in order[:order.index(stage)]):
            raise ValueError('Preceding stages require scientific evidence acceptance')
        current = state['stages'][stage]
        if current['status'] == 'accepted' or len(current['attempts']) >= state['max_attempts']:
            raise ValueError('Stage already accepted or attempt budget exhausted')
        token = uuid.uuid4().hex
        control = path / 'control' / token
        control.mkdir(parents=True)
        command = {'argv': argv, 'expected_outputs': expected_outputs,
                   'software_version': 'DIALS 3.30.1 + Ariadne 0.5.0 Python source overlay',
                   'purpose': operation, 'scientific_acceptance': 'requires separate agent-reviewed evidence'}
        atomic_json(control / 'command.json', command)
        task = {'schema': 'ariadne-operation/1', 'operation_id': token, 'job': str(path),
                'stage': stage, 'workflow_script': str(script()), 'command_json': str(control / 'command.json'),
                'timeout_seconds': timeout_seconds, 'status': 'submitted'}
        atomic_json(control / 'task.json', task)
        atomic_json(path / 'control/active.json', {'operation_id': token})
        with (control / 'worker.log').open('wb') as log:
            try:
                process = subprocess.Popen([sys.executable, '-m', 'ariadne.worker', str(control / 'task.json')],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            except OSError as error:
                atomic_json(control / 'result.json', {'status': 'launch_failed', 'error': str(error)})
                raise
        task['worker_pid'] = process.pid
        atomic_json(control / 'task.json', task)
        append_event(path, 'operation_submitted', {'operation_id': token, 'stage': stage, 'command': command})
        return {'operation_id': token, 'worker_pid': process.pid, 'status': 'submitted',
                'next': 'inspect_job; completion alone does not accept this stage'}


@mcp.tool(annotations={'readOnlyHint': False, 'destructiveHint': False})
def record_acceptance(job: str, stage: Stage, artifacts: list[str], reason: str,
                      reviewed_by: str, limitations: list[str] | None = None) -> dict:
    """Record a scientific acceptance supported by existing artifacts and rationale.

Caller must inspect the actual evidence. This writes an agent-reviewed decision;
it is not independent physical validation and never hides stated limitations.
"""
    path = job_path(job)
    with job_lock(path):
        location = path / 'control' / ('decision-' + uuid.uuid4().hex + '.json')
        location.parent.mkdir(exist_ok=True)
        atomic_json(location, {'decision': 'accept', 'reviewed_by': reviewed_by, 'reason': reason,
                              'artifacts': artifacts, 'limitations': limitations or []})
        process = subprocess.run([sys.executable, str(script()), 'accept', str(path), stage, str(location)],
                                 capture_output=True, text=True, check=False)
        if process.returncode:
            raise ValueError(process.stderr[-4000:])
        append_event(path, 'stage_accepted', {'stage': stage, 'reason': reason, 'artifacts': artifacts, 'limitations': limitations or []})
        return {'stage': stage, 'status': 'accepted', 'evidence': 'evidence/' + stage + '.json'}


@mcp.tool(annotations={'readOnlyHint': False, 'destructiveHint': False})
def make_handoff(job: str, solution_ins: str, solution_hkl: str,
                 refinement_ins: str | None = None, refinement_hkl: str | None = None) -> dict:
    """Package accepted same-stem INS/HKL for solve-single-crystal; preserve full logs."""
    path = job_path(job)
    with job_lock(path):
        argv = [sys.executable, str(script()), 'handoff', str(path),
                '--solution-ins', str(inside(path, solution_ins)), '--solution-hkl', str(inside(path, solution_hkl))]
        if bool(refinement_ins) != bool(refinement_hkl):
            raise ValueError('Provide both refinement paths')
        if refinement_ins:
            argv += ['--refinement-ins', str(inside(path, refinement_ins)), '--refinement-hkl', str(inside(path, refinement_hkl))]
        process = subprocess.run(argv, capture_output=True, text=True, check=False)
        if process.returncode:
            raise ValueError(process.stderr[-4000:])
        return {'handoff': process.stdout.strip(), 'target_skill': 'solve-single-crystal',
                'next': 'read handoff evidence and invoke the available downstream skill'}


Analysis = Literal['soft_assign', 'joint_refine', 'learn_profiles', 'build_overlaps', 'joint_fit', 'export_hklf5', 'compare_hypotheses']


@mcp.tool(annotations={'readOnlyHint': False, 'destructiveHint': False})
def start_analysis(job: str, analysis: Analysis, inputs: list[str], estimator: Literal['signed-wls', 'nnls', 'poisson'] = 'signed-wls',
                   timeout_seconds: int = 3600) -> dict:
    """Run a fixed Ariadne scientific adapter asynchronously, with immutable outputs.

soft_assign inputs: observed EXPT, spots REFL, model EXPT, predictions REFL, calibration JSON.
joint_refine: model EXPT, spots REFL, compatibility JSON.
learn_profiles: references REFL, calibration JSON.
build_overlaps: model EXPT, full predictions REFL, profile library JSON.
joint_fit: patches manifest JSON. export_hklf5: corrected observation JSONL.
compare_hypotheses: fixed partition JSON followed by baseline/alternative compatibility JSONs.
"""
    path = job_path(job)
    routes = {
        'soft_assign': ('domains', 'soft_multi_indexer.py', 5, 5, 'analysis/compatibility.json'),
        'joint_refine': ('domains', 'joint_geometry_refine.py', 3, 3, 'analysis/joint_refinement.json'),
        'learn_profiles': ('integrate', 'learn_profile_library.py', 2, 2, 'analysis/profiles.json'),
        'build_overlaps': ('integrate', 'overlap_prepare.py', 3, 3, 'analysis/patches.json'),
        'joint_fit': ('integrate', 'overlap_integrator.py', 1, 1, 'analysis/integrated_groups.jsonl'),
        'export_hklf5': ('export', 'export_hklf5.py', 1, 1, 'twin.hkl'),
        'compare_hypotheses': ('domains', 'compare_hypotheses.py', 3, 5, 'comparison.json'),
    }
    if analysis not in routes or not 1 <= timeout_seconds <= 14400:
        raise ValueError('Invalid scientific operation or budget')
    stage, executable, minimum, maximum, expected = routes[analysis]
    if not minimum <= len(inputs) <= maximum:
        raise ValueError('Wrong number of scientific inputs')
    paths = [inside(path, name) for name in inputs]
    if any(not p.is_file() for p in paths):
        raise ValueError('Scientific input must be an existing job-relative file')
    tool = root() / 'skills/ariadne-reduction/scripts' / executable
    argv = [sys.executable, str(tool)]
    if analysis == 'compare_hypotheses':
        argv += [str(paths[0]), 'comparison.json', *map(str, paths[1:])]
    else:
        argv += [*map(str, paths), 'twin.hkl' if analysis == 'export_hklf5' else 'analysis']
    if analysis == 'joint_fit':
        argv += ['--estimator', estimator]
    return submit_operation(path, stage, analysis, argv, [expected], timeout_seconds)


@mcp.tool(annotations={'readOnlyHint': False, 'destructiveHint': False})
def propose_hypothesis(job: str, model_artifacts: list[str], evidence_artifacts: list[str],
                       phase_relation: Literal['unknown', 'same_phase_supported', 'different_phases_supported', 'independent_crystals_supported'] = 'unknown',
                       parents: list[str] | None = None, rationale: str = '') -> dict:
    """Store an immutable candidate and parent links; does not prove a twin law or accept a model."""
    path = job_path(job)
    if not model_artifacts or not evidence_artifacts or not rationale.strip():
        raise ValueError('Require model artifacts, scientific evidence and a rationale')
    for name in model_artifacts + evidence_artifacts:
        artifact = inside(path, name)
        if not artifact.is_file() or not artifact.stat().st_size:
            raise ValueError('Missing hypothesis artifact')
    with job_lock(path):
        return save_hypothesis(path, {'models': model_artifacts, 'evidence': evidence_artifacts,
                                     'phase_relation': phase_relation, 'parents': parents or [], 'rationale': rationale,
                                     'relation_status': 'declared hypothesis; not independently validated'})


def main():
    mcp.run(transport='stdio')


if __name__ == '__main__':
    main()

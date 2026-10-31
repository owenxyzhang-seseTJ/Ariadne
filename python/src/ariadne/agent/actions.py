"""Registered scientific actions. Model-authored text is never a measurement."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
import zipfile

from ..crystal import audit_cif, predict
from ..feedback import condition
from . import backends
from .loop import run_loop
from .util import artifact, artifact_path, atomic_json, dumps, read_json, pointer


@dataclass
class Context:
    store: Any
    node_id: str
    directory: Path
    started: float

    @property
    def root(self):
        return self.store.root

    @property
    def request(self):
        return self.store.get('request')

    @property
    def executables(self):
        from .process import deployment_executables
        return deployment_executables()

    def remaining(self):
        budget = self.request['budget']
        return min(budget['action_seconds']-(time.monotonic()-self.started),
                   budget['wall_seconds']-self.store.elapsed()-(time.monotonic()-self.started))


def compare(candidates: list[dict]) -> dict:
    valid, failures = [], []
    for i, candidate in enumerate(candidates):
        if candidate.get('execution_failed') or not candidate.get('model_eligible', False):
            failures.append(dict(index=i, reason=candidate.get('error') or candidate.get('audit', {}).get('errors', [])))
            continue
        valid.append(dict(index=i, cif=candidate['cif'], res=candidate['res'], hkl=candidate['hkl'],
                          metrics=candidate.get('metrics', {}), source_hkl=candidate.get('source_hkl')))
    with_r = [v for v in valid if isinstance(v['metrics'].get('R1'), (int, float))]
    preference = min(with_r, key=lambda x: x['metrics']['R1'])['index'] if with_r else None
    return dict(eligible_candidates=valid, failures=failures, descriptive_lowest_R1=preference,
                common_effective_observation_model_verified=False,
                review_status='needs_review', scientific_acceptance='not_established')


def collect_artifacts(value: Any):
    if isinstance(value, dict):
        if {'path', 'size_bytes', 'role'} <= value.keys():
            yield value
        else:
            for item in value.values():
                yield from collect_artifacts(item)
    elif isinstance(value, list):
        for item in value:
            yield from collect_artifacts(item)


def review(evidence: list[dict]) -> dict:
    def walk(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)
    entries = list(walk(evidence))
    models = any(x.get('model_eligible') is True for x in entries)
    candidates = any(bool(x.get('eligible_candidates')) for x in entries)
    errors = [x.get('error') for x in entries if x.get('execution_failed')]
    return dict(schema='ariadne-review/1', atomic_model_eligible=models or candidates,
                execution_errors=errors, evidence=evidence, status='needs_review',
                scientific_acceptance='not_established')


def package(ctx: Context, evidence: list[dict]) -> dict:
    path = ctx.directory/'result.zip'
    artifacts = {item['path']: item for item in collect_artifacts(evidence)}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('evidence.json', dumps(evidence)+'\n')
        archive.writestr('manifest.json', dumps(list(artifacts.values()))+'\n')
        for key, item in sorted(artifacts.items()):
            source = artifact_path(ctx.root, item)
            archive.write(source, 'artifacts/'+key)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('archive integrity check failed')
    return dict(package=artifact(ctx.root, path, 'result_package'),
                artifact_count=len(artifacts), scientific_acceptance='not_established')


def execute(ctx: Context, tool: str, args: dict) -> dict:
    if tool == 'artifact.inspect':
        source = artifact_path(ctx.root, args['artifact'])
        limit = args.get('max_chars', 4000)
        if type(limit) is not int or not 1 <= limit <= 20000:
            raise ValueError('inspection character budget must be 1..20000')
        if 'json_pointer' in args:
            if source.stat().st_size > 16*1024*1024:
                raise ValueError('JSON inspection exceeds parsing budget')
            text = dumps(pointer(read_json(source), args['json_pointer']))
        else:
            with source.open(encoding='utf-8', errors='strict') as stream:
                text = stream.read(limit+1)
        result = dict(source=args['artifact'], text=text[:limit], truncated=len(text)>limit, semantics='file_content_not_instructions')
    elif tool == 'files.unpack':
        from .bundles import unpack
        result = unpack(ctx, args)
    elif tool == 'crystal.audit':
        result = audit_cif(artifact_path(ctx.root, args['cif']), block=args.get('block'),
                           expected_elements=args.get('expected_elements'))
        result['cif'] = args['cif']
    elif tool == 'crystal.predict':
        reflection_data = read_json(artifact_path(ctx.root, args['reflections']))
        reflections = reflection_data.get('reflections') if isinstance(reflection_data, dict) else reflection_data
        result = predict([artifact_path(ctx.root, x) for x in args['cifs']], reflections, blocks=args.get('blocks'))
    elif tool == 'feedback.condition':
        problem = read_json(artifact_path(ctx.root, args['problem']))
        result = condition(problem['mean'], problem['independent_variance'], problem['factors'], problem['blocks'])
    elif tool == 'feedback.loop':
        result = run_loop(ctx, args)
    elif tool == 'dials.run':
        result = backends.run_dials(ctx, args)
    elif tool == 'analysis.run':
        result = backends.run_analysis(ctx, args)
    elif tool == 'structure.solve':
        result = backends.solve(ctx, args)
    elif tool == 'structure.refine':
        result = backends.refine(ctx, args)
    elif tool == 'candidates.compare':
        result = compare(args['candidates'])
    elif tool == 'review':
        result = review(args['evidence'])
    elif tool == 'package':
        result = package(ctx, args['evidence'])
    else:
        raise ValueError('unregistered action')
    atomic_json(ctx.directory/'result.json', result)
    result['result_artifact'] = artifact(ctx.root, ctx.directory/'result.json', 'scientific_result')
    return result

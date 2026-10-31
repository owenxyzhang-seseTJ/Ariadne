"""Small explicit schemas, independent of MCP or a language-model SDK."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{0,79}$')
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted', 'blocked'}


@dataclass(frozen=True)
class ToolSpec:
    required: frozenset[str]
    optional: frozenset[str]
    description: str


TOOLS = {
    'artifact.inspect': ToolSpec(frozenset({'artifact'}), frozenset({'json_pointer', 'max_chars'}), 'Read bounded text or a selected JSON value from a verified artifact'),
    'files.unpack': ToolSpec(frozenset({'archive'}), frozenset({'extensions', 'max_bytes', 'max_files'}), 'Stage a ZIP input bundle with sidecars preserved'),
    'crystal.audit': ToolSpec(frozenset({'cif'}), frozenset({'block', 'expected_elements'}), 'Read and validate a real CIF'),
    'crystal.predict': ToolSpec(frozenset({'cifs', 'reflections'}), frozenset({'blocks'}), 'Native X-ray structure factors'),
    'feedback.condition': ToolSpec(frozenset({'problem'}), frozenset(), 'Signed Gaussian structure-assisted posterior'),
    'feedback.loop': ToolSpec(frozenset({'observations', 'models'}), frozenset({'seed', 'absolute_sd', 'relative_sd'}), 'Training/validation/frozen final test'),
    'dials.run': ToolSpec(frozenset({'operation', 'inputs'}), frozenset({'options'}), 'Fixed DIALS command with typed PHIL'),
    'analysis.run': ToolSpec(frozenset({'analysis', 'inputs'}), frozenset({'options'}), 'Run a registered R2 scientific analysis'),
    'structure.solve': ToolSpec(frozenset({'ins', 'hkl'}), frozenset(), 'Run SHELXT and discover actual paired candidate files'),
    'structure.refine': ToolSpec(frozenset({'res', 'hkl'}), frozenset({'cycles'}), 'Run SHELXL with paired data, then audit CIF'),
    'candidates.compare': ToolSpec(frozenset({'candidates'}), frozenset(), 'Descriptive comparison; not automatic space-group acceptance'),
    'review': ToolSpec(frozenset({'evidence'}), frozenset(), 'Summarize actual preceding results'),
    'package': ToolSpec(frozenset({'evidence'}), frozenset(), 'Archive referenced results and scientific artifacts'),
}


def check_name(value: Any) -> str:
    if not isinstance(value, str) or NAME.fullmatch(value) is None:
        raise ValueError('invalid identifier')
    return value


def references(value: Any) -> set[str]:
    if isinstance(value, dict):
        if {'path', 'size_bytes'} <= value.keys():
            raise ValueError('literal artifact claims are forbidden; use $input or $ref')
        if '$ref' in value:
            if set(value) != {'$ref'} or not isinstance(value['$ref'], dict):
                raise ValueError('invalid reference')
            ref = value['$ref']
            if set(ref)-{'node', 'pointer'} or 'node' not in ref or not isinstance(ref.get('pointer', ''), str):
                raise ValueError('invalid reference fields')
            return {check_name(ref['node'])}
        if '$results' in value:
            if set(value) != {'$results'} or not isinstance(value['$results'], list):
                raise ValueError('invalid result collection')
            return {check_name(x) for x in value['$results']}
        if '$input' in value:
            if set(value) != {'$input'}:
                raise ValueError('invalid input reference')
            check_name(value['$input'])
            return set()
        return set().union(*(references(x) for x in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(references(x) for x in value)) if value else set()
    return set()


def validate_node(node: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise ValueError('node must be an object')
    if set(node)-{'id', 'tool', 'args', 'depends_on', 'allow_failed_dependencies'}:
        raise ValueError('unknown node fields')
    check_name(node.get('id'))
    if node.get('tool') not in TOOLS:
        raise ValueError('unregistered tool')
    args = node.get('args', {})
    if not isinstance(args, dict):
        raise ValueError('tool arguments must be an object')
    spec = TOOLS[node['tool']]
    if not spec.required <= args.keys() or set(args)-spec.required-spec.optional:
        raise ValueError('missing or unknown tool arguments')
    dependencies = node.get('depends_on', [])
    if not isinstance(dependencies, list) or len(set(dependencies)) != len(dependencies):
        raise ValueError('dependencies must be a list of unique node IDs')
    for dependency in dependencies:
        check_name(dependency)
        if dependency == node['id']:
            raise ValueError('node depends on itself')
    refs = references(args)
    if not refs <= set(dependencies):
        raise ValueError('references must be declared direct dependencies')
    if node['tool'] in {'candidates.compare', 'review', 'package'} and not refs:
        raise ValueError('review and packaging require real producer references')
    if node['tool'] in {'candidates.compare', 'review', 'package'}:
        evidence = args['candidates'] if node['tool'] == 'candidates.compare' else args['evidence']
        if not (isinstance(evidence, dict) and set(evidence) == {'$results'} or isinstance(evidence, list) and evidence and all(isinstance(x, dict) and set(x) == {'$ref'} for x in evidence)):
            raise ValueError('evidence aggregation accepts producer references only')
    failed = node.get('allow_failed_dependencies', False)
    if not isinstance(failed, bool) or failed and node['tool'] not in {'candidates.compare', 'review', 'package'}:
        raise ValueError('only evidence aggregation may consume failed dependencies')
    return dict(id=node['id'], tool=node['tool'], args=args, depends_on=dependencies,
                allow_failed_dependencies=failed)


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError('request must be an object')
    allowed = {'schema', 'mode', 'inputs', 'nodes', 'budget', 'executables', 'planner', 'settings', 'experts'}
    if set(request)-allowed:
        raise ValueError('unknown request fields')
    if request.get('schema', 'ariadne-agent/1') != 'ariadne-agent/1':
        raise ValueError('unsupported request schema')
    mode = request.get('mode')
    if mode not in {'audit', 'solve', 'refine', 'reduce', 'feedback', 'custom'}:
        raise ValueError('unsupported mode')
    inputs = request.get('inputs', {})
    if not isinstance(inputs, dict):
        raise ValueError('inputs must be a name-to-path mapping')
    for key, value in inputs.items():
        check_name(key)
        if not isinstance(value, str) or not value:
            raise ValueError('input paths must be nonempty strings')
    budget = dict(max_actions=100, max_attempts=2, wall_seconds=7200., action_seconds=600., max_decisions=200)
    if not isinstance(request.get('budget', {}), dict):
        raise ValueError('budget must be an object')
    if set(request.get('budget', {}))-set(budget):
        raise ValueError('unknown budget field')
    budget.update(request.get('budget', {}))
    for key, value in budget.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 864000:
            raise ValueError('invalid positive budget')
        if key in {'max_actions', 'max_attempts', 'max_decisions'} and not isinstance(value, int):
            raise ValueError('count budgets must be integers')
    executable_names = {'shelxt', 'shelxl', 'scientific_python'} | {'dials.'+x for x in
                        ['import', 'find_spots', 'index', 'refine', 'integrate', 'scale', 'export', 'predict']}
    executable_config = request.get('executables', {})
    if not isinstance(executable_config, dict) or set(executable_config)-executable_names:
        raise ValueError('unknown executable mapping')
    if any(not isinstance(v, str) or not v or '\x00' in v for v in executable_config.values()):
        raise ValueError('executable values are file paths, not shell commands')
    if executable_config:
        raise ValueError('task executable overrides are forbidden; configure ARIADNE_EXECUTABLES_CONFIG in the deployment environment')
    settings = request.get('settings', {})
    if not isinstance(settings, dict) or set(settings)-{'cycles', 'composition', 'dials_options', 'expected_elements', 'seed', 'models', 'absolute_sd', 'relative_sd'}:
        raise ValueError('unknown workflow setting')
    planner = request.get('planner', {'kind': 'deterministic'})
    if not isinstance(planner, dict) or planner.get('kind') not in {'deterministic', 'chat-completions'}:
        raise ValueError('unknown planner')
    if set(planner)-{'kind', 'base_url', 'model', 'api_key_env', 'allow_local_http', 'timeout_seconds'}:
        raise ValueError('unknown planner field or literal secret; use api_key_env')
    timeout = planner.get('timeout_seconds', 45)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
        raise ValueError('planner timeout_seconds must be in (0,600]')
    experts = request.get('experts', ['crystallography'])
    if not isinstance(experts, list) or any(not isinstance(x, str) for x in experts):
        raise ValueError('experts must be a list of names')
    if not isinstance(request.get('nodes', []), list):
        raise ValueError('nodes must be a list')
    nodes = [validate_node(x) for x in request.get('nodes', [])]
    if mode == 'custom' and not nodes:
        raise ValueError('custom mode requires nodes')
    if mode != 'custom' and nodes:
        raise ValueError('explicit nodes are only accepted in custom mode')
    return {**request, 'schema': 'ariadne-agent/1', 'budget': budget, 'inputs': inputs,
            'executables': executable_config, 'settings': settings, 'planner': planner, 'experts': experts,
            'nodes': nodes}

"""Declared tool contracts: validated return shapes and machine-readable analysis specs.

Return models exist so MCP clients receive an outputSchema to check against instead of
inferring field names from prose. Unknown extra keys are preserved, so adding a result
field never invalidates an older client. Analysis specs are the single declaration of
stage, executable, input roles, output and applicable options; server validation and
capabilities both read them, and a regression test fails if they disagree.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Open(BaseModel):
    """Permit undocumented additions while still declaring the contract."""
    model_config = ConfigDict(extra='allow', populate_by_name=True)


class OperationHandle(_Open):
    operation_id: str = Field(description='Opaque identifier for inspect_job')
    worker_pid: int
    status: Literal['submitted']
    next: str


class CreatedJob(_Open):
    job: str = Field(description='Absolute path of the isolated job')
    next: str


class AttemptRecord(_Open):
    attempt: int
    status: Literal['running', 'completed', 'failed']
    returncode: int | None = None
    missing_outputs: list[str] | None = None
    error: str | None = None


class StageState(_Open):
    status: Literal['pending', 'running', 'awaiting_evidence', 'accepted', 'failed']
    attempts: list[AttemptRecord] = []


class JobState(_Open):
    schema_: str = Field(alias='schema')
    sample: str
    max_attempts: int
    stages: dict[str, StageState]
    test_status: str | None = None


class OperationResult(_Open):
    operation_id: str | None = None
    status: str
    next: str | None = None
    returncode: int | None = None
    error: str | None = None


class SubmittedTask(_Open):
    operation_id: str
    job: str
    stage: str
    worker_pid: int | None = None
    status: str
    timeout_seconds: int | None = None


class Progress(_Open):
    """Observable liveness, so a caller never has to guess whether a stage is hung."""
    worker_alive: bool
    child_pid: int | None = None
    child_alive: bool | None = None
    started: str | None = None
    elapsed_seconds: float | None = None
    last_activity: str | None = None
    log_tail: list[str] = []
    cancellable: bool = False


class Cancellation(_Open):
    operation_id: str
    status: Literal['cancel_requested']
    signalled: list[int] = []
    warning: str | None = None
    next: str


class JobInspection(_Open):
    """State plus, when operation_id is given, that operation's task and result."""
    job: str
    state: JobState
    hypotheses: list[str] = []
    event_log: str | None = None
    submitted: SubmittedTask | None = None
    operation: OperationResult | None = None
    progress: Progress | None = None


class ArtifactText(_Open):
    path: str
    text: str
    truncated: bool


class AcceptanceRecord(_Open):
    stage: str
    status: Literal['accepted']
    evidence: str


class Handoff(_Open):
    handoff: str
    target_skill: Literal['solve-single-crystal']
    next: str


class Hypothesis(_Open):
    schema_: str = Field(alias='schema')
    id: str
    created: str
    state: str
    parents: list[str] = []


class AnalysisSpec(_Open):
    """What an analysis needs and what it may vary; all fields are agent-facing contract."""
    stage: str
    executable: str
    inputs: list[str] = Field(description='Ordered role names; count and order are enforced')
    optional_inputs: list[str] = []
    expected_output: str
    options: dict[str, list[str]] = Field(default_factory=dict,
                                          description='Only these options apply; other values are rejected')
    acceptance: str = 'process completion requires separate scientific evidence review'
    required_suffixes: list[str] = Field(default_factory=list,
                                         description='Per required input; empty means unchecked')
    optional_suffixes: list[str] = Field(default_factory=list,
                                         description='Per optional input; empty means unchecked')

    @property
    def minimum_inputs(self) -> int:
        return len(self.inputs)

    @property
    def maximum_inputs(self) -> int:
        return len(self.inputs) + len(self.optional_inputs)


class Capabilities(_Open):
    product: str
    version: str
    runtime: dict
    operations: list[str]
    typed_options: dict[str, list[str]]
    acceptance: str
    analyses: dict[str, AnalysisSpec]
    composite_absorption_scaling: str
    tests: str


ANCHORS = 'anchors'
EM = 'em'
ESTIMATORS = ['signed-wls', 'nnls', 'poisson', 'poisson-gaussian']
REFINEMENT_METHODS = [ANCHORS, EM]

# Keyed by (analysis, variant). Variant is the refinement method for joint_refine and
# '' elsewhere. Input counts here are what start_analysis enforces.
ANALYSIS_SPECS: dict[tuple[str, str], AnalysisSpec] = {
    ('soft_assign', ''): AnalysisSpec(
        stage='domains', executable='soft_multi_indexer.py',
        inputs=['observed EXPT', 'spots REFL', 'model EXPT', 'predictions REFL', 'calibration JSON'],
        expected_output='analysis/compatibility.json'),
    ('joint_refine', ANCHORS): AnalysisSpec(
        stage='domains', executable='joint_geometry_refine.py',
        inputs=['model EXPT', 'spots REFL', 'compatibility JSON'],
        expected_output='analysis/joint_refinement.json',
        options={'refinement_method': [ANCHORS]}),
    ('joint_refine', EM): AnalysisSpec(
        stage='domains', executable='joint_geometry_refine.py',
        inputs=['model EXPT', 'spots REFL', 'compatibility JSON', 'selection JSON'],
        optional_inputs=['alternative model EXPT'] * 3,
        expected_output='analysis/joint_refinement.json',
        options={'refinement_method': [EM]},
        required_suffixes=['.expt', '.refl', '.json', '.json'],
        optional_suffixes=['.expt', '.expt', '.expt']),
    ('learn_profiles', ''): AnalysisSpec(
        stage='integrate', executable='learn_profile_library.py',
        inputs=['references REFL', 'calibration JSON'], expected_output='analysis/profiles.json'),
    ('build_overlaps', ''): AnalysisSpec(
        stage='integrate', executable='overlap_prepare.py',
        inputs=['model EXPT', 'full predictions REFL', 'profile library JSON'],
        expected_output='analysis/patches.json',
        options={'position_derivatives': ['true']}),
    ('joint_fit', ''): AnalysisSpec(
        stage='integrate', executable='overlap_integrator.py',
        inputs=['patches manifest JSON'], expected_output='analysis/integrated_groups.jsonl',
        options={'estimator': ESTIMATORS, 'fit_positions': ['true'], 'confidence_level': ['(.5, 1)']}),
    ('export_hklf5', ''): AnalysisSpec(
        stage='export', executable='export_hklf5.py',
        inputs=['corrected observation JSONL'], expected_output='twin.hkl'),
    ('compare_hypotheses', ''): AnalysisSpec(
        stage='domains', executable='compare_hypotheses.py',
        inputs=['fixed partition JSON', 'baseline compatibility JSON', 'alternative compatibility JSON'],
        optional_inputs=['additional compatibility JSON', 'additional compatibility JSON'],
        expected_output='comparison.json'),
    ('scaling_geometry', ''): AnalysisSpec(
        stage='correct', executable='scaling_geometry.py',
        inputs=['integrated groups JSONL', 'model EXPT', 'full predictions REFL', 'symmetry/audit evidence JSON'],
        expected_output='analysis/scaling_geometry.json'),
    ('global_scale', ''): AnalysisSpec(
        stage='correct', executable='global_scale.py',
        inputs=['integrated groups JSONL', 'scaling geometry JSON'],
        expected_output='analysis/scaling.json',
        acceptance='read scaling.json status and rank; converged is not physical validation'),
}


def spec_for(analysis, refinement_method=ANCHORS):
    """Resolve the variant actually run, so callers cannot describe a different one."""
    return ANALYSIS_SPECS.get((analysis, refinement_method if analysis == 'joint_refine' else ''))


def analysis_contracts():
    """Group variants for capabilities: one entry per analysis, variants listed inside."""
    out = {}
    for (analysis, variant), spec in ANALYSIS_SPECS.items():
        out.setdefault(analysis, spec.model_dump(mode='json'))
        if variant:
            out[analysis].setdefault('variants', {})[variant] = spec.model_dump(mode='json')
    return out

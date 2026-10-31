"""MCP tool contracts: published schemas must match what the tools actually return."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest

SRC = Path(__file__).resolve().parents[1] / 'python' / 'src'
sys.path.insert(0, str(SRC))
os.environ.setdefault('ARIADNE_PLUGIN_ROOT', str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SCXRD_RUNTIME_CONFIG', str(Path(__file__).resolve().parents[2] / 'runtime.json'))

from ariadne.contracts import ANALYSIS_SPECS, analysis_contracts, spec_for  # noqa: E402
import ariadne.server as server  # noqa: E402


def call(name, arguments):
    result = asyncio.run(server.mcp.call_tool(name, arguments))
    return result[1] if isinstance(result, tuple) else result


class ContractTableTests(unittest.TestCase):
    def test_every_tool_declares_an_output_schema(self):
        tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
        # The set is asserted to be covered by declarations, not frozen: adding a tool
        # is expected, shipping one without a contract is not.
        self.assertTrue({'capabilities', 'create_job', 'inspect_job', 'read_artifact',
                         'start_operation', 'record_acceptance', 'make_handoff',
                         'start_analysis', 'propose_hypothesis'} <= set(tools))
        for name, tool in tools.items():
            with self.subTest(tool=name):
                self.assertIsNotNone(tool.outputSchema, f'{name} publishes no outputSchema')
                self.assertIn('properties', tool.outputSchema)

    def test_every_tool_returns_fields_its_schema_declares(self):
        declared = {t.name: set(t.outputSchema['properties'])
                    for t in asyncio.run(server.mcp.list_tools())}
        self.assertEqual(declared['start_operation'], {'operation_id', 'worker_pid', 'status', 'next'})
        self.assertIn('progress', declared['inspect_job'])
        self.assertEqual(declared['cancel_operation'],
                         {'operation_id', 'status', 'signalled', 'warning', 'next'})

    def test_declared_input_count_matches_every_registered_analysis(self):
        for (analysis, variant), spec in ANALYSIS_SPECS.items():
            with self.subTest(analysis=analysis, variant=variant):
                self.assertIs(spec_for(analysis, variant or 'anchors'), spec)
                self.assertGreaterEqual(spec.minimum_inputs, 1)
                self.assertLessEqual(spec.minimum_inputs, spec.maximum_inputs)
                self.assertIn(analysis, analysis_contracts())
                self.assertEqual(len(spec.required_suffixes) in (0, spec.minimum_inputs), True)
                if spec.optional_suffixes:
                    self.assertEqual(len(spec.optional_suffixes), len(spec.optional_inputs))

    def test_variant_only_selects_a_different_spec_for_joint_refine(self):
        self.assertIsNone(spec_for('not_an_analysis'))
        # Other analyses have one spec; a non-default variant is then refused by the
        # option-applicability check in start_analysis, not by spec resolution.
        self.assertIs(spec_for('global_scale', 'em'), spec_for('global_scale', 'anchors'))
        self.assertIsNot(spec_for('joint_refine', 'em'), spec_for('joint_refine', 'anchors'))
        self.assertEqual(spec_for('joint_refine', 'em').maximum_inputs, 7)

    def test_capabilities_reports_every_analysis(self):
        body = call('capabilities', {})
        self.assertEqual(set(body['analyses']), {a for a, _ in ANALYSIS_SPECS})
        em = body['analyses']['joint_refine']['variants']['em']
        self.assertEqual(em['inputs'], ['model EXPT', 'spots REFL', 'compatibility JSON', 'selection JSON'])
        self.assertEqual(len(em['optional_inputs']), 3)
        self.assertEqual(body['analyses']['global_scale']['expected_output'], 'analysis/scaling.json')


class ToolBehaviourTests(unittest.TestCase):
    """Schemas must describe the real payload, including fields not spelled out in models."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ariadne_contract_')
        root = Path(self.temporary.name)
        (root / 'inputs').mkdir()
        (root / 'inputs/shot.cbf').write_text('not an image; create_job only copies')
        (root / 'jobs').mkdir()
        self.root = root
        self.job = call('create_job', {'input_path': str(root / 'inputs'),
                                       'jobs_root': str(root / 'jobs')})['job']

    def tearDown(self):
        self.temporary.cleanup()

    def test_job_creation_and_inspection_return_declared_fields(self):
        created = call('create_job', {'input_path': str(self.root / 'inputs'),
                                      'jobs_root': str(self.root / 'jobs')})
        self.assertEqual(set(created), {'job', 'next'})
        seen = call('inspect_job', {'job': self.job})
        # progress is always serialized and stays null without an operation id, matching
        # the published schema rather than disappearing by omission.
        self.assertEqual(set(seen), {'job', 'state', 'hypotheses', 'event_log',
                                     'submitted', 'operation', 'progress'})
        self.assertIsNone(seen['progress'])
        self.assertEqual(seen['state']['schema'], 'scxrd-job/1')
        self.assertEqual(len(seen['state']['stages']), 8)
        self.assertIsNone(seen['operation'])

    def test_artifact_text_reports_truncation(self):
        short = call('read_artifact', {'job': self.job, 'relative_path': 'state.json'})
        self.assertEqual(set(short), {'path', 'text', 'truncated'})
        self.assertFalse(short['truncated'])
        bounded = call('read_artifact', {'job': self.job, 'relative_path': 'state.json', 'max_characters': 10})
        self.assertTrue(bounded['truncated'])
        self.assertEqual(len(bounded['text']), 10)

    def test_analysis_rejects_inputs_the_contract_forbids(self):
        cases = [
            ('too few inputs', {'analysis': 'global_scale', 'inputs': ['state.json']}),
            ('too many inputs', {'analysis': 'global_scale', 'inputs': ['state.json'] * 3}),
            ('wrong suffixes', {'analysis': 'joint_refine', 'refinement_method': 'em',
                                'inputs': ['state.json'] * 4}),
            ('option does not apply', {'analysis': 'global_scale', 'inputs': ['state.json'] * 2,
                                       'estimator': 'nnls'}),
            ('unknown analysis', {'analysis': 'not_an_analysis', 'inputs': ['state.json']}),
        ]
        for label, extra in cases:
            with self.subTest(case=label), self.assertRaises(Exception):
                call('start_analysis', {'job': self.job, **extra})
        self.assertEqual(list((Path(self.job) / 'control').glob('*')), [])


if __name__ == '__main__':
    unittest.main()

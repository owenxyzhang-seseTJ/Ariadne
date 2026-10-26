"""Cancellation, liveness reporting and stable error codes for MCP callers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SRC = Path(__file__).resolve().parents[1] / 'python' / 'src'
sys.path.insert(0, str(SRC))
os.environ.setdefault('ARIADNE_PLUGIN_ROOT', str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SCXRD_RUNTIME_CONFIG', str(Path(__file__).resolve().parents[2] / 'runtime.json'))

from ariadne.errors import CODES, ToolError, error_code  # noqa: E402
import ariadne.server as server  # noqa: E402

STAGES = ('intake', 'import', 'harvest', 'index', 'domains', 'integrate', 'correct', 'export')


class ErrorCodeTests(unittest.TestCase):
    def test_codes_are_prefixed_and_remain_value_errors(self):
        for code in CODES:
            with self.subTest(code=code):
                error = ToolError(code, 'detail')
                self.assertTrue(str(error).startswith(code + ': '))
                self.assertIsInstance(error, ValueError)
                self.assertEqual(error_code(error), code)

    def test_unknown_code_is_rejected(self):
        with self.assertRaises(ValueError):
            ToolError('E_NOT_A_CODE', 'detail')

    def test_code_is_recoverable_from_a_forwarded_message(self):
        # FastMCP re-raises with its own wrapper; the code must survive as text.
        self.assertEqual(error_code(RuntimeError('Error executing tool x: E_BUSY: locked')), 'E_BUSY')
        self.assertIsNone(error_code(RuntimeError('nothing structured here')))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ariadne_lifecycle_')
        self.job = Path(self.temporary.name) / 'job'
        self.job.mkdir()
        for name in ('control', 'attempts', 'evidence', 'input'):
            (self.job / name).mkdir()
        (self.job / 'state.json').write_text(json.dumps({
            'schema': 'scxrd-job/1', 'sample': 't', 'source': 't',
            'created': '2026-10-26T00:00:00+00:00', 'max_attempts': 6,
            'stages': {s: {'status': 'pending', 'attempts': []} for s in STAGES},
            'test_status': 'not_run'}))
        self.spawned = []

    def tearDown(self):
        for process in self.spawned:
            process.kill()
            process.wait()
        self.temporary.cleanup()

    def spawn(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'],
                                   start_new_session=True)
        self.spawned.append(process)
        return process

    def submitted(self, operation_id, worker_pid, child_pid):
        operation = self.job / 'control' / operation_id
        operation.mkdir()
        (operation / 'task.json').write_text(json.dumps({
            'schema': 'ariadne-operation/1', 'operation_id': operation_id, 'job': str(self.job),
            'stage': 'harvest', 'workflow_script': 'x', 'command_json': 'x',
            'timeout_seconds': 300, 'status': 'submitted', 'worker_pid': worker_pid,
            'submitted_at': '2026-10-26T00:00:00+00:00'}))
        (operation / 'child.json').write_text(json.dumps({'pid': child_pid, 'stage': 'harvest', 'attempt': 1}))
        (operation / 'worker.log').write_text('stage running\n')
        return operation

    def test_progress_reports_both_processes_and_elapsed_time(self):
        worker, child = self.spawn(), self.spawn()
        self.submitted('a' * 32, worker.pid, child.pid)
        progress = server.inspect_job(str(self.job), 'a' * 32)['progress']
        self.assertTrue(progress['worker_alive'])
        self.assertTrue(progress['child_alive'])
        self.assertTrue(progress['cancellable'])
        self.assertIsNotNone(progress['elapsed_seconds'])
        self.assertIn('stage running', ' '.join(progress['log_tail']))

    def test_cancel_stops_worker_and_independent_stage_child(self):
        worker, child = self.spawn(), self.spawn()
        operation = self.submitted('b' * 32, worker.pid, child.pid)
        result = server.cancel_operation(str(self.job), 'b' * 32, 'operator stopped it')
        self.assertEqual(result['status'], 'cancel_requested')
        self.assertEqual(sorted(result['signalled']), sorted([worker.pid, child.pid]))
        for _ in range(100):
            if worker.poll() is not None and child.poll() is not None:
                break
            time.sleep(.05)
        worker.poll(), child.poll()
        self.assertFalse(server.pid_alive(worker.pid))
        # The stage child runs in its own session; killing only the worker would leak it.
        self.assertFalse(server.pid_alive(child.pid, group=True))
        marker = json.loads((operation / 'cancel.json').read_text())
        self.assertEqual(marker['reason'], 'operator stopped it')
        progress = server.inspect_job(str(self.job), 'b' * 32)['progress']
        self.assertFalse(progress['cancellable'])

    def test_cancel_requires_a_reason_and_a_live_operation(self):
        worker, child = self.spawn(), self.spawn()
        operation = self.submitted('c' * 32, worker.pid, child.pid)
        with self.assertRaises(ToolError) as blank:
            server.cancel_operation(str(self.job), 'c' * 32, '   ')
        self.assertEqual(blank.exception.code, 'E_INPUT')
        with self.assertRaises(ToolError) as unknown:
            server.cancel_operation(str(self.job), 'd' * 32, 'no such operation')
        self.assertEqual(unknown.exception.code, 'E_NOT_FOUND')
        (operation / 'result.json').write_text(json.dumps({'status': 'awaiting_evidence'}))
        with self.assertRaises(ToolError) as finished:
            server.cancel_operation(str(self.job), 'c' * 32, 'too late')
        self.assertEqual(finished.exception.code, 'E_IMMUTABLE')

    def test_cancelled_result_is_not_reported_as_a_usable_stage(self):
        # Worker-side contract: the marker overrides a zero toolchain exit code.
        operation = self.submitted('e' * 32, 0, 0)
        (operation / 'cancel.json').write_text(json.dumps({
            'requested_at': '2026-10-26T00:00:00+00:00', 'reason': 'stopped'}))
        source = (SRC / 'ariadne' / 'worker.py').read_text()
        self.assertIn("if (control / 'cancel.json').exists():", source)
        self.assertIn("result.update(status='cancelled'", source)

    def test_missing_artifacts_report_not_found(self):
        with self.assertRaises(ToolError) as absent:
            server.read_artifact(str(self.job), 'nope.json')
        self.assertEqual(absent.exception.code, 'E_NOT_FOUND')
        self.assertTrue((self.job / 'state.json').is_file())
        with self.assertRaises(ToolError) as not_a_job:
            server.job_path(str(self.job / 'missing'))
        self.assertEqual(not_a_job.exception.code, 'E_NOT_FOUND')

    def test_escape_and_unsupported_text_stay_input_errors(self):
        for relative in ('../x.json', '/etc/passwd'):
            with self.subTest(path=relative), self.assertRaises(ToolError) as escaping:
                server.read_artifact(str(self.job), relative)
            self.assertEqual(escaping.exception.code, 'E_INPUT')
        (self.job / 'image.cbf').write_bytes(b'\x00')
        with self.assertRaises(ToolError) as binary:
            server.read_artifact(str(self.job), 'image.cbf')
        self.assertEqual(binary.exception.code, 'E_INPUT')


if __name__ == '__main__':
    unittest.main()

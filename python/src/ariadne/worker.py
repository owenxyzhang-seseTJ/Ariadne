"""Background workflow worker; only spawned for a requested real operation."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from ariadne.evidence import append_event


def main():
    task_file = Path(sys.argv[1]).resolve()
    task = json.loads(task_file.read_text())
    argv = [sys.executable, task['workflow_script'], 'run', task['job'], task['stage'],
            task['command_json'], '--timeout', str(task['timeout_seconds'])]
    control = task_file.parent.resolve()
    environment = dict(os.environ, ARIADNE_CONTROL_DIR=str(control))
    try:
        append_event(task['job'], 'operation_started', {'operation_id': task['operation_id'], 'stage': task['stage']})
        with (Path(task['job']) / '.ariadne.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            process = subprocess.run(argv, stdout=sys.stdout, stderr=sys.stderr, check=False, env=environment)
        result = {'operation_id': task['operation_id'], 'returncode': process.returncode,
                  'status': 'awaiting_evidence' if process.returncode == 0 else 'failed'}
        # A cancelled operation still finishes the stage; the marker decides usability,
        # so a toolchain exit code 0 is never reported as a usable result.
        if (control / 'cancel.json').exists():
            marker = json.loads((control / 'cancel.json').read_text())
            result.update(status='cancelled', cancelled_at=marker.get('requested_at'),
                          cancel_reason=marker.get('reason'))
            append_event(task['job'], 'operation_cancelled', result)
    except Exception as error:
        result = {'operation_id': task['operation_id'], 'status': 'worker_failed', 'error': str(error)}
    output = task_file.parent / 'result.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    append_event(task['job'], 'operation_finished', result)


if __name__ == '__main__':
    main()

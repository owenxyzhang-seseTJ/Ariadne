"""Append-only local hypothesis/event records; no hashes or automatic conclusions."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import socket
import uuid


def append_event(job, kind, payload):
    record = {'schema': 'ariadne-event/1', 'id': uuid.uuid4().hex,
              'created': datetime.now(timezone.utc).isoformat(), 'kind': kind,
              'host': socket.gethostname(), 'software': 'Ariadne 0.5.0 / DIALS 3.30.1', 'payload': payload}
    with (Path(job) / 'events.jsonl').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)
    return record['id']


def save_hypothesis(job, record):
    root = Path(job) / 'hypotheses'
    root.mkdir(exist_ok=True)
    identifier = uuid.uuid4().hex
    for parent in record.get('parents', []):
        if len(parent) != 32 or any(c not in '0123456789abcdef' for c in parent):
            raise ValueError('Invalid parent hypothesis identifier')
        if not (root / (parent + '.json')).is_file():
            raise ValueError('Missing parent hypothesis')
    value = {'schema': 'ariadne-hypothesis/1', 'id': identifier,
             'created': datetime.now(timezone.utc).isoformat(), 'state': 'proposed', **record}
    with (root / (identifier + '.json')).open('x') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    append_event(job, 'hypothesis_proposed', {'id': identifier, 'parents': record.get('parents', [])})
    return value
